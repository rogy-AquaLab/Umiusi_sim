"""Can the controller work out ITS OWN degradation online — not "which unit died", but "how much
thrust is each unit actually making"?

Failure is the easy, rare case. The common one is drift: a fouled prop, a tired cell, a servo that
has stiffened. tools/dr_ablation.py --loo says this is worth more than the failure case anyway —
pinning per-unit thrust gain (±10 % in DR) takes the attitude error to 0.65x, second only to added
mass. And unlike "which of five discrete hypotheses is true", a gain is a CONTINUOUS quantity that
biases the residual persistently, so it should be far better conditioned than fault ID was
(measured 12-52 % correct in hold, i.e. barely above the 20 % chance level).

Same physics as tools/fault_detect.py, different estimator. Over a window,

    I dw/dt  =  sum_k g_k * (r_k x f_k)  -  drag(w)

is LINEAR in the per-unit gains g, so recursive least squares recovers them without any bank of
hypotheses — and a dead unit is just g_k -> 0, which makes this a strict generalisation. What it
cannot escape is the excitation limit: if the commanded moments do not span enough directions, the
regressor is rank-deficient and no estimator, learned or not, can separate the units. That is the
thing to measure, and it is the same wall fault ID hit.

    python tools/gain_adapt.py --episodes 6 --dr
    python tools/gain_adapt.py --episodes 6 --dr --cruise --prefer-deg 60
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from umiusi_perception.classical import cad_wrench_from_modes  # noqa: E402
from classical_control import (GeneralAllocator, _rep103, body_inertia_tensor,  # noqa: E402
                               build_controller, nominal_plant)  # noqa: E402
from fault_compare import _env, _plant_fault  # noqa: E402


class GainEstimator:
    """Windowed least squares on the per-unit thrust gains, from the gyro residual only.

    NOT a per-step finite difference of omega: at 50 Hz that is differentiation noise and the
    estimate chases it (measured, the first attempt was worse than the 1.0 prior even on a healthy
    vehicle). Integrate instead, exactly as tools/fault_detect.py does — over a window,

        w(t2) - w(t1)  =  sum_steps [ (sum_k g_k * r_k x f_k) / I - drag(w)/I ] dt

    is still LINEAR in g, so each window contributes three well-conditioned rows instead of three
    noisy ones. Ridge toward g = 1 keeps it honest when the window carries no information about a
    unit, which is the normal case in hold-station.
    """

    def __init__(self, sim, r, window=50, ridge=1.0):
        self.I, self.r = body_inertia_tensor(sim, r), r
        self.Iinv = np.linalg.inv(self.I)
        self.lin, self.quad = sim.drag_lin[3:6].copy(), sim.drag_quad[3:6].copy()
        self.plant = nominal_plant(sim)     # calibrated constants only — never the DR truth
        # BUOYANCY RIGHTING MOMENT. The CoB sits above the CoM, so any tilt produces a restoring
        # couple m*g*h*sin(tilt) — and the task commands tilts up to 45 deg. Omitting it left the
        # model explaining only 55 % of the measured dw (relative residual 0.45 with a healthy
        # vehicle, no DR and the TRUE actuator state fed in), and a gain estimator charges the
        # difference to the thrusters.
        c = sim.cfg
        self.f_buoy = float(c["water"]["density"] * c["water"]["displaced_volume"]
                            * abs(c["sim"]["gravity"][1]))
        self.h_cob = float(sim.buoyancy_offset)
        self.window, self.ridge = window, ridge
        self.reset()

    def reset(self):
        self.g = np.ones(4)
        self.buf = []                       # (H [3,4] per-unit angular accel at gain 1, omega)
        self.AtA = np.eye(4) * self.ridge   # ridge prior toward g = 1
        self.Atb = np.ones(4) * self.ridge

    def _H(self, action, actual=None):
        """actual = (servo_rad, thrust_N) measured on the plant. DIAGNOSTIC ONLY — the real vehicle
        has no servo position feedback. It exists to separate two very different failure causes:
        an estimator that cannot see the information, versus one whose regressor is simply wrong
        because the COMMAND is not what the actuator did (23 deg of servo lag before the
        singularity fix, and an ESC that ramps at 4 units/s)."""
        p = self.plant
        if actual is not None:
            servo, th = np.asarray(actual[0], float), np.asarray(actual[1], float)
        else:
            servo = np.asarray(action[:4]) * p.servo_range_rad
            u = np.asarray(action[4:8])
            th = np.sign(u) * np.abs(u) ** p.thrust_curve_exp * p.thrust_per_cmd
        y = np.array([0.0, 1.0, 0.0])
        cols = [np.cross(self.r[k], np.cos(servo[k]) * th[k] * p.thrust_axes[k]
                         + np.sin(servo[k]) * th[k] * y) for k in range(4)]
        return self.Iinv @ np.stack(cols, axis=1)       # [3, 4] angular accel per unit gain

    def _restoring(self, quat):
        """Body-frame moment from buoyancy acting a distance h above the CoM."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
        f_body = R.reshape(3, 3).T @ np.array([0.0, self.f_buoy, 0.0])
        return self.Iinv @ np.cross(np.array([0.0, self.h_cob, 0.0]), f_body)

    def update(self, action, omega_world, quat, dt, actual=None):
        """omega_world = sim.get_state()["ang_vel"], which mj_objectVelocity returns in the WORLD
        frame (flg_local=0). r, thrust_axes and the inertia are all body-local, so it has to be
        rotated first — feeding the world vector straight in silently mixes frames and the
        estimate is then worse than its own prior."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
        omega = R.reshape(3, 3).T @ np.asarray(omega_world, dtype=float)
        self.buf.append((self._H(action, actual), np.asarray(omega, dtype=float).copy(),
                         self._restoring(quat)))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if len(self.buf) == self.window:
            A = np.zeros((3, 4))
            known = np.zeros(3)                     # everything not proportional to g
            for H, w, mr in self.buf[:-1]:
                A += H * dt
                known += (self.Iinv @ (self.lin * w + self.quad * np.abs(w) * w) - mr) * dt
            b = (self.buf[-1][1] - self.buf[0][1]) + known
            self.AtA += A.T @ A
            self.Atb += A.T @ b
            self.g = np.clip(np.linalg.solve(self.AtA, self.Atb), 0.0, 2.0)
        return self.g.copy()


def run(episodes, dr, disturb, cruise, alloc_kw, gains, truth_gain, oracle_act=False):
    env = _env("esc", dr, disturb)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    est = GainEstimator(env.sim, alloc.r)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    err_end, err_half, cond = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        alloc.reset()
        est.reset()
        env._thrust_gain = np.asarray(truth_gain, dtype=float).copy()   # the degradation to find
        w, step, H_log = np.zeros(6), 0, []
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            f_max_tot = ctl.f_max_total(ctl.cap)
            w_des = cad_wrench_from_modes(m, f_max_tot)
            w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)
            a = alloc.allocate(w, ctl.cap)
            st = env.sim.get_state()
            act = (st["servo"], st["thrust"]) if oracle_act else None
            g_hat = est.update(a, st["ang_vel"], st["quat"], dt, actual=act)
            H_log.append(a[4:8].copy())
            obs, _r, term, trunc, _i = env.step(_plant_fault(a, None, 0.0, env.sim.servo_range_rad))
            step += 1
            if step == 300:
                err_half.append(np.abs(g_hat - truth_gain).mean())
            done = term or trunc
        err_end.append(np.abs(est.g - truth_gain).mean())
        s = np.linalg.svd(np.asarray(H_log), compute_uv=False)
        cond.append(s[0] / max(s[-1], 1e-12))
    env.close()
    return np.mean(err_half) if err_half else float("nan"), np.mean(err_end), np.mean(cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--cruise", action="store_true")
    ap.add_argument("--oracle-actuator", action="store_true",
                    help="feed the estimator the TRUE servo angle and thrust (diagnostic)")
    ap.add_argument("--prefer-deg", type=float, default=None)
    ap.add_argument("--w-move", type=float, default=3.0)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    args = ap.parse_args()
    alloc_kw = {} if args.prefer_deg is None else {"prefer_deg": args.prefer_deg, "w_move": args.w_move}
    gains = {"kp": args.kp, "kd": args.kd}

    cases = [("健全 (1,1,1,1)", [1.0, 1.0, 1.0, 1.0]),
             ("lf 70% 劣化", [0.7, 1.0, 1.0, 1.0]),
             ("lf 死 (0)", [0.0, 1.0, 1.0, 1.0]),
             ("全体ばらつき", [0.9, 1.1, 0.85, 1.05])]
    print(f"ユニット推力ゲインのオンライン推定 (RLS)  DR={args.dr} disturb={args.disturb} "
          f"{'cruise' if args.cruise else 'hold'}  alloc={alloc_kw or 'minnorm'}  ep={args.episodes}")
    print(f"{'真のゲイン':<18}{'6s後の誤差':>12}{'12s後':>10}{'指令の条件数':>14}")
    for label, truth in cases:
        half, end, cond = run(args.episodes, args.dr, args.disturb, args.cruise,
                              alloc_kw, gains, truth, args.oracle_actuator)
        print(f"{label:<18}{half:12.3f}{end:10.3f}{cond:14.1f}")
    print("\n  誤差 = |推定 - 真値| の 4 基平均 (0 = 完全、~0.1 = DR のばらつき幅と同程度で無意味)。"
          "\n  条件数が大きいほど指令が特定方向に偏っていて分離できない = 励振不足。")


if __name__ == "__main__":
    main()
