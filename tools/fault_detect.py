"""Can a dead thruster be identified online, from the sensors the robot actually has?

This decides the architecture. If the fault is identifiable analytically, the classical allocator
can simply be told and re-solve (measured: full 6-DOF authority survives on three units), and RL
is not needed for it. If it is not identifiable, we need a learned estimator with memory — and a
one-step MLP will not do, because identification requires integrating a history.

The asymmetry that makes this tractable: TRANSLATION is unobservable (no DVL) but ROTATION is not.
A dead unit removes its moment contribution, and that shows up in the gyro within a few steps.

Method — a bank of hypotheses (healthy, unit k dead). Over a sliding window, each hypothesis
predicts the angular-momentum change its commanded moment would produce:

    I dω/dt = M_hypothesis(commanded forces) - rotational drag(ω)

Score each by squared error against the measured Δω and take the best. No learning, no tuning
beyond the window length.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from classical_control import ClassicalController, GeneralAllocator  # noqa: E402
from fault_compare import _plant_fault  # noqa: E402
from classical_control import _rep103  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402


class FaultBank:
    """Sliding-window multiple-model identification of a single dead thruster."""

    def __init__(self, sim, r, window=25):
        self.sim, self.r, self.window = sim, r, window
        # 系の慣性: base_link 単体ではなくスラスタ 4 基を含めた合成 (平行軸)。
        # 共通の誤差でも仮説間の識別性を鈍らせるので、ここは合わせておく。
        inertia = np.array([sim.model.body_inertia[sim.base_id][i] for i in range(3)], dtype=float)
        m_t = float(sim.model.body_mass[sim.thr_ids[0]])
        for k in range(4):
            d = np.asarray(r[k], dtype=float)
            inertia += m_t * (np.dot(d, d) - d * d)
        self.I = inertia
        self.drag_ang = sim.drag_lin[3:6], sim.drag_quad[3:6]
        self.buf = []                      # (per-unit force vectors, omega)

    def reset(self):
        self.buf.clear()

    def _moment(self, f_units, drop):
        m = np.zeros(3)
        for k in range(4):
            if k == drop:
                continue
            m += np.cross(self.r[k], f_units[k])
        return m

    def push(self, action, omega):
        """action = the command AS ISSUED; omega = measured body angular rate (sim frame)."""
        servo = np.asarray(action[:4]) * self.sim.servo_range_rad
        u = np.asarray(action[4:8])
        th = np.sign(u) * np.abs(u) ** self.sim.thrust_curve_exp * self.sim.thrust_per_cmd
        y = np.array([0.0, 1.0, 0.0])
        f = np.array([np.cos(servo[k]) * th[k] * self.sim.thrust_axes[k] + np.sin(servo[k]) * th[k] * y
                      for k in range(4)])
        self.buf.append((f, np.asarray(omega, dtype=float).copy()))
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def identify(self, dt):
        """-> (best hypothesis: None or unit index, separation margin). None until the window fills."""
        if len(self.buf) < self.window:
            return None, 0.0
        lin, quad = self.drag_ang
        d_omega = self.buf[-1][1] - self.buf[0][1]
        scores = []
        for drop in (None, 0, 1, 2, 3):
            pred = np.zeros(3)
            for f, w in self.buf[:-1]:
                m = self._moment(f, drop) - (lin * w + quad * np.abs(w) * w)
                pred += m / self.I * dt
            scores.append(float(np.sum((pred - d_omega) ** 2)))
        order = np.argsort(scores)
        best = order[0]
        margin = (scores[order[1]] - scores[best]) / max(scores[order[1]], 1e-12)
        return (None if best == 0 else best - 1), margin


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--cruise", action="store_true",
                    help="速度指令あり = 励振あり。同定には persistent excitation が要る")
    args = ap.parse_args()

    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="esc", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = args.dr
    cfg.setdefault("disturbance", {})["enabled"] = False
    c = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    mass = c["hull"]["mass"] + 4 * c["thrusters"]["mass"]
    g = abs(c["sim"]["gravity"][1])
    net_buoy = c["water"]["density"] * c["water"]["displaced_volume"] * g - mass * g

    print(f"window={args.window} steps ({args.window / 50:.2f} s)  DR={args.dr}  "
          f"{'cruise(励振あり)' if args.cruise else 'hold(励振なし)'}  episodes={args.episodes}")
    print(f"{'真の故障':<12}{'正答率':>8}{'検出まで':>10}{'分離マージン':>12}")
    for truth in (None, 0, 1, 2, 3):
        env = UmiusiPoseEnv(cfg)
        if not args.cruise:
            env.vel_cmd_zero_prob = 1.0
        ctl = ClassicalController(env.sim, mass, net_buoy)
        alloc = GeneralAllocator(env.sim)
        bank = FaultBank(env.sim, alloc.r, window=args.window)
        f_max_tot = 4.0 * env.sim.thrust_per_cmd * env.sim.max_duty ** env.sim.thrust_curve_exp
        dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
        hits, tot, first, margins = 0, 0, [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=5000 + ep)
            ctl.reset()
            bank.reset()
            w = np.zeros(6)
            step, detected_at = 0, None
            done = False
            while not done:
                v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
                m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
                w_des = np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f_max_tot
                w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)
                a = alloc.allocate(w, env.sim.max_duty)
                bank.push(a, env.sim.get_state()["ang_vel"])
                obs, _r, term, trunc, _i = env.step(_plant_fault(a, truth, 0.0, env.sim.servo_range_rad))
                guess, margin = bank.identify(dt)
                if step >= args.window:
                    tot += 1
                    hits += int(guess == truth)
                    margins.append(margin)
                    if guess == truth and detected_at is None:
                        detected_at = step
                step += 1
                done = term or trunc
            if detected_at is not None:
                first.append(detected_at / 50.0)
        env.close()
        lbl = "健全" if truth is None else f"{env.sim.unit_names[truth]} 故障"
        ft = f"{np.mean(first):.2f} s" if first else "検出せず"
        print(f"{lbl:<12}{hits / max(tot, 1) * 100:7.0f}%{ft:>10}{np.mean(margins):12.3f}")


if __name__ == "__main__":
    main()
