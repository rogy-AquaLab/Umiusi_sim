"""Non-learned baseline: geometric allocation + PD attitude + a thrust-model velocity observer.

We have been tuning RL for weeks with NO non-learned baseline to compare against. This is that
baseline. It reuses the parts that are already exact and replaces only the part RL was doing:

    ModeMixer          6-D wrench -> 8-D (servo, esc). Already pure kinematics — atan2 fold,
                       per-unit force split, null modes absent from the basis. NOT relearned.
    this controller    what wrench to command. RL's actual job, done classically instead.

Two pathologies of the learned policy are structural and disappear here by construction:
  * hovering at ~90 % of the esc cap — the required wrench is computed, not discovered;
  * commanding heave UPWARD on a positively buoyant vehicle — buoyancy is a known constant
    (+1.17 N), so the trim term is exact.

The velocity observer exists because the deployed observation has no lateral velocity
(obs = ori_err, gyro, v_cmd, prev_action, max_duty — no DVL, no position). A one-step MLP
cannot integrate the force history, so the learned policy could not estimate drift even though
prev_action carries the information. Integrating it explicitly is cheap and self-limiting:
velocity is drag-dominated, so the estimate converges to a bias set by model error rather than
drifting without bound.

    ACCURACY IS GATED BY CALIBRATION. thrust_per_cmd and thrust_curve_exp are both uncalibrated
    and come from a contaminated fit (docs/calibration_plan.md). The observer cannot be better
    than they are; bench calibration (§3) is what improves it.

Usage:
    python tools/classical_control.py --episodes 12            # hold-station diagnostic
    python tools/classical_control.py --episodes 12 --cruise   # with velocity commands
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))

from umiusi_rl.envs.mode_mixer import MODE_NAMES  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP, UmiusiPoseEnv, load_config  # noqa: E402



class VelocityObserver:
    """v̂ in the BODY frame from the commanded thrust and the hydrodynamic model.

    Uses only what the robot knows: the servo/esc command it just issued, and the plant
    constants. No true velocity, no DVL. Integrates

        v̇ = (f_thrust + f_buoy_body - drag(v)) / m_eff

    with drag(v) = lin*v + quad*|v|*v elementwise, m_eff = mass + added mass. Drag-dominated, so
    this settles rather than drifting; the residual is a bias proportional to the model error.
    """

    def __init__(self, sim, mass, net_buoy_up):
        self.m_eff = mass + sim.added_mass_diag[:3]
        self.lin, self.quad = sim.drag_lin[:3], sim.drag_quad[:3]
        self.net_buoy_up = net_buoy_up
        self.sim = sim
        self.v = np.zeros(3)

    def reset(self):
        self.v[:] = 0.0

    def update(self, action, quat, dt):
        """action = [servo x4, esc x4] as commanded (from prev_action); quat from the AHRS."""
        servo = np.asarray(action[:4]) * self.sim.servo_range_rad
        u = np.asarray(action[4:8])
        thrust = np.sign(u) * np.abs(u) ** self.sim.thrust_curve_exp * self.sim.thrust_per_cmd
        # per unit: horizontal along its tangent, vertical along +Y (the mixer's own idealization)
        f = np.zeros(3)
        for k in range(4):
            t = self.sim.thrust_axes[k]
            f += np.cos(servo[k]) * thrust[k] * t + np.sin(servo[k]) * thrust[k] * np.array([0.0, 1.0, 0.0])
        # net buoyancy acts along WORLD +Y; rotate it into the body frame using the AHRS attitude
        # (the real vehicle has absolute orientation, it is just not in the policy's obs vector).
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
        f += R.reshape(3, 3).T @ np.array([0.0, self.net_buoy_up, 0.0])
        drag = self.lin * self.v + self.quad * np.abs(self.v) * self.v
        self.v += (f - drag) / self.m_eff * dt
        return self.v.copy()


class ClassicalController:
    """obs -> 6-D wrench command. Attitude PD + exact buoyancy trim + observer-corrected cruise."""

    def __init__(self, sim, mass, net_buoy_up, kp=2.2, kd=0.45, k_ff=1.0, k_v=1.2):
        self.kp, self.kd, self.k_ff, self.k_v = kp, kd, k_ff, k_v
        self.fz_trim = -net_buoy_up / (4.0 * sim.thrust_per_cmd * sim.max_duty ** sim.thrust_curve_exp)
        self.obs = VelocityObserver(sim, mass, net_buoy_up)
        self.sim = sim

    def reset(self):
        self.obs.reset()

    def wrench(self, ori_err, gyro, v_cmd_body, v_hat_body, max_duty):
        """All 3-vectors REP-103 body (x fwd, y left, z up). Returns modes in [-1, 1]."""
        # Attitude: PD on the rotation-vector error. ori_err points along the rotation that takes
        # the vehicle to its target, so the moment goes the same way.
        tau = self.kp * ori_err - self.kd * gyro
        # Cruise: feed forward the wrench that holds v_cmd against drag, then correct with the
        # observer. v_hat is the only thing standing in for the missing DVL.
        v_ref = float(VEL_PER_CAP) * max_duty
        ff = self.k_ff * v_cmd_body[:2] / max(v_ref, 1e-9)
        fb = self.k_v * (v_cmd_body[:2] - v_hat_body[:2]) / max(v_ref, 1e-9)
        f_xy = ff + fb
        return np.clip([f_xy[0], f_xy[1], self.fz_trim, tau[0], tau[1], tau[2]], -1.0, 1.0)


def _rep103(v_sim):
    """sim/CAD (+Y up) -> REP-103 (x fwd, y left, z up)."""
    return np.array([v_sim[0], -v_sim[2], v_sim[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--cruise", action="store_true", help="sample velocity commands (default: hold)")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--domain-rand", action="store_true",
                    help="model mismatch — the regime RL was trained for; the fair comparison")
    ap.add_argument("--kp", type=float, default=2.2)
    ap.add_argument("--kd", type=float, default=0.45)
    ap.add_argument("--k-v", type=float, default=1.2)
    args = ap.parse_args()

    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="modes", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = args.domain_rand
    cfg.setdefault("disturbance", {})["enabled"] = args.disturb
    env = UmiusiPoseEnv(cfg)
    if not args.cruise:
        env.vel_cmd_zero_prob = 1.0

    import yaml
    c = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    mass = c["hull"]["mass"] + 4 * c["thrusters"]["mass"]
    g = abs(c["sim"]["gravity"][1])
    net_buoy = c["water"]["density"] * c["water"]["displaced_volume"] * g - mass * g

    ctl = ClassicalController(env.sim, mass, net_buoy, kp=args.kp, kd=args.kd, k_v=args.k_v)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    step = env._mode_slew_step or 1.0

    signed, absm, esc, ori, drift, verr = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        m = np.zeros(6)
        done = False
        while not done:
            # obs は実機が持つものだけ: [ori_err 3][gyro 3][v_cmd 3][prev_action 8][max_duty 1]
            ori_err, gyro, v_cmd_o = obs[0:3], obs[3:6], obs[6:9]
            prev_action, cap = obs[9:17], float(obs[17])
            v_hat_sim = ctl.obs.update(prev_action, env.sim.get_state()["quat"], dt)
            m_des = ctl.wrench(ori_err, gyro, v_cmd_o, _rep103(v_hat_sim), cap)
            rate = np.clip((m_des - m) / step, -1.0, 1.0)
            obs, _r, term, trunc, info = env.step(rate)
            m = env._mode_prev_modes.copy()
            signed.append(m)
            absm.append(np.abs(m))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            verr.append(float(np.linalg.norm(v_hat_sim - env.sim.get_state()["lin_vel"])))
            done = term or trunc
    env.close()

    sg, ab = np.array(signed).mean(0), np.array(absm).mean(0)
    print(f"classical  kp={args.kp} kd={args.kd} k_v={args.k_v}  "
          f"{'cruise' if args.cruise else 'hold-station'}  disturb={args.disturb} DR={args.domain_rand}")
    print(f"  median|esc| {np.mean(esc):.4f}   ori {np.mean(ori):.3f} rad   横流れ {np.mean(drift):.4f} m/s")
    print(f"  観測器の速度誤差 {np.mean(verr):.4f} m/s   (真値との差、較正で決まる)")
    for i, n in enumerate(MODE_NAMES):
        r = abs(sg[i]) / ab[i] if ab[i] > 1e-6 else 0.0
        print(f"   {n:3s}  符号付き {sg[i]:+.3f}   |m| {ab[i]:.3f}   比 {r:.2f}")


if __name__ == "__main__":
    main()
