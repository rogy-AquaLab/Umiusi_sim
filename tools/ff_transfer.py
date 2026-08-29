"""FF speed transfer check: command sweep x attitude conditions (Umiusi_sim#3 follow-up).

The velocity channel is pure feedforward (no measured velocity in the obs), so two failure
modes are invisible to the averaged eval metrics and must be checked directly:
  1. FLATNESS — the policy cruises at max speed for any nonzero command (av_mode8: commands
     0.03..0.136 all produced ~0.135 m/s; low-speed approaches become uncommandable).
  2. ATTITUDE DEPENDENCE — the same body-frame command yields different speeds at different
     held attitudes (the FF thrust calibration must be attitude-conditioned).

Usage:
    python -m tools.ff_transfer --model models/av_mode9/final.zip [--max-duty 0.25]

Prints along-speed for a command grid at level attitude (transfer curve + monotonicity), and
a command subset across held attitudes (level / pitch+20 / roll+20). PASS guidance: gain in
[0.7, 1.3] over the mid grid, and attitude spread < 20 % of the commanded speed.
"""
import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO

from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP, UmiusiPoseEnv, load_config


def _quat_rp(roll_deg, pitch_deg):
    # sim frame: +Y up; roll about +X, pitch about +Z (REP-103 pitch = sim z)
    q1, q2, q = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(q1, np.array([1.0, 0.0, 0.0]), np.radians(roll_deg))
    mujoco.mju_axisAngle2Quat(q2, np.array([0.0, 0.0, 1.0]), np.radians(pitch_deg))
    mujoco.mju_mulQuat(q, q1, q2)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-duty", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, default=3, help="episodes per grid point")
    ap.add_argument("--seed", type=int, default=700)
    args = ap.parse_args()

    mdir = Path(args.model).parent
    meta = yaml.safe_load((mdir / "meta.yaml").read_text())
    cfg = load_config(meta.get("config", "configs/train_ppo.yaml"))
    for k in ("task", "obs_mode", "proprio_mode", "obs_frame", "action_mode",
              "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg.setdefault("disturbance", {})["enabled"] = False
    env = UmiusiPoseEnv(cfg)
    env.sim.max_duty = args.max_duty
    env._base["max_duty"] = args.max_duty
    model = PPO.load(args.model, device="cpu")
    vn = pickle.load(open(mdir / "vecnormalize.pkl", "rb"))
    rms = vn.obs_rms

    def norm(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + vn.epsilon),
                       -vn.clip_obs, vn.clip_obs).astype(np.float32)

    def run(vset, quat):
        alongs = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            env.target_quat = quat.copy()
            env.v_cmd = np.array([vset, 0.0, 0.0])
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, env.target_quat)
            env.v_cmd_world = Rt.reshape(3, 3) @ env.v_cmd
            done, steps, acc = False, 0, []
            while not done and steps < 450:
                a, _ = model.predict(norm(obs), deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                if steps > 150:  # steady portion only
                    acc.append(info.get("vel_along", 0.0))
                steps += 1
                done = term or trunc
            alongs.append(float(np.mean(acc)))
        return float(np.mean(alongs))

    vmax = 0.8 * VEL_PER_CAP * args.max_duty
    grid = [0.0] + [round(f * vmax, 3) for f in (0.25, 0.5, 0.75, 1.0)]
    level = _quat_rp(0, 0)
    print(f"[ff_transfer] {args.model} @ cap {args.max_duty} (vmax_cmd {vmax:.3f} m/s)")
    print("-- transfer curve (level) --")
    curve = []
    for v in grid:
        a = run(v, level)
        curve.append((v, a))
        gain = f"  gain {a / v:.2f}" if v > 1e-9 else ""
        print(f"  cmd {v:.3f} -> along {a:+.3f} m/s{gain}")
    mono = all(b[1] >= a[1] - 0.01 for a, b in zip(curve[1:], curve[2:]))
    print(f"  monotonic (mid grid, 1cm/s slack): {'yes' if mono else 'NO'}")
    print("-- attitude conditions (cmd = 0.5 * vmax) --")
    v = grid[2]
    speeds = {}
    for name, q in (("level", level), ("pitch+20", _quat_rp(0, 20)), ("roll+20", _quat_rp(20, 0))):
        speeds[name] = run(v, q)
        print(f"  {name:9s}: {speeds[name]:+.3f} m/s")
    spread = max(speeds.values()) - min(speeds.values())
    print(f"  spread {spread:.3f} m/s ({spread / max(v, 1e-9) * 100:.0f}% of cmd; guide < 20%)")
    env.close()


if __name__ == "__main__":
    main()
