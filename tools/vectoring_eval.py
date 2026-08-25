"""Vectoring scorecard — the acceptance test for velocity-command policies.

Sweeps commanded directions over a grid (8 azimuths x 5 elevations, plus hold-station), runs
each deterministically to steady state, and scores the achieved velocity against the command:
along-speed, perpendicular leak, direction error, attitude error. This is the standard gate for
the MULTIMODAL problem (the vehicle moves vertically far more easily than horizontally — "drone
mode" vs tangential cruise — and a 3-D-trained policy can silently lose one mode; av_cal2/3_3d
lost horizontal entirely while acing vertical).

Pass guideline (nominal physics, duty<=0.4): dir_err <= 30 deg and v_along >= 0.05 m/s on EVERY
grid cell, hold-station speed <= 0.10 m/s.

Usage:
    python -m tools.vectoring_eval --model models/av_cal4_3d           # final.zip + vecnormalize
    python -m tools.vectoring_eval --model ... --ckpt <steps>          # a specific checkpoint
    python -m tools.vectoring_eval --model ... --speed 0.2 --duty 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

_AZ = (0, 45, 90, 135, 180, -135, -90, -45)
_EL = (-60, -30, 0, 30, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", type=int, default=None, help="checkpoint step count instead of final")
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--duty", type=float, default=0.4, help="esc clamp emulating the node max_duty")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "sim" / "src"))
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

    d = Path(args.model)
    if args.ckpt:
        ck = d / "checkpoints" / f"ppo_{args.ckpt}_steps.zip"
        vec = d / "checkpoints" / f"ppo_vecnormalize_{args.ckpt}_steps.pkl"
    else:
        ck, vec = d / "final.zip", d / "vecnormalize.pkl"

    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["obs_mode"] = "imu"
    cfg["env"]["vel_cmd_horizontal"] = False
    # obs contract of the target bundle: absent in its meta = trained WITHOUT the cap dim
    meta = yaml.safe_load((d / "meta.yaml").read_text()) if (d / "meta.yaml").exists() else {}
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    cfg["domain_rand"] = {"enabled": False}
    model = PPO.load(str(ck), device="cpu")
    vn = VecNormalize.load(str(vec), DummyVecEnv([lambda: UmiusiPoseEnv(cfg)]))
    vn.training = False

    def run(v_cmd):
        env = UmiusiPoseEnv(cfg)
        obs, _ = env.reset(seed=0)
        env.target_quat = np.array([1.0, 0, 0, 0])
        env.v_cmd = np.array(v_cmd, float)
        env.v_cmd_world = env.v_cmd.copy()
        st = env.sim.get_state()
        R, oe = env._errors(st)
        obs = env._get_obs(st, R, oe)
        vs, oris = [], []
        T = int(args.seconds * 50)
        for t in range(T):
            a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
            plant = a.copy()
            plant[4:] = np.clip(plant[4:], -args.duty, args.duty)
            st2 = env.sim.step(plant)
            env.prev_action = a.copy()
            R, oe = env._errors(st2)
            obs = env._get_obs(st2, R, oe)
            env.step_count += 1
            if t > T - 300:
                vs.append(st2["lin_vel"].copy())
                oris.append(np.linalg.norm(oe))
        return np.mean(vs, 0), float(np.degrees(np.mean(oris)))

    print(f"vectoring scorecard: {ck.name}  speed {args.speed} m/s  duty<={args.duty}")
    print("rows: elevation [deg] / cols: azimuth [deg].  cell = dir_err[deg] (v_along[m/s])")
    worst = (0.0, None)
    fails = 0
    print("        " + "".join(f"{az:>13d}" for az in _AZ))
    for el in _EL:
        row = [f"el {el:+4d}:"]
        for az in _AZ:
            a, e = np.radians(az), np.radians(el)
            dhat = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
            v, ori = run(args.speed * dhat)
            va = float(v @ dhat)
            perp = float(np.linalg.norm(v - va * dhat))
            err = float(np.degrees(np.arctan2(perp, max(va, 1e-9))))
            bad = err > 30.0 or va < 0.05
            fails += bad
            if err > worst[0]:
                worst = (err, (az, el))
            row.append(f"{err:5.0f} ({va:+.2f})" + ("!" if bad else " "))
        print(" ".join(row))
    v, ori = run([0.0, 0.0, 0.0])
    hold = float(np.linalg.norm(v))
    hold_bad = hold > 0.10
    fails += hold_bad
    print(f"hold-station: speed {hold:.3f} m/s{' !' if hold_bad else ''}   worst dir_err {worst[0]:.0f} deg at az/el {worst[1]}")
    print(f"{'PASS' if fails == 0 else f'FAIL ({fails} cells out of spec)'}")
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
