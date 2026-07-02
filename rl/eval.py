"""Evaluate / watch a trained go-to-pose policy on UmiusiPoseEnv.

Usage:
    python -m rl.eval --model models/ppo/final.zip                 # headless metrics
    python -m rl.eval --model models/ppo/final.zip --episodes 20   # more episodes
    python -m rl.eval --model models/ppo/final.zip --render        # watch in the GUI viewer

The algorithm and config are read from the run's meta.yaml when present; override with
--algo / --config. Reports per-episode return, final position error, and the fraction of
steps spent within the goal tolerance (station-keeping quality).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC, TD3

from rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="path to a trained policy .zip")
    ap.add_argument("--config", default=None, help="env config (default: from run meta.yaml)")
    ap.add_argument("--algo", choices=list(ALGOS), default=None, help="default: from run meta.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="watch in the MuJoCo GUI viewer")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    algo = args.algo or meta.get("algo", "ppo")
    config = args.config or meta.get("config", "configs/train_ppo.yaml")

    cfg = load_config(config)
    for k in ("task", "obs_mode"):  # match the task + sensor suite the policy was trained with
        if k in meta:
            cfg["env"][k] = meta[k]
    env = UmiusiPoseEnv(cfg, render_mode="human" if args.render else None)
    control_dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    model = ALGOS[algo].load(str(model_path), device="cpu")

    returns, pos_errs, ori_errs, depth_errs, hold_fracs, successes = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret, steps, in_tol = 0.0, 0, 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            in_tol += int(info.get("is_success", False))
            if args.render:
                env.render()
                time.sleep(control_dt)
            done = terminated or truncated
        returns.append(ep_ret)
        pos_errs.append(info["pos_err"])
        ori_errs.append(info["ori_err"])
        depth_errs.append(abs(info["depth_err"]))
        hold_fracs.append(in_tol / max(steps, 1))
        successes.append(info.get("is_success", False))
        print(f"ep {ep:2d}: return={ep_ret:8.1f}  ori_err={info['ori_err']:.3f} rad  "
              f"pos_err={info['pos_err']:.3f} m  depth_err={abs(info['depth_err']):.3f} m  "
              f"hold={hold_fracs[-1] * 100:4.0f}%")

    env.close()
    print("-" * 64)
    print(f"episodes={args.episodes}  task={meta.get('task', '?')}  algo={algo}  obs_mode={meta.get('obs_mode', '?')}")
    print(f"mean return        : {np.mean(returns):8.1f} +/- {np.std(returns):.1f}")
    print(f"mean final ori err : {np.mean(ori_errs):.3f} rad")
    print(f"mean final pos err : {np.mean(pos_errs):.3f} m")
    print(f"mean final depth err: {np.mean(depth_errs):.3f} m")
    print(f"mean hold fraction : {np.mean(hold_fracs) * 100:.0f}%   (steps within tolerance)")
    print(f"final-step success : {np.mean(successes) * 100:.0f}%")


if __name__ == "__main__":
    main()
