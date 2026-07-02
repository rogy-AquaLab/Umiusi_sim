"""Train a go-to-pose policy on UmiusiPoseEnv (algorithm-agnostic; PPO default).

Usage:
    python -m rl.train                                   # PPO, config defaults
    python -m rl.train --config configs/train_ppo.yaml   # explicit config
    python -m rl.train --algo sac --timesteps 200000     # switch algorithm
    python -m rl.train --n-envs 12 --run-name ppo_v1     # more parallel envs, named run

Artifacts (all under the gitignored models/<run-name>/):
    final.zip          trained policy
    checkpoints/       periodic checkpoints
    tb/                tensorboard logs   (tensorboard --logdir models/<run-name>/tb)
    meta.yaml          algo + config, so eval can reload without extra flags
"""

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl.envs.umiusi_pose_env import _DEFAULT_OBS, UmiusiPoseEnv, load_config

_ROOT = Path(__file__).resolve().parents[1]
ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def build_model(algo, cfg, venv, seed, tb_dir):
    policy_kwargs = {"net_arch": cfg["policy"]["net_arch"]}
    common = {"policy": "MlpPolicy", "env": venv, "seed": seed, "verbose": 1,
              "device": "cpu", "tensorboard_log": str(tb_dir), "policy_kwargs": policy_kwargs}
    if algo == "ppo":
        p = cfg["ppo"]
        return PPO(n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
                   gamma=p["gamma"], gae_lambda=p["gae_lambda"], ent_coef=p["ent_coef"],
                   learning_rate=p["learning_rate"], clip_range=p["clip_range"], **common)
    # SAC / TD3: off-policy, use SB3 defaults (they ignore PPO-only knobs) + shared net_arch.
    return ALGOS[algo](**common)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/train_ppo.yaml")
    ap.add_argument("--algo", choices=list(ALGOS), default=None, help="override config algo")
    ap.add_argument("--task", choices=list(_DEFAULT_OBS), default=None,
                    help="override task (attitude | attitude_depth | pose)")
    ap.add_argument("--obs-mode", choices=["auto", "full", "imu", "imu_depth", "imu_depth_dvl"],
                    default=None, help="override sensor suite (env.obs_mode)")
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--n-envs", type=int, default=None, help="override number of parallel envs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--vec", choices=["auto", "dummy", "subproc"], default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg["env"]["task"] = args.task
    task = cfg["env"].get("task", "pose")
    if args.obs_mode:
        cfg["env"]["obs_mode"] = args.obs_mode
    obs_mode = cfg["env"].get("obs_mode", "auto")
    if obs_mode == "auto":
        obs_mode = _DEFAULT_OBS[task]
    cfg["env"]["obs_mode"] = obs_mode  # store the resolved suite so eval matches it
    algo = args.algo or cfg.get("algo", "ppo")
    total_timesteps = args.timesteps or cfg["total_timesteps"]
    n_envs = args.n_envs or cfg["n_envs"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    run_name = args.run_name or algo

    run_dir = _ROOT / "models" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    vec_cls = DummyVecEnv if (args.vec == "dummy" or (args.vec == "auto" and n_envs == 1)) else SubprocVecEnv
    venv = make_vec_env(UmiusiPoseEnv, n_envs=n_envs, seed=seed,
                        env_kwargs={"config": cfg}, vec_env_cls=vec_cls)

    model = build_model(algo, cfg, venv, seed, run_dir / "tb")

    # Checkpoint ~10x over the run (save_freq is per-env steps).
    save_freq = max(total_timesteps // (10 * n_envs), 1)
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=str(run_dir / "checkpoints"),
                              name_prefix=algo)

    print(f"[train] task={task} algo={algo} obs_mode={obs_mode} n_envs={n_envs} "
          f"timesteps={total_timesteps} seed={seed} -> {run_dir}")
    model.learn(total_timesteps=total_timesteps, callback=ckpt)

    model.save(str(run_dir / "final"))
    with open(run_dir / "meta.yaml", "w") as f:
        yaml.safe_dump({"algo": algo, "task": task, "obs_mode": obs_mode, "config": args.config,
                        "seed": seed, "total_timesteps": total_timesteps}, f)
    venv.close()
    print(f"[train] done. policy -> {run_dir / 'final.zip'}")
    print(f"[train] eval:  python -m rl.eval --model {run_dir / 'final.zip'}")


if __name__ == "__main__":
    main()
