"""Train a go-to-pose policy on UmiusiPoseEnv (algorithm-agnostic; PPO default).

Usage:
    python -m umiusi_rl.train                                   # PPO, config defaults
    python -m umiusi_rl.train --config configs/train_ppo.yaml   # explicit config
    python -m umiusi_rl.train --algo sac --timesteps 200000     # switch algorithm
    python -m umiusi_rl.train --n-envs 12 --run-name ppo_v1     # more parallel envs, named run

Artifacts (all under the gitignored models/<run-name>/):
    final.zip          trained policy
    checkpoints/       periodic checkpoints
    tb/                tensorboard logs   (tensorboard --logdir models/<run-name>/tb)
    meta.yaml          algo + config, so eval can reload without extra flags
"""

import argparse
import pickle
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from umiusi_rl.envs.umiusi_pose_env import _DEFAULT_OBS, UmiusiPoseEnv, load_config

_ROOT = Path(__file__).resolve().parents[4]        # repo root (packages/sim/src/umiusi_rl/..)
ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


class CurriculumCallback(BaseCallback):
    """Widen the attitude_velocity difficulty (v_cmd direction cone + yaw range) from 0 to the
    config targets over the first `frac` of training, so the policy first learns to cruise in a
    single direction (easy) and then generalizes. Avoids the from-scratch do-nothing local optimum."""

    def __init__(self, total, cone_max, yaw_max, tilt_max, frac, elev_max=0.0):
        super().__init__()
        self.total, self.cone_max, self.yaw_max, self.tilt_max, self.frac = total, cone_max, yaw_max, tilt_max, frac
        # 3-D velocity commands: the elevation range also ramps 0 -> elev_max (vel_cmd_elev_deg).
        # Ramping matters here for MULTIMODALITY: vertical ("drone-mode") locomotion is much
        # easier than tangential horizontal cruise, and a policy exposed to wide elevations too
        # early collapses into the vertical basin (av_cal2/3_3d lesson).
        self.elev_max = elev_max
        self._last_pct = -1

    def _on_step(self):
        p = min(1.0, self.num_timesteps / max(self.frac * self.total, 1.0))
        pct = int(p * 100)
        if pct != self._last_pct:  # throttle the set_attr IPC to ~100 updates
            self._last_pct = pct
            self.training_env.set_attr("vel_cmd_cone_deg", p * self.cone_max)
            self.training_env.set_attr("yaw_target_deg", p * self.yaw_max)
            self.training_env.set_attr("tilt_target_deg", p * self.tilt_max)
            if self.elev_max > 0.0:
                self.training_env.set_attr("vel_cmd_elev_deg", p * self.elev_max)
        return True


class EconRampCallback(BaseCallback):
    """Ramp the economy penalties (w_effort + w_null, via env.econ_ramp) 0 -> 1 over the first
    `frac` of training: learn/keep the task first, then economize. A strong effort penalty from
    step 0 collapses into the do-nothing local optimum (att_v3-era lesson)."""

    def __init__(self, total, frac):
        super().__init__()
        self.total, self.frac = total, frac
        self._last_pct = -1

    def _on_step(self):
        p = min(1.0, self.num_timesteps / max(self.frac * self.total, 1.0))
        pct = int(p * 100)
        if pct != self._last_pct:  # throttle the set_attr IPC
            self._last_pct = pct
            self.training_env.set_attr("econ_ramp", p)
        return True


class CapRangeCurriculumCallback(BaseCallback):
    """Widen the DR esc-cap range from [hi, hi] down to [lo, hi] over the first `frac` of training.

    At the low cap (0.2 -> 4.8 N total thrust) cruise is barely discoverable from scratch — the
    av_cap3/4 runs learned clean allocation but never formed the cruise skill (along 0.016/0.006
    m/s). Let the skill FORM at the easy cap first; the cap is observed (observe_max_duty), so the
    policy can then specialize downward instead of averaging over caps it cannot cruise at."""

    def __init__(self, total, base_dr, lo, hi, frac):
        super().__init__()
        self.total, self.base_dr, self.lo, self.hi, self.frac = total, base_dr, lo, hi, frac
        self._last_pct = -1

    def _on_step(self):
        p = min(1.0, self.num_timesteps / max(self.frac * self.total, 1.0))
        pct = int(p * 100)
        if pct != self._last_pct:
            self._last_pct = pct
            lo_now = self.hi - p * (self.hi - self.lo)
            self.training_env.set_attr("dr", {**self.base_dr, "max_duty_range": [lo_now, self.hi]})
        return True


def warm_start(model, init_path, venv):
    """Copy policy/value weights (and optimizer-free state) from a previous run into `model`.

    Handles a GROWN observation vector (e.g. observe_max_duty appends 1 dim): new obs dims must be
    appended LAST, then every first-layer weight matrix is zero-padded on the input side — the
    loaded policy initially IGNORES the new inputs and learns to use them, keeping everything it
    knows. Also returns the previous VecNormalize obs stats zero-padded the same way (mean 0 /
    var 1 for new dims, so the new obs pass through unscaled at first).
    """
    init_path = Path(init_path)
    zip_path = init_path if init_path.suffix == ".zip" else init_path / "final.zip"
    old = type(model).load(str(zip_path), device="cpu")
    old_dim = int(old.observation_space.shape[0])
    new_dim = int(model.observation_space.shape[0])
    if old.action_space != model.action_space:
        raise ValueError(f"action space mismatch: {old.action_space} vs {model.action_space}")
    if new_dim < old_dim:
        raise ValueError(f"obs shrank {old_dim} -> {new_dim}; warm start only supports growth")
    pad = new_dim - old_dim
    sd, new_sd = old.policy.state_dict(), model.policy.state_dict()
    for k, w in sd.items():
        if pad and w.dim() == 2 and w.shape[1] == old_dim and new_sd[k].shape[1] == new_dim:
            w = torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype)], dim=1)
        if new_sd[k].shape != w.shape:
            raise ValueError(f"cannot adapt {k}: {tuple(w.shape)} -> {tuple(new_sd[k].shape)}")
        new_sd[k] = w
    model.policy.load_state_dict(new_sd)

    stats_path = zip_path.parent / "vecnormalize.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:   # unpickle directly: VecNormalize.load would demand a
            old_vn = pickle.load(f)          # matching (old-dim) env just to read the stats
        rms = old_vn.obs_rms
        if pad:
            import numpy as np
            rms.mean = np.concatenate([rms.mean, np.zeros(pad)])
            rms.var = np.concatenate([rms.var, np.ones(pad)])
        venv.obs_rms = rms
        venv.ret_rms = old_vn.ret_rms
        # FREEZE the normalization: the policy's input scaling must not drift under the new
        # reward/plant, or the transferred weights see a shifting input distribution.
        venv.training = False
        return True
    return False


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
                    help="override task (attitude | attitude_depth | attitude_velocity | pose)")
    ap.add_argument("--obs-mode", choices=["auto", "full", "imu", "imu_depth", "imu_depth_dvl"],
                    default=None, help="override sensor suite (env.obs_mode)")
    ap.add_argument("--vel-cone", type=float, default=None, help="override env.vel_cmd_cone_deg [deg]")
    ap.add_argument("--yaw-target", type=float, default=None, help="override env.yaw_target_deg [deg]")
    ap.add_argument("--curriculum-frac", type=float, default=None,
                    help="attitude_velocity: widen cone/yaw 0->config over this fraction of training (0=off)")
    ap.add_argument("--disturb", action="store_true", help="enable disturbances (water current + impulses)")
    ap.add_argument("--domain-rand", action="store_true",
                    help="enable domain randomization (buoyancy/thrust/drag + obs noise + action latency; sim2real)")
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a previous run (dir with final.zip, or a .zip): copies the "
                         "policy weights (zero-padding first layers if the obs vector GREW, e.g. "
                         "observe_max_duty) and loads + FREEZES its VecNormalize stats")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="override ppo.learning_rate (use a reduced LR when continuing a run)")
    ap.add_argument("--cap-curriculum-frac", type=float, default=0.0,
                    help="ramp domain_rand.max_duty_range from [hi,hi] down to [lo,hi] over this "
                         "fraction of training (needs --domain-rand + max_duty_range): let cruise "
                         "form at the easy cap before exposing the barely-propelled low caps")
    ap.add_argument("--n-envs", type=int, default=None, help="override number of parallel envs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--vec", choices=["auto", "dummy", "subproc"], default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.init_from:
        # Inherit the donor's task / sensor-suite contract (CLI flags still override below): the
        # transferred weights only make sense on the obs layout they were trained with (plus any
        # NEW dims appended last, e.g. observe_max_duty — warm_start zero-pads those).
        donor_meta = Path(args.init_from)
        donor_meta = (donor_meta if donor_meta.suffix != ".zip" else donor_meta.parent) / "meta.yaml"
        if donor_meta.exists():
            dm = yaml.safe_load(donor_meta.read_text())
            for k in ("task", "obs_mode", "proprio_mode", "obs_frame"):
                if dm.get(k) is not None:
                    cfg["env"][k] = dm[k]
            print(f"[train] init-from contract: " +
                  " ".join(f"{k}={cfg['env'].get(k)}" for k in ("task", "obs_mode", "proprio_mode", "obs_frame")))
    if args.task:
        cfg["env"]["task"] = args.task
    task = cfg["env"].get("task", "pose")
    if args.obs_mode:
        cfg["env"]["obs_mode"] = args.obs_mode
    if args.disturb:
        cfg.setdefault("disturbance", {})["enabled"] = True
    if args.domain_rand:
        cfg.setdefault("domain_rand", {})["enabled"] = True
    if args.vel_cone is not None:
        cfg["env"]["vel_cmd_cone_deg"] = args.vel_cone
    if args.yaw_target is not None:
        cfg["env"]["yaw_target_deg"] = args.yaw_target
    obs_mode = cfg["env"].get("obs_mode", "auto")
    if obs_mode == "auto":
        obs_mode = _DEFAULT_OBS[task]
    cfg["env"]["obs_mode"] = obs_mode  # store the resolved suite so eval matches it
    algo = args.algo or cfg.get("algo", "ppo")
    if args.learning_rate is not None:
        cfg.setdefault("ppo", {})["learning_rate"] = args.learning_rate
    total_timesteps = args.timesteps or cfg["total_timesteps"]
    n_envs = args.n_envs or cfg["n_envs"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    run_name = args.run_name or algo

    run_dir = _ROOT / "models" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    vec_cls = DummyVecEnv if (args.vec == "dummy" or (args.vec == "auto" and n_envs == 1)) else SubprocVecEnv
    venv = make_vec_env(UmiusiPoseEnv, n_envs=n_envs, seed=seed,
                        env_kwargs={"config": cfg}, vec_env_cls=vec_cls)
    # Normalize observations (and, for on-policy PPO, rewards) — stabilizes value learning with
    # these large returns. Stats are saved and reloaded by eval so inference matches training.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=(algo == "ppo"), clip_obs=10.0)

    model = build_model(algo, cfg, venv, seed, run_dir / "tb")

    if args.init_from:
        froze = warm_start(model, args.init_from, venv)
        print(f"[train] warm-started from {args.init_from} "
              f"(obs {model.observation_space.shape[0]}-D, "
              f"vecnormalize {'loaded+frozen' if froze else 'NOT found — fresh stats'})")

    # Checkpoint ~10x over the run (save_freq is per-env steps).
    save_freq = max(total_timesteps // (10 * n_envs), 1)
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=str(run_dir / "checkpoints"),
                              name_prefix=algo)
    callbacks = [ckpt]

    # Curriculum (attitude_velocity): start fixed +X / level, widen to the config cone + yaw range.
    # A warm-started run inherits a policy that already masters the full ranges — re-narrowing them
    # would waste steps and shift the data distribution, so the task curriculum defaults OFF there
    # (the econ ramp below still applies; --curriculum-frac overrides).
    cfrac = args.curriculum_frac if args.curriculum_frac is not None else (
        0.0 if args.init_from else (0.5 if task == "attitude_velocity" else 0.0))
    if task == "attitude_velocity" and cfrac > 0:
        cone_max = float(cfg["env"].get("vel_cmd_cone_deg", 180.0))
        yaw_max = float(cfg["env"].get("yaw_target_deg", 180.0))
        tilt_max = float(cfg["env"].get("tilt_target_deg", 45.0))
        elev_max = 0.0
        if not bool(cfg["env"].get("vel_cmd_horizontal", True)):
            elev_max = float(cfg["env"].get("vel_cmd_elev_deg", 60.0))
            venv.set_attr("vel_cmd_elev_deg", 0.0)
        venv.set_attr("vel_cmd_cone_deg", 0.0)
        venv.set_attr("yaw_target_deg", 0.0)
        venv.set_attr("tilt_target_deg", 0.0)
        callbacks.append(CurriculumCallback(total_timesteps, cone_max, yaw_max, tilt_max, cfrac,
                                            elev_max=elev_max))
        print(f"[train] curriculum: cone/yaw/tilt/elev 0 -> {cone_max:.0f}/{yaw_max:.0f}/{tilt_max:.0f}"
              f"/{elev_max:.0f} over {cfrac * 100:.0f}% of steps")

    # Cap curriculum: start every episode at the easy cap (hi), widen down to [lo, hi].
    md_range = cfg.get("domain_rand", {}).get("max_duty_range")
    if args.cap_curriculum_frac > 0.0 and cfg.get("domain_rand", {}).get("enabled") and md_range:
        lo, hi = float(md_range[0]), float(md_range[1])
        venv.set_attr("dr", {**cfg["domain_rand"], "max_duty_range": [hi, hi]})
        callbacks.append(CapRangeCurriculumCallback(total_timesteps, cfg["domain_rand"], lo, hi,
                                                    args.cap_curriculum_frac))
        print(f"[train] cap curriculum: max_duty_range [{hi},{hi}] -> [{lo},{hi}] "
              f"over {args.cap_curriculum_frac * 100:.0f}% of steps")

    # Economy-penalty curriculum: ramp w_effort + w_null in 0 -> full over econ_ramp_frac of steps.
    econ_frac = float(cfg["reward"].get("econ_ramp_frac", 0.0))
    if econ_frac > 0.0:
        venv.set_attr("econ_ramp", 0.0)
        callbacks.append(EconRampCallback(total_timesteps, econ_frac))
        print(f"[train] econ curriculum: w_effort/w_null 0 -> full over {econ_frac * 100:.0f}% of steps")

    print(f"[train] task={task} algo={algo} obs_mode={obs_mode} n_envs={n_envs} "
          f"timesteps={total_timesteps} seed={seed} -> {run_dir}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(str(run_dir / "final"))
    venv.save(str(run_dir / "vecnormalize.pkl"))  # obs/reward normalization stats for eval
    with open(run_dir / "meta.yaml", "w") as f:
        yaml.safe_dump({"algo": algo, "task": task, "obs_mode": obs_mode, "vecnormalize": True,
                        "proprio_mode": cfg["env"].get("proprio_mode"),
                        "obs_frame": cfg["env"].get("obs_frame"),
                        "observe_max_duty": bool(cfg["env"].get("observe_max_duty", False)),
                        "init_from": args.init_from,
                        "vel_cmd_cone_deg": cfg["env"].get("vel_cmd_cone_deg"),
                        "yaw_target_deg": cfg["env"].get("yaw_target_deg"),
                        "tilt_target_deg": cfg["env"].get("tilt_target_deg"),
                        "disturbance": cfg.get("disturbance", {}).get("enabled", False),
                        "domain_rand": cfg.get("domain_rand", {}).get("enabled", False),
                        "config": args.config, "seed": seed, "total_timesteps": total_timesteps}, f)
    venv.close()
    print(f"[train] done. policy -> {run_dir / 'final.zip'}")
    print(f"[train] eval:  python -m umiusi_rl.eval --model {run_dir / 'final.zip'}")


if __name__ == "__main__":
    main()
