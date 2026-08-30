"""Distill an esc-action teacher policy into a wrench-mode student (BC warm start).

Umiusi_sim#3 follow-up: scratch training in mode space (av_mode1) did not discover cruise
(along 32 % of commanded at cap 0.25), while the esc-action av_cap1 teacher cruises but
wastes ~20 % of vertical power in the null mode. This script combines them:

1. Roll the TEACHER through a MODES env: at each step the teacher's 8-D [servo x4, esc x4]
   action is converted to per-unit (horizontal, vertical) forces and PROJECTED onto the
   6 wrench-mode coordinates (the Walsh basis columns are orthogonal, so the projection is
   S^T f / (4 f_max) per group) — the null components are simply dropped. The env executes
   the projected action, so the collected states are the distribution the student will
   actually visit (DAgger-style, not off-policy replay of raw teacher trajectories).
2. Behavior-clone the projected actions (and the teacher's value estimates) into a fresh
   PPO student with the standard train.py architecture, then save a run dir that
   train.py --init-from can warm-start for RL fine-tuning (same obs contract, same
   VecNormalize stats, action space = 6 modes).

The rollout metrics printed at the end are the projected teacher's own performance — the
ceiling BC can reach. If along-speed collapses here, the projection (not BC) is the problem.

Usage:
    python -m umiusi_rl.distill --teacher models/av_cap1_rep103 --out bc_mode1
    python -m umiusi_rl.train --action-mode modes --obs-frame rep103 --domain-rand \
        --init-from models/bc_mode1 --run-name av_mode2 ...
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from umiusi_rl.envs.mode_mixer import MODE_DIM
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config
from umiusi_rl.train import _ROOT, build_model

# meta.yaml keys that define the obs/task contract; copied teacher -> student (eval.py reads
# the same set, so the student run dir evaluates with the correct contract).
_CONTRACT_KEYS = ("task", "obs_mode", "proprio_mode", "obs_frame",
                  "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg")


def project_action(a8, mixer, max_duty):
    """8-D [servo x4, esc x4] -> 6 wrench modes [fx, fy, fz, tx, ty, tz] in [-1, 1].

    Inverse of ModeMixer.mix under the NOMINAL plant constants: per unit the commanded
    thrust splits into horizontal/vertical components at the commanded servo angle; each
    Walsh-basis column has squared norm 4, so modes = S^T f / (4 f_max). Null components
    of the teacher's pattern are orthogonal to every column and vanish.
    """
    servo_rad = np.asarray(a8[:4], dtype=float) * mixer.servo_range_rad
    u = np.asarray(a8[4:8], dtype=float)
    thrust = np.sign(u) * np.abs(u) ** mixer.thrust_curve_exp * mixer.thrust_per_cmd
    h = thrust * np.cos(servo_rad)
    v = thrust * np.sin(servo_rad)
    f_max = mixer.thrust_per_cmd * float(max_duty) ** mixer.thrust_curve_exp
    fx, fy, tz = mixer._Sh.T @ h / (4.0 * f_max)
    fz, tx, ty = mixer._Sv.T @ v / (4.0 * f_max)
    return np.clip([fx, fy, fz, tx, ty, tz], -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", required=True,
                    help="teacher run dir (final.zip + vecnormalize.pkl + meta.yaml), esc action mode")
    ap.add_argument("--out", required=True, help="student run name (written to models/<out>)")
    ap.add_argument("--steps", type=int, default=120_000, help="rollout steps to collect")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-std", type=float, default=-1.0,
                    help="policy log_std after BC (SB3 default 0.0 would swamp the cloned "
                         "behavior with exploration noise at the start of fine-tuning)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    teacher_dir = Path(args.teacher)
    meta = yaml.safe_load((teacher_dir / "meta.yaml").read_text())
    if meta.get("action_mode", "esc") != "esc":
        raise SystemExit("teacher must be an esc-action policy (the projection is the point)")

    cfg = load_config(meta.get("config", "configs/train_ppo.yaml"))
    for k in _CONTRACT_KEYS:
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    cfg["env"]["action_mode"] = "modes"
    cfg.setdefault("disturbance", {})["enabled"] = bool(meta.get("disturbance"))
    # DR on (as the teacher trained): varies the cap per episode, so the dataset teaches the
    # cap-conditioning input; the mixer itself always uses the nominal constants.
    cfg.setdefault("domain_rand", {})["enabled"] = bool(meta.get("domain_rand"))

    env = UmiusiPoseEnv(cfg)
    assert env._mixer is not None

    teacher = PPO.load(str(teacher_dir / "final.zip"), device="cpu")
    import pickle
    with open(teacher_dir / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rms, clip_obs, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    if rms.mean.shape[0] != env.observation_space.shape[0]:
        raise SystemExit(f"obs dim mismatch: teacher stats {rms.mean.shape[0]} vs env "
                         f"{env.observation_space.shape[0]} (contract keys not aligned?)")

    def norm_obs(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip_obs, clip_obs).astype(np.float32)

    print(f"[distill] rollout: {args.steps} steps of the projected teacher (modes env)")
    obs_buf = np.empty((args.steps, env.observation_space.shape[0]), dtype=np.float32)
    act_buf = np.empty((args.steps, MODE_DIM), dtype=np.float32)
    vel_alongs, vel_cmds, null_fracs, vert_powers = [], [], [], []
    obs, _ = env.reset(seed=args.seed)
    ep = 0
    for t in range(args.steps):
        no = norm_obs(obs)
        a8, _ = teacher.predict(no, deterministic=True)
        m6 = project_action(a8, env._mixer, env.sim.max_duty)
        obs_buf[t] = no
        act_buf[t] = m6
        obs, _r, terminated, truncated, info = env.step(m6)
        null_fracs.append(float(info.get("null_frac", 0.0)))
        vert_powers.append(float(info.get("vert_power", 0.0)))
        if terminated or truncated:
            vel_alongs.append(float(info.get("vel_along", 0.0)))
            vel_cmds.append(float(info.get("vel_cmd_speed", 0.0)))
            ep += 1
            obs, _ = env.reset(seed=args.seed + ep)
        if (t + 1) % 20_000 == 0:
            print(f"[distill]   {t + 1}/{args.steps} steps, {ep} episodes")
    wp, nf = np.array(vert_powers), np.array(null_fracs)
    null_pw = float((nf * wp).sum() / wp.sum()) * 100.0 if wp.sum() > 1e-12 else 0.0
    print(f"[distill] projected teacher: along {np.mean(vel_alongs):.3f} / desired "
          f"{np.mean(vel_cmds):.3f} m/s over {ep} episodes, null share (pw) {null_pw:.1f}% "
          f"(the student's BC ceiling)")

    # Teacher value targets on the same normalized obs (the copied VecNormalize also carries
    # ret_rms, so the fine-tune's normalized-return scale matches these values).
    with torch.no_grad():
        val_buf = teacher.policy.predict_values(
            torch.as_tensor(obs_buf)).squeeze(-1).numpy().astype(np.float32)

    run_dir = _ROOT / "models" / args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
    student = build_model(meta.get("algo", "ppo"), cfg, venv, args.seed, run_dir / "tb")
    if not isinstance(student, PPO):
        raise SystemExit("distillation supports the PPO student only")

    print(f"[distill] BC: {args.epochs} epochs x {args.steps} samples (batch {args.batch_size})")
    policy = student.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    obs_t = torch.as_tensor(obs_buf)
    act_t = torch.as_tensor(act_buf)
    val_t = torch.as_tensor(val_buf)
    for epoch in range(args.epochs):
        perm = torch.randperm(args.steps)
        a_loss_sum = v_loss_sum = n_batches = 0.0
        for i in range(0, args.steps, args.batch_size):
            idx = perm[i:i + args.batch_size]
            ob, tgt, vt = obs_t[idx], act_t[idx], val_t[idx]
            features = policy.extract_features(ob)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            mean = policy.action_net(latent_pi)
            value = policy.value_net(latent_vf).squeeze(-1)
            a_loss = torch.nn.functional.mse_loss(mean, tgt)
            v_loss = torch.nn.functional.mse_loss(value, vt)
            loss = a_loss + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            a_loss_sum += a_loss.item()
            v_loss_sum += v_loss.item()
            n_batches += 1
        print(f"[distill]   epoch {epoch + 1:2d}/{args.epochs}  action mse {a_loss_sum / n_batches:.5f}"
              f"  value mse {v_loss_sum / n_batches:.4f}")
    with torch.no_grad():
        policy.log_std.fill_(args.log_std)

    student.save(str(run_dir / "final.zip"))
    shutil.copy(teacher_dir / "vecnormalize.pkl", run_dir / "vecnormalize.pkl")
    out_meta = {k: meta[k] for k in ("algo", "config", "disturbance", "domain_rand", "obs_mode",
                                     "proprio_mode", "task", "tilt_target_deg", "vel_cmd_cone_deg",
                                     "yaw_target_deg") if k in meta}
    out_meta.update({
        "obs_frame": meta.get("obs_frame", "sim"),
        "observe_max_duty": bool(meta.get("observe_max_duty", False)),
        "action_mode": "modes",
        "vecnormalize": True,
        "distilled_from": str(teacher_dir),
        "bc_steps": args.steps,
        "bc_epochs": args.epochs,
    })
    with open(run_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(out_meta, f, sort_keys=True)
    env.close()
    venv.close()
    print(f"[distill] done. student -> {run_dir}/final.zip  "
          f"(eval it directly, then train.py --init-from {run_dir})")


if __name__ == "__main__":
    main()
