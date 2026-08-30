"""Capsize gate: is it safe to raise the deploy esc cap for this policy? (Umiusi_sim#3)

Raising max_duty buys thrust but also buys capsize speed: a full roll-mode command flips the
vehicle at EVERY cap (measured 2026-08-28 — roll authority is not the limit, precision is), and
the righting moment is small (CoB ~1 cm above CoM). Before recommending a cap on the robot,
check what the policy's own transients do at that cap, with disturbances on.

The measured quantity is the roll EXCURSION BEYOND THE COMMANDED ROLL during the HOLDING phase
(after --settle steps): the task commands tilts up to tilt_target_deg (45), so absolute roll
conflates "did as told" with "tipped over", and the initial slew from level to a tilted target
would otherwise be counted as an excursion by construction. Reports, per cap: max excursion, p95, the fraction of episodes exceeding
45 deg (near-capsize) and 90 deg (capsized), and the mean settled attitude error.

    python -m tools.capsize_gate --model models/av_mode15/final.zip --caps 0.25,0.4,0.6

GATE (suggested): capsized 0 %, near-capsize < 5 %, p95 excursion < 30 deg. A cap that fails
this should not be recommended to the operator regardless of how good its cruise numbers look.
"""
import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config


def _roll_deg(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    return float(np.degrees(np.arctan2(R[1, 2], R[1, 1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--caps", default="0.25,0.4", help="comma-separated esc caps to gate")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--no-disturb", action="store_true", help="disable the disturbance stress")
    ap.add_argument("--settle", type=int, default=150,
                    help="steps to ignore: at t=0 the vehicle is level and the target may be tilted, "
                         "so the commanded slew would otherwise count as an excursion")
    args = ap.parse_args()

    mdir = Path(args.model).parent
    meta = yaml.safe_load((mdir / "meta.yaml").read_text())
    cfg = load_config(meta.get("config", "configs/train_ppo.yaml"))
    for k in ("task", "obs_mode", "proprio_mode", "obs_frame", "action_mode",
              "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    cfg.setdefault("domain_rand", {})["enabled"] = True          # model mismatch is part of the risk
    cfg.setdefault("domain_rand", {}).pop("max_duty_range", None)  # the pinned cap must survive DR
    cfg.setdefault("disturbance", {})["enabled"] = not args.no_disturb
    model = PPO.load(args.model, device="cpu")
    vn = pickle.load(open(mdir / "vecnormalize.pkl", "rb"))
    rms = vn.obs_rms

    def norm(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + vn.epsilon),
                       -vn.clip_obs, vn.clip_obs).astype(np.float32)

    print(f"[capsize_gate] {args.model}  episodes={args.episodes} "
          f"disturb={'off' if args.no_disturb else 'on'} DR=on")
    print(f"{'cap':>6s} {'max excur':>10s} {'p95':>8s} {'>45deg':>8s} {'>90deg':>8s} {'ori end':>9s}  verdict")
    for cap in [float(c) for c in args.caps.split(",")]:
        env = UmiusiPoseEnv(cfg)
        env.sim.max_duty = cap
        env._base["max_duty"] = cap
        peaks, oris = [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=3000 + ep)
            env.sim.max_duty = cap          # reset() re-samples DR; re-pin the cap under test
            env._base["max_duty"] = cap
            target_roll = _roll_deg(env.target_quat)   # the tilt the task asked for
            done, peak = False, 0.0
            while not done:
                a, _ = model.predict(norm(obs), deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                if info.get("step_idx", 0) > args.settle:   # holding phase only (see --settle)
                    peak = max(peak, abs(_roll_deg(env.sim.get_state()["quat"]) - target_roll))
                done = term or trunc
            peaks.append(peak)
            oris.append(info["ori_err"])
        env.close()
        p = np.array(peaks)
        near, over = float((p > 45).mean()), float((p > 90).mean())
        ok = over == 0.0 and near < 0.05 and np.percentile(p, 95) < 30.0
        print(f"{cap:6.2f} {p.max():10.1f} {np.percentile(p, 95):8.1f} "
              f"{near * 100:7.0f}% {over * 100:7.0f}% {np.mean(oris):9.3f}  "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
