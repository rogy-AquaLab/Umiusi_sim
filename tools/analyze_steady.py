"""Diagnose steady-state behavior of a trained attitude policy: which episodes settle vs vibrate.

Runs the policy for N episodes and, over the TAIL (last steps, after it should have settled),
measures per-episode: steady orientation error, servo motion (deg/step), servo oscillation
amplitude (std, deg), and thrust change. Correlates with the target tilt (angle of the target
body-up from world-up) — the component buoyancy fights — to see if vibration tracks difficulty.

    MUJOCO_GL=egl python -m tools.analyze_steady --model models/att_v5/final.zip
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def target_tilt_deg(quat):
    """Angle of the target body +Y axis from world +Y (the tilt buoyancy resists)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    up = R.reshape(3, 3)[:, 1]  # body-y in world
    return float(np.degrees(np.arccos(np.clip(up[1], -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--tail", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()

    md = Path(args.model).parent
    meta = yaml.safe_load((md / "meta.yaml").read_text()) if (md / "meta.yaml").exists() else {}
    cfg = load_config(meta.get("config", "configs/train_ppo.yaml"))
    for k in ("task", "obs_mode"):
        if k in meta:
            cfg["env"][k] = meta[k]
    env = UmiusiPoseEnv(cfg)

    rms = None
    stats = md / "vecnormalize.pkl"
    if stats.exists():
        d = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
        vn = VecNormalize.load(str(stats), d)
        d.close()
        rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

    def norm(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32) if rms else o

    model = ALGOS[meta.get("algo", "ppo")].load(str(md / "final.zip"), device="cpu")

    print(f"{'ep':>3} {'tilt°':>6} {'ori_ss':>7} {'servo_mot°/st':>13} {'servo_std°':>10} {'thr_chg':>8}  class")
    rows = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        tilt = target_tilt_deg(env.target_quat)
        servos, escs, oris = [], [], []
        done = False
        while not done:
            a, _ = model.predict(norm(obs), deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            servos.append(info["servo"].copy())
            escs.append(info["esc_cmd"].copy())
            oris.append(info["ori_err"])
            done = term or trunc
        s, e, o = np.array(servos), np.array(escs), np.array(oris)
        ts, te = s[-args.tail:], e[-args.tail:]
        servo_mot = float(np.degrees(np.mean(np.abs(np.diff(ts, axis=0)))))
        servo_std = float(np.degrees(np.mean(np.std(ts, axis=0))))
        thr_chg = float(np.mean(np.abs(np.diff(te, axis=0))))
        ori_ss = float(np.mean(o[-args.tail:]))
        cls = "VIBRATING" if servo_mot > 0.3 else "settled"
        rows.append((tilt, ori_ss, servo_mot, servo_std, thr_chg, cls))
        print(f"{ep:>3} {tilt:6.1f} {ori_ss:7.3f} {servo_mot:13.3f} {servo_std:10.2f} {thr_chg:8.4f}  {cls}")

    vib = [r for r in rows if r[5] == "VIBRATING"]
    settled = [r for r in rows if r[5] == "settled"]
    print("-" * 70)
    print(f"settled: {len(settled)}/{len(rows)}   vibrating: {len(vib)}/{len(rows)}")
    if settled:
        print(f"  settled  : mean tilt {np.mean([r[0] for r in settled]):5.1f}°  ori_ss {np.mean([r[1] for r in settled]):.3f}"
              f"  servo_mot {np.mean([r[2] for r in settled]):.3f}°/st")
    if vib:
        print(f"  vibrating: mean tilt {np.mean([r[0] for r in vib]):5.1f}°  ori_ss {np.mean([r[1] for r in vib]):.3f}"
              f"  servo_mot {np.mean([r[2] for r in vib]):.3f}°/st")


if __name__ == "__main__":
    main()
