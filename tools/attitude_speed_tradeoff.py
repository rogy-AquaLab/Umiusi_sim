"""Is the attitude-vs-cruise seesaw PHYSICS or optimization? (Umiusi_sim#3)

Every rate-action run trades one for the other (mode10/13 attitude-good tracking-weak; mode12/14
tracking-good attitude-weak) even after the constraint signal was fixed. If holding attitude while
cruising is physically expensive — the azimuth thrusters tilt to make horizontal thrust, and the
CoP offset turns translation into a moment — then no reward balance can satisfy both, and the
acceptance criteria have to be speed-conditional.

This measures steady-state attitude error as a function of COMMANDED SPEED for a given policy
(same held attitude target, eval conditions). A flat curve => optimization problem. A rising
curve => physical coupling, and its slope is the price of speed in radians.

    python -m tools.attitude_speed_tradeoff --model models/av_mode14/final.zip [--max-duty 0.25]
"""
import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO

from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP, UmiusiPoseEnv, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-duty", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--settle", type=int, default=150, help="steps to ignore (initial slew)")
    ap.add_argument("--tilts", default="0",
                    help="comma-separated held ROLL targets [deg] to sweep (default level only). "
                         "The eval battery samples tilts up to 45 deg, and holding a tilt fights "
                         "the buoyancy righting moment with the same thrusters that cruise — "
                         "this separates 'cruise coupling' from 'tilt holding'")
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

    vmax = 0.8 * VEL_PER_CAP * args.max_duty
    tilts = [float(t) for t in args.tilts.split(",")]
    print(f"[tradeoff] {args.model} @ cap {args.max_duty}  (steady-state means)")
    print(f"{'tilt [deg]':>10s} {'cmd [m/s]':>10s} {'ori err [rad]':>14s} {'|w| [rad/s]':>12s} {'v_along':>9s}")
    for tilt in tilts:
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0.0, 0.0]), np.radians(tilt))
        for f in (0.0, 0.5, 1.0):
            vset = f * vmax
            oris, ws, vs = [], [], []
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=2000 + ep)
                env.target_quat = q.copy()
                env.v_cmd = np.array([vset, 0.0, 0.0])
                Rt = np.zeros(9)
                mujoco.mju_quat2Mat(Rt, env.target_quat)
                env.v_cmd_world = Rt.reshape(3, 3) @ env.v_cmd
                done, k = False, 0
                while not done:
                    a, _ = model.predict(norm(obs), deterministic=True)
                    obs, _r, term, trunc, info = env.step(a)
                    k += 1
                    if k > args.settle:
                        oris.append(info["ori_err"])
                        ws.append(info.get("ang_speed", 0.0))
                        vs.append(info.get("vel_along", 0.0))
                    done = term or trunc
            print(f"{tilt:10.0f} {vset:10.3f} {np.mean(oris):14.3f} {np.mean(ws):12.3f} {np.mean(vs):9.3f}")
    env.close()


if __name__ == "__main__":
    main()
