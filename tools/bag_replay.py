"""Validate / calibrate the simulator against a real-vehicle rosbag (exported to npz).

Two modes, both operating on an npz export of a pool-test bag (see the exporter snippet in
docs/calibration_plan.md — reading mcap needs the ROS python, so export once, analyse here):

  policy   Feed the RECORDED observations through a policy exactly as rl_attitude_node builds
           them (servo/thrust = the telemetry ECHO, target from the recorded setpoints) and
           score the predicted servo commands against the RECORDED commands. Validates that the
           sim-side understanding of the deployed pipeline is right: on the 2026-08-21
           servo-debug bag, av_curr4 — the policy DEPLOYED AT THE TIME, kept here as the
           historical example — reproduces the recorded commands with sign agreement 100 %
           (|cmd| > 5 deg) and per-channel correlation 0.92-0.94 at 1 tick of publish latency.

  physics  Open-loop k-step replay: initialise the sim from each recorded attitude/gyro window,
           drive it with the RECORDED commands, and score the k-step body-gyro prediction RMSE
           (persistence = repeating the initial gyro — the floor for smooth signals). Optionally
           grid-fit the most-uncertain scalars. This is how the 2026-08-21 calibration was made:
           buoyancy offset 0.05 -> ~0.01 m and the propeller-law thrust curve (exp ~2) came out
           of exactly this replay (linear map RMSE 0.28 -> 0.06 with both fixes; floor 0.047).

The IMU->CAD frame remap (--frame) matters: the BNO055 world frame is z-up while the sim frame
is +Y-up. From the bag itself (yaw excursion about world z, body z pinned to vertical) the
vehicle's up-axis is IMU z, so recorded quats/gyros are permuted into the CAD frame before use.
NOTE the deployed rl_attitude_node does NOT do this remap — the policy on the vehicle received
pitch/yaw-swapped observations (reported in the sim2real issue).

Usage:
    python -m tools.bag_replay policy  --npz out/servo_debug.npz --model models/av_curr4  # (historical bundle)
    python -m tools.bag_replay physics --npz out/servo_debug.npz [--thrust-exp 2.0] [--k 25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from umiusi_sim.simulator import UmiusiSimulator

POS = ("lf", "lb", "rb", "rf")
# IMU (z-up world) -> CAD (+Y-up) proper rotation: cad(x, y, z) = imu(x, -z, y).
FRAME_P = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def load(npz_path):
    d = np.load(npz_path)
    ct = d["cmd_lf_t"]

    def aligned(name):
        out = np.zeros((len(ct), 4))
        for k, p in enumerate(POS):
            ts, aa = d[f"cmd_{p}_t"], d[f"cmd_{p}_{name}"]
            idx = np.clip(np.searchsorted(ts, ct), 0, len(ts) - 1)
            prev = np.clip(idx - 1, 0, len(ts) - 1)
            pick = np.where(np.abs(ts[idx] - ct) <= np.abs(ts[prev] - ct), idx, prev)
            out[:, k] = aa[pick]
        return out

    gi = np.clip(np.searchsorted(d["imu_t"], ct), 0, len(d["imu_t"]) - 1)
    quat = d["imu_quat"][gi]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return dict(d=d, ct=ct, cmd_angle=aligned("angle"), cmd_duty=aligned("duty"),
                gyro=d["imu_gyro"][gi], quat=quat)


def remap_frame(quat, gyro, P=FRAME_P):
    """Rotate IMU-frame orientation/gyro into the CAD frame: R' = P R P^T, w' = P w."""
    g2 = gyro @ P.T
    q2 = np.zeros_like(quat)
    R = np.zeros(9)
    for i in range(len(quat)):
        mujoco.mju_quat2Mat(R, quat[i])
        M = P @ R.reshape(3, 3) @ P.T
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, M.flatten())
        q2[i] = q
    return q2, g2


def run_policy(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

    b = load(args.npz)
    d, ct, cmd_angle = b["d"], b["ct"], b["cmd_angle"]
    cfg = load_config(args.config)
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["obs_mode"] = "imu"
    cfg["env"]["proprio_mode"] = "full"      # the deployed policies are 25-D
    # obs contract of the target bundle: absent in its meta = trained WITHOUT the cap dim
    _meta_p = Path(args.model) / "meta.yaml"
    _meta = yaml.safe_load(_meta_p.read_text()) if _meta_p.exists() else {}
    cfg["env"]["observe_max_duty"] = bool(_meta.get("observe_max_duty", False))
    model = PPO.load(f"{args.model}/final.zip", device="cpu")
    venv = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
    vn = VecNormalize.load(f"{args.model}/vecnormalize.pkl", venv)
    vn.training = False

    def latest(ts, series, t):
        return series[max(np.searchsorted(ts, t, side="right") - 1, 0)]

    def sub_quat(qa, qb):
        out = np.zeros(3)
        mujoco.mju_subQuat(out, np.asarray(qa, float), np.asarray(qb, float))
        return out

    prev_a = np.zeros(8)
    pred = np.zeros((len(ct), 8))
    for i, t in enumerate(ct):     # replicate rl_attitude_node._build_obs, echoes and all
        quat = latest(d["imu_t"], d["imu_quat"], t).copy()
        n = np.linalg.norm(quat)
        quat = quat / n if n > 1e-9 else np.array([1.0, 0, 0, 0])
        obs = np.concatenate([
            sub_quat(latest(d["sp_t"], d["sp_quat"], t), quat),
            latest(d["imu_t"], d["imu_gyro"], t), args.v_cmd * np.array([1.0, 0, 0]),
            np.radians(latest(d["thr_t"], d["thr_angle"], t)) / np.radians(90.0),
            latest(d["thr_t"], d["thr_rpm"], t) / 1000.0, prev_a])
        a, _ = model.predict(vn.normalize_obs(obs.astype(np.float32)), deterministic=True)
        prev_a = np.clip(a, -1, 1)
        pred[i] = prev_a

    pa = pred[:, :4] * 90.0
    print(f"policy replay: {args.model} on {args.npz}  (lag +1 tick = publish latency)")
    for k in range(4):
        c = np.corrcoef(pa[:-1, k], cmd_angle[1:, k])[0, 1]
        print(f"  {POS[k]}: corr {c:+.3f}")
    m = np.abs(cmd_angle[1:]) > 5
    agree = np.mean(np.sign(pa[:-1][m]) == np.sign(cmd_angle[1:][m]))
    print(f"  sign agreement (|cmd|>5 deg): {agree * 100:.1f} %")


def run_physics(args):
    b = load(args.npz)
    q2, g2 = remap_frame(b["quat"], b["gyro"])
    cmd_angle, cmd_duty, ct = b["cmd_angle"], b["cmd_duty"], b["ct"]
    sim = UmiusiSimulator()
    if args.thrust_exp is not None:
        sim.thrust_curve_exp = args.thrust_exp
    if args.buoyancy_offset is not None:
        sim.set_buoyancy_offset(args.buoyancy_offset)

    K = args.k
    errs, base = [], []
    for w0 in range(50, len(ct) - K - 1, args.stride):
        sim.reset(pos=(0, 0, 0), quat=tuple(q2[w0]))
        sim.data.qvel[3:6] = g2[w0]
        mujoco.mj_forward(sim.model, sim.data)
        pred = []
        for t in range(K):
            a = np.concatenate([np.clip(cmd_angle[w0 + t] / 90.0, -1, 1),
                                np.clip(cmd_duty[w0 + t], -1, 1)])
            sim.step(a)
            Rm = sim.data.xmat[sim.base_id].reshape(3, 3)
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY,
                                     sim.base_id, vel6, 0)
            pred.append(Rm.T @ vel6[:3])
        errs.append(np.sqrt(np.mean((np.array(pred) - g2[w0 + 1:w0 + K + 1]) ** 2)))
        base.append(np.sqrt(np.mean((np.tile(g2[w0], (K, 1)) - g2[w0 + 1:w0 + K + 1]) ** 2)))
    print(f"physics replay: {args.npz}  K={K} steps ({K * 0.02:.1f} s), {len(errs)} windows")
    print(f"  gyro RMSE {np.mean(errs):.4f} rad/s   persistence floor {np.mean(base):.4f}")
    print(f"  (config: thrust exp {sim.thrust_curve_exp}, CoB offset {sim.buoyancy_offset} m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("policy", help="recorded obs -> policy -> compare with recorded commands")
    p1.add_argument("--npz", required=True)
    p1.add_argument("--model", default="models/av_curr4")
    p1.add_argument("--config", default="configs/train_ppo.yaml")
    p1.add_argument("--v-cmd", type=float, default=0.0,
                    help="node vel_cmd during the recording (servo-debug bag: 0)")
    p2 = sub.add_parser("physics", help="recorded commands -> sim -> compare gyro with recorded")
    p2.add_argument("--npz", required=True)
    p2.add_argument("--k", type=int, default=25, help="prediction horizon [control steps]")
    p2.add_argument("--stride", type=int, default=50)
    p2.add_argument("--thrust-exp", type=float, default=None)
    p2.add_argument("--buoyancy-offset", type=float, default=None)
    args = ap.parse_args()
    (run_policy if args.mode == "policy" else run_physics)(args)


if __name__ == "__main__":
    main()
