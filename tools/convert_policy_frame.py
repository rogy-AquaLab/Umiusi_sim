"""Convert a trained policy between observation FRAMES — exactly, without retraining.

Why: the sim's body frame is the CAD frame (+X fwd, +Y up, +Z starboard), while the robot's
IMU / ROS side speaks REP-103 (+X fwd, +Y left, +Z up). The 2026-08-21 pool test showed what
happens when the two are confused: the deployed policy received pitch/yaw-swapped observations
and could not control attitude at all (sim2real issue #3). The DEPLOYMENT CONTRACT going
forward: **a policy artifact handed to the robot consumes REP-103 body-frame observations** —
the node feeds the IMU quaternion/gyro without any hand-written axis shuffling.

How: every frame-dependent block of the observation (ori_err, angular velocity, v_cmd — all
3-vectors) transforms as v' = P v with P a SIGNED PERMUTATION (proper rotation with entries
0/±1). For an MLP policy the first layer computes W @ ((o - mean)/std); feeding o' = B o
(B = block-diagonal obs transform) is EXACTLY compensated by
    mean' = B mean,   var' = |B| var (signs square away),   W' = W B^T
(B orthogonal). Later layers, biases, log_std, action head are untouched — the converted
policy is bit-for-bit equivalent on remapped inputs. A built-in self-test verifies this on
random observations before anything is written.

Frames (as seen from the sim/CAD frame):
    sim     identity (the training-native frame)
    rep103  x fwd, y left(= -z_sim), z up(= +y_sim)     <- the deployment contract
    ned     x fwd, y right(= +z_sim), z down(= -y_sim)  (if the IMU turns out to publish NED)

Usage:
    python -m tools.convert_policy_frame --model models/av_cal1 --out models/av_cal1_rep103
    python -m tools.convert_policy_frame --model ... --to ned      # NED-consuming variant
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Rows = target-frame axes expressed in sim/CAD coordinates.
FRAMES = {
    "sim": np.eye(3),
    "rep103": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    "ned": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
}

# Observation layouts: list of (block_size, is_vector3) in order. v_cmd counts as a 3-vector.
LAYOUTS = {
    17: [(3, True), (3, True), (3, True), (8, False)],                    # imu+vel, proprio action
    25: [(3, True), (3, True), (3, True), (4, False), (4, False), (8, False)],  # proprio full
    22: [(3, True), (3, True), (4, False), (4, False), (8, False)],       # attitude, proprio full
    14: [(3, True), (3, True), (8, False)],                               # attitude, proprio action
}


def obs_transform(dim, P):
    """Block-diagonal signed-permutation B for the obs layout of width `dim`."""
    if dim not in LAYOUTS:
        raise SystemExit(f"unknown obs layout of dim {dim}; known: {sorted(LAYOUTS)}")
    B = np.zeros((dim, dim))
    i = 0
    for size, is_vec in LAYOUTS[dim]:
        B[i:i + size, i:i + size] = P if is_vec else np.eye(size)
        i += size
    return B


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="run dir containing final.zip + vecnormalize.pkl")
    ap.add_argument("--out", default=None, help="output dir (default: <model>_<to>)")
    ap.add_argument("--to", default="rep103", choices=[k for k in FRAMES if k != "sim"],
                    help="target frame the converted policy consumes (default rep103)")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym
    from gymnasium import spaces

    src = Path(args.model)
    out = Path(args.out) if args.out else src.with_name(src.name + f"_{args.to}")
    model = PPO.load(str(src / "final.zip"), device="cpu")
    dim = int(np.prod(model.observation_space.shape))
    B = obs_transform(dim, FRAMES[args.to])

    sd = model.policy.state_dict()
    Bt = torch.tensor(B, dtype=sd["mlp_extractor.policy_net.0.weight"].dtype)
    for key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        sd[key] = sd[key] @ Bt.T          # W' = W B^T  (input o' = B o)
    model.policy.load_state_dict(sd)

    # VecNormalize stats: mean' = B mean; var' = |B| var.
    class _Stub(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
            self.action_space = model.action_space
        def reset(self, *, seed=None, options=None):
            return np.zeros(dim, np.float32), {}
        def step(self, a):
            return np.zeros(dim, np.float32), 0.0, False, False, {}
    venv = DummyVecEnv([_Stub])
    vn = VecNormalize.load(str(src / "vecnormalize.pkl"), venv)
    mean0, var0 = vn.obs_rms.mean.copy(), vn.obs_rms.var.copy()
    vn.obs_rms.mean = B @ mean0
    vn.obs_rms.var = np.abs(B) @ var0

    # --- self-test: converted(B o) == original(o) on random obs -------------------------------
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, dim)).astype(np.float32)
    def norm(o, mean, var):
        return np.clip((o - mean) / np.sqrt(var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
    ref = PPO.load(str(src / "final.zip"), device="cpu")
    a_ref, _ = ref.predict(norm(obs, mean0, var0).astype(np.float32), deterministic=True)
    a_new, _ = model.predict(norm(obs @ B.T, vn.obs_rms.mean, vn.obs_rms.var).astype(np.float32),
                             deterministic=True)
    err = float(np.abs(a_ref - a_new).max())
    assert err < 1e-5, f"conversion self-test failed: max action diff {err}"

    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "final.zip")
    vn.save(str(out / "vecnormalize.pkl"))
    meta_src = src / "meta.yaml"
    meta = meta_src.read_text() if meta_src.exists() else ""
    (out / "meta.yaml").write_text(
        meta.rstrip() + f"\nobs_frame: {args.to}   # converted from {src.name} (sim frame) by "
        "tools/convert_policy_frame.py; consumes REP-103/NED body-frame obs directly\n")
    print(f"converted {src} (sim frame, {dim}-D obs) -> {out} ({args.to} frame)")
    print(f"self-test: max action diff {err:.2e} on 64 random obs  OK")


if __name__ == "__main__":
    main()
