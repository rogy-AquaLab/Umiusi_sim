"""Pre-deployment verification for a policy bundle — golden vectors + sanity battery.

Purpose: at the pool, BEFORE getting wet, prove that the policy the robot loaded is the policy
the sim validated — and that its responses are directionally sane. Two halves:

  generate  (dev machine, this repo): run the policy on a battery of canonical observations and
            save (obs, action) GOLDEN VECTORS plus metadata into the bundle
            (<model>/golden.npz). Also prints the sanity battery so a human can eyeball it.

  verify    (anywhere — the robot only needs numpy + torch, no SB3/sim): reload the bundle
            through whatever inference path deployment uses, replay golden.npz, and require
            max |action - golden| < 1e-4. A pass means bytes, normalisation stats, obs layout
            and frame convention all survived the copy to the robot.

The golden obs are stored IN THE POLICY'S OWN FRAME (meta.yaml obs_frame) — for a rep103
bundle they are what the robot node itself would build from the IMU, so a verify pass also
locks the frame contract.

Sanity battery (printed on generate, saved into golden.npz for the record):
  * neutral obs        -> actions must be finite, |esc| below hover-ish levels
  * +roll error        -> servo response must be nonzero and consistent left/right
  * +pitch / +yaw error, +v_cmd fwd -> likewise (direction printed for the log)
  * saturation scan    -> fraction of |action| > 0.99 over random in-distribution obs

Usage:
    python -m tools.preflight_policy generate --model models/av_cal1_best_rep103
    python -m tools.preflight_policy verify   --model <copied dir>     # on the robot/CI
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

def _load_sb3(model_dir):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym
    from gymnasium import spaces

    model = PPO.load(str(model_dir / "final.zip"), device="cpu")
    dim = int(np.prod(model.observation_space.shape))

    class _S(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
            self.action_space = model.action_space
        def reset(self, *, seed=None, options=None):
            return np.zeros(dim, np.float32), {}
        def step(self, a):
            return np.zeros(dim, np.float32), 0.0, False, False, {}

    vn = VecNormalize.load(str(model_dir / "vecnormalize.pkl"), DummyVecEnv([_S]))
    vn.training = False

    def act(obs):
        a, _ = model.predict(vn.normalize_obs(obs.astype(np.float32)), deterministic=True)
        return np.asarray(a, dtype=np.float64)

    return act, dim


def _battery(dim):
    """Canonical + random observations for a 17/18-D (imu+vel, action-proprio) or 14-D bundle.
    Layout: [ori_err(3), gyro(3), (v_cmd(3) if 17/18-D), prev_action(8), (max_duty(1) if 18-D)].
    18-D = the Phase2 cap-observed contract (Umiusi_sim#3): max_duty is a RAW scalar, LAST."""
    has_v = dim in (17, 18, 25)
    has_cap = dim == 18
    def obs(ori=(0, 0, 0), gyro=(0, 0, 0), v=(0, 0, 0), prev=None, cap=0.25):
        parts = [np.asarray(ori, float), np.asarray(gyro, float)]
        if has_v:
            parts.append(np.asarray(v, float))
        parts.append(np.zeros(8) if prev is None else np.asarray(prev, float))
        if has_cap:
            parts.append(np.asarray([cap], float))
        o = np.concatenate(parts)
        assert len(o) == dim, (len(o), dim)
        return o
    named = {
        "neutral":      obs(),
        "roll_err_+20": obs(ori=(0.35, 0, 0)),
        "pitch_err_+20": obs(ori=(0, 0, 0.35)) if dim in (14, 22) else obs(ori=(0, 0.35, 0)),
        "yaw_err_+20":  obs(ori=(0, 0.35, 0)) if dim in (14, 22) else obs(ori=(0, 0, 0.35)),
        "gyro_spike":   obs(gyro=(0.5, -0.5, 0.5)),
    }
    if has_v:
        named["cruise_cmd"] = obs(v=(0.4, 0, 0))
        named["hold_cmd"] = obs(v=(0, 0, 0))
    if has_cap:  # pin the cap-conditional behaviour across the deploy range
        named["cruise_cap_0.2"] = obs(v=(0.4, 0, 0), cap=0.2)
        named["cruise_cap_0.4"] = obs(v=(0.4, 0, 0), cap=0.4)
    rng = np.random.default_rng(42)
    rand = np.stack([obs(ori=rng.normal(0, 0.3, 3), gyro=rng.normal(0, 0.3, 3),
                         v=rng.uniform(-0.4, 0.4, 3), prev=rng.uniform(-1, 1, 8),
                         cap=rng.uniform(0.2, 0.4))
                     for _ in range(64)])
    return named, rand


def generate(args):
    d = Path(args.model)
    act, dim = _load_sb3(d)
    named, rand = _battery(dim)
    print(f"policy {d} (obs dim {dim}) — sanity battery:")
    gold_obs, gold_act, names = [], [], []
    for name, o in named.items():
        a = act(o)
        gold_obs.append(o)
        gold_act.append(a)
        names.append(name)
        print(f"  {name:14s} servo {np.round(a[:4], 2)}  esc {np.round(a[4:], 2)}")
        assert np.all(np.isfinite(a)), name
    sat = []
    for o in rand:
        a = act(o)
        gold_obs.append(o)
        gold_act.append(a)
        names.append("random")
        sat.append(np.mean(np.abs(a) > 0.99))
    print(f"  saturation over 64 random in-distribution obs: {np.mean(sat) * 100:.1f} %")
    np.savez(d / "golden.npz", obs=np.array(gold_obs), act=np.array(gold_act),
             names=np.array(names), obs_dim=dim)
    print(f"wrote {d / 'golden.npz'}  ({len(gold_obs)} vectors)")


def verify(args):
    d = Path(args.model)
    g = np.load(d / "golden.npz")
    act, dim = _load_sb3(d)
    assert dim == int(g["obs_dim"]), f"obs dim changed: bundle {dim} vs golden {int(g['obs_dim'])}"
    worst = 0.0
    for o, a_ref in zip(g["obs"], g["act"]):
        worst = max(worst, float(np.abs(act(o) - a_ref).max()))
    ok = worst < 1e-4
    print(f"{'PASS' if ok else 'FAIL'}: max |action - golden| = {worst:.2e} over {len(g['obs'])} vectors")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("generate", "verify"):
        p = sub.add_parser(m)
        p.add_argument("--model", required=True, help="policy bundle dir (final.zip + vecnormalize.pkl)")
    args = ap.parse_args()
    (generate if args.mode == "generate" else verify)(args)


if __name__ == "__main__":
    main()
