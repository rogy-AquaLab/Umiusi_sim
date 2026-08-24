"""Frame-contract tests: env obs_frame presets and the exact policy frame conversion.

Runnable two ways:
    python -m pytest tests/test_obs_frame.py
    python tests/test_obs_frame.py            # standalone (plain asserts)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "sim" / "src"))

from umiusi_rl.envs.umiusi_pose_env import _OBS_FRAMES, UmiusiPoseEnv, load_config  # noqa: E402


def _cfg(**env_over):
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"] = {**cfg["env"], "task": "attitude_velocity", "obs_mode": "imu",
                  "proprio_mode": "action", **env_over}
    cfg["domain_rand"] = {"enabled": False}
    return cfg


def test_frames_are_proper_rotations():
    for name, P in _OBS_FRAMES.items():
        assert np.allclose(P @ P.T, np.eye(3)), name
        assert np.isclose(np.linalg.det(P), 1.0), f"{name} must be right-handed"


def test_obs_frame_permutes_vector_blocks():
    """rep103 obs must equal the sim-frame obs with each 3-vector block mapped by P."""
    e_sim = UmiusiPoseEnv(_cfg(obs_frame="sim"))
    e_rep = UmiusiPoseEnv(_cfg(obs_frame="rep103"))
    P = _OBS_FRAMES["rep103"]
    o_sim, _ = e_sim.reset(seed=7)
    o_rep, _ = e_rep.reset(seed=7)
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.uniform(-1, 1, size=8)
        o_sim = e_sim.step(a)[0]
        o_rep = e_rep.step(a)[0]
        for i in (0, 3, 6):        # ori_err, gyro, v_cmd blocks
            assert np.allclose(o_rep[i:i + 3], P @ o_sim[i:i + 3], atol=1e-6)
        assert np.allclose(o_rep[9:], o_sim[9:], atol=1e-6)   # prev_action untouched


def test_converted_policy_closed_loop_equivalence():
    """(converted policy, rep103 env) must fly identically to (original, sim env)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:  # RL extra not installed
        return
    src = Path("models/av_sim2real2")
    if not (src / "final.zip").exists():
        return  # models/ is gitignored; skip on a clean checkout
    import subprocess
    out = Path("/tmp/umiusi_obs_frame_test")
    subprocess.run([sys.executable, "-m", "tools.convert_policy_frame",
                    "--model", str(src), "--out", str(out)], check=True,
                   cwd=Path(__file__).resolve().parents[1])
    # Teacher-forced comparison: ONE simulation (sim-frame env drives the plant); at every step
    # both policies see the same physical state — the original through sim-frame obs, the
    # converted one through rep103-frame obs — and must emit the same action. (A free-running
    # closed-loop comparison would amplify float32 noise through the pitch-unstable plant.)
    models = {}
    for frame, mdir in (("sim", src), ("rep103", out)):
        model = PPO.load(str(mdir / "final.zip"), device="cpu")
        venv = DummyVecEnv([lambda f=frame: UmiusiPoseEnv(_cfg(obs_frame=f))])
        vn = VecNormalize.load(str(mdir / "vecnormalize.pkl"), venv)
        vn.training = False
        models[frame] = (model, vn)
    from umiusi_rl.envs.umiusi_pose_env import _OBS_FRAMES
    P = _OBS_FRAMES["rep103"]

    def to_rep103(o):
        o2 = o.copy()
        for i in (0, 3, 6):
            o2[i:i + 3] = P @ o[i:i + 3]
        return o2

    env = UmiusiPoseEnv(_cfg(obs_frame="sim"))
    obs, _ = env.reset(seed=3)
    worst = 0.0
    rng = np.random.default_rng(5)
    for t in range(100):
        m_sim, vn_sim = models["sim"]
        m_rep, vn_rep = models["rep103"]
        a_sim, _ = m_sim.predict(vn_sim.normalize_obs(obs), deterministic=True)
        a_rep, _ = m_rep.predict(vn_rep.normalize_obs(to_rep103(obs)), deterministic=True)
        worst = max(worst, float(np.abs(a_sim - a_rep).max()))
        # drive the plant with the original policy (plus occasional exploration kicks so the
        # comparison covers a spread of states, not one attractor)
        drive = a_sim if t % 7 else np.clip(a_sim + rng.uniform(-0.3, 0.3, 8), -1, 1)
        obs = env.step(drive)[0]
    assert worst < 1e-4, f"converted policy disagrees with original: max action diff {worst}"


if __name__ == "__main__":
    for fn in (test_frames_are_proper_rotations, test_obs_frame_permutes_vector_blocks,
               test_converted_policy_closed_loop_equivalence):
        fn()
        print(f"PASS  {fn.__name__}")
    print("all passed")
