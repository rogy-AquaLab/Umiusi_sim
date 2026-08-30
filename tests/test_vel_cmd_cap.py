"""vel_cmd_cap_frac: velocity commands stay within the episode cap's reachable speed.

The open-loop surge ceiling is ~0.68 m/s per unit of max_duty (measured 2026-08-26);
with vel_cmd_cap_frac set, sampled commands must respect frac * 0.68 * max_duty, and
with the key absent the legacy flat U(0, vel_cmd_max) sampling must be unchanged.
"""

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config


def _cfg(**env_overrides):
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["vel_cmd_zero_prob"] = 0.0
    cfg["env"].update(env_overrides)
    return cfg


def test_vel_cmd_respects_cap_ceiling():
    cfg = _cfg(vel_cmd_cap_frac=0.8)
    cfg.setdefault("domain_rand", {})["enabled"] = True  # cap varies per episode
    env = UmiusiPoseEnv(cfg)
    try:
        for i in range(30):
            env.reset(seed=i)
            # expected ceiling hardcoded (NOT read from the implementation): 0.8 * 0.68 * cap
            assert np.linalg.norm(env.v_cmd) <= 0.8 * 0.68 * env.sim.max_duty + 1e-9
    finally:
        env.close()


def test_vel_cmd_cap_off_by_default():
    env = UmiusiPoseEnv(_cfg())
    try:
        speeds = [float(np.linalg.norm((env.reset(seed=i), env.v_cmd)[1])) for i in range(40)]
    finally:
        env.close()
    # legacy flat U(0, 0.4): commands above any capped ceiling must still occur
    assert max(speeds) > 0.30
