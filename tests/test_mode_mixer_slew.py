"""Mode-command slew (env.mode_slew_per_s): the wrench command is rate-limited in MODE
coordinates, so fast dithering cannot reach the plant and every intermediate command stays
in the null-free subspace.

Expected step size hardcoded (not read from the implementation): 2.0/s at 50 Hz = 0.04/step.
"""

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config


def _env(**env_overrides):
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["action_mode"] = "modes"
    cfg["env"].update(env_overrides)
    return UmiusiPoseEnv(cfg)


def test_mode_command_rate_is_limited():
    env = _env(mode_slew_per_s=2.0)
    try:
        env.reset(seed=0)
        prev = env._mode_prev_modes.copy()
        for k in range(30):  # alternate extreme commands: the applied modes must walk, not jump
            target = np.full(6, 1.0 if k % 2 == 0 else -1.0)
            env.step(target)
            applied = env._mode_prev_modes
            assert np.all(np.abs(applied - prev) <= 0.04 + 1e-9), \
                f"mode command jumped {np.abs(applied - prev).max():.3f} (> 0.04/step)"
            prev = applied.copy()
    finally:
        env.close()


def test_mode_command_converges_to_held_target():
    env = _env(mode_slew_per_s=2.0)
    try:
        env.reset(seed=0)
        for _ in range(60):  # 1.2 s at 50 Hz: full swing takes 1.0 s
            env.step([1, 0, 0, 0, 0, 0])
        assert np.allclose(env._mode_prev_modes, [1, 0, 0, 0, 0, 0], atol=1e-6)
    finally:
        env.close()


def test_rate_action_integrates_and_holds():
    env = _env(mode_slew_per_s=2.0, mode_rate_action=True)
    try:
        env.reset(seed=0)
        # a = +1 on fx: the applied mode walks up by exactly the slew step (0.04/step)
        for k in range(10):
            env.step([1, 0, 0, 0, 0, 0])
            assert abs(env._mode_prev_modes[0] - 0.04 * (k + 1)) < 1e-9
        held = env._mode_prev_modes.copy()
        for _ in range(5):  # a = 0 means HOLD
            env.step(np.zeros(6))
        assert np.allclose(env._mode_prev_modes, held)
        for _ in range(60):  # saturates at +1, never beyond
            env.step([1, 0, 0, 0, 0, 0])
        assert abs(env._mode_prev_modes[0] - 1.0) < 1e-9
    finally:
        env.close()


def test_hi_speed_episodes_are_sampled():
    env = _env(mode_slew_per_s=2.0, vel_cmd_cap_frac=0.8, vel_cmd_hi_prob=1.0,
               vel_cmd_zero_prob=0.0)
    try:
        for i in range(10):
            env.reset(seed=i)
            hi = 0.8 * 0.68 * env.sim.max_duty
            v = float(np.linalg.norm(env.v_cmd))
            assert 0.9 * hi - 1e-9 <= v <= hi + 1e-9, f"hi episode sampled {v} vs ceiling {hi}"
    finally:
        env.close()


def test_slew_off_is_backward_compatible():
    a = {}
    for slew in (0.0, 2.0):
        env = _env(mode_slew_per_s=slew)
        try:
            env.reset(seed=7)
            # constant zero command: slewed and unslewed paths must act identically
            for _ in range(5):
                obs, r, *_ = env.step(np.zeros(6))
            a[slew] = (obs.copy(), r)
        finally:
            env.close()
    assert np.allclose(a[0.0][0], a[2.0][0]) and a[0.0][1] == a[2.0][1]
