"""domain_rand.thrust_slew_range: the ESC ramp is sampled per episode.

The deploy node applies its own runtime-settable thrust_slew_per_s, and the physical ESC/prop
ramp is unmeasured and sits in series with it, so the real rate is min(parameter, hardware) and
a policy must not lean on the one value the sim happened to train with — the same argument that
already randomizes the servo rate.
"""

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

_RANGE = [1.0, 10.0]


def _cfg(dr_enabled, **dr_overrides):
    cfg = load_config("configs/train_ppo.yaml")
    cfg.setdefault("domain_rand", {})["enabled"] = dr_enabled
    cfg["domain_rand"].update(dr_overrides)
    return cfg


def test_sampled_per_episode_and_inside_the_range():
    env = UmiusiPoseEnv(_cfg(True, thrust_slew_range=_RANGE))
    seen = []
    for ep in range(24):
        env.reset(seed=ep)
        seen.append(env.sim.thrust_slew)
    env.close()
    assert min(seen) >= _RANGE[0] and max(seen) <= _RANGE[1], (min(seen), max(seen))
    assert len(set(seen)) > 1, "the ramp must vary per episode, not be pinned"
    # Spread over the band, not clustered at one end (a mis-wired sampler can still stay in range).
    assert max(seen) - min(seen) > 0.5 * (_RANGE[1] - _RANGE[0]), sorted(seen)


def test_absent_key_leaves_the_config_value_untouched():
    """DR on but no range configured -> the plant keeps configs/umiusi.yaml's value."""
    cfg = _cfg(True)
    cfg["domain_rand"].pop("thrust_slew_range", None)
    env = UmiusiPoseEnv(cfg)
    nominal = env.sim.cfg["thrusters"]["thrust_slew_per_s"]
    for ep in range(5):
        env.reset(seed=ep)
        assert env.sim.thrust_slew == nominal
    env.close()


def test_dr_off_restores_the_nominal_ramp():
    """Even with a range configured, DR disabled must give the nominal plant (eval conditions)."""
    env = UmiusiPoseEnv(_cfg(False, thrust_slew_range=_RANGE))
    nominal = env.sim.cfg["thrusters"]["thrust_slew_per_s"]
    for ep in range(5):
        env.reset(seed=ep)
        assert env.sim.thrust_slew == nominal
    env.close()


def test_the_sampled_ramp_actually_limits_the_plant():
    """A range is worthless if the value never reaches the ESC integrator — pin the effect."""
    env = UmiusiPoseEnv(_cfg(True, thrust_slew_range=[1.0, 1.0]))
    env.reset(seed=0)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    act = np.zeros(env.action_space.shape[0])
    act[4:] = 1.0                      # full-scale esc step: the ramp is the only thing limiting it
    prev = env.sim.esc_current.copy()
    rates = []
    for _ in range(10):
        _o, _r, _t, _tr, info = env.step(act)
        cur = info["esc_applied"]
        rates.append(float(np.max(np.abs(cur - prev)) / dt))
        prev = cur.copy()
    env.close()
    assert max(rates) <= 1.0 + 1e-9, max(rates)
