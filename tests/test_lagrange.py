"""LagrangeCallback: adaptive constraint multipliers move toward explicit targets, and the
env actually applies the multipliers to the reward."""

import numpy as np
import pytest

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config
from umiusi_rl.train import LagrangeCallback


class _StubVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, **kwargs):
        self.calls.append((name, kwargs))


def _cb(eta=0.5, probe=(0.4, 0.0, None), **cfg_extra):
    import types

    cb = LagrangeCallback({"eta": eta, "ori_target": 0.2, "track_target": 0.15, "lambda_max": 8.0,
                           "probe_every": 1, **cfg_extra}, {})
    stub = _StubVecEnv()
    cb.model = types.SimpleNamespace(get_env=lambda: stub)  # training_env property reads this
    cb._stub = stub
    cb._probe = lambda: probe          # stub the deterministic probe
    return cb


_INFO = {"ori_err": 0.4, "vel_track": 0.0, "step_idx": 400, "vel_cmd_speed": 0.1}


def test_violation_grows_multiplier_and_satisfaction_shrinks_it():
    cb = _cb(probe=(0.4, 0.0, None))   # ori 0.4 violates 0.2; track 0.0 satisfies 0.15
    cb._on_rollout_end()
    # ori violated (0.4 > 0.2) -> lambda up; track satisfied (0.0 < 0.15) -> lambda down
    assert cb.lam["ori"] > 1.0
    assert cb.lam["track"] < 1.0
    assert cb._stub.calls and cb._stub.calls[-1][0] == "apply_train_ctx"


def test_multiplier_is_clipped():
    cb = _cb(eta=5.0, probe=(2.0, 1.0, None))
    for _ in range(20):
        cb._on_rollout_end()
    assert cb.lam["ori"] <= 8.0 + 1e-9
    assert cb.lam["track"] <= 8.0 + 1e-9


def test_probe_without_samples_leaves_multipliers_alone():
    cb = _cb(probe=(None, None, None))  # e.g. an episode with no commanded velocity
    lam_before = dict(cb.lam)
    cb._on_rollout_end()
    assert cb.lam == lam_before


def test_apply_train_ctx_pierces_monitor_wrapper():
    # REGRESSION (2026-08-27): plain venv.set_attr sets attributes on the Monitor wrapper,
    # not the env — every curriculum and the Lagrange multipliers were silently inert up to
    # av_mode9. env_method("apply_train_ctx", ...) resolves through wrapper getattr and must
    # reach the inner env.
    from stable_baselines3.common.env_util import make_vec_env

    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["action_mode"] = "modes"
    venv = make_vec_env(UmiusiPoseEnv, n_envs=1, seed=0, env_kwargs={"config": cfg})
    try:
        venv.env_method("apply_train_ctx", econ_ramp=0.25, lagrange={"ori": 2.5})
        inner = venv.envs[0].unwrapped
        assert inner.econ_ramp == 0.25
        assert inner.lagrange == {"ori": 2.5}
    finally:
        venv.close()


def test_apply_train_ctx_pierces_subproc_workers():
    # The training runs use n_envs=8 -> SubprocVecEnv, a different code path than the
    # DummyVecEnv above (worker process, pickled call). Both resolve the method with
    # get_wrapper_attr, but the training path is the one that actually matters.
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["action_mode"] = "modes"
    venv = make_vec_env(UmiusiPoseEnv, n_envs=2, seed=0, env_kwargs={"config": cfg},
                        vec_env_cls=SubprocVecEnv)
    try:
        venv.env_method("apply_train_ctx", econ_ramp=0.25, lagrange={"ori": 2.5})
        # read back THROUGH the worker (get_attr would hit the Monitor wrapper, so ask the env)
        assert venv.env_method("apply_train_ctx") is not None  # no-op call must not raise
        econ = [e["econ_ramp"] for e in venv.env_method("_train_ctx_snapshot")]
        assert econ == [0.25, 0.25]
    finally:
        venv.close()


def test_env_applies_ori_multiplier():
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["action_mode"] = "modes"
    rewards = {}
    for lam in (1.0, 5.0):
        env = UmiusiPoseEnv(cfg)
        try:
            env.reset(seed=3)  # random tilted target -> nonzero ori_err from step one
            env.lagrange = {"ori": lam}
            _, r, *_ = env.step(np.zeros(6))
            rewards[lam] = r
        finally:
            env.close()
    # a larger ori multiplier makes the same (erring) state strictly worse
    assert rewards[5.0] < rewards[1.0]


def test_effort_constraint_is_opt_in_and_tracks_hover_duty():
    """The effort constraint only exists when a target is configured, and it responds to the
    HOVER duty fraction — the signal the whole cap-normalization fix is about."""
    off = _cb(probe=(0.1, 0.0, 0.9))
    assert "effort" not in off.targets
    off._on_rollout_end()
    assert "effort" not in off.lam, "no effort_target configured -> no multiplier"

    on = _cb(probe=(0.1, 0.0, 0.9), effort_target=0.25)   # hovering at 90 % of cap: violated
    on._on_rollout_end()
    assert on.lam["effort"] > 1.0

    ok = _cb(probe=(0.1, 0.0, 0.1), effort_target=0.25)   # hovering at 10 % of cap: satisfied
    ok._on_rollout_end()
    assert ok.lam["effort"] < 1.0


def test_env_applies_the_effort_multiplier_to_the_reward():
    """A multiplier nobody multiplies by is worthless — pin that the env reads it."""
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"]["action_mode"] = "esc"
    cfg.setdefault("domain_rand", {})["enabled"] = False
    env = UmiusiPoseEnv(cfg)
    action = np.zeros(env.action_space.shape[0])
    action[4:] = env.sim.max_duty            # full duty -> a large effort penalty
    env.reset(seed=0)
    env.lagrange = {}
    _o, r_base, *_ = env.step(action)
    env.reset(seed=0)
    env.lagrange = {"effort": 4.0}
    _o, r_high, *_ = env.step(action)
    env.close()
    assert r_high < r_base - 1e-9, (r_base, r_high)


def _modes_env(**reward_overrides):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"]["action_mode"] = "modes"
    cfg["env"]["task"] = "attitude_velocity"
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg.setdefault("disturbance", {})["enabled"] = False
    cfg["reward"].update(reward_overrides)
    return UmiusiPoseEnv(cfg)


def _cmd_perp_after(env, v_cmd, rate):
    """Drive one mode-rate step and read back the uncommanded-translation signal."""
    env.reset(seed=0)
    env.v_cmd = np.asarray(v_cmd, dtype=float)
    _o, _r, _t, _tr, info = env.step(np.asarray(rate, dtype=float))
    return info["cmd_perp"]


def test_cmd_perp_ignores_wrench_along_the_commanded_direction():
    """Sway is a legitimate DOF: commanding it must not be penalised (the user's requirement)."""
    env = _modes_env()
    # v_cmd along sim -z == REP-103 +y (left): a pure fy wrench is exactly what was asked for.
    along = _cmd_perp_after(env, [0.0, 0.0, -0.3], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # the same fy wrench with the command pointing forward instead is entirely uncommanded
    across = _cmd_perp_after(env, [0.3, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    env.close()
    assert along == pytest.approx(0.0, abs=1e-9), f"commanded sway must be free, got {along}"
    assert across > 0.0, across


def test_cmd_perp_penalises_any_translation_while_holding_station():
    """v_cmd = 0 -> nothing is commanded, so the whole (fx, fy) magnitude counts."""
    env = _modes_env()
    hold = _cmd_perp_after(env, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    env.close()
    assert hold > 0.0


def test_cmd_perp_ignores_heave_and_attitude():
    """fz must stay free (depth holding needs steady heave) and so must the moments."""
    env = _modes_env()
    for rate in ([0, 0, 1.0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0], [0, 0, 0, 0, 1.0, 0], [0, 0, 0, 0, 0, 1.0]):
        assert _cmd_perp_after(env, [0.0, 0.0, 0.0], rate) == pytest.approx(0.0, abs=1e-9), rate
    env.close()


def test_w_cmd_perp_is_off_by_default_and_lowers_reward_when_enabled():
    off, on = _modes_env(), _modes_env(w_cmd_perp=5.0)
    rate = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    for env in (off, on):
        env.reset(seed=0)
        env.v_cmd = np.zeros(3)
    _o, r_off, *_ = off.step(rate)
    _o, r_on, *_ = on.step(rate)
    off.close(); on.close()
    assert r_on < r_off - 1e-9, (r_off, r_on)
