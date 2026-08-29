"""LagrangeCallback: adaptive constraint multipliers move toward explicit targets, and the
env actually applies the multipliers to the reward."""

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config
from umiusi_rl.train import LagrangeCallback


class _StubVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, **kwargs):
        self.calls.append((name, kwargs))


def _cb(eta=0.5, probe=(0.4, 0.0)):
    import types

    cb = LagrangeCallback({"eta": eta, "ori_target": 0.2, "track_target": 0.15, "lambda_max": 8.0,
                           "probe_every": 1}, {})
    stub = _StubVecEnv()
    cb.model = types.SimpleNamespace(get_env=lambda: stub)  # training_env property reads this
    cb._stub = stub
    cb._probe = lambda: probe          # stub the deterministic probe
    return cb


_INFO = {"ori_err": 0.4, "vel_track": 0.0, "step_idx": 400, "vel_cmd_speed": 0.1}


def test_violation_grows_multiplier_and_satisfaction_shrinks_it():
    cb = _cb(probe=(0.4, 0.0))         # ori 0.4 violates 0.2; track 0.0 satisfies 0.15
    cb._on_rollout_end()
    # ori violated (0.4 > 0.2) -> lambda up; track satisfied (0.0 < 0.15) -> lambda down
    assert cb.lam["ori"] > 1.0
    assert cb.lam["track"] < 1.0
    assert cb._stub.calls and cb._stub.calls[-1][0] == "apply_train_ctx"


def test_multiplier_is_clipped():
    cb = _cb(eta=5.0, probe=(2.0, 1.0))
    for _ in range(20):
        cb._on_rollout_end()
    assert cb.lam["ori"] <= 8.0 + 1e-9
    assert cb.lam["track"] <= 8.0 + 1e-9


def test_probe_without_samples_leaves_multipliers_alone():
    cb = _cb(probe=(None, None))       # e.g. an episode with no commanded velocity
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
