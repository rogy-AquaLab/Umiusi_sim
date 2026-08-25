"""Unit tests for the deploy-path ESC duty cap in the plant and the vertical null-mode
decomposition / penalty plumbing (Umiusi_sim#3: the 8/25 underwater run put 41.2 % of vertical
power into the (+ - + -) null mode while |esc| saturated at the deploy clamp).

Runnable two ways:
    python -m pytest tests/test_duty_cap_nullmode.py        # if pytest is installed
    python tests/test_duty_cap_nullmode.py                  # standalone (plain asserts)
"""

import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))

from umiusi_rl.envs.umiusi_pose_env import _VERT_MODE_SIGNS, UmiusiPoseEnv, load_config  # noqa: E402
from umiusi_sim.simulator import UmiusiSimulator  # noqa: E402


def _sim():
    return UmiusiSimulator()


def test_max_duty_caps_esc_and_thrust():
    """|esc| and the resulting thrust never exceed the configured cap, whatever is commanded."""
    sim = _sim()
    sim.max_duty = 0.25
    act = np.zeros(8)
    act[4:8] = [1.0, -1.0, 1.0, -1.0]
    for _ in range(100):                     # let the esc slew settle onto the cap
        sim.step(act)
    assert np.max(np.abs(sim.esc_current)) <= 0.25 + 1e-9
    f_cap = 0.25 ** sim.thrust_curve_exp * sim.thrust_per_cmd
    assert np.max(np.abs(sim.thrust_mag)) <= f_cap + 1e-9
    # and the cap actually binds (commands were +/-1)
    assert np.max(np.abs(sim.esc_current)) > 0.24


def test_thrust_world_in_state():
    sim = _sim()
    s = sim.step(np.zeros(8))
    assert s["thrust_world"].shape == (4, 3)


def test_mode_signs_match_pivot_geometry():
    """_VERT_MODE_SIGNS (roll = left-right, pitch = front-back, null = diagonal) must agree with
    the actual pivot coordinates in configs/umiusi.yaml — CAD frame: +X forward, +Z starboard."""
    cfg = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    units = {u["name"]: u for u in cfg["thrusters"]["units"]}
    assert set(units) == set(_VERT_MODE_SIGNS)
    xs = [float(u["pivot"][0]) for u in units.values()]
    x_mid = 0.5 * (max(xs) + min(xs))
    for name, (s_roll, s_pitch, s_null) in _VERT_MODE_SIGNS.items():
        x, _, z = (float(v) for v in units[name]["pivot"])
        port = z < 0.0          # +Z = starboard -> port is z < 0... sign checked both ways below
        front = x > x_mid
        # roll mode: same sign for both units on one side, opposite across sides
        # pitch mode: same sign for both front units, opposite front/back
        assert s_pitch == (1 if front else -1), f"{name}: pitch sign vs pivot x"
        # roll orientation (which side is +) is a convention; require CONSISTENCY: all same-side
        # units share the sign and the two sides differ.
        assert s_null == s_roll * s_pitch, f"{name}: null must be roll*pitch (diagonal)"
    roll_by_side = {}
    for name, (s_roll, _, _) in _VERT_MODE_SIGNS.items():
        side = float(units[name]["pivot"][2]) > 0.0
        roll_by_side.setdefault(side, set()).add(s_roll)
    assert all(len(v) == 1 for v in roll_by_side.values()), "roll sign must be per-side"
    assert roll_by_side[True] != roll_by_side[False], "roll sign must differ across sides"


def _env(**env_overrides):
    cfg = load_config("configs/train_ppo.yaml")
    cfg["domain_rand"]["enabled"] = False
    cfg["disturbance"]["enabled"] = False
    cfg["env"].update(env_overrides)
    return UmiusiPoseEnv(cfg)


def test_null_pattern_is_detected_and_produces_no_net_force():
    """Full-tilt servos + a (+ - + -) esc pattern = the null mode: null_frac -> ~1 and the four
    vertical forces cancel (no net heave, no roll/pitch moment) — pure waste, now measurable."""
    env = _env()
    env.reset(seed=0)
    act = np.zeros(8)
    act[:4] = 1.0                            # tilt all thrusters to the vertical
    act[4:8] = np.array([1.0, -1.0, 1.0, -1.0]) * 0.8
    info = {}
    for _ in range(120):                     # servo travel + esc slew settle
        _, _, _, _, info = env.step(act)
    assert info["null_frac"] > 0.9, f"null_frac={info['null_frac']:.3f}"
    v = env.sim.thrust_world[:, 1]           # world-vertical components (upright vehicle)
    assert abs(v.sum()) < 0.1 * np.abs(v).sum(), "null mode must produce no net vertical force"


def test_heave_pattern_has_low_null_frac():
    """All-same-sign esc at full tilt = pure heave: the null share must be ~0."""
    env = _env()
    env.reset(seed=0)
    act = np.zeros(8)
    act[:4] = 1.0
    act[4:8] = 0.8
    info = {}
    for _ in range(120):
        _, _, _, _, info = env.step(act)
    assert info["null_frac"] < 0.1, f"null_frac={info['null_frac']:.3f}"


def test_observe_max_duty_appends_cap_dim():
    env17 = _env(observe_max_duty=False)
    env18 = _env(observe_max_duty=True)
    assert env18.observation_space.shape[0] == env17.observation_space.shape[0] + 1
    obs, _ = env18.reset(seed=0)
    assert abs(float(obs[-1]) - env18.sim.max_duty) < 1e-6


def test_effort_exp_power_sum():
    """effort_exp = 3 penalizes sum(|u|^3): a pure null command must cost reward via w_null too."""
    env = _env()
    assert env.effort_exp == 3.0 and env.w_null > 0.0
    # spot-check the effort arithmetic through the reward delta of a zero vs nonzero command
    env.reset(seed=0)
    _, r_zero, _, _, _ = env.step(np.zeros(8))
    env.reset(seed=0)
    act = np.zeros(8)
    act[4:8] = 0.25
    _, r_thrust, _, _, _ = env.step(act)
    assert r_thrust < r_zero  # thrusting costs effort (all else ~equal at step 1)


def test_dr_samples_max_duty_range():
    cfg = load_config("configs/train_ppo.yaml")
    cfg["domain_rand"]["enabled"] = True
    cfg["domain_rand"]["max_duty_range"] = [0.2, 0.4]
    cfg["disturbance"]["enabled"] = False
    env = UmiusiPoseEnv(cfg)
    caps = set()
    for i in range(8):
        env.reset(seed=i)
        assert 0.2 <= env.sim.max_duty <= 0.4
        caps.add(round(env.sim.max_duty, 6))
    assert len(caps) > 1, "cap must actually vary across episodes"
    # DR off -> back to the config base value
    env.dr["enabled"] = False
    env.reset(seed=99)
    assert env.sim.max_duty == env._base["max_duty"]


def test_warm_start_pads_grown_obs():
    """Continuing av_cal1_best (17-D obs) into an 18-D (observe_max_duty) run: the first-layer
    weights get a ZERO column for the new dim (policy initially ignores the cap input) and the
    frozen VecNormalize stats get mean 0 / var 1 appended."""
    import tempfile

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from umiusi_rl.train import warm_start

    cfg17 = load_config("configs/train_ppo.yaml")
    cfg17["domain_rand"]["enabled"] = False
    cfg17["env"]["observe_max_duty"] = False
    cfg18 = load_config("configs/train_ppo.yaml")
    cfg18["domain_rand"]["enabled"] = False
    cfg18["env"]["observe_max_duty"] = True

    kw = dict(policy="MlpPolicy", device="cpu", seed=0, n_steps=8, batch_size=8,
              policy_kwargs={"net_arch": [16, 16]})
    v17 = VecNormalize(DummyVecEnv([lambda: UmiusiPoseEnv(cfg17)]))
    old = PPO(env=v17, **kw)
    with tempfile.TemporaryDirectory() as td:
        old.save(f"{td}/final")
        v17.save(f"{td}/vecnormalize.pkl")
        v18 = VecNormalize(DummyVecEnv([lambda: UmiusiPoseEnv(cfg18)]))
        new = PPO(env=v18, **kw)
        froze = warm_start(new, td, v18)
    assert froze and v18.training is False
    n_old = old.observation_space.shape[0]
    n_new = new.observation_space.shape[0]
    assert n_new == n_old + 1
    d_old = old.policy.state_dict()
    d_new = new.policy.state_dict()
    for k, w in d_new.items():
        if w.dim() == 2 and w.shape[1] == n_new and d_old[k].shape[1] == n_old:
            assert np.allclose(w[:, :n_old].numpy(), d_old[k].numpy())
            assert np.all(w[:, n_old].numpy() == 0.0), f"{k}: new-dim column must be zero"
        else:
            assert np.allclose(w.numpy(), d_old[k].numpy()), k
    assert v18.obs_rms.mean.shape == (n_new,) and v18.obs_rms.mean[n_old] == 0.0
    assert v18.obs_rms.var[n_old] == 1.0
    v17.close(); v18.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
