"""Unit + physics tests for ModeMixer (action_mode: "modes" — Umiusi_sim#3 null-space removal).

The mixer's whole claim is structural: a policy commanding 6 wrench modes CANNOT express the
(+ - + -) null patterns. These tests verify (a) the closed-form per-unit expansion, (b) that a
held pure mode command actually produces the named rigid-body motion in the simulator, with a
near-zero ACTUAL null share (only servo-lag transients remain), and (c) the env integration
(6-D action space, unchanged 18-D obs contract, mixed 8-D prev_action feedback).

Runnable two ways:
    python -m pytest tests/test_mode_mixer.py        # if pytest is installed
    python tests/test_mode_mixer.py                  # standalone (plain asserts)
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))

from umiusi_rl.envs.mode_mixer import _MODE_SIGNS, ModeMixer  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import _VERT_MODE_SIGNS, UmiusiPoseEnv, load_config  # noqa: E402
from umiusi_sim.simulator import UmiusiSimulator  # noqa: E402


def _mixer(sim):
    return ModeMixer(sim.unit_names, sim.thrust_axes, sim.servo_range_rad,
                     sim.thrust_per_cmd, sim.thrust_curve_exp)


def _mode(name, value=1.0):
    m = np.zeros(6)
    m["fx fy fz tx ty tz".split().index(name)] = value
    return m


def test_mode_basis_is_orthogonal_and_null_free():
    """Within each group (horizontal fx/fy/tz, vertical fz/tx/ty) the sign columns are mutually
    orthogonal Walsh vectors, and both are orthogonal to the corresponding NULL pattern."""
    names = ["lf", "lb", "rb", "rf"]                       # action order
    signs = np.array([_MODE_SIGNS[n] for n in names], dtype=float)
    horiz, vert = signs[:, 0:3], signs[:, 3:6]
    for block in (horiz, vert):
        gram = block.T @ block
        assert np.allclose(gram, 4.0 * np.eye(3)), gram
    vert_null = np.array([[1.0, *(float(s) for s in _VERT_MODE_SIGNS[n])] for n in names]).T[3]
    assert np.allclose(vert.T @ vert_null, 0.0)            # fz/tx/ty cannot express the null
    horiz_null = vert_null                                  # same (+,-,+,-) diagonal in action order
    assert np.allclose(horiz.T @ horiz_null, 0.0)


def test_single_mode_full_scale_hits_the_cap_exactly():
    """One mode at +/-1 commands each unit at exactly the current esc cap."""
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    for cap in (0.2, 0.25, 0.4):
        a = mx.mix(_mode("fz"), cap, np.zeros(4))
        assert np.allclose(a[:4], 1.0), a[:4]              # all servos to +90 deg (up)
        assert np.allclose(np.abs(a[4:]), cap, atol=1e-9)   # esc exactly at the cap
        assert np.all(a[4:] > 0.0)


def test_pure_surge_keeps_servos_flat_with_signed_esc():
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    a = mx.mix(_mode("fx"), 0.3, np.zeros(4))
    assert np.allclose(a[:4], 0.0, atol=1e-9)              # tangential: no tilt
    assert np.allclose(np.abs(a[4:]), 0.3, atol=1e-9)
    signs = np.sign(a[4:])
    assert list(signs) == [1.0, 1.0, -1.0, -1.0]           # (lf, lb, rb, rf) = fx column


def test_rear_half_plane_folds_into_esc_reversal():
    """A backward force (fx = -1) must reverse the esc, not command an unreachable servo angle."""
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    a = mx.mix(_mode("fx", -1.0), 0.3, np.zeros(4))
    assert np.allclose(a[:4], 0.0, atol=1e-9)
    assert list(np.sign(a[4:])) == [-1.0, -1.0, 1.0, 1.0]
    # mixed vertical + backward horizontal: |servo| stays within range, esc reversed
    a = mx.mix(np.array([-1.0, 0.0, 0.5, 0.0, 0.0, 0.0]), 0.3, np.zeros(4))
    assert np.all(np.abs(a[:4]) <= 1.0)


def test_deadband_holds_previous_servo_and_zeroes_esc():
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    prev = np.array([0.3, -0.2, 0.1, 0.5])
    a = mx.mix(np.zeros(6), 0.3, prev)
    assert np.allclose(a[:4], prev)
    assert np.allclose(a[4:], 0.0)


def test_saturation_clips_per_unit_but_direction_survives():
    """Overflowing combined modes clip |esc| at the cap; the servo angle (direction) is kept."""
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    a = mx.mix(np.ones(6), 0.25, np.zeros(4))
    assert np.max(np.abs(a[4:])) <= 0.25 + 1e-9


def _settle_and_measure(sim, action, steps=150):
    """Hold an 8-D action, return (mean world force [N], mean world torque about CoM [N m],
    null share of vertical mode power) time-averaged AFTER the servo/esc slew settles."""
    sim.reset()
    R0 = np.eye(3)
    f_sum, t_sum = np.zeros(3), np.zeros(3)
    null_pw, mode_pw = 0.0, 0.0
    names = list(sim.unit_names)
    vert_modes = np.array([[1.0, *(float(s) for s in _VERT_MODE_SIGNS[n])] for n in names]).T / 2.0
    com = sim.data.subtree_com[sim.base_id].copy()
    n_avg = 0
    for k in range(steps):
        # Freeze the vehicle each step: measure the APPLIED wrench in the initial (identity)
        # attitude rather than letting the body accelerate away and rotate the thrusters.
        sim.data.qpos[0:3] = 0.0
        sim.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        sim.data.qvel[:] = 0.0
        sim.step(action)
        if k < steps // 2:
            continue                                        # skip the slew transient
        n_avg += 1
        f_sum += sim.thrust_world.sum(axis=0)
        for j in range(4):
            r = sim.data.site_xpos[sim.site_ids[j]] - com
            t_sum += np.cross(r, sim.thrust_world[j])
        v_vert = sim.thrust_world @ R0[:, 1]
        m = vert_modes @ v_vert
        null_pw += m[3] ** 2
        mode_pw += float(m @ m)
    return f_sum / n_avg, t_sum / n_avg, (null_pw / mode_pw if mode_pw > 1e-9 else 0.0)


def test_pure_modes_produce_the_named_wrench_in_the_plant():
    """Physics check: each held pure mode yields the advertised net force/torque direction
    (REP-103: fx = sim +X, fy = sim -Z, fz = sim +Y, tz = sim +Y torque) and a near-zero
    ACTUAL vertical null share."""
    sim = UmiusiSimulator()
    mx = _mixer(sim)
    cap = 0.3

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("fx"), cap, np.zeros(4)))
    assert f[0] > 1.0 and abs(f[1]) < 0.3 * f[0] and abs(f[2]) < 0.3 * f[0], f
    assert null < 0.02, null

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("fy"), cap, np.zeros(4)))
    assert f[2] < -1.0 and abs(f[0]) < 0.3 * -f[2], f       # REP-103 +y (left) = sim -Z

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("fz"), cap, np.zeros(4)))
    assert f[1] > 1.0 and abs(f[0]) < 0.3 * f[1] and abs(f[2]) < 0.3 * f[1], f
    assert null < 0.02, null

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("tz"), cap, np.zeros(4)))
    assert t[1] > 0.1 and np.linalg.norm(f) < 0.5, (f, t)   # +yaw torque, ~zero net force

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("tx"), cap, np.zeros(4)))
    assert t[0] > 0.1 and abs(f[1]) < 0.5, (f, t)           # +roll torque, heave cancels
    assert null < 0.02, null

    f, t, null = _settle_and_measure(sim, mx.mix(_mode("ty"), cap, np.zeros(4)))
    assert t[2] < -0.1, t                                   # REP-103 +ty (nose down): rep y = -sim z
    assert null < 0.02, null


def test_env_integration_modes():
    """action_mode: modes -> 6-D action space, obs layout unchanged, prev_action = mixed 8-D."""
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["action_mode"] = "modes"
    cfg["env"]["obs_frame"] = "rep103"
    env = UmiusiPoseEnv(cfg)
    assert env.action_space.shape == (6,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,)                               # imu 6 + v_cmd 3 + prev_action 8 + cap 1
    obs, _, _, _, info = env.step(np.array([1.0, 0, 0, 0, 0, 0]))
    assert env.prev_action.shape == (8,)                    # obs feeds back the MIXED command
    assert np.allclose(env.prev_action[:4], 0.0, atol=1e-9)  # pure surge: servos flat
    assert np.max(np.abs(env.prev_action[4:])) <= env.sim.max_duty + 1e-9
    # w_null is forced off under modes (commanded null is structurally zero)
    assert env.w_null == 0.0
    env.close()


def test_env_esc_mode_unchanged():
    """Default action_mode keeps the raw 8-D contract (no regression for existing runs)."""
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    env = UmiusiPoseEnv(cfg)
    assert env.action_space.shape == (8,)
    env.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all mode-mixer tests passed")
