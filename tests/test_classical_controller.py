"""Two invariants the classical controller silently broke, so they are tests now.

1. THE CONTROLLER MAY NOT READ THE RANDOMIZED PLANT. `_apply_domain_rand` REBINDS
   sim.thrust_per_cmd / thrust_curve_exp / drag / added mass on every reset. The observer and the
   allocator held a reference to `sim` and read those attributes at control time, so under DR they
   were using the episode's true values — the very numbers the controller is supposed to be robust
   to. Every DR result was flattered by an amount nobody could see. A deployed controller carries
   the CALIBRATED constants and nothing else.

2. THE ATTITUDE GAINS MUST MEAN A TORQUE, NOT A MODE. A mode is normalized by the full-cap wrench,
   so a fixed gain in mode units delivers a torque proportional to cap**exp — the loop gain swings
   4x across the deploy range (max_duty 0.2-0.4, moved by hand in the field). tools/dr_ablation.py
   measured the cap alone at 2.3x the attitude error despite the cap being OBSERVED.

Both are the same class of bug as tests/test_reward_cap_invariance.py: a quantity that must not
depend on the cap, quietly depending on the cap.
"""

import sys
from pathlib import Path

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))          # the cross-cutting glue lives in tools/

from classical_control import build_controller  # noqa: E402

CAPS = (0.2, 0.4)


def _env(dr):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="esc", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = dr
    cfg.setdefault("disturbance", {})["enabled"] = False
    return UmiusiPoseEnv(cfg)


def test_controller_constants_do_not_follow_domain_randomization():
    env = _env(dr=True)
    ctl = build_controller(env)
    nominal = (ctl.plant.thrust_per_cmd, ctl.plant.thrust_curve_exp)
    drag = ctl.obs.lin.copy()
    for ep in range(8):
        env.reset(seed=ep)
    # the plant really did move, otherwise this test proves nothing
    assert (env.sim.thrust_per_cmd, env.sim.thrust_curve_exp) != nominal
    assert not np.allclose(env.sim.drag_lin[:3], drag)
    # ...and the controller did not
    assert (ctl.plant.thrust_per_cmd, ctl.plant.thrust_curve_exp) == nominal
    assert (ctl.obs.plant.thrust_per_cmd, ctl.obs.plant.thrust_curve_exp) == nominal
    assert np.allclose(ctl.obs.lin, drag)
    env.close()


def _attitude_torque(ctl, cap, ori_err):
    """The physical moment the controller asks for at this cap [N] in mode-x-wrench units."""
    ctl.reset()
    m = ctl.wrench(ori_err, np.zeros(3), np.zeros(3), np.zeros(3), cap)
    assert np.all(np.abs(m[3:6]) < 1.0), "saturated — the test would be measuring the clip"
    return np.asarray(m[3:6]) * ctl.f_max_total(ctl.cap)


def test_attitude_torque_is_cap_invariant():
    env = _env(dr=False)
    ori_err = np.array([0.05, -0.03, 0.02])
    ctl = build_controller(env, cap_tau=0.0)      # no cap filter lag, so one step is enough
    tau = [_attitude_torque(ctl, cap, ori_err) for cap in CAPS]
    assert np.allclose(tau[0], tau[1], rtol=1e-9), f"attitude torque moved with the cap: {tau}"

    # and the normalization is what does it: without it the same gains scale as cap ** exp
    old = build_controller(env, cap_tau=0.0, cap_norm=False)
    tau_old = [_attitude_torque(old, cap, ori_err) for cap in CAPS]
    ratio = np.linalg.norm(tau_old[1]) / np.linalg.norm(tau_old[0])
    assert ratio > 3.0, f"expected the un-normalized gains to scale with the cap, got {ratio:.2f}x"
    env.close()


def test_buoyancy_trim_holds_the_same_force_at_every_cap():
    """Same argument for the heave trim: net buoyancy is a constant, so the mode must move."""
    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0)
    force = []
    for cap in CAPS:
        ctl.reset()
        m = ctl.wrench(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), cap)
        force.append(m[2] * ctl.f_max_total(ctl.cap))
    assert np.allclose(force[0], force[1], rtol=1e-9), f"trim force moved with the cap: {force}"
    assert force[0] < 0.0, "positively buoyant hull: the trim must push DOWN"
    env.close()


def test_cruise_reference_speed_is_the_plant_solve_not_the_linear_constant():
    """`VEL_PER_CAP * cap` is a line fitted at the deploy cap; the cruise loop must not use it.

    Thrust goes as cap**exp and drag as v**2, so reachable speed is NOT proportional to the cap.
    Measured on the nominal plant: cap 0.5 gives 0.458 m/s where the line says 0.340 (41 % low).
    Normalizing the feedforward by the low number made the same command ask for ~1.35x the thrust
    it needed, and it got worse the further the operator moved the cap from 0.25.
    """
    from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP

    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0)
    for cap in CAPS + (0.5,):
        v = ctl.reachable_speed(cap)
        lin, quad = ctl.plant.drag_lin[0], ctl.plant.drag_quad[0]
        thrust = 4.0 * ctl.plant.thrust_per_cmd * cap ** ctl.plant.thrust_curve_exp
        assert np.isclose(lin * v + quad * v * v, thrust), f"cap {cap}: thrust != drag at v={v}"
    # and it is genuinely a different number from the line, well outside the fit point
    assert ctl.reachable_speed(0.5) > 1.3 * VEL_PER_CAP * 0.5
    env.close()


def test_unreachable_velocity_command_is_clamped_not_saturated():
    """Above the cap's reachable speed the command must be scaled back, keeping its direction.

    Otherwise the horizontal modes pin at the clip and the saturation scaling takes the authority
    away from the attitude axes — the operator lowering the cap would cost attitude, not speed.
    """
    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0)
    cap = 0.25
    v_ref = ctl.reachable_speed(cap)
    direction = np.array([0.6, 0.8, 0.0])          # unit, off-axis so a per-axis clip would show
    # Feed the observer's estimate back at the ceiling so the velocity error is zero and only the
    # feedforward is under test — otherwise both cases saturate and the assertions are vacuous.
    v_hat = direction * v_ref

    ctl.reset()
    at = ctl.wrench(np.zeros(3), np.zeros(3), direction * v_ref, v_hat, cap)
    ctl.reset()
    over = ctl.wrench(np.zeros(3), np.zeros(3), direction * v_ref * 5.0, v_hat, cap)
    assert np.allclose(at[:2], over[:2]), f"command above the ceiling was not clamped: {at} {over}"
    # unsaturated, so the agreement above is the clamp and not two commands hitting the same clip
    assert np.max(np.abs(over[:2])) < 1.0 - 1e-6, f"still saturating: {over}"
    assert np.linalg.norm(at[:2]) > 1e-6, "the reachable command produced no horizontal force"
    env.close()


def test_cruise_feedforward_is_the_drag_model_at_every_cap():
    """Commanding the reachable speed must ask for exactly full thrust — at ANY cap.

    A mode of 1.0 is the full-cap thrust and the reachable speed is where drag equals it, so
    ff(v_ref(cap)) == 1.0 is an identity, and it is the whole cap dependence of the cruise loop.
    `k_ff * v_cmd / v_ref` only satisfied it by accident: it is a straight line through a
    quadratic, so it undershot everywhere in between and the error moved with the cap.
    """
    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0, k_v=0.0)     # feedforward alone
    for cap in CAPS + (0.5,):
        v_ref = ctl.reachable_speed(cap)
        ctl.reset()
        m = ctl.wrench(np.zeros(3), np.zeros(3), np.array([v_ref, 0.0, 0.0]), np.zeros(3), cap)
        assert np.isclose(m[0], 1.0, rtol=1e-9), f"cap {cap}: surge feedforward {m[0]} != 1.0"
        # half the reachable speed needs far LESS than half the thrust — the quadratic the old
        # linear normalization flattened away
        ctl.reset()
        half = ctl.wrench(np.zeros(3), np.zeros(3), np.array([v_ref / 2, 0.0, 0.0]), np.zeros(3), cap)
        assert half[0] < 0.4, f"cap {cap}: half-speed feedforward {half[0]} looks linear, not drag"
    env.close()


def test_singularity_avoidance_does_not_change_the_wrench():
    """The null space is the whole licence for `prefer_deg`: moving in it must be wrench-neutral.

    Holding station the required per-unit force is nearly vertical, which parks every servo on the
    +-90 deg fold (measured: |phi| median 84 deg, 59 % of steps above 80 deg, horizontal component
    changing sign on 1.3 % of steps — a ~180 deg servo command each time). Spending the two spare
    actuator DOF on staying off that boundary cut the commanded servo rate 344 -> 47 deg/s and the
    attitude error 0.33 -> 0.09 rad under disturbance. All of that is only legitimate if the
    offset really does leave A x unchanged, so check the basis, and then check the solver actually
    used it (a silently empty null space would make prefer_deg a no-op that still looks fine).
    """
    from classical_control import GeneralAllocator
    env = _env(dr=False)
    a = GeneralAllocator(env.sim, prefer_deg=60.0, w_move=3.0)
    assert a.null.shape[1] == 2, f"4 live units, 6-DOF wrench: expected 2 null dims, got {a.null.shape}"
    assert np.allclose(a.A[:, a.cols] @ a.null, 0.0, atol=1e-9), "null basis is not in the null space"

    plain = GeneralAllocator(env.sim)
    # a wrench that is mostly vertical — i.e. exactly the hold-station case that sits on the fold
    w = np.array([0.05, 1.2, 0.02, 0.03, -0.02, 0.01])
    cap = 0.25
    for _ in range(5):                        # let the warm-started search settle
        act_a = a.allocate(w, cap)
        act_p = plain.allocate(w, cap)
    assert not np.allclose(act_a[:4], act_p[:4], atol=1e-3), \
        "prefer_deg changed nothing — the null space is not being used"

    def realised(act):
        servo = act[:4] * env.sim.servo_range_rad
        thrust = np.sign(act[4:]) * np.abs(act[4:]) ** a.plant.thrust_curve_exp * a.plant.thrust_per_cmd
        return a.A @ np.concatenate([thrust * np.cos(servo), thrust * np.sin(servo)])

    # Both must land on the SAME wrench, and on the commanded one. The tolerance is not numerical:
    # `allocate` zeroes any unit below 2 % of f_max (a servo angle for a unit making no thrust is
    # meaningless), and dropping it costs a few percent of the wrench. That truncation is the
    # design; what must not happen is the two variants disagreeing, which is what a wrong null
    # basis would look like.
    ra, rp = realised(act_a), realised(act_p)
    assert np.allclose(ra, rp, atol=0.03 * np.linalg.norm(w)), \
        f"the null-space offset moved the realised wrench: {ra} vs {rp}"
    assert np.allclose(ra, w, atol=0.03 * np.linalg.norm(w)), f"realised {ra} != commanded {w}"
    env.close()
