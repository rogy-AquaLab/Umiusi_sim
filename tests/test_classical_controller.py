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
    """Same argument for the heave trim: net buoyancy is a constant, so the mode must move.

    OFF by default since 2026-09-30 (the hull is ballasted instead), so this asks for it — the
    term still has to be correct for anyone comparing against the pre-2026-09-30 results.
    """
    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0, buoy_trim=True)
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


def test_singularity_avoidance_actually_leaves_the_fold_at_hold_station():
    """The regime the feature exists for, and the one where it was silently inert.

    Holding station the required wrench is pure buoyancy trim, so the minimum-norm solution has
    h = 0 for every unit and every servo sits at EXACTLY +-90 deg — on the fold, where the sign of
    the next horizontal demand decides a 180 deg servo command. Scoring distance-to-singularity on
    the unfolded angle made every escape look worse than staying (each null direction gives h with
    alternating signs, so half the units land past 90 deg), and argmin never moved. Anything that
    re-introduces that will leave these servos pinned at the limit again.
    """
    from classical_control import GeneralAllocator
    from umiusi_perception.classical import cad_wrench_from_modes

    env = _env(dr=False)
    # buoy_trim=True ON PURPOSE: the fold problem at hold station is CREATED by the trim. Holding a
    # buoyant hull down needs a near-vertical force, which is what parks every servo at +-90 deg.
    # With the trim off (the default since 2026-09-30) the hold wrench is zero and there is nothing
    # to avoid — see test_hold_station_costs_nothing_without_the_trim below.
    ctl = build_controller(env, cap_tau=0.0, kp=1.0, kd=0.35, buoy_trim=True)
    plain = GeneralAllocator(env.sim)
    avoid = GeneralAllocator(env.sim, prefer_deg=60.0, w_move=3.0, dead_hold=True)
    cap = 0.25
    for _ in range(8):                        # let the warm-started search settle
        m = ctl.wrench(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), cap)
        w = cad_wrench_from_modes(m, ctl.f_max_total(ctl.cap))
        a_plain, a_avoid = plain.allocate(w, cap), avoid.allocate(w, cap)
    env.close()

    # without avoidance the solution is exactly vertical: every servo on the fold
    assert np.allclose(np.abs(a_plain[:4]), 1.0, atol=1e-6), \
        f"expected the minimum-norm hold solution to sit on the fold, got {a_plain[:4]}"
    # with it, every servo must be clear of the limit — this is the assertion that was failing
    deg = np.degrees(a_avoid[:4] * env.sim.servo_range_rad)
    assert np.max(np.abs(deg)) < 60.0, f"avoidance left a servo on the fold: {np.round(deg, 1)}"


def test_hold_station_costs_nothing_without_the_trim():
    """With `buoy_trim=False` the z channel means commanded heave and nothing else: 0 in, 0 out.

    This is the whole point of dropping the trim. On the vehicle, idle duty decomposed as
    0.107 constant + 0.25*cap, and the constant was the trim — roughly 63 % of idle duty at cap
    0.25, spent continuously just to stay level. It also created the azimuth problem: the required
    force was near-vertical, which is exactly the fold. Both go away together, and the hull gets
    ballasted near neutral instead.
    """
    from classical_control import GeneralAllocator
    from umiusi_perception.classical import cad_wrench_from_modes

    env = _env(dr=False)
    ctl = build_controller(env, cap_tau=0.0, kp=1.0, kd=0.35)          # default: no trim
    alloc = GeneralAllocator(env.sim, prefer_deg=60.0, w_move=3.0, dead_hold=True)
    for _ in range(8):
        m = ctl.wrench(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.25)
        act = alloc.allocate(cad_wrench_from_modes(m, ctl.f_max_total(ctl.cap)), 0.25)
    env.close()
    assert m[2] == 0.0, f"no command and no trim must give no heave mode, got {m[2]}"
    assert np.allclose(act[4:], 0.0), f"idle must burn no duty at all, got {act[4:]}"
    # ...and a commanded heave still reaches the channel (the trim is gone, the command is not)
    ctl2_env = _env(dr=False)
    ctl2 = build_controller(ctl2_env, cap_tau=0.0)
    up = ctl2.wrench(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 0.05]), np.zeros(3), 0.25)
    ctl2_env.close()
    assert up[2] > 0.0, f"commanded ascent must still produce upward heave, got {up[2]}"


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


def test_null_space_circulation_does_not_creep_to_the_cap():
    """A tiny vertical demand must not cost the duty cap.

    `_away_from_singularity` re-centres its search window on the previous choice every step, and
    once every unit is inside `prefer_deg` nothing in the cost grows with the circulation's size —
    so it creeps until the cap term stops it at `cap_margin * f_max`. Measured 2026-10-01 in the
    deploy chain (neutral hull, CoB 0.5 mm ahead of the CoM): the units pushed +-2.1 N against each
    other to hold a 0.06 N/unit pitch torque, mean |duty| 0.265 = 88 % of cap 0.3, standing still.
    `w_effort` prices the circulation; 0.0 (the default) keeps the old behaviour, which this test
    also pins so the defect stays visible until the bundle is re-exported with it on.
    """
    from classical_control import GeneralAllocator

    env = _env(dr=False)
    cap = 0.3
    # a small pure pitch torque (CAD rot+Z) — what a slightly bow-heavy hull asks for at hold
    w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25])
    kw = dict(prefer_deg=60.0, w_move=3.0, dead_hold=True, cap_margin=0.85, w_cap=50.0)
    old = GeneralAllocator(env.sim, **kw)
    new = GeneralAllocator(env.sim, **kw, w_effort=2.0)
    for _ in range(200):                      # the creep is a drift over many steps
        a_old, a_new = old.allocate(w, cap), new.allocate(w, cap)
    env.close()

    def per_unit_force(alloc, act):
        p = alloc.plant
        return np.abs(act[4:]) ** p.thrust_curve_exp * p.thrust_per_cmd

    plain = GeneralAllocator(env.sim)         # minimum norm = exactly the force the wrench needs
    f_need = per_unit_force(plain, plain.allocate(w, cap)).max()
    f_old, f_new = per_unit_force(old, a_old).max(), per_unit_force(new, a_new).max()
    # static input creeps "only" ~5-15x (to ~1 N); in the closed loop it reached cap_margin * f_max
    assert f_old > 2.0 * f_need, \
        "the creep is gone with w_effort=0 — if that is intended, flip the default and drop this half"
    assert f_new < 0.5 * f_old, f"w_effort did not stop the circulation: {f_new:.3f} vs old {f_old:.3f} N"
    assert f_new < 2.0 * f_need, f"w_effort still spends {f_new:.3f} N/unit for a {f_need:.3f} N demand"
    # NOTE: for a near-zero demand w_effort lets the servos sit on the fold again. That is the
    # intended trade — reversing a unit that is barely pushing costs almost nothing — and the
    # closed-loop check (tools/competition_eval.py --w-effort) showed no esc reversals at hold.
