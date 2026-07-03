"""Simulator validation — the hard gate that must pass before any RL (phase 3).

Standalone (no RL, no ROS). Runs the analytical simulator through a set of physical
sanity checks and prints a PASS/FAIL report plus calibration numbers for the
hardware-pending values in configs/umiusi.yaml (drag, thrust map, displaced volume).

Checks (see the project README):
  1. buoyancy       — net vertical force sign matches the configured volume vs. neutral,
                      i.e. the vehicle floats/sinks as configured (near-neutral by design).
  2. restoring      — a tilt produces a righting moment back toward upright (CoB above CoM).
  3. drag           — the damping wrench opposes velocity on every axis; motion decays.
  4. thrust_dir     — a single thruster at neutral servo pushes horizontally along its
                      configured mounting direction (cross-checks configs vs. the MJCF).
  5. servo_tilt     — servos track the command, and a positive servo tilts thrust UP (+Y).
  6. ff_alloc       — the analytical feed-forward controller yields decoupled motion (pure heave
                      rises without spinning; pure surge stays horizontal) — guards the 8x6 matrix.
  7. open_loop      — a scripted command sequence stays finite and bounded (no runaway).

Sign conventions are hard-asserted against their physical design intent, so a wiring or
sign bug fails the gate. The id -> lf/lb/rb/rf name mapping has no ground truth in the
model, so it is reported as a note rather than asserted.

Usage:
    python -m tools.validate_sim            # run the gate; exit 1 if any check fails
    python -m tools.validate_sim -v         # also print the per-thruster / calibration detail

Frame: CAD frame, +Y up. Units: SI, radians internally.
"""

import argparse
import sys

import mujoco
import numpy as np

from umiusi_sim.simulator import UmiusiSimulator

# Tolerances kept loose on purpose: these are direction/sign checks, not precision tests.
_EPS = 1e-6
_ALIGN = 0.9  # cos-similarity required between observed and expected thrust direction


class Report:
    """Collects PASS/FAIL rows and free-form info/calibration lines, then renders them."""

    def __init__(self):
        self.rows = []  # (name, ok, detail)
        self.notes = []  # (title, [lines])

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        return bool(ok)

    def note(self, title, lines):
        self.notes.append((title, lines))

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.rows)

    def render(self, verbose):
        out = ["", "=" * 72, "UMIUSI simulator validation", "=" * 72]
        for name, ok, detail in self.rows:
            tag = "PASS" if ok else "FAIL"
            out.append(f"  [{tag}] {name}" + (f"  —  {detail}" if detail else ""))
        if verbose:
            for title, lines in self.notes:
                out.append("")
                out.append(f"  · {title}")
                out.extend(f"      {ln}" for ln in lines)
        out.append("=" * 72)
        n_fail = sum(1 for _, ok, _ in self.rows if not ok)
        out.append(f"{'ALL CHECKS PASSED' if not n_fail else f'{n_fail} CHECK(S) FAILED'}"
                   f"  ({len(self.rows) - n_fail}/{len(self.rows)} passed)")
        if not verbose:
            out.append("(run with -v for per-thruster detail and calibration numbers)")
        out.append("")
        return "\n".join(out)


# -- helpers ------------------------------------------------------------------
def _rot_y(deg):
    """Rotation matrix about +Y (world up) by `deg` degrees; R @ x expressed in world."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _com_vel(sim):
    """Whole-vehicle center-of-mass velocity in world frame (no body-origin lever coupling)."""
    mujoco.mj_subtreeVel(sim.model, sim.data)
    return sim.data.subtree_linvel[sim.base_id].copy()


def _total_mass(sim):
    return float(sum(sim.model.body_mass))


def _quat_about_x(deg):
    a = np.radians(deg) / 2.0
    return (np.cos(a), np.sin(a), 0.0, 0.0)


def _settle_velocity(sim, action, steps):
    """Run `action` from rest and return the resulting CoM velocity vector."""
    sim.reset(pos=(0.0, 0.0, 0.0))
    for _ in range(steps):
        sim.step(action)
    return _com_vel(sim)


# -- checks -------------------------------------------------------------------
def check_buoyancy(sim, rep):
    """Net vertical force sign matches configured volume vs. neutral (foam is a volume knob)."""
    g = abs(sim.gravity[1])
    mass = _total_mass(sim)
    neutral_vol = mass / sim.density
    net_force = (sim.density * sim.volume - mass) * g  # +up
    expect_up = net_force > 0.0

    # Observe: from rest, zero action -> the CoM should drift the way net_force points.
    v = _settle_velocity(sim, np.zeros(8), steps=15)
    obs_up = v[1] > 0.0
    ok = abs(net_force) < _EPS or (obs_up == expect_up)
    trend = "rises (微浮き)" if obs_up else "sinks (微沈み)"
    rep.check("buoyancy: float/sink matches config", ok,
              f"net {net_force:+.2f} N, vehicle {trend}")

    # Terminal vertical speed under zero action (drag-limited): a calibration handle.
    v_term = _settle_velocity(sim, np.zeros(8), steps=400)[1]
    band = 0.02 * mass / sim.density  # ±2% of mass worth of volume, a reasonable DR band
    rep.note("buoyancy calibration (foam not in CAD — tune displaced_volume)", [
        f"total mass (CAD, no foam) : {mass:.3f} kg",
        f"neutral displaced_volume  : {neutral_vol:.6f} m^3",
        f"configured displaced_volume: {sim.volume:.6f} m^3  ({'+' if sim.volume >= neutral_vol else '-'}"
        f"{abs(sim.volume - neutral_vol) * 1e6:.0f} cm^3 vs neutral)",
        f"net buoyancy at rest      : {net_force:+.2f} N  ({trend})",
        f"terminal vertical speed   : {v_term:+.3f} m/s (drag-limited under zero thrust)",
        f"suggested near-neutral DR band: {neutral_vol - band:.6f} .. {neutral_vol + band:.6f} m^3",
    ])


def check_restoring(sim, rep):
    """A roll tilt should produce a righting angular acceleration back toward upright."""
    roll = 25.0  # degrees about world +X
    sim.reset(pos=(0.0, 0.0, 0.0), quat=_quat_about_x(roll))
    for _ in range(8):
        sim.step(np.zeros(8))
    wx = sim.get_state()["ang_vel"][0]  # world angular velocity about X
    # Rolled by +X; a restoring moment drives the roll back down -> angular velocity about -X.
    ok = wx < 0.0
    rep.check("restoring: tilt self-levels", ok,
              f"roll +{roll:.0f}° -> ang_vel_x {wx:+.4f} rad/s (want < 0)")


def check_drag(sim, rep):
    """Damping opposes velocity on every axis (analytic), and horizontal motion decays (sim)."""
    from umiusi_sim.physics import hydrodynamics as hydro

    # Analytic: for a unit velocity on each of the 6 axes, the drag wrench must oppose it.
    signs_ok = True
    detail_axes = []
    for i in range(6):
        v = np.zeros(6)
        v[i] = 1.0
        w = hydro.drag_wrench_body(v, sim.drag_lin, sim.drag_quad)
        signs_ok &= w[i] < 0.0
        detail_axes.append(f"axis{i}: v=+1 -> w={w[i]:+.1f}")
    rep.check("drag: opposes velocity on all 6 axes (analytic)", signs_ok, "sign = -v on every axis")

    # Sim: give a horizontal +X velocity, no thrust; speed must drop (drag decelerates).
    sim.reset(pos=(0.0, 0.0, 0.0))
    sim.data.qvel[0] = 1.0
    mujoco.mj_forward(sim.model, sim.data)
    v0 = abs(_com_vel(sim)[0])
    for _ in range(10):
        sim.step(np.zeros(8))
    v1 = abs(_com_vel(sim)[0])
    rep.check("drag: horizontal motion decays (sim)", v1 < v0,
              f"|vx| {v0:.3f} -> {v1:.3f} m/s")
    rep.note("drag detail (analytic sign per axis)", detail_axes)


def check_thrust_direction(sim, rep):
    """Each thruster at neutral servo (=0) pushes horizontally along its neutral (tangential) axis."""
    base_v = _settle_velocity(sim, np.zeros(8), steps=3)  # buoyancy/gravity baseline to subtract

    all_ok = True
    lines = []
    for k in range(4):
        act = np.zeros(8)
        act[4 + k] = 0.5  # fire only thruster k forward, servos at neutral (0)
        v = _settle_velocity(sim, act, steps=3) - base_v  # isolate the thrust contribution
        speed = np.linalg.norm(v)
        # At servo=0 the body is upright, so the world thrust dir is the configured neutral axis.
        expect = sim.thrust_axes[k] / np.linalg.norm(sim.thrust_axes[k])
        obs = v / speed if speed > _EPS else v
        align = float(obs @ expect)
        horizontal = abs(v[1]) < 0.25 * speed if speed > _EPS else False
        ok = speed > _EPS and align > _ALIGN and horizontal
        all_ok &= ok
        lines.append(f"id{k + 1}: align={align:+.2f}, |Δv|={speed:.3f}, dir={np.round(obs, 2)}")
    rep.check("thrust_dir: neutral thrust horizontal (tangential) per unit", all_ok,
              "all 4 thrusters match configured neutral thrust_axis")
    rep.note("thrust direction per unit (observed vs configured neutral thrust_axis)", lines)


def check_servo_tilt(sim, rep):
    """Servos track the command; a positive servo command tilts thrust UP (+Y)."""
    # Tracking: command +0.5 on all servos, let them slew, compare to target.
    target = 0.5 * sim.servo_range_rad
    sim.reset(pos=(0.0, 0.0, 0.0))
    for _ in range(60):
        sim.step(np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]))
    servo = sim.get_state()["servo"]
    track_ok = np.allclose(servo, target, atol=np.radians(3.0))
    rep.check("servo: tracks commanded angle", track_ok,
              f"cmd {np.degrees(target):.0f}° -> {np.round(np.degrees(servo), 1)}°")

    # Direction: with a positive servo, each thruster's world thrust axis gains +Y (tilts up).
    up_ok = True
    lines = []
    for k, bid in enumerate(sim.thr_ids):
        y_up = float((sim.data.xmat[bid].reshape(3, 3) @ sim.thrust_axes[k])[1])
        up_ok &= y_up > 0.0
        lines.append(f"thruster_{k + 1}: thrust·ŷ = {y_up:+.2f} (want > 0 for +servo)")
    rep.check("servo: +command tilts thrust UP (+Y)", up_ok, "positive servo -> upward thrust")
    rep.note("servo tilt detail (+0.5 command, thrust vertical component)", lines)


def check_ff_allocation(sim, rep):
    """The analytical feed-forward controller (control.py) must produce DECOUPLED motion: a pure
    heave command rises straight up WITHOUT spinning (guards the f4v allocation symmetry), and a
    pure surge command moves horizontally. This catches sign/row regressions in the 8x6 allocation
    matrix that the force-level checks above would miss (e.g. an f_iv V_z that breaks heave)."""
    from umiusi_sim.control import feedforward_allocation

    def run(ori, vel, steps=150):
        sim.reset(pos=(0.0, 0.0, 0.0))
        a = feedforward_allocation(ori, vel)
        for _ in range(steps):
            sim.step(a)
        return _com_vel(sim), float(np.linalg.norm(sim.get_state()["ang_vel"]))

    v, spin = run([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])    # pure heave (+V_z)
    speed = np.linalg.norm(v)
    heave_ok = speed > _EPS and v[1] > 0.0 and abs(v[1]) > 0.7 * speed and spin < 0.15
    rep.check("ff_alloc: pure heave rises without spinning", heave_ok,
              f"v={np.round(v, 2)} m/s, |ang_vel|={spin:.3f} rad/s (want +Y, low spin)")

    v2, spin2 = run([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])  # pure surge (+V_x)
    speed2 = np.linalg.norm(v2)
    surge_ok = speed2 > _EPS and abs(v2[1]) < 0.35 * speed2 and spin2 < 0.20
    rep.check("ff_alloc: pure surge stays horizontal", surge_ok,
              f"v={np.round(v2, 2)} m/s, |ang_vel|={spin2:.3f} rad/s (want horizontal, low spin)")


def check_open_loop(sim, rep):
    """A scripted servo sweep + thrust pulse must stay finite and bounded (no runaway)."""
    sim.reset(pos=(0.0, 0.5, 0.0))
    control_dt = 1.0 / sim.cfg["sim"]["control_rate_hz"]
    max_speed = 0.0
    finite = True
    t = 0.0
    for _ in range(300):
        servo = 0.6 * np.sin(2.0 * np.pi * 0.2 * t) * np.ones(4)
        esc = 0.4 * np.ones(4)
        s = sim.step(np.concatenate([servo, esc]))
        finite &= np.all(np.isfinite(s["pos"])) and np.all(np.isfinite(s["lin_vel"]))
        max_speed = max(max_speed, float(np.linalg.norm(s["lin_vel"])))
        t += control_dt
    ok = finite and max_speed < 50.0  # 50 m/s would be an obvious blow-up for this vehicle
    rep.check("open_loop: scripted run stays bounded", ok,
              f"finite={finite}, peak speed {max_speed:.2f} m/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print per-thruster detail and calibration numbers")
    args = ap.parse_args()

    sim = UmiusiSimulator()
    rep = Report()

    check_buoyancy(sim, rep)
    check_restoring(sim, rep)
    check_drag(sim, rep)
    check_thrust_direction(sim, rep)
    check_servo_tilt(sim, rep)
    check_ff_allocation(sim, rep)
    check_open_loop(sim, rep)

    # No ground truth in the model for the id -> name mapping; report the assumption.
    rep.note("id -> name mapping (ASSUMED — verify against hardware)", [
        "1=lf  2=lb  3=rb  4=rf   (from configs/umiusi.yaml; not physically checkable here)",
    ])

    print(rep.render(args.verbose))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
