"""Analytical hydrodynamics — the primary, reference physics model.

Each effect is an explicit, individually testable function operating on plain numpy
arrays (SI units). This is deliberately simple and readable so it can be tuned and
validated before RL (see the project README).

Body-frame 6-vectors use the order [x, y, z, roll, pitch, yaw] = [linear(3), angular(3)].
Gravity is applied by the MuJoCo engine via body mass; only buoyancy/drag/added-mass
are added here.
"""

import numpy as np


def buoyancy_force_world(density, volume, gravity):
    """Upward buoyancy force in world frame: opposes gravity, magnitude rho*V*|g|.

    Applied at the center of buoyancy (handled by the caller).
    """
    return -density * volume * np.asarray(gravity, dtype=float)


def drag_wrench_body(vel_body, linear_coef, quadratic_coef):
    """Damping wrench in body frame: -(D_lin * v + D_quad * |v| * v), componentwise.

    vel_body, *_coef: length-6 arrays [linear(3), angular(3)].
    Returns a length-6 wrench [force(3), torque(3)] in the body frame.
    """
    v = np.asarray(vel_body, dtype=float)
    lin = np.asarray(linear_coef, dtype=float)
    quad = np.asarray(quadratic_coef, dtype=float)
    return -(lin * v + quad * np.abs(v) * v)


def added_mass_wrench_body(acc_body, diag):
    """Diagonal added-mass reaction in body frame: -M_A * a.

    acc_body, diag: length-6 arrays. With diag all-zero this returns zeros (default OFF).
    """
    return -np.asarray(diag, dtype=float) * np.asarray(acc_body, dtype=float)


def lift_force_body(vel_lin_body, coef, ref_axis):
    """Hydrodynamic LIFT: a force PERPENDICULAR to the body-frame translational velocity, growing
    with the angle of attack between the flow and the longitudinal reference axis, magnitude ~|v|^2.

    Thin-body form  L = coef * |v|^2 * sin(a)cos(a) * n_hat, where `a` is the angle of attack (flow
    vs `ref_axis`), `n_hat` is the unit vector perpendicular to v in the (v, ref_axis) plane pointing
    toward ref_axis, and coef = 1/2 * rho * Cl * A (lumped). Equivalently coef*|v|^2*(v_hat . ref)*
    (ref - (v_hat . ref) v_hat).

    The sin*cos (= 1/2 sin 2a) shape makes lift VANISH both for flow ALONG the reference axis (a=0)
    AND for flow PERPENDICULAR to it (a=90 deg): so pure surge / pure heave / pure sway produce ZERO
    lift, and only genuinely OBLIQUE (angled) flow lifts. This keeps the axis-aligned validate_sim
    invariants (surge stays horizontal, heave rises) intact. coef == 0 -> OFF.

    vel_lin_body: length-3 body-frame linear velocity. Returns a length-3 body-frame force.
    """
    v = np.asarray(vel_lin_body, dtype=float)
    speed2 = float(v @ v)
    if coef == 0.0 or speed2 < 1e-12:
        return np.zeros(3)
    vhat = v / np.sqrt(speed2)
    ref = np.asarray(ref_axis, dtype=float)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    c = float(vhat @ ref)  # cos(alpha)
    perp = ref - c * vhat  # perpendicular to v, magnitude sin(alpha), toward ref
    return coef * speed2 * c * perp  # coef*|v|^2 * cos(a) * (sin(a) n_hat)


def coupling_moment_body(vel_lin_body, sway_yaw, heave_pitch):
    """Optional OFF-DIAGONAL damping: a translational velocity induces a CROSS-axis body moment
    (linear + quadratic), M = -(c_lin * v + c_quad * |v| * v). Body axes: X fwd, Y up, Z lateral;
    moments [Mx(roll,+X), My(yaw/heading,+Y), Mz(pitch,+Z)].

    sway_yaw    [lin, quad]: body-Z (sway) velocity -> yaw moment about +Y (weathercocking).
    heave_pitch [lin, quad]: body-Y (heave) velocity -> pitch moment about +Z.
    Returns a length-3 body moment. All-zero coefs -> zeros (default OFF).
    """
    v = np.asarray(vel_lin_body, dtype=float)
    m = np.zeros(3)
    vz, vy = v[2], v[1]
    m[1] -= sway_yaw[0] * vz + sway_yaw[1] * abs(vz) * vz  # sway -> yaw (about +Y)
    m[2] -= heave_pitch[0] * vy + heave_pitch[1] * abs(vy) * vy  # heave -> pitch (about +Z)
    return m
