"""Analytical hydrodynamics — the primary, reference physics model.

Each effect is an explicit, individually testable function operating on plain numpy
arrays (SI units). This is deliberately simple and readable so it can be tuned and
validated before RL (see ai/project_spec.yaml, task_request #3/#5).

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
