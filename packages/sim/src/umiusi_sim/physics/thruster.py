"""Azimuth-thruster model: servo angle tracking + ESC command -> thrust force.

Kept intentionally simple: a linear command->thrust map and a rate-limited servo.
The thrust acts along the thruster's (servo-rotated) local axis; the caller supplies
the body rotation matrix to express it in world coordinates.
"""

import numpy as np


def slew(current, target, max_rate, dt):
    """Rate-limit `current` toward `target` by at most max_rate*dt (all in the same unit)."""
    step = max_rate * dt
    return current + np.clip(target - current, -step, step)


def thrust_to_world(magnitude, thrust_axis_local, body_xmat):
    """World-frame thrust force vector.

    magnitude: scalar thrust [N] (sign allowed for reverse).
    thrust_axis_local: length-3 unit vector in the thruster body frame.
    body_xmat: 3x3 rotation matrix (body -> world), e.g. MjData.xmat[body].reshape(3, 3).
    """
    axis = np.asarray(thrust_axis_local, dtype=float)
    return magnitude * (np.asarray(body_xmat, dtype=float).reshape(3, 3) @ axis)
