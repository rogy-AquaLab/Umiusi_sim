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


def track(current, target, max_rate, tau, dt):
    """RC-servo tracking: first-order lag saturated by a rate limit.

    rate = clip((target - current) / tau, +/- max_rate): far from the target the servo moves at
    its slew rate; within max_rate * tau of it, the motion is an exponential convergence with
    time constant tau — so a CONSTANT command is actually reached (the position control loop a
    real RC servo runs), unlike pure slew which crawls at full rate forever under a flapping
    command. tau <= 0 falls back to slew().
    """
    if tau <= 0.0:
        return slew(current, target, max_rate, dt)
    err = target - current
    step = np.clip(err / tau, -max_rate, max_rate) * dt
    # never overshoot the target within one step (guards dt >= tau)
    return current + np.where(np.abs(step) > np.abs(err), err, step)


def thrust_to_world(magnitude, thrust_axis_local, body_xmat):
    """World-frame thrust force vector.

    magnitude: scalar thrust [N] (sign allowed for reverse).
    thrust_axis_local: length-3 unit vector in the thruster body frame.
    body_xmat: 3x3 rotation matrix (body -> world), e.g. MjData.xmat[body].reshape(3, 3).
    """
    axis = np.asarray(thrust_axis_local, dtype=float)
    return magnitude * (np.asarray(body_xmat, dtype=float).reshape(3, 3) @ axis)
