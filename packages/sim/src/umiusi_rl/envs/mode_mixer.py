"""ModeMixer — wrench-mode action -> 8-D thruster action, with NO null-space coordinate.

Action contract (action_mode: "modes"): a = [fx, fy, fz, tx, ty, tz] in [-1, 1], a normalized
body-frame wrench command in REP-103 axes (x fwd, y left, z up — the deployed obs_frame):
    fx  surge (forward force)         tx  roll  (+ = left side up)
    fy  sway  (leftward force)        ty  pitch (+ = nose down, REP-103 right-hand)
    fz  heave (upward force)          tz  yaw   (+ = counter-clockwise seen from above)

One mode at +/-1 commands each thruster at the CURRENT esc cap, so "1.0" always means "full
available authority"; combined modes that overflow it are clipped per unit.

Two things must stay true of any edit here, and neither is checkable by a test:
  * the fold to (servo, esc) must match the deployed feed-forward allocation
    (umiusi_perception.control / feed_forward.hpp) — the robot runs that one;
  * the plant constants must be the NOMINAL ones, never the DR-perturbed episode values,
    because the deployed mixer runs with the nominals.
The "horizontal along t_i, vertical along +Y" idealization is deliberate for the same reason
(it matches deploy; the real hinge tilts ~7 deg up and the residual is part of the plant).
"""

import numpy as np

# Per-unit mode signs in action order, keyed by unit name (configs/umiusi.yaml `units[].name`).
# __init__ re-derives every column from the config geometry and rejects a mismatch.
_MODE_SIGNS = {  # name: (fx, fy, tz, fz, tx, ty)
    "lf": (+1, -1, -1, +1, +1, -1),
    "lb": (+1, +1, -1, +1, +1, +1),
    "rb": (-1, +1, -1, +1, -1, +1),
    "rf": (-1, -1, -1, +1, -1, -1),
}
MODE_DIM = 6
MODE_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")
# Part of the DEPLOY contract (tools/export_policy.py reads it from here): the deployed mixer
# must use the same value.
DEADBAND_FRAC = 0.02


class ModeMixer:
    """modes[6] + max_duty + previous servo command -> action [servo x4, esc x4].

    Rate-limit the MODE vector (in the env / deploy wrapper), never this mixer's servo/esc
    output: the null-free commands are a linear subspace and the modes are linear coordinates
    of it, so a slewed mode vector stays null-free, while slewing the folded output
    interpolates outside the subspace and raises the realized null share.
    """

    def __init__(self, unit_names, thrust_axes, unit_pivots, servo_range_rad, thrust_per_cmd,
                 thrust_curve_exp, deadband_frac=DEADBAND_FRAC):
        names = list(unit_names)
        if set(names) != set(_MODE_SIGNS):
            raise ValueError(f"mode mixer needs geometric unit names {sorted(_MODE_SIGNS)}, "
                             f"got {names}")
        signs = np.array([_MODE_SIGNS[n] for n in names], dtype=float)  # 4x6, action order
        self._Sh = signs[:, 0:3]                      # horizontal: columns (fx, fy, tz)
        self._Sv = signs[:, 3:6]                      # vertical:   columns (fz, tx, ty)
        self.servo_range_rad = float(servo_range_rad)
        self.thrust_per_cmd = float(thrust_per_cmd)
        self.thrust_curve_exp = float(thrust_curve_exp)
        self.deadband_frac = float(deadband_frac)
        # Re-derive the table from the config geometry: a wrong sign never crashes, it just makes
        # "pure roll" produce something else, so BOTH groups must be checked here.
        # Horizontal (fx, fy): against the configured tangent directions.
        ax = np.asarray(thrust_axes, dtype=float)
        if not (np.all(np.sign(ax[:, 0]) == self._Sh[:, 0])
                and np.all(-np.sign(ax[:, 2]) == self._Sh[:, 1])
                and np.allclose(ax[:, 1], 0.0, atol=1e-6)):
            raise ValueError("thrust_axes do not match the mode sign table (geometry changed?)")
        # Vertical (fz, tx, ty): against the mounting-pivot square, in CAD axes (+x fwd, -z port).
        pv = np.asarray(unit_pivots, dtype=float)
        if pv.shape != (4, 3):
            raise ValueError(f"unit_pivots must be (4, 3) in action order, got {pv.shape}")
        roll_sign = -np.sign(pv[:, 2] - pv[:, 2].mean())     # port (CAD -z) side up
        pitch_sign = -np.sign(pv[:, 0] - pv[:, 0].mean())    # front (CAD +x) side down
        if not (np.all(self._Sv[:, 0] == 1.0)
                and np.all(roll_sign == self._Sv[:, 1])
                and np.all(pitch_sign == self._Sv[:, 2])):
            raise ValueError("unit_pivots do not match the mode sign table (geometry changed?)")

    def mix(self, modes, max_duty, prev_servo_cmd):
        """Return action[8] = [servo x4, esc x4], each channel in [-1, 1].

        max_duty is the plant's CURRENT esc cap (the value the policy observes); a unit whose
        force falls in the deadband holds prev_servo_cmd[i] (atan2 is undefined at zero).
        """
        m = np.clip(np.asarray(modes, dtype=float).reshape(MODE_DIM), -1.0, 1.0)
        f_max = self.thrust_per_cmd * float(max_duty) ** self.thrust_curve_exp
        h = self._Sh @ np.array([m[0], m[1], m[5]]) * f_max   # tangential force per unit [N]
        v = self._Sv @ np.array([m[2], m[3], m[4]]) * f_max   # vertical force per unit [N]

        phi = np.arctan2(v, h)                                # (-pi, pi]
        rear = np.abs(phi) > np.pi / 2.0                      # unreachable half-plane ->
        phi = np.where(rear, phi - np.sign(phi) * np.pi, phi)  # fold and reverse the esc
        esc_sign = np.where(rear, -1.0, 1.0)

        mag = np.hypot(h, v)
        u = esc_sign * (np.minimum(mag, f_max) / self.thrust_per_cmd) ** (1.0 / self.thrust_curve_exp)

        servo = phi / self.servo_range_rad
        dead = mag < self.deadband_frac * f_max
        servo = np.where(dead, np.asarray(prev_servo_cmd, dtype=float), servo)
        u = np.where(dead, 0.0, u)
        return np.concatenate([np.clip(servo, -1.0, 1.0), np.clip(u, -1.0, 1.0)])
