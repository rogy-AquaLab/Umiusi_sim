"""ModeMixer — wrench-mode action -> 8-D thruster action, with NO null-space coordinate.

Umiusi_sim#3 follow-up (2026-08-26): reward shaping could not remove the vertical null mode
(the (+,-,+,-) diagonal pattern; 41.2 % of vertical power in the 8/25 run, still ~22 % after
w_null=1..5 continuation runs) because "null == 0" is an exact linear constraint on the raw
8-D action that gradient descent only approaches. This mixer removes the constraint by
re-parameterizing the ACTION: the policy commands 6 wrench modes and the two do-nothing
patterns (vertical AND horizontal null) simply do not exist in the basis.

Action contract (action_mode: "modes"): a = [fx, fy, fz, tx, ty, tz] in [-1, 1],
a normalized body-frame wrench command in REP-103 axes (x fwd, y left, z up — the same
convention as the deployed obs_frame):
    fx  surge (forward force)         tx  roll  (+ = left side up)
    fy  sway  (leftward force)        ty  pitch (+ = nose down, REP-103 right-hand)
    fz  heave (upward force)          tz  yaw   (+ = counter-clockwise seen from above)

Scaling: a single mode at +/-1 commands each thruster at the CURRENT esc cap, i.e. the
per-unit force f_max = thrust_per_cmd * max_duty**thrust_curve_exp. The cap is observed by
the policy (observe_max_duty), so "1.0" always means "full available authority", whatever
the operator set. Combined modes that overflow f_max are per-unit clipped (saturation the
policy sees through prev_action and learns to moderate).

Per unit the horizontal (tangential t_i) and vertical components are combined exactly like
the deployed feed-forward allocation (umiusi_perception.control / feed_forward.hpp):
    servo = atan2(v, h)   (folded into +/-90 deg by reversing the esc for rear half-plane)
    esc   = sign * (min(|f|, f_max) / thrust_per_cmd) ** (1 / thrust_curve_exp)
The mixer uses the NOMINAL plant constants (thrust_per_cmd / thrust_curve_exp from the sim
config, not the DR-perturbed episode values): the deployed mixer will run with the same
nominal constants, and the mismatch is exactly what the policy's feedback learns to absorb.

The idealization "horizontal component along t_i, vertical along +Y" matches the deploy
allocation; the real hinge axis has a ~7 deg upward tilt (MJCF), whose small cross-coupling
is part of the plant the policy corrects for.
"""

import numpy as np

# Per-unit mode signs, keyed by geometric unit name (configs/umiusi.yaml `units[].name`).
# Columns: (fx, fy, tz | fz, tx, ty). Derived from the thruster geometry:
#   fx =  sign(thrust_axis.x)      fy = -sign(thrust_axis.z)   (REP-103 y = -sim z)
#   tz = -1 for all units (the four tangents share one rotational sense: uniform +h -> -yaw)
#   fz = +1 (all up)               tx / ty from the pivot square (tx = roll = left - right,
#   ty = -(front - back): REP-103 +pitch about y(left) is nose-DOWN)
# Every column is a Walsh vector; any two columns within a group are orthogonal, and the
# missing 4th vectors — vertical (+,-,+,-) and horizontal (+,-,+,-) in action order — are
# exactly the null modes this parameterization cannot express.
_MODE_SIGNS = {  # name: (fx, fy, tz, fz, tx, ty)
    "lf": (+1, -1, -1, +1, +1, -1),
    "lb": (+1, +1, -1, +1, +1, +1),
    "rb": (-1, +1, -1, +1, -1, +1),
    "rf": (-1, -1, -1, +1, -1, -1),
}
MODE_DIM = 6
MODE_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")
# Below this fraction of f_max a unit's force direction is numerically meaningless (atan2 at
# zero), so it holds its previous servo angle and zeroes the esc. Part of the DEPLOY contract
# (tools/export_policy.py reads it from here) — the deployed mixer must use the same value.
DEADBAND_FRAC = 0.02


class ModeMixer:
    """modes[6] + max_duty + previous servo command -> action [servo x4, esc x4].

    NOTE on rate limiting (2026-08-26, null diagnosis): temporal smoothing of the command
    belongs on the MODE vector (in the env / deploy wrapper), never on this mixer's
    servo/esc output. The null-free commands form a LINEAR subspace of the per-unit (h, v)
    force space, and the mode coordinates are linear coordinates of it — a slewed mode
    vector stays null-free at every intermediate step. Slewing the folded (servo, esc)
    output instead interpolates OUTSIDE the subspace and was measured to RAISE the realized
    null share (smoke 19.7% vs ~14% baseline) while freezing the servos.
    """

    def __init__(self, unit_names, thrust_axes, servo_range_rad, thrust_per_cmd,
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
        # Geometry consistency: the fx/fy signs must match the configured tangent directions
        # (guards a config edit silently breaking the hardcoded table).
        ax = np.asarray(thrust_axes, dtype=float)
        if not (np.all(np.sign(ax[:, 0]) == self._Sh[:, 0])
                and np.all(-np.sign(ax[:, 2]) == self._Sh[:, 1])
                and np.allclose(ax[:, 1], 0.0, atol=1e-6)):
            raise ValueError("thrust_axes do not match the mode sign table (geometry changed?)")

    def mix(self, modes, max_duty, prev_servo_cmd):
        """Return action[8] = [servo x4, esc x4], each channel in [-1, 1].

        modes:          array-like (6,) in [-1, 1] — [fx, fy, fz, tx, ty, tz]
        max_duty:       the plant's CURRENT esc cap (the same value the policy observes)
        prev_servo_cmd: (4,) previous mixed servo command; held when a unit's force is in
                        the deadband (atan2 undefined at zero — avoids servo flapping)
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
