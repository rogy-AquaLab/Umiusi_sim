"""Analytical feed-forward low-level controller (AttitudeController allocation port).

Mirrors the real ``sinsei_umiusi_control`` ``feed_forward.hpp`` allocation: it maps a
desired 6-DOF command (body-frame orientation + velocity) to the 4 azimuth thrusters,
each described by a HORIZONTAL and a VERTICAL force component. Per thruster the two
components are combined into a servo azimuth angle (which way the ducted prop points)
and a signed thrust magnitude, then mapped onto ``UmiusiSimulator`` 's 8-D action
``[servo_1..4, esc_1..4]`` (each in ``[-1, 1]``).

This is the "driver's actuator layer": high-level code asks for a body-frame velocity /
orientation, this turns it into servo + ESC commands with no feedback (pure feed-forward).

Convention note (sim vs. controller frame): the allocation matrix comes from the real
controller, whose axes do NOT line up 1:1 with this MuJoCo model (sim: +X forward, +Y up,
+Z lateral). Empirically (see ``__main__`` self-test and ``tools/competition_run.py``):
  * command ``Vz``  -> net +Y  (heave / up)         -> the "vertical" thruster component
  * command ``Vx``  -> net -X  (surge, sign flipped) -> horizontal component
  * command ``Vy``  -> a yaw couple (this sim's symmetric thruster axes cancel sway)
Callers therefore map their sim-frame intent into this command convention explicitly
rather than assuming Vx==+X; the driver documents that mapping in one place.
"""

from __future__ import annotations

import math

import numpy as np

_R = math.sqrt(2.0)

# 8x6 allocation matrix. Rows: [f1h, f1v, f2h, f2v, f3h, f3v, f4h, f4v].
# Columns (input order): [Phi_x, Phi_y, Phi_z, V_x, V_y, V_z].
# Verified against the authoritative source (sinsei_umiusi_control feed_forward.hpp): every vertical
# row (f_iv) has V_z = 1.0, so a pure heave (+V_z) drives all four thrusters up symmetrically (no yaw
# couple). (An earlier port had f4v's V_z = 0, which made pure heave spin the vehicle ~0.84 rad/s.)
_ALLOC = np.array(
    [
        [0.0, 0.0, 1.0, -_R,  _R, 0.0],  # f1h
        [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # f1v
        [0.0, 0.0, 1.0, -_R, -_R, 0.0],  # f2h
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # f2v
        [0.0, 0.0, 1.0,  _R, -_R, 0.0],  # f3h
        [-1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # f3v
        [0.0, 0.0, 1.0,  _R,  _R, 0.0],  # f4h
        [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # f4v
    ],
    dtype=float,
)

# Servo mechanical range is +/-90 deg (config: thrusters.servo_range_deg); UmiusiSimulator
# exposes it as ``self.servo_range_rad``. The action's servo channel is angle / range in [-1, 1].
SERVO_RANGE_DEG = 90.0

_EPS = 1e-9


def feedforward_allocation(target_orientation, target_velocity, servo_range_deg=SERVO_RANGE_DEG):
    """Feed-forward 6-DOF command -> 8-D sim action ``[servo_1..4, esc_1..4]``.

    Parameters
    ----------
    target_orientation : array-like, shape (3,)
        Desired body-frame orientation command ``[Phi_x, Phi_y, Phi_z]`` (roll/pitch/yaw).
    target_velocity : array-like, shape (3,)
        Desired body-frame velocity command ``[V_x, V_y, V_z]`` (in the controller convention;
        see the module docstring for how this maps onto sim axes).
    servo_range_deg : float
        Servo half-range in degrees used to normalise the azimuth angle to ``[-1, 1]``.

    Returns
    -------
    np.ndarray, shape (8,)
        Action ``[servo_1..4, esc_1..4]``, each clipped to ``[-1, 1]``.
    """
    u = np.concatenate([
        np.asarray(target_orientation, dtype=float).reshape(3),
        np.asarray(target_velocity, dtype=float).reshape(3),
    ])
    f = _ALLOC @ u  # [f1h, f1v, f2h, f2v, f3h, f3v, f4h, f4v]

    action = np.zeros(8)
    for i in range(4):
        fih = f[2 * i]
        fiv = f[2 * i + 1]

        # Servo azimuth = atan(f_vertical / f_horizontal) in [-90, 90] deg (0 if both ~0).
        if abs(fih) < _EPS and abs(fiv) < _EPS:
            servo_deg = 0.0
        elif abs(fih) < _EPS:
            servo_deg = 90.0 * math.copysign(1.0, fiv)
        else:
            servo_deg = math.degrees(math.atan(fiv / fih))

        mag = math.hypot(fih, fiv)

        # Sign: with a +/-90 deg servo, directions whose horizontal component points "backward"
        # (left half-plane) are reached by reversing the ESC instead. atan2 -> [0, 2*pi) so the
        # (pi/2, 3*pi/2) test is meaningful (Python's atan2 alone returns (-pi, pi]).
        ang = math.atan2(fiv, fih)
        if ang < 0.0:
            ang += 2.0 * math.pi
        sign = -1.0 if (math.pi / 2.0 < ang < 3.0 * math.pi / 2.0) else 1.0
        thrust = sign * mag / _R  # /sqrt(2) so |thrust| <= 1 for unit commands

        action[i] = float(np.clip(servo_deg / servo_range_deg, -1.0, 1.0))
        action[4 + i] = float(np.clip(thrust, -1.0, 1.0))
    return action


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    def show(label, ori, vel):
        a = feedforward_allocation(ori, vel)
        print(f"{label:28s} servo(deg)={np.round(a[:4] * SERVO_RANGE_DEG, 1)}  esc={np.round(a[4:], 3)}")

    print("feed-forward allocation self-test (orientation=[0,0,0] unless noted)")
    print("-" * 78)
    show("pure +Vx (surge)",  [0, 0, 0], [1, 0, 0])
    show("pure -Vx",          [0, 0, 0], [-1, 0, 0])
    show("pure +Vy (sway)",   [0, 0, 0], [0, 1, 0])
    show("pure +Vz (heave)",  [0, 0, 0], [0, 0, 1])
    show("pure -Vz",          [0, 0, 0], [0, 0, -1])
    show("pure +Phi_z (yaw)", [0, 0, 1], [0, 0, 0])
    print("-" * 78)
    # Sanity assertions (faithful-port expectations):
    a = feedforward_allocation([0, 0, 0], [1, 0, 0])
    assert np.allclose(a[:4], 0.0, atol=1e-6), "pure surge should keep servos ~0 deg"
    assert np.allclose(np.abs(a[4:]), 1.0), "pure surge should be symmetric full thrust"
    az = feedforward_allocation([0, 0, 0], [0, 0, 1])
    assert np.allclose(az[:3], 1.0), "pure heave should tilt servos toward +90 deg"
    print("self-test OK: surge -> ~0 deg symmetric thrust; heave -> servos tilt up.")
