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
  * command ``Vy``  -> net +Z sway (starboard), small yaw residual from the fore/aft pivot
    asymmetry. (An earlier port swapped the starboard rows, which cancelled the sway and
    left only a yaw couple — the old "Vy -> yaw couple" note described that bug.)
Callers therefore map their sim-frame intent into this command convention explicitly
rather than assuming Vx==+X; the driver documents that mapping in one place.
"""

from __future__ import annotations

import math

import numpy as np

_R = math.sqrt(2.0)

# 8x6 allocation matrix. Rows: [lf_h, lf_v, lb_h, lb_v, rb_h, rb_v, rf_h, rf_v].
# Columns (input order): [Phi_x, Phi_y, Phi_z, V_x, V_y, V_z].
# Row order is the real stack's POSITIONS order (lf, lb, rb, rf) — feed_forward.hpp's row labels
# (スラスタ1 :lf .. スラスタ4 :rf) match controllers.yaml's hardware ids (lf=1, lb=2, rb=3, rf=4).
# NOTE this hardware numbering is NOT the sim's CAD unit-id numbering (configs/umiusi.yaml, where
# geometrically id3=rf, id4=rb): an earlier port read the rows as CAD-id order and permuted with
# [0,1,3,2], swapping the starboard pair — which turned pure Phi_y (pitch) into the zero-wrench
# diagonal null mode and flipped V_y's starboard signs (wiring verified on the vehicle 2026-08-25,
# autonomy known_issues A-12: no correction needed).
# Also verified against feed_forward.hpp: every vertical row has V_z = 1.0, so a pure heave (+V_z)
# drives all four thrusters up symmetrically (no yaw couple). (An earlier port had the last vertical
# row's V_z = 0, which made pure heave spin the vehicle ~0.84 rad/s.)
_ALLOC = np.array(
    [
        [0.0, 0.0, 1.0, -_R,  _R, 0.0],  # lf_h
        [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # lf_v
        [0.0, 0.0, 1.0, -_R, -_R, 0.0],  # lb_h
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # lb_v
        [0.0, 0.0, 1.0,  _R, -_R, 0.0],  # rb_h
        [-1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # rb_v
        [0.0, 0.0, 1.0,  _R,  _R, 0.0],  # rf_h
        [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # rf_v
    ],
    dtype=float,
)

# Servo mechanical range is +/-90 deg (config: thrusters.servo_range_deg); UmiusiSimulator
# exposes it as ``self.servo_range_rad``. The action's servo channel is angle / range in [-1, 1].
SERVO_RANGE_DEG = 90.0

_EPS = 1e-9


def feedforward_allocation(target_orientation, target_velocity, servo_range_deg=SERVO_RANGE_DEG,
                           thrust_curve_exp=1.0):
    """Feed-forward 6-DOF command -> 8-D sim action ``[servo x4, esc x4]``.

    The output channel order is the ACTION contract (lf, lb, rb, rf) shared by the simulator
    (configs/umiusi.yaml ``action_order``) and sinsei_UMIUSI_autonomy's POSITIONS. The
    allocation matrix rows are already in this order (see ``_ALLOC``), so no permutation.

    Parameters
    ----------
    target_orientation : array-like, shape (3,)
        Desired body-frame orientation command ``[Phi_x, Phi_y, Phi_z]`` (roll/pitch/yaw).
    target_velocity : array-like, shape (3,)
        Desired body-frame velocity command ``[V_x, V_y, V_z]`` (in the controller convention;
        see the module docstring for how this maps onto sim axes).
    servo_range_deg : float
        Servo half-range in degrees used to normalise the azimuth angle to ``[-1, 1]``.
    thrust_curve_exp : float
        The plant's thrust-curve exponent (configs/umiusi.yaml ``thrust_curve_exp``); the
        commanded esc is pre-warped by the INVERSE curve ``u = sign(t)|t|^(1/exp)`` so the
        realised thrust is the linear allocation. 1.0 (old linear plant) = no warp.

    Returns
    -------
    np.ndarray, shape (8,)
        Action ``[servo_1..4, esc_1..4]``, each clipped to ``[-1, 1]``.
    """
    u = np.concatenate([
        np.asarray(target_orientation, dtype=float).reshape(3),
        np.asarray(target_velocity, dtype=float).reshape(3),
    ])
    f = _ALLOC @ u  # [lf_h, lf_v, lb_h, lb_v, rb_h, rb_v, rf_h, rf_v]

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
        if thrust_curve_exp != 1.0:  # invert the plant's propeller-law curve
            thrust = math.copysign(abs(thrust) ** (1.0 / thrust_curve_exp), thrust)

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
    show("pure +Phi_x (roll)", [1, 0, 0], [0, 0, 0])
    show("pure +Phi_y (pitch)", [0, 1, 0], [0, 0, 0])
    print("-" * 78)
    # Sanity assertions (faithful-port expectations):
    a = feedforward_allocation([0, 0, 0], [1, 0, 0])
    assert np.allclose(a[:4], 0.0, atol=1e-6), "pure surge should keep servos ~0 deg"
    assert np.allclose(np.abs(a[4:]), 1.0), "pure surge should be symmetric full thrust"
    az = feedforward_allocation([0, 0, 0], [0, 0, 1])
    assert np.allclose(az[:4], 1.0), "pure heave should tilt all servos toward +90 deg"
    # Couples in action order (lf, lb, rb, rf): vertical force sign = servo sign here (esc > 0).
    ar = feedforward_allocation([1, 0, 0], [0, 0, 0])
    assert np.allclose(np.sign(ar[:4]), [1, 1, -1, -1]), "pure roll: port up, starboard down"
    ap = feedforward_allocation([0, 1, 0], [0, 0, 0])
    assert np.allclose(np.sign(ap[:4]), [-1, 1, 1, -1]), \
        "pure pitch: front down, back up (a diagonal +-+- pattern is the null mode — wrong)"
    print("self-test OK: surge/heave symmetric; roll and pitch form proper couples.")
