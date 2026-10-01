"""FSM {surge, heave, yaw} -> the DEPLOY control chain -> 8-D sim action.

The competition harnesses used to drive the plant with `feedforward_allocation`, which is not what
the robot runs: no attitude loop, a linear thrust curve against a plant whose exponent is 2.0, and
(measured 2026-10-01) a yaw sign that turns the nose AWAY from the FSM's +yaw. This module runs the
same chain the robot does instead:

    FSM {surge, heave, yaw}            [-1, 1] each (behavior.py)
      -> navigator "setpoint"          v = [surge, 0, heave] * surge_scale [m/s], yaw -> RATE
      -> control yaw (D-1)             rate loop; hold_yaw adds an outer heading loop with relatch
      -> ClassicalController.wrench    attitude PID (roll/pitch level) + cruise, buoy_trim OFF (D-3)
      -> GeneralAllocator.allocate     -> [servo x4, esc x4]
      -> esc / servo slew              (as classical_attitude_node._emit)

YAW (control `docs/attitude_controller.md`, feat/hold-yaw): yaw is a RATE command by default,
`moment = kp_yaw_rate * (rate - omega_z)`. ClassicalController has no rate mode, but its yaw torque
is `kp*e - kd*w`, so feeding `e = (kd/kp) * rate` gives exactly `kd * (rate - w)` — a pure rate loop
with gain kd. The roll/pitch target is "level at the CURRENT heading", so the yaw part of ori_err is
zero before that substitution (control's reduced-attitude rule: target yaw is ignored).

YAW SIGN. behavior.py's contract is "+az = target toward body +Z (image right), and a POSITIVE yaw
turns the nose that way". Turning toward +Z is a NEGATIVE rotation about +Y (CAD up) = a negative
REP-103 yaw rate, hence `yaw_sign=-1` by default. `yaw_sign=+1` reproduces the robot navigator,
which integrates `+cmd["yaw"]` straight into a REP-103 heading setpoint — see the report this tool
prints; whether the ROBOT is inverted also depends on how its camera is mounted, which the sim
cannot tell.

NOT modelled: the control C++ allocator (velocity is a normalized [-1,1] command there, D-2; here it
is m/s through the classical controller's drag feed-forward, S-6). Both are open-loop force commands
on the vertical axis (k_v_vert = 0), which is what matters for the buoyancy study.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pathlib import Path
import json

from umiusi_perception.classical import (ClassicalController, GeneralAllocator, PlantContract,
                                         cad_wrench_from_modes, rep103_from_cad)

# The bundle the Pi loads (gitignored; `python tools/export_classical.py` writes it). Gains AND
# allocator knobs come from here, never from library defaults — falling back to defaults is how
# A-11 happened twice (prefer_deg / dead_hold missing = the vertical command sits on the fold and
# the slewed output averages to ~0, measured 2026-10-01).
DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models" / "classical" / "classical_bundle.json"

# CAD (+X fwd, +Y up, +Z starboard) -> REP-103 (x fwd, y left, z up), as a matrix on 3-vectors.
_P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_about_up(R):
    """Heading [rad] as a rotation about CAD +Y (= REP-103 yaw, + = nose to port / left)."""
    fwd = R @ np.array([1.0, 0.0, 0.0])
    return math.atan2(-fwd[2], fwd[0])


class DeployDriver:
    def __init__(self, sim, bundle=DEFAULT_BUNDLE, max_duty=0.3, surge_scale=0.35, yaw_rate_scale=0.6, yaw_sign=-1.0,
                 hold_yaw=False, kp_yaw_hold=1.0, relatch_rad=math.radians(90.0),
                 servo_slew_deg_s=None, thrust_slew_per_s=None, alloc_overrides=None,
                 gain_overrides=None, servo_aware=False, contract_overrides=None):
        self.sim = sim
        sim.max_duty = float(max_duty)
        self.max_duty = float(max_duty)
        b = json.loads(Path(bundle).read_text())
        # The controller sees the BUNDLE's plant, not this episode's: a scenario that changes the
        # sim's buoyancy or CoB is exactly the mismatch the robot will meet.
        plant = PlantContract.from_dict({**b["contract"], **(contract_overrides or {})})
        self.ctl = ClassicalController(plant, **{**b["gains"], **(gain_overrides or {})})
        self.alloc = GeneralAllocator(plant, **{**b["allocator"], **(alloc_overrides or {})})
        self.fingerprint = b.get("fingerprint")
        self.plant = plant
        # SERVO-AWARE DUTY (2026-10-01, experiment). The allocator solves (servo, esc) for the TARGET angle
        # and the node slews the two independently, so while a servo is still turning (250 deg/s) the
        # esc already pushes with the full target magnitude along the OLD angle — a force pointing the
        # wrong way. That was measured to drive a 25-40 deg attitude limit cycle at hold when the real
        # thrust curve is flatter than the model (exp 1.4-1.6), and to vanish with a 1000 deg/s servo.
        # This projects each unit's desired force onto the angle the servo has actually reached.
        self.servo_aware = bool(servo_aware)
        self.surge_scale, self.yaw_rate_scale = float(surge_scale), float(yaw_rate_scale)
        self.yaw_sign = float(yaw_sign)
        self.hold_yaw, self.kp_yaw_hold, self.relatch = bool(hold_yaw), float(kp_yaw_hold), relatch_rad
        self.dt = 1.0 / float(sim.cfg["sim"]["control_rate_hz"])
        # The node rate-limits what it emits, and the observer must be fed what was EMITTED.
        self.servo_slew = (math.degrees(sim.servo_slew_rad) if servo_slew_deg_s is None
                           else float(servo_slew_deg_s)) / 90.0           # [action units / s]
        self.thrust_slew = sim.thrust_slew if thrust_slew_per_s is None else float(thrust_slew_per_s)
        self.reset()

    def reset(self):
        self.ctl.reset()
        self.alloc.reset()
        self.action = np.zeros(8)
        self.latched = None
        self.relatches = 0

    def _yaw_rate_cmd(self, yaw_cmd, heading):
        rate = self.yaw_sign * float(yaw_cmd) * self.yaw_rate_scale
        if not self.hold_yaw:
            self.latched = None
            return rate
        if self.latched is None:
            self.latched = heading                       # false -> true edge: latch the measurement
        self.latched = _wrap(self.latched + rate * self.dt)
        err = _wrap(self.latched - heading)
        if abs(err) > self.relatch:                      # heading jumped: do not chase it
            self.latched, err = heading, 0.0
            self.relatches += 1
        return rate + self.kp_yaw_hold * err

    def step(self, cmd):
        """FSM command -> 8-D action (already slewed, ready for `sim.step`)."""
        d = self.sim.data
        R = d.xmat[self.sim.base_id].reshape(3, 3)
        q = d.xquat[self.sim.base_id].copy()
        psi = heading_about_up(R)
        q_level = np.array([math.cos(psi / 2.0), 0.0, math.sin(psi / 2.0), 0.0])
        e = np.zeros(3)
        mujoco.mju_subQuat(e, q_level, q)                # body-frame rot-vec, current -> target
        ori_err = _P @ e
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, d, mujoco.mjtObj.mjOBJ_BODY, self.sim.base_id, vel6, 0)
        gyro = _P @ (R.T @ vel6[:3])
        rate = self._yaw_rate_cmd(cmd.get("yaw", 0.0), psi)
        ori_err[2] = (self.ctl.kd / self.ctl.kp) * rate   # kp*e - kd*w == kd*(rate - w)
        v_cmd = np.array([cmd.get("surge", 0.0), 0.0, cmd.get("heave", 0.0)]) * self.surge_scale
        v_hat = self.ctl.obs.update(self.action, q, self.dt)
        modes = self.ctl.wrench(ori_err, gyro, v_cmd, rep103_from_cad(v_hat), self.max_duty)
        wrench = cad_wrench_from_modes(modes, self.ctl.f_max_total(self.ctl.cap))
        target = self.alloc.allocate(wrench, self.max_duty)
        ds, de = self.servo_slew * self.dt, self.thrust_slew * self.dt
        servo = self.action[:4] + np.clip(target[:4] - self.action[:4], -ds, ds)
        esc_t = target[4:]
        if self.servo_aware:
            p = self.plant
            rng = float(p.servo_range_rad)
            f_t = np.sign(esc_t) * np.abs(esc_t) ** p.thrust_curve_exp      # in units of thrust_per_cmd
            f_a = f_t * np.cos((target[:4] - servo) * rng)                    # component along the real angle
            esc_t = np.sign(f_a) * np.abs(f_a) ** (1.0 / p.thrust_curve_exp)
        self.action = np.concatenate([
            servo,
            self.action[4:] + np.clip(np.clip(esc_t, -self.max_duty, self.max_duty) - self.action[4:], -de, de)])
        return self.action.copy()
