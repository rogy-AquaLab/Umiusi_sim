"""Classical attitude + cruise controller — the SAME code in sim and on the robot.

This is the deployable half of `tools/classical_control.py`. It is here, and not in the sim
wheel, for one reason: every sim2real failure this project has recorded was an INTERFACE bug, not
a physics bug — pitch/yaw swapped on the vehicle (2026-08-21), world-frame omega fed to a
body-frame model, a `forces` policy scored in `esc` mode. In the sim both sides of an interface
use the same wrong convention and it cancels; on the robot it does not. A second implementation
of this controller in the ROS node would be one more such interface. So there is one
implementation, in the one package that is installed on the Pi (see this package's pyproject),
and the ROS node and `tools/classical_control.py` both drive it.

Pure numpy. No MuJoCo, no `umiusi_sim`, no `umiusi_rl` — those must never reach the robot.

FRAME. Internally everything is the sim/CAD frame (+X forward, +Y UP, +Z starboard), because
that is the frame the thruster geometry was measured in and the frame the allocation was
validated in. The controller's PUBLIC vectors — `ClassicalController.wrench` arguments — are
REP-103 body (x fwd, y left, z up), which is the deployment contract (docs/rl.md): a policy or a
controller in that frame consumes the robot's IMU with no axis shuffling. `rep103_from_cad` /
`cad_from_rep103` are the ONLY place the two meet; do not open-code the swap at a call site.

CALIBRATION GATES ACCURACY. `thrust_per_cmd` and `thrust_curve_exp` are both uncalibrated and
come from a contaminated fit; the observer and the feedforward cannot be better than they are.
The contract carries them explicitly so the robot reproduces the plant the controller was tuned
against, the same way `export_policy.py`'s `action_contract` does for a learned policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
_Y_UP = np.array([0.0, 1.0, 0.0])
_ARRAY_FIELDS = ("thrust_axes", "pivots_from_com", "drag_lin", "drag_quad", "added_mass_diag")


def rep103_from_cad(v):
    """sim/CAD (+X fwd, +Y up, +Z starboard) -> REP-103 (x fwd, y left, z up)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[2], v[1]])


def cad_from_rep103(v):
    """REP-103 (x fwd, y left, z up) -> sim/CAD (+X fwd, +Y up, +Z starboard)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[0], v[2], -v[1]])


def quat_to_mat(quat):
    """[w, x, y, z] -> 3x3 rotation matrix. Here so the robot needs no MuJoCo for `mju_quat2Mat`."""
    w, x, y, z = (float(c) for c in quat)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:            # a zero quaternion is not normalizable; the caller's AHRS glitched
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class PlantContract:
    """The calibrated constants and geometry a deployed controller carries.

    SNAPSHOTTED, NEVER READ LIVE. In sim, `domain_rand` rebinds thrust/drag/added mass on every
    reset; a controller that read them at control time would be seeing the very truth it is
    supposed to be robust to, and every DR number was flattered by an amount nobody could see.
    On the robot the same discipline means the controller cannot silently follow a config edit
    that the tuning was never checked against.

    Geometry is CAD frame. `pivots_from_com` is each thruster pivot relative to the CENTRE OF
    MASS (not the body origin) — the moment arm. It is baked in at export because computing it
    needs the full mass tree, which only the simulator has.
    """

    thrust_per_cmd: float           # N at |u| = 1, per thruster
    thrust_curve_exp: float         # F = sign(u) * |u|**exp * thrust_per_cmd
    servo_range_rad: float          # mechanical half-range of the azimuth servo
    thrust_axes: np.ndarray         # (4, 3) neutral thrust direction per unit, CAD frame
    pivots_from_com: np.ndarray     # (4, 3) pivot - CoM, CAD frame [m]
    drag_lin: np.ndarray            # (3,) N/(m/s), CAD axis order
    drag_quad: np.ndarray           # (3,) N/(m/s)^2, CAD axis order
    added_mass_diag: np.ndarray     # (3,) translational added mass [kg], CAD axis order
    mass: float                     # kg, hull + thrusters
    net_buoy_up: float              # N, positive = buoyant (this hull is, by ~1.17 N)
    control_rate_hz: float
    cap_ref: float                  # the esc cap the attitude gains are QUOTED at
    version: int = CONTRACT_VERSION

    def to_dict(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        got = d.pop("version", 0)
        if got != CONTRACT_VERSION:
            raise ValueError(
                f"classical contract version {got} != {CONTRACT_VERSION}. Re-export it with "
                "tools/export_classical.py — a stale contract ships a different plant than the "
                "one the controller was tuned against, which is the A-11 failure mode.")
        for k in _ARRAY_FIELDS:
            d[k] = np.asarray(d[k], dtype=float)
        return cls(**d)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def f_max_total(plant, max_duty):
    """Wrench magnitude one mode unit stands for at this cap [N] — 4 units at full thrust."""
    return 4.0 * plant.thrust_per_cmd * max(float(max_duty), 1e-9) ** plant.thrust_curve_exp


def reachable_speed(thrust_per_cmd, thrust_curve_exp, drag_lin, drag_quad, max_duty):
    """Terminal surge speed of this plant at this cap [m/s]: solve thrust = drag for v.

    `VEL_PER_CAP * max_duty` is a straight line fitted near the deploy cap 0.25 and it is only
    right there, because thrust goes as cap**exp while drag goes as v**2: at cap 0.4 the line
    says 0.272 m/s and the solve says 0.358 (24 % low), at cap 0.5 0.340 vs 0.458. Anything that
    scales a velocity by the cap must use this.

    DELIBERATELY DUPLICATED in `umiusi_rl.envs.umiusi_pose_env`: the sim wheel imports nothing
    from this package (a training box installs `packages/sim[rl]` alone), and this is a physics
    identity rather than a tunable fact. `tests/test_classical_contract.py` pins the two equal —
    if you change one, that test fails until you change the other.
    """
    f = 4.0 * float(thrust_per_cmd) * max(float(max_duty), 0.0) ** float(thrust_curve_exp)
    lin, quad = float(drag_lin), float(drag_quad)
    if quad <= 1e-9:
        return f / max(lin, 1e-9)
    return float((-lin + np.sqrt(lin * lin + 4.0 * quad * f)) / (2.0 * quad))


class VelocityObserver:
    """v̂ in the CAD body frame from the commanded thrust and the hydrodynamic model.

    Uses only what the robot knows: the servo/esc command it just issued, the plant constants and
    the AHRS attitude. No true velocity, no DVL. Integrates

        v̇ = (f_thrust + f_buoy_body - drag(v)) / m_eff

    with drag(v) = lin*v + quad*|v|*v elementwise, m_eff = mass + added mass. Drag-dominated, so
    this settles rather than drifting; the residual is a bias set by model error. That is the
    whole reason the LEARNED policy could not do this: a one-step MLP cannot integrate the force
    history, so it could not estimate drift even though prev_action carries the information.
    """

    def __init__(self, plant):
        self.plant = plant
        self.m_eff = plant.mass + plant.added_mass_diag
        self.lin, self.quad = plant.drag_lin, plant.drag_quad
        self.v = np.zeros(3)

    def reset(self):
        self.v[:] = 0.0

    def update(self, action, quat, dt):
        """action = [servo x4, esc x4] as commanded (from prev_action); quat from the AHRS."""
        p = self.plant
        servo = np.asarray(action[:4]) * p.servo_range_rad
        u = np.asarray(action[4:8])
        thrust = np.sign(u) * np.abs(u) ** p.thrust_curve_exp * p.thrust_per_cmd
        # per unit: horizontal along its tangent, vertical along +Y (the mixer's own idealization)
        f = np.zeros(3)
        for k in range(4):
            t = p.thrust_axes[k]
            f += np.cos(servo[k]) * thrust[k] * t + np.sin(servo[k]) * thrust[k] * _Y_UP
        # net buoyancy acts along WORLD +Y; rotate it into the body frame using the AHRS attitude
        # (the real vehicle has absolute orientation, it is just not in the policy's obs vector).
        f += quat_to_mat(quat).T @ np.array([0.0, p.net_buoy_up, 0.0])
        drag = self.lin * self.v + self.quad * np.abs(self.v) * self.v
        self.v += (f - drag) / self.m_eff * dt
        return self.v.copy()


class GeneralAllocator:
    """Desired 6-D wrench (CAD frame) -> [servo x4, esc x4], by pseudo-inverse of the geometry.

    ModeMixer uses a hardcoded Walsh sign table that assumes ALL FOUR units work. This solves the
    allocation from the actual thruster positions instead, which buys two things the table cannot:

      * FAULT TOLERANCE. Drop a unit's columns and re-solve. The reduced 6x6 has rank 6 and
        condition 8.1 for any single failure, so the vehicle keeps FULL 6-DOF wrench authority on
        three thrusters — it just has no margin left. A fixed table cannot express this: it keeps
        commanding the dead unit and the realised wrench is silently wrong.
      * SERVO BIAS COMPENSATION. A known per-unit angle offset rotates that unit's thrust; folding
        it into the solve puts the intended force back where it belongs.

    x = [h_1..h_4, v_1..v_4] are per-unit horizontal (along the unit tangent) and vertical (+Y)
    force components; the minimum-norm pinv solution is also the null-free one, so this agrees
    with ModeMixer when all four units are live.

    NULL SPACE (`prefer_deg`). Eight actuator DOF carry a six-DOF wrench, so every wrench has a
    2-D family of solutions and minimum norm is only ONE of them — chosen memorylessly, with no
    idea where the servos currently are or how fast they can turn. Measured cost of that choice
    (tools/alloc_rate_audit.py, hold, no DR): the commanded servo rate exceeds the 250 deg/s slew
    limit on 15 % of steps and the servos sit 23 deg (p95 124 deg) behind the command, so the
    wrench the vehicle actually produces is not the one that was solved for.

    The measured cause is the AZIMUTH SINGULARITY, not the null space per se. The servo range is
    +-90 deg, so a force direction maps to exactly one servo angle and the 180 deg fold at
    |phi| = 90 is forced — there is no branch to choose and hysteresis is not available. And that
    is precisely where the vehicle lives: holding station, the required per-unit force is nearly
    pure vertical (buoyancy trim), so |phi| has median 84 deg and exceeds 80 deg on 59 % of steps,
    with the horizontal component changing sign on 1.3 % of steps — each of which is a ~180 deg
    servo command. It is the classic marine-DP azimuth problem.

    `prefer_deg` is the classic answer: spend the null space on staying AWAY from the singularity.
    The two null directions are horizontal circulations that produce no wrench at all, so they can
    lift every |h| off zero for free. Note that "stay near the previous solution" does NOT work as
    an objective — every minimum-norm solution lies in the row space of A, so the null component
    of the difference between two of them is identically zero and the projection returns where it
    started; and aiming at the previous ANGLE only chases the singularity, since that is where the
    previous angle already was (measured: it made the commanded rate worse, p95 725 -> 6696 deg/s).
    """

    def __init__(self, plant, servo_offset_rad=None, live=None, prefer_deg=None, dead_hold=False,
                 w_move=1.0, cap_margin=0.85, w_cap=50.0):
        self.plant = plant
        self.r = np.asarray(plant.pivots_from_com, dtype=float)
        self.offset = np.zeros(4) if servo_offset_rad is None else np.asarray(servo_offset_rad, float)
        self.prefer = None if prefer_deg is None else np.radians(float(prefer_deg))
        self.w_move = float(w_move)
        self.cap_margin, self.w_cap = float(cap_margin), float(w_cap)
        self.dead_hold, self.phi_prev, self.z_prev = dead_hold, None, None
        self.hv_prev = np.zeros(8)
        self.set_live(np.ones(4, dtype=bool) if live is None else np.asarray(live, dtype=bool))

    def reset(self):
        self.phi_prev, self.z_prev = None, None

    def set_live(self, live):
        self.live = np.asarray(live, dtype=bool)
        A = np.zeros((6, 8))
        for k in range(4):
            t = self.plant.thrust_axes[k]
            A[0:3, k], A[3:6, k] = t, np.cross(self.r[k], t)
            A[0:3, 4 + k], A[3:6, 4 + k] = _Y_UP, np.cross(self.r[k], _Y_UP)
        cols = [k for k in range(4) if self.live[k]] + [4 + k for k in range(4) if self.live[k]]
        self.A, self.cols, self.pinv = A, cols, np.linalg.pinv(A[:, cols])
        # orthonormal basis of the null space of the LIVE columns: the directions we may move in
        # without changing the realised wrench at all. Empty (rank 6, 3 units) when a unit is dead.
        _u, s, vt = np.linalg.svd(A[:, cols])
        rank = int((s > s.max() * max(A[:, cols].shape) * np.finfo(float).eps).sum())
        self.null = vt[rank:].T
        self.phi_prev, self.z_prev = None, None

    def _away_from_singularity(self, xr, f_max):
        """Pick the null-space offset by a cost over BOTH distance-to-singularity and servo motion.

        Two terms, and both are needed. Singularity alone re-optimises from scratch every step and
        the argmin hops between grid cells — measured, that is worse than doing nothing (commanded
        rate 344 -> 500+ deg/s). Continuity alone just sits at the singularity. Together this is
        the standard marine-DP azimuth allocation with rate weighting; the search is centred on
        the previous offset so the sequence is smooth by construction.

        z is 2-D (0-D with a unit dead), so a local grid beats a solver: a few hundred vectorised
        8-vectors, no local-minimum trouble, and the cost is not smooth near the fold anyway.
        """
        if not self.null.size:
            return xr
        # Nothing to avoid? Then do not spend anything. The circulation that keeps a unit off the
        # singularity is force it would not otherwise make, paid for out of the duty cap. Cruising,
        # the commanded horizontal force already holds every unit off the fold (|phi| median 23 deg
        # against 84 deg in hold), so the search is skipped and costs nothing.
        x0 = np.zeros(8)
        x0[self.cols] = xr
        if np.max(np.abs(np.arctan2(x0[4:], x0[:4]))) <= self.prefer:
            self.z_prev = None
            return xr
        prefer = self.prefer
        n = self.null.shape[1]
        z0 = self.z_prev if self.z_prev is not None else np.zeros(n)
        g = np.linspace(-0.5, 0.5, 11) * f_max
        z = np.stack(np.meshgrid(*([g] * n)), axis=-1).reshape(-1, n) + z0
        cand = xr[None, :] + z @ self.null.T                       # [G, n_cols]
        x = np.zeros((cand.shape[0], 8))
        x[:, self.cols] = cand
        h, v = x[:, :4], x[:, 4:]
        mag = np.hypot(h, v)
        phi = np.arctan2(v, h)
        into = np.maximum(0.0, np.abs(phi) - prefer)               # rad past the preferred angle
        # the servo angle is the FOLDED one, and that is what has to move smoothly
        phi_s = np.where(np.abs(phi) > np.pi / 2.0, phi - np.sign(phi) * np.pi, phi)
        move = np.zeros_like(phi_s) if self.phi_prev is None else phi_s - self.phi_prev[None, :]
        # price approaching the cap, not only exceeding it: a candidate that sits AT the cap has no
        # margin left for the next disturbance, and `preserve_direction` will then shrink the whole
        # wrench rather than just that unit.
        over = np.maximum(0.0, mag - self.cap_margin * f_max) / max(f_max, 1e-9)
        cost = (np.sum(mag * (into ** 2 + self.w_move * move ** 2), axis=1)
                + self.w_cap * np.sum(over ** 2, axis=1))
        i = int(np.argmin(cost))
        self.z_prev = z[i]
        return cand[i]

    def allocate(self, wrench, max_duty, preserve_direction=True):
        p = self.plant
        x = np.zeros(8)
        xr = self.pinv @ np.asarray(wrench, dtype=float)
        f_max = p.thrust_per_cmd * max_duty ** p.thrust_curve_exp
        if self.prefer is not None:
            xr = self._away_from_singularity(xr, f_max)
        x[self.cols] = xr
        h, v = x[:4], x[4:]
        # Saturation handling. pinv minimises ||x||, which is the wrong objective once the duty cap
        # binds: clipping each unit independently changes the DIRECTION of the realised wrench, and
        # for attitude control a wrench pointing the wrong way is worse than one that is too small.
        # This matters most with a unit dead — three thrusters must each work harder, so the cap
        # binds far sooner (measured: fault-AWARE allocation was worse than fault-unaware on
        # attitude until this was added, purely because the naive solve saturated).
        # Scale the whole solution instead: same direction, reduced magnitude.
        if preserve_direction:
            worst = np.max(np.hypot(h, v)) / max(f_max, 1e-9)
            if worst > 1.0:
                h, v = h / worst, v / worst
        # a known servo bias rotates the (h, v) pair the other way before folding
        c, s_ = np.cos(self.offset), np.sin(self.offset)
        h, v = c * h + s_ * v, -s_ * h + c * v
        h = np.where(np.abs(h) < 1e-9, 0.0, h)   # 折返し境界での符号ノイズを潰す
        phi = np.arctan2(v, h)
        rear = np.abs(phi) > np.pi / 2.0 + 1e-9
        phi = np.where(rear, phi - np.sign(phi) * np.pi, phi)
        mag = np.hypot(h, v)
        u = np.where(rear, -1.0, 1.0) * (np.minimum(mag, f_max) / p.thrust_per_cmd) \
            ** (1.0 / p.thrust_curve_exp)
        dead = mag < 0.02 * f_max
        # A unit below the dead zone makes no thrust, so its ANGLE is free. Snapping it to 0
        # spends the servo's whole travel budget on a unit that is not pushing, and in hold-station
        # the required force sits near this threshold, so units cross it constantly: the servo is
        # driven to 0 and back every few steps. dead_hold parks it where it already is instead.
        idle = self.phi_prev / p.servo_range_rad if (self.dead_hold and self.phi_prev is not None) \
            else np.zeros(4)
        servo = np.where(dead, idle, phi / p.servo_range_rad)
        u = np.where(dead | ~self.live, 0.0, u)
        servo = np.clip(servo, -1.0, 1.0)
        # the angle actually COMMANDED, which is what the next warm start must aim at (post fold,
        # post clip, post the dead-zone snap to 0 — all of them move the servo)
        self.phi_prev = servo * p.servo_range_rad
        # The (h, v) this solved for, normalised by the cap force. This IS the action of the env's
        # "forces" mode, so it is the label to clone when the student works in that space — a
        # continuous target, unlike the folded servo angle above.
        self.hv_prev = np.concatenate([h, v]) / max(f_max, 1e-9)
        return np.concatenate([servo, np.clip(u, -1.0, 1.0)])


class ClassicalController:
    """obs -> 6-D wrench command. Attitude PID + exact buoyancy trim + observer-corrected cruise.

    Two pathologies of the learned policy are structural and disappear here by construction:
      * hovering at ~90 % of the esc cap — the required wrench is computed, not discovered;
      * commanding heave UPWARD on a positively buoyant vehicle — buoyancy is a known constant,
        so the trim term is exact.

    The integral term exists for MODEL MISMATCH, which is the one regime the PD version lost in.
    Everything domain_rand shakes — CoB height (±60 %), displaced volume (±5 %), per-unit servo
    neutral (±3 deg), per-unit thrust gain (±10 %) — enters the attitude loop as a torque that is
    CONSTANT over an episode. A PD leaves exactly that as steady-state error; an integrator is the
    textbook answer and needs no extra sensor. ki=0 reproduces the PD behaviour.

    EVERY CAP-DEPENDENT QUANTITY IS SOLVED, NOT TUNED. There are exactly four, and all of them
    reduce to f_max(cap) = 4 * thrust_per_cmd * cap**exp:
        attitude gains   tau_mode = (kp*e - kd*w + ki*int e) * f_max(cap_ref) / f_max(cap)
        buoyancy trim    fz_mode  = -net_buoy_up / f_max(cap)
        reachable speed  v_ref    = the positive root of lin*v + quad*v^2 = f_max(cap)
        cruise feedfwd   ff_mode  = drag(v_cmd) / f_max(cap)
    Do NOT bake a constant tuned at cap 0.25 into a deploy node. The operator moves the cap by
    hand (0.25 -> 0.4 as trust builds) and the loop gain goes as cap**exp.
    """

    def __init__(self, plant, kp=2.2, kd=0.45, k_ff=1.0, k_v=1.2,
                 ki=0.0, i_max=0.35, cap_norm=True, cap_tau=1.0):
        self.kp, self.kd, self.k_ff, self.k_v = kp, kd, k_ff, k_v
        self.ki, self.i_max = ki, i_max
        self.plant = plant
        self.net_buoy_up = plant.net_buoy_up
        self.dt = 1.0 / float(plant.control_rate_hz)
        self.obs = VelocityObserver(plant)
        self.i_err = np.zeros(3)
        # The cap the attitude gains are QUOTED at. A mode is normalized by the full-cap wrench, so
        # a fixed kp in mode units delivers a physical torque proportional to cap ** exp — the loop
        # gain swings 4x across the deploy range (max_duty 0.2 -> 0.4, and the operator moves it by
        # hand). Measured: the cap alone cost 2.3x in attitude error even though it is OBSERVED.
        # Quoting the gains against a reference cap and dividing by the live one makes them mean a
        # torque, and the cap stops being a disturbance. cap_norm=False restores the old behaviour.
        self.cap_norm = cap_norm
        self.cap_ref = float(plant.cap_ref)
        self.cap_tau, self.cap = cap_tau, None

    def f_max_total(self, max_duty):
        return f_max_total(self.plant, max_duty)

    def reachable_speed(self, max_duty):
        """Terminal surge speed the CALIBRATED plant can hold at this cap [m/s].

        Surge drag stands in for the whole horizontal plane, the same approximation the env's
        command ceiling makes, so the controller normalizes by the same number the command was
        sampled against.
        """
        p = self.plant
        return reachable_speed(p.thrust_per_cmd, p.thrust_curve_exp,
                               p.drag_lin[0], p.drag_quad[0], max_duty)

    def reset(self):
        self.obs.reset()
        self.i_err[:] = 0.0
        self.cap = None

    def filter_cap(self, max_duty):
        """The ESC cap is a PARAMETER the operator sets, not a 50 Hz measurement.

        It reaches the controller through the observation vector, which carries sensor noise
        (obs_noise 0.005 — about 2 % of a 0.25 cap), and the cap enters the loop gain SQUARED. Raw,
        it jitters the attitude gain a few percent every step for nothing. The operator moves the
        cap on a timescale of minutes, so filter it; callers should use this value, not the raw obs,
        wherever the cap converts between mode units and newtons.
        """
        a = self.dt / max(self.cap_tau, self.dt)
        self.cap = float(max_duty) if self.cap is None else (1.0 - a) * self.cap + a * float(max_duty)
        return self.cap

    def wrench(self, ori_err, gyro, v_cmd_body, v_hat_body, max_duty):
        """All 3-vectors REP-103 body (x fwd, y left, z up). Returns modes in [-1, 1]."""
        # Attitude: PID on the rotation-vector error. ori_err points along the rotation that takes
        # the vehicle to its target, so the moment goes the same way.
        cap = self.filter_cap(max_duty)
        g_cap = (f_max_total(self.plant, self.cap_ref) / f_max_total(self.plant, cap)
                 if self.cap_norm else 1.0)
        tau = (self.kp * np.asarray(ori_err) - self.kd * np.asarray(gyro)
               + self.ki * self.i_err) * g_cap
        if self.ki > 0.0:
            # Anti-windup, both halves: stop integrating any axis whose command already saturates
            # (otherwise the integrator charges up while the plant cannot respond, and overshoots
            # on the way back), and clamp the integral's own authority to i_max mode units.
            grow = np.abs(tau) < 1.0
            self.i_err = np.where(grow, self.i_err + np.asarray(ori_err) * self.dt, self.i_err)
            lim = self.i_max / self.ki
            self.i_err = np.clip(self.i_err, -lim, lim)
        # Cruise: feed forward the wrench that holds v_cmd against drag, then correct with the
        # observer. v_hat is the only thing standing in for the missing DVL.
        # Normalize by what the cap can actually hold, solved from the plant instead of a line
        # fitted at the deploy cap 0.25 — that line is 41 % low at cap 0.5.
        v_ref = self.reachable_speed(cap)
        # ...and clamp the command to it, keeping its DIRECTION (same rule as the saturation
        # scaling in the allocator). An unreachable command never closes its error: the feedback
        # term pins the horizontal modes at the clip, and the clip then takes the authority away
        # from the attitude axes. The operator lowering the cap must cost speed, not attitude.
        v_cmd_xy = np.asarray(v_cmd_body[:2], dtype=float)
        speed = float(np.linalg.norm(v_cmd_xy))
        if v_ref > 0.0 and speed > v_ref:
            v_cmd_xy = v_cmd_xy * (v_ref / speed)
        # The thrust that HOLDS v_cmd is its drag, so compute it rather than interpolate. Drag goes
        # as v**2 while `k_ff * v_cmd / v_ref` is a straight line through it — equal only at 0 and
        # at v_ref, and up to ~2x low in between. In these units a mode of 1.0 IS the full-cap
        # thrust, so dividing the drag by it is the whole cap dependence, exactly.
        # FRAME TRAP: v_cmd is REP-103 [surge, sway] but the drag arrays are CAD, where +Y is UP —
        # so the horizontal plane is rows [0, 2], not [0, 1]. Row 1 is heave.
        d_lin, d_quad = self.plant.drag_lin[[0, 2]], self.plant.drag_quad[[0, 2]]
        drag_cmd = d_lin * v_cmd_xy + d_quad * np.abs(v_cmd_xy) * v_cmd_xy
        ff = self.k_ff * drag_cmd / self.f_max_total(cap)
        fb = self.k_v * (v_cmd_xy - np.asarray(v_hat_body)[:2]) / max(v_ref, 1e-9)
        f_xy = ff + fb
        # Buoyancy trim, in the mode units of THIS episode's cap. A mode is normalized by the
        # full-cap wrench, so the same physical force is a different mode value at a different
        # max_duty — and the cap is a runtime parameter the operator raises (0.25 -> 0.4). Fixing
        # the trim at the nominal cap left a standing heave error at every other cap.
        fz_trim = -self.net_buoy_up / f_max_total(self.plant, cap)
        return np.clip([f_xy[0], f_xy[1], fz_trim, tau[0], tau[1], tau[2]], -1.0, 1.0)
