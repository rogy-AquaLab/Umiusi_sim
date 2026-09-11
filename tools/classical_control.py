"""Non-learned baseline: geometric allocation + PD attitude + a thrust-model velocity observer.

We have been tuning RL for weeks with NO non-learned baseline to compare against. This is that
baseline. It reuses the parts that are already exact and replaces only the part RL was doing:

    ModeMixer          6-D wrench -> 8-D (servo, esc). Already pure kinematics — atan2 fold,
                       per-unit force split, null modes absent from the basis. NOT relearned.
    this controller    what wrench to command. RL's actual job, done classically instead.

Two pathologies of the learned policy are structural and disappear here by construction:
  * hovering at ~90 % of the esc cap — the required wrench is computed, not discovered;
  * commanding heave UPWARD on a positively buoyant vehicle — buoyancy is a known constant
    (+1.17 N), so the trim term is exact.

The velocity observer exists because the deployed observation has no lateral velocity
(obs = ori_err, gyro, v_cmd, prev_action, max_duty — no DVL, no position). A one-step MLP
cannot integrate the force history, so the learned policy could not estimate drift even though
prev_action carries the information. Integrating it explicitly is cheap and self-limiting:
velocity is drag-dominated, so the estimate converges to a bias set by model error rather than
drifting without bound.

    ACCURACY IS GATED BY CALIBRATION. thrust_per_cmd and thrust_curve_exp are both uncalibrated
    and come from a contaminated fit (docs/calibration_plan.md). The observer cannot be better
    than they are; bench calibration (§3) is what improves it.

Usage:
    python tools/classical_control.py --episodes 12            # hold-station diagnostic
    python tools/classical_control.py --episodes 12 --cruise   # with velocity commands
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))

from umiusi_rl.envs.mode_mixer import MODE_NAMES  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config, reachable_speed  # noqa: E402


def nominal_plant(sim):
    """The plant constants a DEPLOYED controller carries — snapshotted, never read live.

    domain_rand REBINDS sim.thrust_per_cmd / thrust_curve_exp / drag / added mass on every reset
    (umiusi_pose_env._apply_domain_rand). Reading them through `sim` at control time let the
    controller see the very truth it is supposed to be robust to, which silently flattered every
    DR number. Geometry (thruster positions, axes, servo range) is fixed by the CAD, so it is
    honest either way. `max_duty` is NOT here: the cap is observed (obs[17]), so it is an input.

    Drag lives here too — it is a calibrated constant, not geometry, and DR shakes it. The observer
    integrates it and `ClassicalController.reachable_speed` solves against it; those two must be
    the same numbers, or the cruise loop feeds forward for a plant its own observer disagrees with.
    """
    return SimpleNamespace(
        thrust_per_cmd=float(sim.thrust_per_cmd),
        thrust_curve_exp=float(sim.thrust_curve_exp),
        servo_range_rad=float(sim.servo_range_rad),
        thrust_axes=np.array(sim.thrust_axes, dtype=float),
        drag_lin=np.array(sim.drag_lin[:3], dtype=float),
        drag_quad=np.array(sim.drag_quad[:3], dtype=float),
    )


def f_max_total(plant, max_duty):
    """Wrench magnitude one mode unit stands for at this cap [N] — 4 units at full thrust."""
    return 4.0 * plant.thrust_per_cmd * max(float(max_duty), 1e-9) ** plant.thrust_curve_exp


def body_inertia_tensor(sim, r):
    """Full 3x3 rotational inertia in the BODY frame: hull + thrusters + rotational added mass.

    Two traps, both of which silently halve or rotate the predicted angular acceleration:
      * `model.body_inertia` is diagonal in the body's PRINCIPAL frame, and `body_iquat` here is a
        ~120 deg rotation, not identity. Dividing a body-frame moment by it elementwise mixes axes.
      * underwater the fluid entrained by a rotating hull is not negligible next to the structure
        (added_mass_diag[3:6] vs body_inertia ~ 0.08-0.17 vs 0.19-0.47), so it belongs here.
    Thrusters enter by the parallel-axis theorem as point masses at their pivots.
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(sim.model.body_iquat[sim.base_id], dtype=float))
    R = R.reshape(3, 3)
    tensor = R @ np.diag(np.asarray(sim.model.body_inertia[sim.base_id], dtype=float)) @ R.T
    m_t = float(sim.model.body_mass[sim.thr_ids[0]])
    for k in range(4):
        d = np.asarray(r[k], dtype=float)
        tensor = tensor + m_t * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return tensor + np.diag(np.asarray(sim.added_mass_diag[3:6], dtype=float))


class VelocityObserver:
    """v̂ in the BODY frame from the commanded thrust and the hydrodynamic model.

    Uses only what the robot knows: the servo/esc command it just issued, and the plant
    constants. No true velocity, no DVL. Integrates

        v̇ = (f_thrust + f_buoy_body - drag(v)) / m_eff

    with drag(v) = lin*v + quad*|v|*v elementwise, m_eff = mass + added mass. Drag-dominated, so
    this settles rather than drifting; the residual is a bias proportional to the model error.
    """

    def __init__(self, sim, mass, net_buoy_up):
        self.m_eff = mass + sim.added_mass_diag[:3]
        self.net_buoy_up = net_buoy_up
        self.plant = nominal_plant(sim)
        self.lin, self.quad = self.plant.drag_lin, self.plant.drag_quad
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
            f += np.cos(servo[k]) * thrust[k] * t + np.sin(servo[k]) * thrust[k] * np.array([0.0, 1.0, 0.0])
        # net buoyancy acts along WORLD +Y; rotate it into the body frame using the AHRS attitude
        # (the real vehicle has absolute orientation, it is just not in the policy's obs vector).
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
        f += R.reshape(3, 3).T @ np.array([0.0, self.net_buoy_up, 0.0])
        drag = self.lin * self.v + self.quad * np.abs(self.v) * self.v
        self.v += (f - drag) / self.m_eff * dt
        return self.v.copy()


class GeneralAllocator:
    """Desired 6-D wrench -> [servo x4, esc x4], by pseudo-inverse of the real geometry.

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

    NULL SPACE (`warm_start`). Eight actuator DOF carry a six-DOF wrench, so every wrench has a
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

    def __init__(self, sim, servo_offset_rad=None, live=None, prefer_deg=None, dead_hold=False,
                 w_move=1.0, cap_margin=0.85, w_cap=50.0):
        mujoco.mj_forward(sim.model, sim.data)
        com = sim.data.subtree_com[sim.base_id] - sim.data.xpos[sim.base_id]
        com_local = sim.data.xmat[sim.base_id].reshape(3, 3).T @ com
        self.r = sim.unit_pivots - com_local
        self.plant = nominal_plant(sim)
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
        y = np.array([0.0, 1.0, 0.0])
        A = np.zeros((6, 8))
        for k in range(4):
            t = self.plant.thrust_axes[k]
            A[0:3, k], A[3:6, k] = t, np.cross(self.r[k], t)
            A[0:3, 4 + k], A[3:6, 4 + k] = y, np.cross(self.r[k], y)
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

    The integral term exists for MODEL MISMATCH, which is the one regime the PD version lost in.
    Everything domain_rand shakes — CoB height (±60 %), displaced volume (±5 %), per-unit servo
    neutral (±3 deg), per-unit thrust gain (±10 %) — enters the attitude loop as a torque that is
    CONSTANT over an episode. A PD leaves exactly that as steady-state error; an integrator is the
    textbook answer and needs no extra sensor. ki=0 reproduces the PD behaviour.
    """

    def __init__(self, sim, mass, net_buoy_up, kp=2.2, kd=0.45, k_ff=1.0, k_v=1.2,
                 ki=0.0, i_max=0.35, cap_norm=True, cap_tau=1.0):
        self.kp, self.kd, self.k_ff, self.k_v = kp, kd, k_ff, k_v
        self.ki, self.i_max = ki, i_max
        self.net_buoy_up = net_buoy_up
        self.plant = nominal_plant(sim)
        self.dt = 1.0 / float(sim.cfg["sim"]["control_rate_hz"])
        self.obs = VelocityObserver(sim, mass, net_buoy_up)
        self.i_err = np.zeros(3)
        # The cap the attitude gains are QUOTED at. A mode is normalized by the full-cap wrench, so
        # a fixed kp in mode units delivers a physical torque proportional to cap ** exp — the loop
        # gain swings 4x across the deploy range (max_duty 0.2 -> 0.4, and the operator moves it by
        # hand). Measured: the cap alone cost 2.3x in attitude error even though it is OBSERVED.
        # Quoting the gains against a reference cap and dividing by the live one makes them mean a
        # torque, and the cap stops being a disturbance. cap_norm=False restores the old behaviour.
        self.cap_norm = cap_norm
        self.cap_ref = float(sim.max_duty)
        self.cap_tau, self.cap = cap_tau, None

    def f_max_total(self, max_duty):
        return f_max_total(self.plant, max_duty)

    def reachable_speed(self, max_duty):
        """Terminal surge speed the CALIBRATED plant can hold at this cap [m/s].

        Surge drag stands in for the whole horizontal plane, the same approximation the env's
        command ceiling makes (`umiusi_pose_env._reachable_speed`), so the controller normalizes
        by the same number the command was sampled against.
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
        tau = (self.kp * ori_err - self.kd * gyro + self.ki * self.i_err) * g_cap
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
        # Normalize by what the cap can actually hold, solved from the plant instead of
        # VEL_PER_CAP's straight line: the line is fitted at the deploy cap 0.25 and is 41 % low at
        # cap 0.5 (0.340 vs 0.458 m/s), so raising the cap used to shrink v_ref far faster than the
        # vehicle's real speed grew, and the same command asked for progressively more thrust.
        v_ref = self.reachable_speed(cap)
        # ...and clamp the command to it, keeping its DIRECTION (same rule as the saturation
        # scaling below). An unreachable command never closes its error: the feedback term pins the
        # horizontal modes at the clip, and the clip then takes the authority away from the
        # attitude axes. The operator lowering the cap must cost speed, not attitude.
        v_cmd_xy = np.asarray(v_cmd_body[:2], dtype=float)
        speed = float(np.linalg.norm(v_cmd_xy))
        if v_ref > 0.0 and speed > v_ref:
            v_cmd_xy = v_cmd_xy * (v_ref / speed)
        # The thrust that HOLDS v_cmd is its drag, so compute it rather than interpolate. Drag goes
        # as v**2 while `k_ff * v_cmd / v_ref` is a straight line through it — equal only at 0 and
        # at v_ref, and up to ~2x low in between, which is why the too-small old v_ref used to look
        # better: two errors of opposite sign. In these units a mode of 1.0 IS the full-cap thrust,
        # so dividing the drag by it is the whole cap dependence, exactly.
        # FRAME TRAP: v_cmd is REP-103 [surge, sway] but the drag arrays are in the SIM frame,
        # where +Y is UP — so the horizontal plane is rows [0, 2], not [0, 1]. Row 1 is heave.
        d_lin, d_quad = self.plant.drag_lin[[0, 2]], self.plant.drag_quad[[0, 2]]
        drag_cmd = d_lin * v_cmd_xy + d_quad * np.abs(v_cmd_xy) * v_cmd_xy
        ff = self.k_ff * drag_cmd / self.f_max_total(cap)
        fb = self.k_v * (v_cmd_xy - v_hat_body[:2]) / max(v_ref, 1e-9)
        f_xy = ff + fb
        # Buoyancy trim, in the mode units of THIS episode's cap. A mode is normalized by the
        # full-cap wrench, so the same physical force is a different mode value at a different
        # max_duty — and the cap is a runtime parameter the operator raises (0.25 -> 0.4). Fixing
        # the trim at the nominal cap left a standing heave error at every other cap.
        fz_trim = -self.net_buoy_up / f_max_total(self.plant, cap)
        return np.clip([f_xy[0], f_xy[1], fz_trim, tau[0], tau[1], tau[2]], -1.0, 1.0)


def build_controller(env, **gains):
    """A ClassicalController for this env: mass and net buoyancy come from configs/umiusi.yaml.

    The nominal config, not the episode's randomized plant — same reason as `nominal_plant`.
    """
    c = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    mass = c["hull"]["mass"] + 4 * c["thrusters"]["mass"]
    g = abs(c["sim"]["gravity"][1])
    net_buoy = c["water"]["density"] * c["water"]["displaced_volume"] * g - mass * g
    return ClassicalController(env.sim, mass, net_buoy, **gains)


def _rep103(v_sim):
    """sim/CAD (+Y up) -> REP-103 (x fwd, y left, z up)."""
    return np.array([v_sim[0], -v_sim[2], v_sim[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--cruise", action="store_true", help="sample velocity commands (default: hold)")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--domain-rand", action="store_true",
                    help="model mismatch — the regime RL was trained for; the fair comparison")
    ap.add_argument("--kp", type=float, default=2.2)
    ap.add_argument("--kd", type=float, default=0.45)
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--k-v", type=float, default=1.2)
    args = ap.parse_args()

    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="modes", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = args.domain_rand
    cfg.setdefault("disturbance", {})["enabled"] = args.disturb
    env = UmiusiPoseEnv(cfg)
    if not args.cruise:
        env.vel_cmd_zero_prob = 1.0

    ctl = build_controller(env, kp=args.kp, kd=args.kd, ki=args.ki, k_v=args.k_v)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    step = env._mode_slew_step or 1.0

    signed, absm, esc, ori, drift, verr = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        m = np.zeros(6)
        done = False
        while not done:
            # obs は実機が持つものだけ: [ori_err 3][gyro 3][v_cmd 3][prev_action 8][max_duty 1]
            ori_err, gyro, v_cmd_o = obs[0:3], obs[3:6], obs[6:9]
            prev_action, cap = obs[9:17], float(obs[17])
            v_hat_sim = ctl.obs.update(prev_action, env.sim.get_state()["quat"], dt)
            m_des = ctl.wrench(ori_err, gyro, v_cmd_o, _rep103(v_hat_sim), cap)
            rate = np.clip((m_des - m) / step, -1.0, 1.0)
            obs, _r, term, trunc, info = env.step(rate)
            m = env._mode_prev_modes.copy()
            signed.append(m)
            absm.append(np.abs(m))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            verr.append(float(np.linalg.norm(v_hat_sim - env.sim.get_state()["lin_vel"])))
            done = term or trunc
    env.close()

    sg, ab = np.array(signed).mean(0), np.array(absm).mean(0)
    print(f"classical  kp={args.kp} kd={args.kd} ki={args.ki} k_v={args.k_v}  "
          f"{'cruise' if args.cruise else 'hold-station'}  disturb={args.disturb} DR={args.domain_rand}")
    print(f"  median|esc| {np.mean(esc):.4f}   ori {np.mean(ori):.3f} rad   横流れ {np.mean(drift):.4f} m/s")
    print(f"  観測器の速度誤差 {np.mean(verr):.4f} m/s   (真値との差、較正で決まる)")
    for i, n in enumerate(MODE_NAMES):
        r = abs(sg[i]) / ab[i] if ab[i] > 1e-6 else 0.0
        print(f"   {n:3s}  符号付き {sg[i]:+.3f}   |m| {ab[i]:.3f}   比 {r:.2f}")


if __name__ == "__main__":
    main()
