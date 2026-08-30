"""UmiusiPoseEnv — attitude / depth / pose control for the UMIUSI ROV.

Algorithm-agnostic Gymnasium environment (NO PPO-specific assumptions) wrapping the
standalone UmiusiSimulator. The agent commands the 8-D per-thruster action
([servo x4, esc x4] in [-1, 1]). Three selectable tasks, matched to realistic (cheap)
sensor suites — the reward/success are always computed from the true state, so a limited
sensor set simply leaves part of the task unobservable:

    task = "attitude"       track a random target ORIENTATION (roll/pitch/yaw).
                            Cheap sensor: a 9-DOF AHRS (e.g. BNO055).   obs_mode "imu".
    task = "attitude_depth" track a random orientation AND a random depth.
                            AHRS + pressure/depth sensor.               obs_mode "imu_depth".
    task = "pose"           go-to-pose: random target POSITION (upright orientation).
                            Needs velocity (DVL) + a position reference. obs_mode "full".
    task = "attitude_velocity"  hold a random target orientation (upright + random yaw, plus bounded
                            tilt) (feedback) AND cruise in a commanded DIRECTION (feedforward: obs adds
                            the 3-D velocity command, NOT measured velocity — speed magnitude is
                            unobservable without a DVL, so only the direction is controlled). The
                            command is horizontal by default, or 3-D when vel_cmd_horizontal=false
                            (needed to reach different depths). obs_mode "imu".

Observation = exteroceptive (sensor-suite dependent) ++ proprioception (proprio_mode):
    proprio_mode "full"    servo/range (4) + thrust/thrust_per_cmd (4) + prev action (8) = 16
    proprio_mode "action"  prev action (8) only — the sim2real-safe suite: the real vehicle
                           cannot measure servo angle or (reliably) rpm, so nothing else is fed
                           back (issue #3, item 4). Dims below are for proprio_mode "full";
                           subtract 8 for "action". attitude_velocity adds v_cmd (3) to either.

    obs_mode "full"          pos_err(3) + ori_err(3) + lin_vel(3) + ang_vel(3)      -> 28
    obs_mode "imu"           ori_err(3) + ang_vel(3)                                -> 22
    obs_mode "imu_depth"     ori_err(3) + ang_vel(3) + depth_err(1)                 -> 23
    obs_mode "imu_depth_dvl" ori_err(3) + ang_vel(3) + depth_err(1) + lin_vel(3)    -> 26

ori_err is the rotation-vector error to the (task-dependent) TARGET orientation, which an
AHRS supplies (absolute attitude incl. magnetometer heading). Horizontal position (X, Z) is
only in "full" — with imu/imu_depth/imu_depth_dvl it is unobservable (no GPS underwater), so
imu* modes cannot do absolute horizontal station-keeping (imu_depth_dvl can still reject
drift via velocity). See the project README (sensor / observability notes).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from umiusi_sim.simulator import UmiusiSimulator

from umiusi_rl.envs.mode_mixer import MODE_DIM, ModeMixer

_ROOT = Path(__file__).resolve().parents[5]        # repo root (packages/sim/src/umiusi_rl/envs/..)

ACT_DIM = 8
# Open-loop terminal surge speed per unit of esc cap [m/s per max_duty], measured 2026-08-26
# (servo 0, esc full: 0.171 m/s @ cap 0.25, 0.297 m/s @ cap 0.4). Used by vel_cmd_cap_frac to
# keep velocity commands physically reachable; re-measure if thrust/drag calibration changes.
VEL_PER_CAP = 0.68
# Proprioception width per proprio_mode (see below): "full" feeds back the sim's servo angle and
# thrust state; "action" feeds back ONLY the previous action. The real vehicle cannot measure
# servo angle (RC servos have no position feedback — the bag showed /state angle is an echo of
# the command) and its rpm telemetry is partly dead, so "action" is the sim2real-safe suite
# (issue #3, item 4): everything in it exists identically on the robot.
_PROPRIO_DIM = {"full": 16, "action": 8}
# Exteroceptive (navigation-sensor) dimensions per observation mode.
_EXTERO_DIM = {"full": 12, "imu": 6, "imu_depth": 7, "imu_depth_dvl": 10}
# Default sensor suite per task (overridable with obs_mode / --obs-mode).
_DEFAULT_OBS = {"pose": "full", "attitude": "imu", "attitude_depth": "imu_depth",
                "attitude_velocity": "imu"}
_Y_UP = np.array([0.0, 1.0, 0.0])
# Observation frame presets: rows = target-frame axes in sim/CAD coordinates. Every 3-vector in
# the OBSERVATION (ori_err, angular velocity, lin_vel/pos_err, v_cmd) is expressed in this frame;
# physics, reward and success stay in the sim frame. "rep103" (x fwd, y left, z up — the ROS
# convention the robot's IMU speaks) is the DEPLOYMENT CONTRACT: a policy trained with it consumes
# the robot's IMU data with no hand-written axis shuffling (the missing remap scrambled pitch/yaw
# on the 2026-08-21 pool test — sim2real issue #3). Default "sim" keeps every existing run valid;
# tools/convert_policy_frame.py converts trained policies between frames exactly.
_OBS_FRAMES = {
    "sim": np.eye(3),
    "rep103": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "ned": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}
# Vertical-thrust mode decomposition over the 4 thruster positions (the pivots form a square, so
# these four vectors are exactly orthogonal): heave (+ + + +), roll = left - right, pitch = front -
# back, and the NULL mode (+ - + -), which produces no net force and no moment — pure waste (heat /
# battery / servo wear). The 8/25 underwater run put 41.2 % of the vertical power into the null
# mode (Umiusi_sim#3); the reward names and penalizes it directly, because a magnitude-only effort
# term cannot see it (a null command has the same action-space norm as a useful one).
# Signs per unit name: (roll, pitch, null); heave is all +1.
_VERT_MODE_SIGNS = {"lf": (1, 1, 1), "lb": (1, -1, -1), "rb": (-1, -1, 1), "rf": (-1, 1, -1)}


def load_config(path):
    """Load a training/env YAML config into a dict (path relative to the repo root is OK)."""
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


class UmiusiPoseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config, render_mode=None):
        super().__init__()
        cfg = config if isinstance(config, dict) else load_config(config)
        self.cfg = cfg

        sim_config = cfg.get("sim_config", "configs/umiusi.yaml")
        sim_config = sim_config if Path(sim_config).is_absolute() else _ROOT / sim_config
        self.sim = UmiusiSimulator(config_path=sim_config)

        e = cfg["env"]
        self.task = e.get("task", "pose")
        if self.task not in _DEFAULT_OBS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {list(_DEFAULT_OBS)}")
        self.obs_mode = e.get("obs_mode", "auto")
        if self.obs_mode == "auto":
            self.obs_mode = _DEFAULT_OBS[self.task]
        if self.obs_mode not in _EXTERO_DIM:
            raise ValueError(f"unknown obs_mode {self.obs_mode!r}; expected one of {list(_EXTERO_DIM)}")

        self.horizon = int(e["horizon"])
        self.target_box = np.array(e["target_box"], dtype=float)
        self.workspace_bounds = np.array(e["workspace_bounds"], dtype=float)
        self.start_jitter = float(e["start_jitter"])
        self.depth_target_range = float(e.get("depth_target_range", 0.5))
        self.tilt_target_deg = float(e.get("tilt_target_deg", 45.0))
        self.yaw_target_deg = float(e.get("yaw_target_deg", 180.0))
        self.vel_cmd_max = float(e.get("vel_cmd_max", 0.4))  # max commanded speed [m/s] (attitude_velocity)
        # Cap-aware command ceiling (0 = off): sample |v_cmd| <= vel_cmd_cap_frac * (VEL_PER_CAP *
        # max_duty), the episode's PHYSICALLY REACHABLE speed. Open-loop full surge tops out at
        # ~0.171 m/s (cap 0.25) / 0.297 m/s (cap 0.4) — roughly linear in cap (F ~ cap^2, drag ~ v^2)
        # — so commands beyond it (the old flat U(0, 0.4)) are unsatisfiable and cruise never forms.
        self.vel_cmd_cap_frac = float(e.get("vel_cmd_cap_frac", 0.0))
        # P(high-speed episode): command U(0.9, 1.0) * the sampled ceiling instead of U(0, hi),
        # so near-top-speed cruise is actually in the training distribution (the flat U(0, hi)
        # mean is ~hi/2 and top-speed tracking would stay untrained/unmeasured).
        self.vel_cmd_hi_prob = float(e.get("vel_cmd_hi_prob", 0.0))
        self.vel_cmd_horizontal = bool(e.get("vel_cmd_horizontal", True))  # sample v_cmd in the x-z plane
        self.vel_cmd_cone_deg = float(e.get("vel_cmd_cone_deg", 180.0))  # v_cmd dir within +/- this of +X (curriculum)
        self.vel_cmd_zero_prob = float(e.get("vel_cmd_zero_prob", 0.0))  # P(v_cmd == 0): hold-station episodes
        # 3-D command shaping (vel_cmd_horizontal: false). The vehicle is effectively MULTIMODAL —
        # "drone mode" (all thrusters tilted vertical) moves vertically far more easily than the
        # tangential-thrust horizontal mode, and naive 3-D training collapses into the vertical
        # basin, catastrophically forgetting horizontal cruise (av_cal2/3_3d lesson). Two guards:
        #   vel_cmd_elev_deg        max |elevation| of the command [deg] — a CURRICULUM can ramp
        #                           this from 0 (pure horizontal) upward via set_attr.
        #   vel_cmd_horizontal_prob per-episode probability of forcing elevation = 0, keeping a
        #                           FLOOR of pure-horizontal episodes so that skill never leaves
        #                           the training distribution.
        self.vel_cmd_elev_deg = float(e.get("vel_cmd_elev_deg", 90.0))
        self.vel_cmd_horizontal_prob = float(e.get("vel_cmd_horizontal_prob", 0.0))
        self.vel_tol = float(e.get("vel_tol", 0.10))         # velocity match tolerance [m/s]
        self.vel_deadband = float(e.get("vel_deadband", 0.0))  # m/s: no sideways-drift penalty inside this
        self.pos_tol = float(e["pos_tol"])
        self.ori_tol = float(e["ori_tol"])
        self.depth_tol = float(e.get("depth_tol", 0.10))
        self.near_goal_dist = float(e["near_goal_dist"])
        self.near_goal_ori = float(e.get("near_goal_ori", 0.20))  # rad: within this, press to settle
        self.ori_deadband = float(e.get("ori_deadband", 0.0))     # rad: no ori reward gradient inside this
        self.proprio_mode = e.get("proprio_mode", "full")
        if self.proprio_mode not in _PROPRIO_DIM:
            raise ValueError(f"unknown proprio_mode {self.proprio_mode!r}; expected one of {list(_PROPRIO_DIM)}")
        obs_frame = e.get("obs_frame", "sim")
        if obs_frame not in _OBS_FRAMES:
            raise ValueError(f"unknown obs_frame {obs_frame!r}; expected one of {list(_OBS_FRAMES)}")
        self.obs_frame = obs_frame
        self._obs_P = _OBS_FRAMES[obs_frame]
        self.rw = cfg["reward"]
        # --- economy shaping (Umiusi_sim#3: the deploy cap + null-mode waste) -------------------
        # effort_exp > 0 switches the effort penalty from the legacy L2 norm to the POWER-dimension
        # sum(|u_i|^exp): thrust ~ u^2 but electrical power ~ u^3 (propeller law), so exp = 3 makes
        # the penalty proportional to the real cost (heat / battery). w_null names the vertical
        # null mode (see _VERT_MODE_SIGNS) — pure waste a magnitude-only term cannot see.
        # econ_ramp is a 0..1 multiplier on BOTH (curriculum: learn the task first, then economize —
        # a strong effort penalty from step 0 collapses into the do-nothing local optimum, an
        # att_v3-era lesson). Default 1.0 so eval and non-curriculum runs are unaffected.
        self.effort_exp = float(self.rw.get("effort_exp", 0.0))   # 0 = legacy ||esc||_2
        self.w_null = float(self.rw.get("w_null", 0.0))
        self.econ_ramp = 1.0
        # Vertical mode decomposition (heave/roll/pitch/null), rows orthonormal over the 4 units in
        # ACTION order. Needs the geometric unit names; with an old no-name config the null penalty
        # and its diagnostics are simply off.
        names = list(self.sim.unit_names)
        if set(names) == set(_VERT_MODE_SIGNS):
            self._vert_modes = np.array(
                [[1.0, *(float(s) for s in _VERT_MODE_SIGNS[n])] for n in names]).T / 2.0
        else:
            self._vert_modes = None
        # Observe the plant's ESC cap (1 obs dim, appended LAST so existing layouts are a prefix):
        # with max_duty domain-randomized but unobserved, the economical policy converges onto the
        # lowest sampled cap and a field cap raise buys nothing; observing it makes the behaviour
        # cap-conditional. The deploy node already owns a max_duty parameter, so feeding it into the
        # obs adds no new cross-package dependency — only one more entry in the obs contract.
        self.observe_max_duty = bool(e.get("observe_max_duty", False))
        self.dr = cfg.get("domain_rand", {"enabled": False})
        # sim2real: control->actuation delay (steps); only applied when domain_rand is enabled.
        self.action_latency = int(self.dr.get("action_latency_steps", 0))
        self.dist = cfg.get("disturbance", {"enabled": False})  # water current + random impulses
        self._impulse_left = 0
        self._act_buf = []      # delayed-action buffer (sim2real latency); filled in reset()
        self._act_latency = 0

        self._base = {
            "volume": self.sim.volume,
            "thrust_per_cmd": self.sim.thrust_per_cmd,
            "drag_lin": self.sim.drag_lin.copy(),
            "drag_quad": self.sim.drag_quad.copy(),
            "added_mass": self.sim.added_mass_diag.copy(),
            "servo_slew": self.sim.servo_slew_rad,
            "servo_tau": self.sim.servo_tau,
            "thrust_exp": self.sim.thrust_curve_exp,
            "buoy_offset": self.sim.buoyancy_offset,
            "max_duty": self.sim.max_duty,
        }
        self._servo_offset = np.zeros(4)   # per-episode servo neutral offset [rad] (DR)
        self._thrust_gain = np.ones(4)     # per-episode per-thruster thrust asymmetry (DR)

        # attitude_velocity adds the feedforward velocity command (3), no measured velocity (no DVL).
        obs_dim = (_EXTERO_DIM[self.obs_mode] + _PROPRIO_DIM[self.proprio_mode]
                   + (3 if self.task == "attitude_velocity" else 0)
                   + (1 if self.observe_max_duty else 0))
        # Action parameterization (Umiusi_sim#3 null-space follow-up):
        #   "esc"   (default) raw 8-D [servo x4, esc x4] — every existing run.
        #   "modes" 6-D wrench modes [fx, fy, fz, tx, ty, tz] (REP-103 axes) expanded to the
        #           8-D action by ModeMixer — the two null patterns are unrepresentable, so the
        #           commanded null share is structurally ~0 instead of reward-shaped. Everything
        #           downstream (latency buffer, DR perturbation, reward terms, prev_action and
        #           thus the OBS CONTRACT) keeps seeing the mixed 8-D actuator command.
        self.action_mode = e.get("action_mode", "esc")
        if self.action_mode not in ("esc", "modes"):
            raise ValueError(f"unknown action_mode {self.action_mode!r}; expected 'esc' or 'modes'")
        if self.action_mode == "modes":
            # NOMINAL plant constants (not the DR-perturbed episode values): the deployed mixer
            # runs with the same nominals, and the mismatch is the policy's feedback problem.
            self._mixer = ModeMixer(self.sim.unit_names, self.sim.thrust_axes,
                                    self.sim.unit_pivots, self.sim.servo_range_rad,
                                    self.sim.thrust_per_cmd, self.sim.thrust_curve_exp)
            # Mode-command slew (mode_slew_per_s, action units/s; 0 = off): rate-limits the
            # 6-D MODE vector before mixing, so the realized wrench cannot change faster than
            # this — the bang-bang fast-dither strategy (null diagnosis 2026-08-26: servo cmd
            # jumps > 65 deg median, ~200 deg/s perpetual servo racing) is structurally
            # unreachable. Slewing MUST happen in mode coordinates: they are linear coordinates
            # of the null-free force subspace, so every intermediate command stays null-free
            # (slewing the folded servo/esc instead RAISES realized null — see ModeMixer note).
            slew = float(e.get("mode_slew_per_s", 0.0))
            dt = 1.0 / float(self.sim.cfg["sim"]["control_rate_hz"])
            self._mode_slew_step = slew * dt if slew > 0.0 else None
            # mode_rate_action: the action IS the mode RATE (a in [-1,1] scales the slew step:
            # m += a * step). Under the target+limiter form the policy learns to ride the
            # limiter with saturated targets (av_mode3: 100 % of steps at full slew, |raw -
            # applied| ~ 1.1, std blown up to 1.88 by ent_coef because raw noise became
            # physically free). As a rate, a = 0 means HOLD, the raw signal is the physical
            # quantity itself, and a rate penalty (w_mode_rate) finally has a real gradient.
            self._mode_rate_action = bool(e.get("mode_rate_action", False))
            if self._mode_rate_action and self._mode_slew_step is None:
                raise ValueError("mode_rate_action requires mode_slew_per_s > 0 (the rate scale)")
            # Commanded null is structurally zero; the residual ACTUAL null (servo-lag
            # transients, DR offsets) is not usefully controllable — don't reward-chase it.
            self.w_null = 0.0
        else:
            self._mixer = None
        self._mode_prev_servo = np.zeros(4)
        self._mode_prev_modes = np.zeros(MODE_DIM)
        # Adaptive constraint multipliers (name -> lambda, default 1.0), updated from outside
        # by train.py's LagrangeCallback via venv.set_attr("lagrange", ...).
        self.lagrange = {}
        act_dim = MODE_DIM if self._mixer is not None else ACT_DIM
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self._viewer = None
        # Optional target-pose marker (mocap body in the MJCF); -1 if the model has none.
        try:
            self._mocap_id = int(self.sim.model.body_mocapid[self.sim.model.body("target_marker").id])
        except (KeyError, ValueError):
            self._mocap_id = -1

        self.target_pos = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.track_position = self.task == "pose"  # full 3-D position (vertical included)
        self.track_depth = self.task == "attitude_depth"  # depth is a separate objective only here
        self.track_velocity = self.task == "attitude_velocity"  # feedforward velocity command
        self.v_cmd = np.zeros(3)        # commanded velocity in the TARGET-BODY frame (obs; yaw-independent)
        self.v_cmd_world = np.zeros(3)  # same command rotated to world (for the reward/direction)
        self.prev_action = np.zeros(ACT_DIM)
        self.step_count = 0

    # -- target sampling -------------------------------------------------------
    def _sample_target_quat(self, tilt_deg=None):
        """Random target orientation: yaw about +Y plus a bounded tilt about a random horizontal axis."""
        tilt_deg = self.tilt_target_deg if tilt_deg is None else tilt_deg
        yaw = np.radians(self.np_random.uniform(-self.yaw_target_deg, self.yaw_target_deg))
        tilt = np.radians(self.np_random.uniform(0.0, tilt_deg))
        phi = self.np_random.uniform(0.0, 2.0 * np.pi)
        tilt_axis = np.array([np.cos(phi), 0.0, np.sin(phi)])
        q_yaw, q_tilt, q = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_yaw, _Y_UP, yaw)
        mujoco.mju_axisAngle2Quat(q_tilt, tilt_axis, tilt)
        mujoco.mju_mulQuat(q, q_yaw, q_tilt)
        return q

    # -- domain randomization --------------------------------------------------
    def _apply_domain_rand(self):
        b = self._base
        if not self.dr.get("enabled", False):
            self.sim.volume = b["volume"]
            self.sim.thrust_per_cmd = b["thrust_per_cmd"]
            self.sim.drag_lin = b["drag_lin"].copy()
            self.sim.drag_quad = b["drag_quad"].copy()
            self.sim.added_mass_diag = b["added_mass"].copy()
            self.sim.servo_slew_rad = b["servo_slew"]
            self.sim.servo_tau = b["servo_tau"]
            self.sim.thrust_curve_exp = b["thrust_exp"]
            self.sim.set_buoyancy_offset(b["buoy_offset"])
            self.sim.max_duty = b["max_duty"]
            self._servo_offset = np.zeros(4)
            self._thrust_gain = np.ones(4)
            return
        u = self.np_random.uniform
        self.sim.volume = b["volume"] * (1.0 + u(-1, 1) * self.dr["buoyancy_frac"])
        self.sim.thrust_per_cmd = b["thrust_per_cmd"] * (1.0 + u(-1, 1) * self.dr["thrust_frac"])
        self.sim.drag_lin = b["drag_lin"] * (1.0 + u(-1, 1) * self.dr["drag_frac"])
        self.sim.drag_quad = b["drag_quad"] * (1.0 + u(-1, 1) * self.dr["drag_frac"])
        # Added mass is an estimate with a wide error bar (tools/estimate_hydro): randomize it.
        am_frac = self.dr.get("added_mass_frac", 0.0)
        if am_frac > 0.0:
            self.sim.added_mass_diag = b["added_mass"] * (1.0 + u(-1, 1, size=6) * am_frac)
        # The real servo rate is only estimated (datasheet 300-350 deg/s no-load, derated in
        # water/under load) and the bag analysis showed the policy behaves well over 100-500 deg/s
        # WHEN it has learned not to lean on one specific rate (issue #2, proposal C): sample the
        # slew limit per episode so the policy works for whatever the true rate is.
        slew_range = self.dr.get("servo_slew_range_deg_s")
        if slew_range:
            self.sim.servo_slew_rad = np.radians(u(float(slew_range[0]), float(slew_range[1])))
        tau_frac = self.dr.get("servo_tau_frac", 0.0)
        if tau_frac > 0.0 and b["servo_tau"] > 0.0:
            self.sim.servo_tau = b["servo_tau"] * (1.0 + u(-1, 1) * tau_frac)
        # Thrust-curve exponent: the low-duty thrust shape is only bag-fitted (exp vs max thrust
        # confounded below |u| = 0.2), so randomize the exponent over its plausible band.
        exp_range = self.dr.get("thrust_exp_range")
        if exp_range:
            self.sim.thrust_curve_exp = u(float(exp_range[0]), float(exp_range[1]))
        # CoB height: bag-fitted to ~5-10 mm (was 50); randomize generously until the static-tilt
        # calibration pins it.
        bo_frac = self.dr.get("buoyancy_offset_frac", 0.0)
        if bo_frac > 0.0:
            self.sim.set_buoyancy_offset(b["buoy_offset"] * (1.0 + u(-1, 1) * bo_frac))
        # ESC cap: sample the deploy clamp per episode (the operator raises max_duty in the field
        # from 0.25 toward 0.4 as trust builds — Umiusi_sim#3) so one policy stays valid at any cap.
        # Pair with env.observe_max_duty so the policy can actually EXPLOIT a higher cap.
        md_range = self.dr.get("max_duty_range")
        self.sim.max_duty = (u(float(md_range[0]), float(md_range[1])) if md_range
                             else b["max_duty"])
        # Per-thruster imperfections: servo neutral offset [deg] and thrust-gain asymmetry.
        off = self.dr.get("servo_offset_deg", 0.0)
        self._servo_offset = np.radians(u(-off, off, size=4)) if off > 0.0 else np.zeros(4)
        tuf = self.dr.get("thrust_unit_frac", 0.0)
        self._thrust_gain = 1.0 + u(-1, 1, size=4) * tuf if tuf > 0.0 else np.ones(4)

    def _place_marker(self, state):
        """Move the visual target marker to show the commanded pose (rendering only)."""
        if self._mocap_id < 0:
            return
        if self.task == "pose":
            self.sim.data.mocap_pos[self._mocap_id] = self.target_pos
            self.sim.data.mocap_quat[self._mocap_id] = (1.0, 0.0, 0.0, 0.0)
        else:  # attitude tasks: park the marker beside the vehicle, oriented to the target
            self.sim.data.mocap_pos[self._mocap_id] = state["pos"] + np.array([0.7, 0.0, 0.0])
            self.sim.data.mocap_quat[self._mocap_id] = self.target_quat

    # -- disturbances (water current + random impulses) ------------------------
    def _sample_current(self):
        """Per-episode water current; also clears any impulse. No-op if disturbance is disabled."""
        self._impulse_left = 0
        self.sim.ext_force_world = np.zeros(3)
        if not self.dist.get("enabled", False):
            self.sim.current_world = np.zeros(3)
            return
        d = self.np_random.normal(size=3)
        if self.dist.get("current_horizontal", True):
            d[1] = 0.0
        d = d / (np.linalg.norm(d) + 1e-9)
        self.sim.current_world = d * self.np_random.uniform(0.0, self.dist.get("current_max", 0.0))

    def _apply_disturbance(self):
        """Occasionally fire a short world-frame force impulse (waves/bumps)."""
        if not self.dist.get("enabled", False):
            return
        if self._impulse_left > 0:
            self._impulse_left -= 1
            if self._impulse_left == 0:
                self.sim.ext_force_world = np.zeros(3)
        elif self.np_random.uniform() < self.dist.get("impulse_prob", 0.0):
            d = self.np_random.normal(size=3)
            self.sim.ext_force_world = d / (np.linalg.norm(d) + 1e-9) * self.dist.get("impulse_force", 0.0)
            self._impulse_left = int(self.dist.get("impulse_steps", 5))

    # -- errors / observation --------------------------------------------------
    def _errors(self, state):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, state["quat"])
        R = R.reshape(3, 3)
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, self.target_quat, state["quat"])  # rot-vec: current -> target
        return R, ori_err

    def _get_obs(self, state, R, ori_err):
        P = self._obs_P                  # sim body frame -> obs frame (identity for "sim")
        w_body = P @ (R.T @ state["ang_vel"])  # gyro
        ori_err = P @ ori_err
        servo_n = state["servo"] / self.sim.servo_range_rad
        thrust_n = state["thrust"] / max(self.sim.thrust_per_cmd, 1e-9)

        if self.obs_mode == "full":
            pos_err_body = P @ (R.T @ (self.target_pos - state["pos"]))
            extero = [pos_err_body, ori_err, P @ (R.T @ state["lin_vel"]), w_body]
        elif self.obs_mode == "imu":
            extero = [ori_err, w_body]
        elif self.obs_mode == "imu_depth":
            extero = [ori_err, w_body, np.array([self.target_pos[1] - state["pos"][1]])]
        else:  # imu_depth_dvl: adds body-frame velocity (DVL) for drift rejection
            extero = [ori_err, w_body, np.array([self.target_pos[1] - state["pos"][1]]),
                      P @ (R.T @ state["lin_vel"])]

        if self.track_velocity:  # feedforward velocity command only (no measured velocity / DVL)
            extero = extero + [P @ self.v_cmd]
        if self.proprio_mode == "full":
            proprio = [servo_n, thrust_n, self.prev_action]
        else:  # "action": only signals that exist identically on the real vehicle
            proprio = [self.prev_action]
        if self.observe_max_duty:  # appended LAST: existing layouts stay a prefix (warm-start friendly)
            proprio = proprio + [np.array([self.sim.max_duty])]
        obs = np.concatenate(extero + proprio)
        if self.dr.get("enabled", False) and self.dr.get("obs_noise", 0.0) > 0.0:
            obs = obs + self.np_random.normal(0.0, self.dr["obs_noise"], size=obs.shape)
        return obs.astype(np.float32)

    def apply_train_ctx(self, **kwargs):
        """Set training-context attributes from a VecEnv callback.

        MUST be called via ``venv.env_method("apply_train_ctx", ...)`` — attribute LOOKUP
        forwards through gymnasium wrappers (Monitor), so the method binds to this env.
        Plain ``venv.set_attr`` does NOT forward: it sets the attribute on the Monitor
        wrapper while the env keeps its own — the silent bug that disabled every curriculum
        (econ_ramp, cap range, cone/yaw/tilt) and the Lagrange multipliers up to av_mode9
        (discovered 2026-08-27: mode9 trained bit-identically to mode8).
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"apply_train_ctx: env has no attribute {k!r}")
            setattr(self, k, v)

    def _train_ctx_snapshot(self):
        """Read back what apply_train_ctx set (tests / diagnostics; works across VecEnv workers)."""
        return {"econ_ramp": self.econ_ramp, "lagrange": dict(self.lagrange),
                "max_duty_range": self.dr.get("max_duty_range"),
                "vel_cmd_cone_deg": self.vel_cmd_cone_deg}

    # -- lifecycle -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_domain_rand()

        start = self.np_random.uniform(-self.start_jitter, self.start_jitter, size=3)
        if self.task == "pose":
            self.target_pos = self.np_random.uniform(-self.target_box, self.target_box)
            self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # upright
        else:
            self.target_quat = self._sample_target_quat(self.tilt_target_deg)
            depth = self.np_random.uniform(-self.depth_target_range, self.depth_target_range)
            self.target_pos = np.array([start[0], depth if self.track_depth else start[1], start[2]])
        if self.track_velocity:  # body-frame velocity command within +/- cone of body +X (yaw-independent)
            ang = np.radians(self.np_random.uniform(-self.vel_cmd_cone_deg, self.vel_cmd_cone_deg))
            if self.vel_cmd_horizontal:
                dhat = np.array([np.cos(ang), 0.0, np.sin(ang)])  # horizontal cruise (x-z plane)
            else:  # 3-D command: also tilt in elevation — needed to reach balloons at different depths.
                # Elevation is sampled SPHERE-UNIFORM within +/- vel_cmd_elev_deg (sin(elev) uniform
                # in +/- sin(limit)), not uniform in angle: uniform-in-angle piles probability onto
                # the poles (straight up/down) and the policy loses horizontal cruise.
                if (self.vel_cmd_horizontal_prob > 0.0
                        and self.np_random.uniform() < self.vel_cmd_horizontal_prob):
                    elev = 0.0   # horizontal-episode floor (multimodality guard, see __init__)
                else:
                    s_lim = np.sin(np.radians(np.clip(self.vel_cmd_elev_deg, 0.0, 90.0)))
                    elev = np.arcsin(self.np_random.uniform(-s_lim, s_lim))
                dhat = np.array([np.cos(elev) * np.cos(ang), np.sin(elev), np.cos(elev) * np.sin(ang)])
            speed_hi = self.vel_cmd_max
            if self.vel_cmd_cap_frac > 0.0:  # _apply_domain_rand already set this episode's cap
                speed_hi = min(speed_hi, self.vel_cmd_cap_frac * VEL_PER_CAP * self.sim.max_duty)
            if self.vel_cmd_hi_prob > 0.0 and self.np_random.uniform() < self.vel_cmd_hi_prob:
                speed_cmd = self.np_random.uniform(0.9, 1.0) * speed_hi
            else:
                speed_cmd = self.np_random.uniform(0.0, speed_hi)
            if self.vel_cmd_zero_prob > 0.0 and self.np_random.uniform() < self.vel_cmd_zero_prob:
                speed_cmd = 0.0  # explicit hold-station episodes (dry tests / narrow pools)
            self.v_cmd = dhat * speed_cmd
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, self.target_quat)
            self.v_cmd_world = Rt.reshape(3, 3) @ self.v_cmd  # command expressed in world for the reward

        state = self.sim.reset(pos=tuple(start), quat=(1.0, 0.0, 0.0, 0.0))
        self._sample_current()
        # sim2real action latency: only when domain_rand is enabled (a control->actuation delay).
        self._act_latency = self.action_latency if self.dr.get("enabled", False) else 0
        self._act_buf = [np.zeros(ACT_DIM) for _ in range(self._act_latency)]
        self.prev_action = np.zeros(ACT_DIM)
        self._mode_prev_servo = np.zeros(4)
        self._mode_prev_modes = np.zeros(MODE_DIM)
        self.prev_servo = state["servo"].copy()  # from the post-reset state (avoid a step-1 servo-rate spike)
        self.step_count = 0
        self._place_marker(state)
        R, ori_err = self._errors(state)
        return self._get_obs(state, R, ori_err), self._info(state, 0.0, float(np.linalg.norm(ori_err)), 0.0, False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        mode_rate_mag = 0.0
        if self._mixer is not None:  # wrench modes -> 8-D actuator command; downstream unchanged
            raw = action                                   # already clipped to [-1, 1] above
            if self._mode_rate_action:  # action = mode RATE: integrate (slew limit is inherent)
                mode_rate_mag = float(np.mean(np.abs(raw)))
                action = np.clip(self._mode_prev_modes + raw * self._mode_slew_step, -1.0, 1.0)
                self._mode_prev_modes = action.copy()
            elif self._mode_slew_step is not None:  # target form: rate-limit toward the target
                action = self._mode_prev_modes + np.clip(
                    raw - self._mode_prev_modes, -self._mode_slew_step, self._mode_slew_step)
                self._mode_prev_modes = action.copy()
            else:
                action = raw
            action = self._mixer.mix(action, self.sim.max_duty, self._mode_prev_servo)
            self._mode_prev_servo = action[:4].copy()
        if self._act_latency > 0:  # sim2real: apply a delayed command (control->actuation lag)
            self._act_buf.append(action)
            action = self._act_buf.pop(0)
        self._apply_disturbance()
        # Per-thruster DR imperfections (identity when DR is off): the POLICY's action is what is
        # observed/penalized; the PLANT receives the perturbed version.
        plant_action = action
        if self._servo_offset.any() or not np.all(self._thrust_gain == 1.0):
            plant_action = action.copy()
            plant_action[:4] = np.clip(action[:4] + self._servo_offset / self.sim.servo_range_rad, -1.0, 1.0)
            plant_action[4:] = np.clip(action[4:] * self._thrust_gain, -1.0, 1.0)
        state = self.sim.step(plant_action)
        self.step_count += 1
        R, ori_err_vec = self._errors(state)

        pos_err = float(np.linalg.norm(self.target_pos - state["pos"]))
        ori_err = float(np.linalg.norm(ori_err_vec))
        depth_err = float(self.target_pos[1] - state["pos"][1])
        speed = float(np.linalg.norm(state["lin_vel"]))
        # Velocity command is DIRECTION-only controllable without a velocity sensor: split the actual
        # velocity into the along-command speed and the perpendicular (sideways) drift.
        v_along = v_perp = vcn = vel_over = vel_track = 0.0
        if self.track_velocity:
            v = state["lin_vel"]
            vcn = float(np.linalg.norm(self.v_cmd_world))  # world-frame command (norm == body command)
            dhat = self.v_cmd_world / vcn if vcn > 1e-6 else np.zeros(3)
            v_along = float(v @ dhat)
            v_perp = float(np.linalg.norm(v - v_along * dhat)) if vcn > 1e-6 else speed
        vel_err = v_perp  # only the perpendicular drift is a controllable error (magnitude isn't observable)
        ang_speed = float(np.linalg.norm(state["ang_vel"]))
        # Effort: legacy = L2 norm of the esc command; effort_exp > 0 = sum(|u|^exp), the POWER
        # dimension (exp 3) so the penalty tracks the real cost (heat / battery), not duty count.
        if self.effort_exp > 0.0:
            effort = float(np.sum(np.abs(action[4:8]) ** self.effort_exp))
        else:
            effort = float(np.linalg.norm(action[4:8]))          # thrust magnitude (legacy)
        # Vertical thrust mode decomposition (BODY frame — the allocation geometry lives there):
        # null_n = null-mode amplitude in units of the per-thruster cap force, null_frac = the null
        # share of vertical mode power (the 8/25 run: 41.2 %), roll_use = roll-mode amplitude over
        # its cap-limited maximum (8/25: 19 % of authority used).
        null_n = null_frac = roll_use = vert_power = 0.0
        if self._vert_modes is not None:
            v_vert = state["thrust_world"] @ R[:, 1]             # body-frame vertical force per unit [N]
            m_h, m_r, m_p, m_n = self._vert_modes @ v_vert
            f_cap = self.sim.max_duty ** self.sim.thrust_curve_exp * self.sim.thrust_per_cmd
            null_n = abs(m_n) / max(f_cap, 1e-9)
            mode_power = m_h * m_h + m_r * m_r + m_p * m_p + m_n * m_n
            null_frac = float(m_n * m_n / mode_power) if mode_power > 1e-12 else 0.0
            roll_use = abs(m_r) / max(2.0 * f_cap, 1e-9)
            vert_power = float(mode_power)   # weight for the POWER-WEIGHTED null share (accept metric)
        action_rate = float(np.linalg.norm(action - self.prev_action))
        servo_rate = float(np.linalg.norm(state["servo"] - self.prev_servo))       # actual servo motion
        thrust_rate = float(np.linalg.norm(action[4:8] - self.prev_action[4:8]))   # thrust command change
        # Gap between the commanded and the actual servo angle: nonzero while the servo is still
        # travelling; a persistently large gap means the policy demands angles the servo can never
        # reach (the +/-90 flapping pathology of issue #2 — 100 % duty, 0.1 % of time on target).
        cmd_gap = float(np.linalg.norm(action[:4] * self.sim.servo_range_rad - state["servo"]))
        rw = self.rw

        # Penalty terms give a gradient everywhere; dense exp() "closeness" bonuses add a smooth
        # pull toward the goal (reduces reliance on the sparse goal_bonus, lowers variance).
        def prox(err, scale):
            return float(np.exp(-((err / scale) ** 2)))

        # Economy terms ride econ_ramp (0..1 curriculum): task first, then economize.
        reward = -self.econ_ramp * rw["w_effort"] * effort - rw["w_action_rate"] * action_rate
        if self.w_null > 0.0:
            reward -= self.econ_ramp * self.w_null * null_n
        reward -= rw.get("w_servo_rate", 0.0) * servo_rate      # penalize servo chatter (smooth steering)
        reward -= rw.get("w_thrust_rate", 0.0) * thrust_rate    # penalize thrust command changes
        reward -= rw.get("w_cmd_gap", 0.0) * cmd_gap            # penalize unreachable servo commands
        # mode_rate_action: penalize rate USE (0 = hold). Rides econ_ramp — from step 0 it is
        # exactly the "strong effort penalty -> do-nothing local optimum" trap (see w_effort note).
        reward -= self.econ_ramp * rw.get("w_mode_rate", 0.0) * mode_rate_mag
        # Near the target attitude, damp actuation. HOLD task: press both servo AND thrust to a stop
        # (kills limit cycles). CRUISE task: once the drift is small too (steadily on-course), damp the
        # SERVO (steering) chatter — but NOT the thrust, which may still modulate to hold speed. This is
        # the steady-cruise servo-settle that cuts the residual servo vibration.
        if ori_err < self.near_goal_ori:
            if not self.track_velocity:
                reward -= rw.get("w_settle_servo", 0.0) * servo_rate
                reward -= rw.get("w_settle_thrust", 0.0) * thrust_rate
            elif v_perp < self.vel_tol:
                reward -= rw.get("w_settle_servo_cruise", 0.0) * servo_rate
        # Deadband: no orientation reward gradient once inside ori_deadband, so the policy has no
        # incentive to chatter for sub-deadband precision — it can settle and hold still.
        ori_eff = max(0.0, ori_err - self.ori_deadband)
        # self.lagrange: adaptive constraint multipliers (LagrangeCallback in train.py updates
        # them toward explicit targets, e.g. "mean ori_err <= 0.2 rad" — replaces hand-tuning
        # the attitude-vs-cruise balance). Default 1.0 = the static config weights.
        reward -= self.lagrange.get("ori", 1.0) * rw["w_ori"] * ori_eff
        reward += rw.get("w_ori_bonus", 0.0) * prox(ori_eff, rw.get("ori_scale", 0.35))
        if self.track_position:
            reward -= rw["w_pos"] * pos_err
            reward += rw.get("w_pos_bonus", 0.0) * prox(pos_err, rw.get("pos_scale", 0.5))
            if pos_err < self.near_goal_dist:
                reward -= rw["w_vel"] * speed
        else:
            reward -= rw.get("w_angvel", 0.05) * ang_speed  # damp rotation to hold attitude
        if self.track_depth:
            reward -= rw.get("w_depth", 1.0) * abs(depth_err)
            reward += rw.get("w_depth_bonus", 0.0) * prox(abs(depth_err), rw.get("depth_scale", 0.25))
        if self.track_velocity:  # aim propulsion in the commanded DIRECTION (speed is not observable)
            # w_vel_dir_ratio > 0 switches the cruise reward to TRACKING-RATIO units
            # (full reward at v_along == commanded, independent of |v_cmd|): with cap-aware
            # commands the absolute form w_vel_dir * v scales down with the cap and drowns in
            # the attitude terms — av_mode2 rationally abandoned cruising for that reason.
            # The 0.05 m/s floor keeps near-zero commands from amplifying velocity noise.
            wvr = rw.get("w_vel_dir_ratio", 0.0)
            if wvr > 0.0:
                # TRACK the commanded speed, don't just reach it (av_mode5 cruised at 1.6x the
                # command — the capped form min(v, vcn) gives overshoot for free, but FF
                # deployment needs the commanded speed). The overshoot penalty must stay GENTLE:
                # the av_mode6 1:1 ratio form blew up at small vcn (denominator 0.05 -> a 5 cm/s
                # excess on a 6 cm/s command cost 3.3/step) and its gradient noise wrecked the
                # shared trunk — attitude collapsed to 0.44-0.75 rad. Deadband 2 cm/s, slope
                # 0.5, denominator floored at 0.10 m/s.
                # SYMMETRIC tracking penalty (av_mode10 lesson: penalizing only overshoot left a
                # monotonic but weak transfer curve, gain 0.5-0.6 — undershoot was free). |v - vcn|
                # past a 2 cm/s deadband, clipped at 1.0 (bounded gradient), "track" multiplier
                # adapted by the Lagrange callback toward mean <= track_target.
                den = max(vcn, 0.10)
                vel_track = min(1.0, max(0.0, abs(v_along - vcn) - 0.02) / den)
                reward += wvr * (min(max(v_along, 0.0), vcn) / den
                                 - self.lagrange.get("track", 1.0) * vel_track)
                vel_over = min(1.0, max(0.0, v_along - vcn - 0.02) / den)  # diagnostics only
            else:
                reward += rw.get("w_vel_dir", 8.0) * min(max(v_along, 0.0), vcn)  # move toward goal, cap at desired
            # Deadband: no drift penalty below vel_deadband, so the policy has no incentive to chatter the
            # servos chasing sub-deadband drift precision (a root cause of the steady-state servo vibration).
            reward -= rw.get("w_vel_perp", 4.0) * max(0.0, v_perp - self.vel_deadband)  # no sideways drift

        success = ori_err < self.ori_tol
        if self.track_position:
            success = success and pos_err < self.pos_tol
        if self.track_depth:
            success = success and abs(depth_err) < self.depth_tol
        if self.track_velocity:
            if vcn > 1e-6:  # moving the right way, up to speed, without sideways drift
                success = success and v_along > 0.7 * vcn and v_perp < self.vel_tol
            else:
                success = success and speed < self.vel_tol  # zero command -> hold still
        if success:
            reward += rw["goal_bonus"]

        # Out-of-bounds only ends go-to-pose episodes; attitude tasks let the vehicle drift.
        out_of_bounds = self.track_position and bool(np.any(np.abs(state["pos"]) > self.workspace_bounds))
        if out_of_bounds:
            reward -= rw["out_of_bounds_penalty"]
        terminated = out_of_bounds
        truncated = self.step_count >= self.horizon

        self.prev_action = action.copy()
        self.prev_servo = state["servo"].copy()
        self._place_marker(state)
        info = self._info(state, pos_err, ori_err, depth_err, success)
        info["out_of_bounds"] = out_of_bounds
        info["vel_err"] = vel_err                          # perpendicular drift (attitude_velocity)
        info["vel_along"] = v_along                         # speed along the commanded direction [m/s]
        info["vel_cmd_speed"] = vcn                         # commanded (desired) speed [m/s]
        info["ang_speed"] = ang_speed                       # ||angular velocity|| [rad/s] (wobble diagnostic)
        info["servo"] = state["servo"].copy()             # actual servo angles (motion diagnostics)
        info["esc_cmd"] = action[4:8].copy()               # raw policy command (thrust use)
        info["esc_applied"] = self.sim.esc_current.copy()  # slew-limited applied esc (true thrust change)
        info["cmd_gap"] = cmd_gap                          # ||servo command - actual|| [rad] (reachability)
        info["null_frac"] = null_frac                      # null share of vertical mode power (accept: <= 5 %)
        info["vert_power"] = vert_power                    # total vertical mode power [N^2] (null weighting)
        info["roll_use"] = roll_use                        # roll-mode amplitude / cap max (accept: >= 50 %)
        info["max_duty"] = self.sim.max_duty               # plant esc cap this episode (DR-sampled)
        info["mode_rate_mag"] = mode_rate_mag              # mean |mode rate action| (limiter-riding diagnostic)
        info["vel_over"] = vel_over                        # clipped overshoot ratio (diagnostics)
        info["vel_track"] = vel_track                      # clipped |v-vcn| ratio (Lagrange constraint signal)
        info["step_idx"] = self.step_count                 # for steady-state-only constraint accounting
        return self._get_obs(state, R, ori_err_vec), reward, terminated, truncated, info

    def _info(self, state, pos_err, ori_err, depth_err, success):
        return {"pos_err": pos_err, "ori_err": ori_err, "depth_err": depth_err,
                "is_success": success, "target_pos": self.target_pos.copy()}

    # -- rendering -------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            # Route through the shared viewer so eval --render matches every other tool: the
            # fixed "track" camera (upright, follows the vehicle). Lazy import keeps headless
            # eval (no --render) free of any GUI dependency.
            from umiusi_sim.viewer import UmiusiViewer

            self._viewer = UmiusiViewer(
                self.sim.model, self.sim.data, base_id=self.sim.base_id, cam="track",
                control_rate_hz=self.sim.cfg["sim"]["control_rate_hz"],
            ).launch()
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
