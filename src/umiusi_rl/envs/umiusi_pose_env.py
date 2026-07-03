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
    task = "attitude_velocity"  hold an upright + random-yaw orientation (feedback) AND cruise at a
                            random commanded horizontal velocity (feedforward: obs adds the 3-D
                            velocity command, NOT measured velocity — the deterministic sim makes the
                            v_cmd->thrust map static, so no DVL is needed). obs_mode "imu".

Observation = exteroceptive (sensor-suite dependent) ++ proprioception (ALWAYS):
    proprioception: servo/range (4) + thrust/thrust_per_cmd (4) + previous action (8) = 16

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

_ROOT = Path(__file__).resolve().parents[3]        # repo root (src/umiusi_rl/envs/..)

ACT_DIM = 8
PROPRIO_DIM = 16  # servo(4) + thrust(4) + prev_action(8), always present
# Exteroceptive (navigation-sensor) dimensions per observation mode.
_EXTERO_DIM = {"full": 12, "imu": 6, "imu_depth": 7, "imu_depth_dvl": 10}
# Default sensor suite per task (overridable with obs_mode / --obs-mode).
_DEFAULT_OBS = {"pose": "full", "attitude": "imu", "attitude_depth": "imu_depth",
                "attitude_velocity": "imu"}
_Y_UP = np.array([0.0, 1.0, 0.0])


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
        self.vel_cmd_horizontal = bool(e.get("vel_cmd_horizontal", True))  # sample v_cmd in the x-z plane
        self.vel_cmd_cone_deg = float(e.get("vel_cmd_cone_deg", 180.0))  # v_cmd dir within +/- this of +X (curriculum)
        self.vel_tol = float(e.get("vel_tol", 0.10))         # velocity match tolerance [m/s]
        self.pos_tol = float(e["pos_tol"])
        self.ori_tol = float(e["ori_tol"])
        self.depth_tol = float(e.get("depth_tol", 0.10))
        self.near_goal_dist = float(e["near_goal_dist"])
        self.near_goal_ori = float(e.get("near_goal_ori", 0.20))  # rad: within this, press to settle
        self.ori_deadband = float(e.get("ori_deadband", 0.0))     # rad: no ori reward gradient inside this
        self.rw = cfg["reward"]
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
        }

        # attitude_velocity adds the feedforward velocity command (3), no measured velocity (no DVL).
        obs_dim = _EXTERO_DIM[self.obs_mode] + PROPRIO_DIM + (3 if self.task == "attitude_velocity" else 0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
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
            return
        u = self.np_random.uniform
        self.sim.volume = b["volume"] * (1.0 + u(-1, 1) * self.dr["buoyancy_frac"])
        self.sim.thrust_per_cmd = b["thrust_per_cmd"] * (1.0 + u(-1, 1) * self.dr["thrust_frac"])
        self.sim.drag_lin = b["drag_lin"] * (1.0 + u(-1, 1) * self.dr["drag_frac"])
        self.sim.drag_quad = b["drag_quad"] * (1.0 + u(-1, 1) * self.dr["drag_frac"])

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
        w_body = R.T @ state["ang_vel"]  # gyro
        servo_n = state["servo"] / self.sim.servo_range_rad
        thrust_n = state["thrust"] / max(self.sim.thrust_per_cmd, 1e-9)

        if self.obs_mode == "full":
            pos_err_body = R.T @ (self.target_pos - state["pos"])
            extero = [pos_err_body, ori_err, R.T @ state["lin_vel"], w_body]
        elif self.obs_mode == "imu":
            extero = [ori_err, w_body]
        elif self.obs_mode == "imu_depth":
            extero = [ori_err, w_body, np.array([self.target_pos[1] - state["pos"][1]])]
        else:  # imu_depth_dvl: adds body-frame velocity (DVL) for drift rejection
            extero = [ori_err, w_body, np.array([self.target_pos[1] - state["pos"][1]]), R.T @ state["lin_vel"]]

        if self.track_velocity:  # feedforward velocity command only (no measured velocity / DVL)
            extero = extero + [self.v_cmd]
        obs = np.concatenate(extero + [servo_n, thrust_n, self.prev_action])
        if self.dr.get("enabled", False) and self.dr.get("obs_noise", 0.0) > 0.0:
            obs = obs + self.np_random.normal(0.0, self.dr["obs_noise"], size=obs.shape)
        return obs.astype(np.float32)

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
        if self.track_velocity:  # body-frame horizontal command within +/- cone of body +X (yaw-independent)
            ang = np.radians(self.np_random.uniform(-self.vel_cmd_cone_deg, self.vel_cmd_cone_deg))
            dhat = np.array([np.cos(ang), 0.0, np.sin(ang)])
            self.v_cmd = dhat * self.np_random.uniform(0.0, self.vel_cmd_max)
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, self.target_quat)
            self.v_cmd_world = Rt.reshape(3, 3) @ self.v_cmd  # command expressed in world for the reward

        state = self.sim.reset(pos=tuple(start), quat=(1.0, 0.0, 0.0, 0.0))
        self._sample_current()
        # sim2real action latency: only when domain_rand is enabled (a control->actuation delay).
        self._act_latency = self.action_latency if self.dr.get("enabled", False) else 0
        self._act_buf = [np.zeros(ACT_DIM) for _ in range(self._act_latency)]
        self.prev_action = np.zeros(ACT_DIM)
        self.prev_servo = np.zeros(4)
        self.step_count = 0
        self._place_marker(state)
        R, ori_err = self._errors(state)
        return self._get_obs(state, R, ori_err), self._info(state, 0.0, float(np.linalg.norm(ori_err)), 0.0, False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self._act_latency > 0:  # sim2real: apply a delayed command (control->actuation lag)
            self._act_buf.append(action)
            action = self._act_buf.pop(0)
        self._apply_disturbance()
        state = self.sim.step(action)
        self.step_count += 1
        R, ori_err_vec = self._errors(state)

        pos_err = float(np.linalg.norm(self.target_pos - state["pos"]))
        ori_err = float(np.linalg.norm(ori_err_vec))
        depth_err = float(self.target_pos[1] - state["pos"][1])
        speed = float(np.linalg.norm(state["lin_vel"]))
        # Velocity command is DIRECTION-only controllable without a velocity sensor: split the actual
        # velocity into the along-command speed and the perpendicular (sideways) drift.
        v_along = v_perp = vcn = 0.0
        if self.track_velocity:
            v = state["lin_vel"]
            vcn = float(np.linalg.norm(self.v_cmd_world))  # world-frame command (norm == body command)
            dhat = self.v_cmd_world / vcn if vcn > 1e-6 else np.zeros(3)
            v_along = float(v @ dhat)
            v_perp = float(np.linalg.norm(v - v_along * dhat)) if vcn > 1e-6 else speed
        vel_err = v_perp  # only the perpendicular drift is a controllable error (magnitude isn't observable)
        ang_speed = float(np.linalg.norm(state["ang_vel"]))
        effort = float(np.linalg.norm(action[4:8]))              # thrust magnitude
        action_rate = float(np.linalg.norm(action - self.prev_action))
        servo_rate = float(np.linalg.norm(state["servo"] - self.prev_servo))       # actual servo motion
        thrust_rate = float(np.linalg.norm(action[4:8] - self.prev_action[4:8]))   # thrust command change
        rw = self.rw

        # Penalty terms give a gradient everywhere; dense exp() "closeness" bonuses add a smooth
        # pull toward the goal (reduces reliance on the sparse goal_bonus, lowers variance).
        def prox(err, scale):
            return float(np.exp(-((err / scale) ** 2)))

        reward = -rw["w_effort"] * effort - rw["w_action_rate"] * action_rate
        reward -= rw.get("w_servo_rate", 0.0) * servo_rate      # penalize servo chatter (smooth steering)
        reward -= rw.get("w_thrust_rate", 0.0) * thrust_rate    # penalize thrust command changes
        # Near the target orientation, press hard to stop moving (kills limit cycles). NOT for the
        # velocity task, where the vehicle must keep modulating thrust to cruise.
        if ori_err < self.near_goal_ori and not self.track_velocity:
            reward -= rw.get("w_settle_servo", 0.0) * servo_rate
            reward -= rw.get("w_settle_thrust", 0.0) * thrust_rate
        # Deadband: no orientation reward gradient once inside ori_deadband, so the policy has no
        # incentive to chatter for sub-deadband precision — it can settle and hold still.
        ori_eff = max(0.0, ori_err - self.ori_deadband)
        reward -= rw["w_ori"] * ori_eff
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
            reward += rw.get("w_vel_dir", 8.0) * min(max(v_along, 0.0), vcn)  # move toward goal, cap at desired
            reward -= rw.get("w_vel_perp", 4.0) * v_perp                       # no sideways drift

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
        return self._get_obs(state, R, ori_err_vec), reward, terminated, truncated, info

    def _info(self, state, pos_err, ori_err, depth_err, success):
        return {"pos_err": pos_err, "ori_err": ori_err, "depth_err": depth_err,
                "is_success": success, "target_pos": self.target_pos.copy()}

    # -- rendering -------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            # "track" follows the vehicle so it stays in frame even as it drifts.
            self._viewer.cam.fixedcamid = self.sim.model.camera("track").id
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
