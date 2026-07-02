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
drift via velocity). See ai/project_spec.yaml (rl section) and ai/architecture.md (section 7).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from sim.simulator import UmiusiSimulator

_ROOT = Path(__file__).resolve().parents[2]

ACT_DIM = 8
PROPRIO_DIM = 16  # servo(4) + thrust(4) + prev_action(8), always present
# Exteroceptive (navigation-sensor) dimensions per observation mode.
_EXTERO_DIM = {"full": 12, "imu": 6, "imu_depth": 7, "imu_depth_dvl": 10}
# Default sensor suite per task (overridable with obs_mode / --obs-mode).
_DEFAULT_OBS = {"pose": "full", "attitude": "imu", "attitude_depth": "imu_depth"}
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
        self.pos_tol = float(e["pos_tol"])
        self.ori_tol = float(e["ori_tol"])
        self.depth_tol = float(e.get("depth_tol", 0.10))
        self.near_goal_dist = float(e["near_goal_dist"])
        self.rw = cfg["reward"]
        self.dr = cfg.get("domain_rand", {"enabled": False})

        self._base = {
            "volume": self.sim.volume,
            "thrust_per_cmd": self.sim.thrust_per_cmd,
            "drag_lin": self.sim.drag_lin.copy(),
            "drag_quad": self.sim.drag_quad.copy(),
        }

        obs_dim = _EXTERO_DIM[self.obs_mode] + PROPRIO_DIM
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
        self.prev_action = np.zeros(ACT_DIM)
        self.step_count = 0

    # -- target sampling -------------------------------------------------------
    def _sample_target_quat(self):
        """Random target orientation: yaw about +Y plus a bounded tilt about a random horizontal axis."""
        yaw = np.radians(self.np_random.uniform(-self.yaw_target_deg, self.yaw_target_deg))
        tilt = np.radians(self.np_random.uniform(0.0, self.tilt_target_deg))
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
            self.target_quat = self._sample_target_quat()
            depth = self.np_random.uniform(-self.depth_target_range, self.depth_target_range)
            self.target_pos = np.array([start[0], depth if self.track_depth else start[1], start[2]])

        state = self.sim.reset(pos=tuple(start), quat=(1.0, 0.0, 0.0, 0.0))
        self.prev_action = np.zeros(ACT_DIM)
        self.prev_servo = np.zeros(4)
        self.step_count = 0
        self._place_marker(state)
        R, ori_err = self._errors(state)
        return self._get_obs(state, R, ori_err), self._info(state, 0.0, float(np.linalg.norm(ori_err)), 0.0, False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        state = self.sim.step(action)
        self.step_count += 1
        R, ori_err_vec = self._errors(state)

        pos_err = float(np.linalg.norm(self.target_pos - state["pos"]))
        ori_err = float(np.linalg.norm(ori_err_vec))
        depth_err = float(self.target_pos[1] - state["pos"][1])
        speed = float(np.linalg.norm(state["lin_vel"]))
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
        reward -= rw["w_ori"] * ori_err
        reward += rw.get("w_ori_bonus", 0.0) * prox(ori_err, rw.get("ori_scale", 0.35))
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

        success = ori_err < self.ori_tol
        if self.track_position:
            success = success and pos_err < self.pos_tol
        if self.track_depth:
            success = success and abs(depth_err) < self.depth_tol
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
        info["servo"] = state["servo"].copy()   # for eval diagnostics (servo motion)
        info["esc_cmd"] = action[4:8].copy()     # for eval diagnostics (thrust use / change)
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
