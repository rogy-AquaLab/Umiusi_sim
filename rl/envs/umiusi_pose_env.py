"""UmiusiPoseEnv — go-to-pose / station-keeping for the UMIUSI ROV.

Algorithm-agnostic Gymnasium environment (NO PPO-specific assumptions) wrapping the
standalone UmiusiSimulator. The agent commands the 8-D per-thruster action
([servo x4, esc x4] in [-1, 1]) and must reach and hold a target pose. The target
POSITION is randomized each episode; the target ORIENTATION is upright (the +Y-up
model's identity orientation).

Observation (28-D, float32):
    position error to target, body frame   (3)
    orientation error to target (rot-vec)  (3)
    body linear velocity                   (3)
    body angular velocity                  (3)
    servo angles / servo range             (4)
    thrust estimate / thrust_per_cmd       (4)
    previous action                        (8)

See ai/project_spec.yaml (rl section) and ai/architecture.md (section 7).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from sim.simulator import UmiusiSimulator

_ROOT = Path(__file__).resolve().parents[2]
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

OBS_DIM = 28  # 3 pos_err + 3 ori_err + 3 lin_vel + 3 ang_vel + 4 servo + 4 thrust + 8 prev_action
ACT_DIM = 8


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
        # `config` may be a dict (from load_config) or a path to a YAML file.
        cfg = config if isinstance(config, dict) else load_config(config)
        self.cfg = cfg

        sim_config = cfg.get("sim_config", "configs/umiusi.yaml")
        sim_config = sim_config if Path(sim_config).is_absolute() else _ROOT / sim_config
        self.sim = UmiusiSimulator(config_path=sim_config)

        e = cfg["env"]
        self.horizon = int(e["horizon"])
        self.target_box = np.array(e["target_box"], dtype=float)
        self.workspace_bounds = np.array(e["workspace_bounds"], dtype=float)
        self.start_jitter = float(e["start_jitter"])
        self.pos_tol = float(e["pos_tol"])
        self.ori_tol = float(e["ori_tol"])
        self.near_goal_dist = float(e["near_goal_dist"])
        self.rw = cfg["reward"]
        self.dr = cfg.get("domain_rand", {"enabled": False})

        # Base physical values, kept so domain randomization perturbs around them each reset.
        self._base = {
            "volume": self.sim.volume,
            "thrust_per_cmd": self.sim.thrust_per_cmd,
            "drag_lin": self.sim.drag_lin.copy(),
            "drag_quad": self.sim.drag_quad.copy(),
        }

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.render_mode = render_mode
        self._viewer = None

        self.target_pos = np.zeros(3)
        self.prev_action = np.zeros(ACT_DIM)
        self.step_count = 0

    # -- domain randomization --------------------------------------------------
    def _apply_domain_rand(self):
        b = self._base
        if not self.dr.get("enabled", False):
            # Ensure the sim always starts from the nominal config values.
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

    # -- observation -----------------------------------------------------------
    def _get_obs(self, state):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, state["quat"])
        R = R.reshape(3, 3)

        pos_err_body = R.T @ (self.target_pos - state["pos"])
        ori_err = np.zeros(3)
        mujoco.mju_subQuat(ori_err, _IDENTITY_QUAT, state["quat"])  # rot-vec from current to upright
        v_body = R.T @ state["lin_vel"]
        w_body = R.T @ state["ang_vel"]
        servo_n = state["servo"] / self.sim.servo_range_rad
        thrust_n = state["thrust"] / max(self.sim.thrust_per_cmd, 1e-9)

        obs = np.concatenate([pos_err_body, ori_err, v_body, w_body, servo_n, thrust_n, self.prev_action])
        if self.dr.get("enabled", False) and self.dr.get("obs_noise", 0.0) > 0.0:
            obs = obs + self.np_random.normal(0.0, self.dr["obs_noise"], size=obs.shape)
        return obs.astype(np.float32)

    # -- lifecycle -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._apply_domain_rand()

        self.target_pos = self.np_random.uniform(-self.target_box, self.target_box)
        start = self.np_random.uniform(-self.start_jitter, self.start_jitter, size=3)
        state = self.sim.reset(pos=tuple(start), quat=(1.0, 0.0, 0.0, 0.0))

        self.prev_action = np.zeros(ACT_DIM)
        self.step_count = 0
        return self._get_obs(state), self._info(state)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        state = self.sim.step(action)
        self.step_count += 1

        pos_err = float(np.linalg.norm(self.target_pos - state["pos"]))
        ori_err_vec = np.zeros(3)
        mujoco.mju_subQuat(ori_err_vec, _IDENTITY_QUAT, state["quat"])
        ori_err = float(np.linalg.norm(ori_err_vec))
        speed = float(np.linalg.norm(state["lin_vel"]))
        effort = float(np.linalg.norm(action[4:8]))
        action_rate = float(np.linalg.norm(action - self.prev_action))
        success = pos_err < self.pos_tol and ori_err < self.ori_tol

        rw = self.rw
        reward = (
            -rw["w_pos"] * pos_err
            - rw["w_ori"] * ori_err
            - (rw["w_vel"] * speed if pos_err < self.near_goal_dist else 0.0)
            - rw["w_effort"] * effort
            - rw["w_action_rate"] * action_rate
            + (rw["goal_bonus"] if success else 0.0)
        )

        out_of_bounds = bool(np.any(np.abs(state["pos"]) > self.workspace_bounds))
        if out_of_bounds:
            reward -= rw["out_of_bounds_penalty"]
        terminated = out_of_bounds
        truncated = self.step_count >= self.horizon

        self.prev_action = action.copy()
        info = self._info(state)
        info.update(pos_err=pos_err, ori_err=ori_err, is_success=success, out_of_bounds=out_of_bounds)
        return self._get_obs(state), reward, terminated, truncated, info

    def _info(self, state):
        return {"target_pos": self.target_pos.copy(), "pos": state["pos"].copy()}

    # -- rendering -------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = self.sim.model.camera("iso").id
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
