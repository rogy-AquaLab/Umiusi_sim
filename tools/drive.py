"""Drive a trained UMIUSI policy interactively — command a direction/orientation and watch.

Loads a trained attitude / attitude_velocity policy (its ``vecnormalize.pkl`` stats too, exactly
like ``umiusi_rl.eval``) and lets you fly the vehicle from the keyboard in the legible default
world (checker floor + axis triad). Each control step it OVERRIDES the env's commanded velocity
direction (``v_cmd``) and target orientation (``target_quat``) from your current keyboard command,
feeds the env's observation to the policy, and steps — so you can confirm "command a direction ->
it goes that way" and "command an orientation -> it turns there". The red/green/blue target marker
shows the commanded orientation.

Everything routes through the shared viewer (``umiusi_sim.viewer``): fixed ``track`` camera, +Y-up
handled, ``[`` / ``]`` cycle cameras.

Keyboard (the GUI window):
    A / D  or  Left / Right : steer cruise DIRECTION (body-frame azimuth) left / right
    W / S  or  Up   / Down  : cruise elevation up / down   (only if the policy is 3-D, i.e.
                              vel_cmd_horizontal=false; ignored for horizontal-only policies)
    Q / E                   : target YAW  (turn the held heading left / right)
    R / F                   : target TILT (nose up / down)
    Space                   : STOP (zero the cruise command; keep holding orientation)
    [  /  ]                 : cycle cameras (handled by the viewer)

Usage:
    python -m tools.drive --model models/av_curr4/final.zip          # interactive (needs a display)
    python -m tools.drive --model models/av_curr4/final.zip --headless 150   # headless self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config
from umiusi_sim.simulator import UmiusiSimulator

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
_ROOT = Path(__file__).resolve().parents[1]
_SCRATCH = Path(
    "/tmp/claude-1000/-home-satoi-mujoco-ws/0cf18c2f-3f06-4906-a070-c3f6db043305/scratchpad"
)
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])

# keyboard step sizes
_D_AZ = np.radians(15.0)
_D_EL = np.radians(10.0)
_D_YAW = np.radians(15.0)
_D_TILT = np.radians(10.0)

# GLFW key codes (mujoco.viewer passes raw GLFW codes to key_callback).
_K_SPACE = 32
_K_RIGHT, _K_LEFT, _K_DOWN, _K_UP = 262, 263, 264, 265


class Command:
    """Mutable keyboard-driven command shared between the viewer callback and the drive loop."""

    def __init__(self, speed, horizontal):
        self.speed = float(speed)          # cruise speed magnitude [m/s]
        self.cruise = float(speed)         # remembered speed to resume after a Stop
        self.horizontal = bool(horizontal)  # policy only controls horizontal cruise direction
        self.az = 0.0                      # body-frame cruise azimuth [rad] (0 = body +X)
        self.el = 0.0                      # body-frame cruise elevation [rad] (3-D policies only)
        self.yaw = 0.0                     # target yaw about +Y [rad]
        self.tilt = 0.0                    # target tilt (nose up/down about +Z) [rad]

    def target_quat(self):
        """Target orientation quaternion = yaw about +Y, then tilt (nose up/down) about +Z."""
        q_yaw, q_tilt, q = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_yaw, _Y, self.yaw)
        mujoco.mju_axisAngle2Quat(q_tilt, _Z, self.tilt)
        mujoco.mju_mulQuat(q, q_yaw, q_tilt)
        return q

    def v_cmd_body(self):
        """Commanded velocity in the TARGET-BODY frame (what the env's obs carries)."""
        if self.horizontal:
            dhat = np.array([np.cos(self.az), 0.0, np.sin(self.az)])
        else:
            dhat = np.array([np.cos(self.el) * np.cos(self.az),
                             np.sin(self.el),
                             np.cos(self.el) * np.sin(self.az)])
        return dhat * self.speed

    def describe(self):
        return (f"cmd: dir az={np.degrees(self.az):+5.0f}deg el={np.degrees(self.el):+4.0f}deg "
                f"speed={self.speed:.2f} m/s | target yaw={np.degrees(self.yaw):+5.0f}deg "
                f"tilt={np.degrees(self.tilt):+4.0f}deg")


def _make_key_callback(cmd, track_velocity):
    """Return a ``fn(keycode)`` that mutates ``cmd`` and prints it on change."""

    def cb(keycode):
        k = keycode
        ch = chr(k).upper() if 0 <= k < 0x110000 else ""
        changed = True
        if k == _K_SPACE:
            cmd.speed = 0.0
        elif ch == "A" or k == _K_LEFT:
            cmd.az -= _D_AZ
            cmd.speed = cmd.cruise
        elif ch == "D" or k == _K_RIGHT:
            cmd.az += _D_AZ
            cmd.speed = cmd.cruise
        elif ch == "W" or k == _K_UP:
            cmd.el += _D_EL
            cmd.speed = cmd.cruise
        elif ch == "S" or k == _K_DOWN:
            cmd.el -= _D_EL
            cmd.speed = cmd.cruise
        elif ch == "Q":
            cmd.yaw += _D_YAW
        elif ch == "E":
            cmd.yaw -= _D_YAW
        elif ch == "R":
            cmd.tilt += _D_TILT
        elif ch == "F":
            cmd.tilt -= _D_TILT
        else:
            changed = False
        if changed:
            if not track_velocity:  # attitude-only policy: no cruise command
                cmd.speed = 0.0
            print(cmd.describe(), flush=True)

    return cb


def _apply_command(env, cmd):
    """Push the current keyboard command into the env (target_quat + v_cmd/v_cmd_world)."""
    env.target_quat = cmd.target_quat()
    if env.track_velocity:
        env.v_cmd = cmd.v_cmd_body()
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, env.target_quat)
        env.v_cmd_world = Rt.reshape(3, 3) @ env.v_cmd


def _build_env(model_path, no_disturb=True):
    """Build UmiusiPoseEnv matching a trained policy + a norm_obs() from its VecNormalize stats."""
    model_path = Path(model_path)
    meta_path = model_path.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    algo = meta.get("algo", "ppo")
    config = meta.get("config", "configs/train_ppo.yaml")

    cfg = load_config(config)
    for k in ("task", "obs_mode", "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    cfg["env"]["task"] = meta.get("task", cfg["env"].get("task", "attitude_velocity"))
    if no_disturb:
        cfg.setdefault("disturbance", {})["enabled"] = False
    cfg.setdefault("domain_rand", {})["enabled"] = False

    env = UmiusiPoseEnv(cfg, render_mode=None)
    policy = ALGOS[algo].load(str(model_path), device="cpu")

    stats_path = model_path.parent / "vecnormalize.pkl"
    if stats_path.exists():
        dummy = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
        vn = VecNormalize.load(str(stats_path), dummy)
        dummy.close()
        rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

        def norm_obs(o):
            return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    else:
        def norm_obs(o):
            return o

    return env, policy, norm_obs, cfg, meta


def _obs_now(env):
    """Compute the env's current observation from the live state + the just-set command."""
    state = env.sim.get_state()
    R, ori_err = env._errors(state)
    return env._get_obs(state, R, ori_err)


def _swap_in_world(env, cfg):
    """Replace the env's simulator with one loaded from the legible default world (visual only).

    default_world is additive (base robot untouched, same body/site/actuator NAMES), so the
    simulator + env keep working; only the scene the viewer shows gains a floor/grid/axes.
    """
    from umiusi_sim.description.scenarios import default_world as scn

    _SCRATCH.mkdir(parents=True, exist_ok=True)
    world_xml = scn.write_xml(_SCRATCH / "drive_world.xml")
    sim_config = cfg.get("sim_config", "configs/umiusi.yaml")
    sim_config = sim_config if Path(sim_config).is_absolute() else _ROOT / sim_config
    env.sim = UmiusiSimulator(model_path=world_xml, config_path=sim_config)
    try:
        env._mocap_id = int(env.sim.model.body_mocapid[env.sim.model.body("target_marker").id])
    except (KeyError, ValueError):
        env._mocap_id = -1


def run_headless(env, policy, norm_obs, n_steps, meta):
    """Self-test WITHOUT a display: fixed body +X command; report measured vs commanded direction."""
    cmd = Command(speed=env.vel_cmd_max, horizontal=env.vel_cmd_horizontal)
    cmd.az = cmd.el = cmd.yaw = cmd.tilt = 0.0  # command straight ahead (body +X), upright target
    env.reset(seed=0)
    _apply_command(env, cmd)
    cmd_world = env.v_cmd_world.copy()
    dhat = cmd_world / (np.linalg.norm(cmd_world) + 1e-9)

    vels, settle = [], max(20, n_steps // 3)  # ignore the initial spin-up transient
    for i in range(n_steps):
        _apply_command(env, cmd)               # re-assert each step (env.reset/step don't touch it here)
        obs = _obs_now(env)
        action, _ = policy.predict(norm_obs(obs), deterministic=True)
        env.step(action)
        if i >= settle:
            vels.append(env.sim.get_state()["lin_vel"].copy())

    mean_v = np.mean(vels, axis=0)
    speed = float(np.linalg.norm(mean_v))
    vhat = mean_v / (speed + 1e-9)
    cos = float(np.dot(vhat, dhat))
    v_along = float(np.dot(mean_v, dhat))
    v_perp = float(np.linalg.norm(mean_v - v_along * dhat))
    print("-" * 68)
    print(f"headless self-test  task={meta.get('task', '?')}  steps={n_steps} "
          f"(averaged last {len(vels)})")
    print(f"  commanded world direction : {np.round(dhat, 3)}  (speed cmd {cmd.speed:.2f} m/s)")
    print(f"  measured mean velocity    : {np.round(mean_v, 3)} m/s  (|v|={speed:.3f})")
    print(f"  along cmd = {v_along:+.3f} m/s   sideways = {v_perp:.3f} m/s")
    print(f"  direction cosine (cmd . measured) = {cos:+.3f}   "
          f"-> {'ALIGNED' if cos > 0.5 else ('positive' if cos > 0 else 'MISALIGNED')}")
    return cos


def run_interactive(env, policy, norm_obs, cfg):
    """Live GUI drive loop in the default world, paced to real time by the shared viewer."""
    from umiusi_sim.viewer import UmiusiViewer

    _swap_in_world(env, cfg)
    cmd = Command(speed=env.vel_cmd_max, horizontal=env.vel_cmd_horizontal)
    env.reset(seed=0)
    print(cmd.describe(), flush=True)

    extra_keys = {
        "A/D or Left/Right": "steer cruise azimuth",
        "W/S or Up/Down": "cruise elevation (3-D policies only)",
        "Q/E": "target yaw", "R/F": "target tilt", "space": "STOP (zero cruise)",
    }
    key_cb = _make_key_callback(cmd, env.track_velocity)

    def step():
        _apply_command(env, cmd)
        obs = _obs_now(env)
        action, _ = policy.predict(norm_obs(obs), deterministic=True)
        env.step(action)

    with UmiusiViewer(env.sim.model, env.sim.data, base_id=env.sim.base_id, cam="track",
                      control_rate_hz=env.sim.cfg["sim"]["control_rate_hz"],
                      key_callback=key_cb, extra_keys=extra_keys) as viewer:
        viewer.run(step)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="trained policy .zip (attitude_velocity or attitude)")
    ap.add_argument("--headless", type=int, default=0, metavar="N",
                    help="run N control steps with a fixed body +X command, no viewer (self-test)")
    args = ap.parse_args()

    env, policy, norm_obs, cfg, meta = _build_env(args.model)
    if not env.track_velocity and args.headless:
        print(f"note: task={meta.get('task')} has no cruise command; headless self-test drives "
              "orientation only (cosine is not meaningful).")
    try:
        if args.headless:
            run_headless(env, policy, norm_obs, args.headless, meta)
        else:
            run_interactive(env, policy, norm_obs, cfg)
    finally:
        env.close()


if __name__ == "__main__":
    main()
