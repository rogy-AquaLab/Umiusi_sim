"""ros_policy — drive the ROS-driven MuJoCo sim with a trained RL policy over rosbridge.

The C++ ros2_control hardware plugin (`umiusi_sim_bridge::MujocoSystem`) runs the real MuJoCo
physics inside the ROS control loop; the sinsei_umiusi_control controllers expose per-thruster
DIRECT-OVERRIDE topics that bypass the feed-forward AttitudeController and pass an ESC/servo
command straight to the sim. This tool makes a trained `umiusi_rl` policy the low-level
controller of that running sim:

    * connect to rosbridge (ws://localhost:9090) with **roslibpy** — no rclpy needed (mirrors
      tools/ros_view.py). Runs in THIS uv venv (torch + SB3 + our umiusi_rl).
    * SUBSCRIBE the vehicle state:
        /state/imu_state          (ImuState)        -> current quaternion + body gyro
        /state/thruster_state_all (ThrusterStateAll)-> per-thruster servo angle + applied esc
    * reconstruct the policy's observation (25-D proprio "full" / 17-D proprio "action") EXACTLY
      as UmiusiPoseEnv._get_obs builds it for
      task=attitude_velocity, obs_mode=imu (reusing the env's own helpers so the layout/scaling
      can't silently drift), apply the training-time VecNormalize, run policy.predict().
    * PUBLISH the four direct-override commands:
        /cmd/direct/thruster_controller/output_{lf,lb,rb,rf}  (ThrusterOutput, runnable=true)
      mapping the 8-D action [servo x4, esc x4] in [-1,1] to {duty_cycle, angle_deg}.

Fixed test command (prove the loop): hold UPRIGHT (target = identity) + CRUISE forward
(v_cmd = body +X at the policy's vel_cmd_max). The vehicle should hold ~level and translate
along +X, versus just floating up in +Y when undriven.

Usage:
    # start the sim first (in ros2_ws): ros2 launch umiusi_sim_bridge sim.launch.py
    uv run --extra viz python -m tools.ros_policy --dry-run          # obs+action sanity, no publish
    uv run --extra viz python -m tools.ros_policy                    # closed loop (publishes)
    uv run --extra viz python -m tools.ros_policy --seconds 10       # run the loop for 10 s then stop
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import mujoco
import numpy as np
import roslibpy
import yaml

_REPO = Path(__file__).resolve().parents[1]  # umiusi_sim repo root

# --- ROS topics / types (verified against a live `ros2 launch umiusi_sim_bridge sim.launch.py`) --
IMU_TOPIC = "/state/imu_state"
IMU_TYPE = "sinsei_umiusi_msgs/msg/ImuState"
THR_TOPIC = "/state/thruster_state_all"
THR_TYPE = "sinsei_umiusi_msgs/msg/ThrusterStateAll"
QPOS_TOPIC = "/umiusi_sim/qpos"
QPOS_TYPE = "std_msgs/Float64MultiArray"
CMD_PREFIX = "/cmd/direct/thruster_controller/output_"
CMD_TYPE = "sinsei_umiusi_msgs/msg/ThrusterOutput"

# Thruster position <-> policy index. UmiusiSimulator now maps action channels by NAME
# (configs/umiusi.yaml `action_order`, default lf, lb, rb, rf — this same tuple), so policy
# index k means the thruster named POSITIONS[k] in BOTH sim and autonomy. NOTE the geometric
# names in the config label unit id3 = rf and id4 = rb (the old assumed "3=rb 4=rf" had the
# starboard pair swapped); which CAN id is wired to which corner still needs a hardware check.
POSITIONS = ("lf", "lb", "rb", "rf")


class StateReceiver:
    """Keeps the latest ImuState / ThrusterStateAll (and optionally qpos) from rosbridge.

    roslibpy delivers messages on its network thread; the control loop reads under a lock.
    """

    def __init__(self, client: roslibpy.Ros, want_qpos: bool = False):
        self._lock = threading.Lock()
        self.imu: dict | None = None
        self.thruster: dict | None = None
        self.qpos: np.ndarray | None = None
        self.imu_count = 0
        self.thr_count = 0
        self._imu_topic = roslibpy.Topic(client, IMU_TOPIC, IMU_TYPE, throttle_rate=0, queue_length=1)
        self._thr_topic = roslibpy.Topic(client, THR_TOPIC, THR_TYPE, throttle_rate=0, queue_length=1)
        self._qpos_topic = (
            roslibpy.Topic(client, QPOS_TOPIC, QPOS_TYPE, throttle_rate=0, queue_length=1)
            if want_qpos else None
        )

    def subscribe(self):
        self._imu_topic.subscribe(self._on_imu)
        self._thr_topic.subscribe(self._on_thr)
        if self._qpos_topic is not None:
            self._qpos_topic.subscribe(self._on_qpos)

    def unsubscribe(self):
        for t in (self._imu_topic, self._thr_topic, self._qpos_topic):
            if t is None:
                continue
            try:
                t.unsubscribe()
            except Exception:
                pass

    def _on_imu(self, msg):
        with self._lock:
            self.imu = msg
            self.imu_count += 1

    def _on_thr(self, msg):
        with self._lock:
            self.thruster = msg
            self.thr_count += 1

    def _on_qpos(self, msg):
        data = msg.get("data")
        if not data:
            return
        with self._lock:
            self.qpos = np.asarray(data, dtype=float)

    def get(self):
        with self._lock:
            return self.imu, self.thruster

    def get_qpos(self):
        with self._lock:
            return None if self.qpos is None else self.qpos.copy()


def build_env_and_policy(model_path: Path, config: str | None, algo: str | None):
    """Instantiate a READ-ONLY UmiusiPoseEnv (attitude_velocity / imu) to borrow _get_obs +
    constants, load the matching VecNormalize stats and the trained policy (mirrors rl/eval.py)."""
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

    algos = {"ppo": PPO, "sac": SAC, "td3": TD3}
    meta_path = model_path.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    algo = algo or meta.get("algo", "ppo")
    config = config or meta.get("config", "configs/train_ppo.yaml")

    cfg = load_config(config)
    # Match the exact task / sensor-suite / curriculum the policy trained with (as rl/eval.py does).
    for k in ("task", "obs_mode", "proprio_mode", "obs_frame",
              "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    # obs-contract key: absent in old runs = trained WITHOUT the cap dim (never let the newer
    # config file grow the obs vector under an old policy)
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    # Nominal model at deploy: no domain-rand obs noise, no disturbances (mirror eval defaults).
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg.setdefault("disturbance", {})["enabled"] = False

    env = UmiusiPoseEnv(cfg)  # read-only: used for _get_obs/_errors + servo_range/thrust_per_cmd
    if env.task != "attitude_velocity" or env.obs_mode != "imu":
        print(f"WARNING: expected task=attitude_velocity/obs_mode=imu, got "
              f"{env.task}/{env.obs_mode}; obs layout may differ.", flush=True)

    model = algos[algo].load(str(model_path), device="cpu")

    stats_path = model_path.parent / "vecnormalize.pkl"
    if stats_path.exists():
        _dummy = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
        vn = VecNormalize.load(str(stats_path), _dummy)
        _dummy.close()
        rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

        def norm_obs(o):
            return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    else:
        print("WARNING: no vecnormalize.pkl next to the model; using raw obs.", flush=True)

        def norm_obs(o):
            return o.astype(np.float32)

    return env, model, norm_obs


def reconstruct_obs(env, imu: dict, thr: dict, target_quat, v_cmd, prev_action):
    """Build the exact observation UmiusiPoseEnv._get_obs produces, from live ROS state.
    (With proprio_mode "action" the servo/thrust entries below are simply not emitted.)

    Mapping (imu + attitude_velocity, obs = [ori_err(3), ang_vel(3), v_cmd(3), servo(4),
    thrust(4), prev_action(8)]):
      ori_err   = mju_subQuat(target_quat, current_quat)      (current -> target rot-vector)
      ang_vel   = body gyro (rad/s) from /state/imu_state.angular_velocity
      v_cmd     = commanded velocity in the TARGET-BODY frame (fixed test: body +X * vel_cmd_max)
      servo     = per-thruster servo angle / servo_range      (deg -> rad, /90 deg)
      thrust    = per-thruster applied esc (= rpm/1000)        (thrust/thrust_per_cmd)
      prev_action = last published policy action
    _get_obs computes the gyro as R^T @ state["ang_vel"] (expects a WORLD-frame angular velocity),
    but the IMU already reports the BODY-frame gyro, so we pre-rotate it by R (R @ gyro_body) and
    _get_obs's R^T undoes it -> exactly the body gyro. Likewise state["servo"] is fed in radians
    and state["thrust"] in Newtons so the env's own normalisation reproduces the training scaling.
    """
    q = imu["quaternion"]
    cur_quat = np.array([q["w"], q["x"], q["y"], q["z"]], dtype=float)  # ROS x,y,z,w -> MuJoCo w,x,y,z
    g = imu["angular_velocity"]
    gyro_body = np.array([g["x"], g["y"], g["z"]], dtype=float)         # body-frame gyro [rad/s]

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, cur_quat)
    R = R.reshape(3, 3)

    servo_deg = np.array([thr[p]["angle"] for p in POSITIONS], dtype=float)   # commanded servo [deg]
    esc_applied = np.array([thr[p]["rpm"] for p in POSITIONS], dtype=float) / 1000.0  # applied esc

    state = {
        "quat": cur_quat,
        "ang_vel": R @ gyro_body,               # world-frame; env applies R^T -> body gyro
        "servo": np.radians(servo_deg),         # env divides by servo_range_rad
        "thrust": esc_applied * env.sim.thrust_per_cmd,  # env divides by thrust_per_cmd
    }
    env.target_quat = np.asarray(target_quat, dtype=float)
    env.v_cmd = np.asarray(v_cmd, dtype=float)
    env.prev_action = np.asarray(prev_action, dtype=float)

    R2, ori_err = env._errors(state)
    obs = env._get_obs(state, R2, ori_err)
    return obs, cur_quat, gyro_body, servo_deg, esc_applied


def action_to_outputs(action, servo_range_deg):
    """Map policy action [servo x4, esc x4] in [-1,1] to per-position ThrusterOutput dicts.

    servo angle [deg] = servo_action * servo_range_deg (plugin converts deg->rad, clamps +/-range;
    servo_action * range_rad reproduces the training servo target). duty_cycle = esc_action (the
    plugin clamps to [-1,1] and multiplies by thrust_per_cmd=30 N, exactly as training does)."""
    outs = {}
    for k, p in enumerate(POSITIONS):
        outs[p] = {
            "runnable": {"esc": True, "servo": True},
            "duty_cycle": float(action[4 + k]),
            "angle": float(action[k]) * servo_range_deg,
        }
    return outs


def _fmt(a, p=3):
    return np.array2string(np.asarray(a), precision=p, suppress_small=True, floatmode="fixed")


def run(env, model, norm_obs, client, receiver, args):
    servo_range_deg = float(np.degrees(env.sim.servo_range_rad))
    vel = float(args.vel_cmd if args.vel_cmd is not None else env.vel_cmd_max)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])           # identity = hold upright/level
    v_cmd = np.array([vel, 0.0, 0.0])                       # cruise along body +X

    pubs = {p: roslibpy.Topic(client, CMD_PREFIX + p, CMD_TYPE) for p in POSITIONS}
    if not args.dry_run:
        for t in pubs.values():
            t.advertise()

    print(f"policy loop: task={env.task} obs_mode={env.obs_mode}  target=identity  "
          f"v_cmd=[{vel:.3f}, 0, 0] m/s (+X)  hz={args.hz}  "
          f"{'DRY-RUN (no publish)' if args.dry_run else 'PUBLISHING'}", flush=True)

    # wait for the first state messages
    t0 = time.time()
    while time.time() - t0 < 10.0:
        imu, thr = receiver.get()
        if imu is not None and thr is not None:
            break
        time.sleep(0.1)
    imu, thr = receiver.get()
    if imu is None or thr is None:
        print("ERROR: no /state/imu_state or /state/thruster_state_all received — is the sim up?",
              flush=True)
        return 1

    prev_action = np.zeros(8)
    dt = 1.0 / args.hz
    period_start = time.time()
    ticks = 0
    printed = 0
    last_report = 0.0
    while client.is_connected:
        tic = time.time()
        imu, thr = receiver.get()
        if imu is None or thr is None:
            time.sleep(dt)
            continue

        obs, cur_quat, gyro_body, servo_deg, esc_applied = reconstruct_obs(
            env, imu, thr, target_quat, v_cmd, prev_action)
        nobs = norm_obs(obs)
        action, _ = model.predict(nobs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(8), -1.0, 1.0)
        outs = action_to_outputs(action, servo_range_deg)

        if not args.dry_run:
            for p in POSITIONS:
                pubs[p].publish(roslibpy.Message(outs[p]))

        if args.dry_run and printed < args.dry_ticks:
            ori_err = obs[0:3]
            print(f"\n--- tick {printed} ---", flush=True)
            print(f"  imu quat(wxyz) = {_fmt(cur_quat, 4)}   gyro_body(rad/s) = {_fmt(gyro_body, 4)}",
                  flush=True)
            print(f"  servo(deg)     = {_fmt(servo_deg, 2)}   esc_applied      = {_fmt(esc_applied, 3)}",
                  flush=True)
            print(f"  obs(25): ori_err={_fmt(obs[0:3])} ang_vel={_fmt(obs[3:6])} "
                  f"v_cmd={_fmt(obs[6:9])}", flush=True)
            print(f"           servo_n={_fmt(obs[9:13])} thrust_n={_fmt(obs[13:17])} "
                  f"prev_act={_fmt(obs[17:25])}", flush=True)
            print(f"  finite={np.all(np.isfinite(obs))} |ori_err|={np.linalg.norm(ori_err):.3f} rad",
                  flush=True)
            print(f"  action servo={_fmt(action[0:4])} esc={_fmt(action[4:8])}", flush=True)
            print("  -> outputs: " + "  ".join(
                f"{p}(duty={outs[p]['duty_cycle']:+.2f},ang={outs[p]['angle']:+.1f}deg)"
                for p in POSITIONS), flush=True)
            printed += 1
            if printed >= args.dry_ticks:
                break

        prev_action = action
        ticks += 1

        # periodic pose report from qpos (run mode), so the loop is self-verifying
        if not args.dry_run and args.report and (tic - last_report) >= 1.0:
            qp = receiver.get_qpos()
            last_report = tic
            if qp is not None and qp.size >= 7:
                print(f"[t={tic - period_start:5.1f}s] base pos={_fmt(qp[0:3], 3)} "
                      f"quat(wxyz)={_fmt(qp[3:7], 3)} servo(rad)={_fmt(qp[7:11], 3)}", flush=True)

        if args.seconds is not None and (tic - period_start) >= args.seconds:
            break
        time.sleep(max(0.0, dt - (time.time() - tic)))

    if not args.dry_run:
        # leave the thrusters commanded to zero so the vehicle doesn't keep driving after we exit
        for p in POSITIONS:
            pubs[p].publish(roslibpy.Message(
                {"runnable": {"esc": True, "servo": True}, "duty_cycle": 0.0, "angle": 0.0}))
            try:
                pubs[p].unadvertise()
            except Exception:
                pass
    print(f"\ndone ({ticks} ticks).", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(_REPO / "examples" / "cruise_policy" / "final.zip"),
                    help="trained policy .zip (default: examples/cruise_policy/final.zip)")
    ap.add_argument("--config", default=None, help="env config (default: from the run's meta.yaml)")
    ap.add_argument("--algo", default=None, help="ppo/sac/td3 (default: from meta.yaml)")
    ap.add_argument("--url", default="ws://localhost:9090", help="rosbridge websocket URL")
    ap.add_argument("--hz", type=float, default=50.0, help="control rate (default 50 Hz)")
    ap.add_argument("--vel-cmd", type=float, default=None,
                    help="forward (+X) commanded speed [m/s] (default: policy vel_cmd_max)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build obs + predict for a few ticks against the live sim, print, DO NOT publish")
    ap.add_argument("--dry-ticks", type=int, default=8, help="ticks to print in --dry-run")
    ap.add_argument("--seconds", type=float, default=None, help="run the closed loop for N s then stop")
    ap.add_argument("--report", action="store_true", default=True,
                    help="in run mode, print base pose from /umiusi_sim/qpos every ~1 s")
    args = ap.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = _REPO / model_path
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", flush=True)
        return 2

    print(f"loading policy + env from {model_path} ...", flush=True)
    env, model, norm_obs = build_env_and_policy(model_path, args.config, args.algo)

    parsed = urlparse(args.url)
    client = roslibpy.Ros(host=parsed.hostname or "localhost", port=parsed.port or 9090,
                          is_secure=parsed.scheme in ("wss", "https"))
    print(f"connecting to rosbridge at {args.url} ...", flush=True)
    try:
        client.run(timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not connect to rosbridge at {args.url}: {e}", flush=True)
        return 2
    if not client.is_connected:
        print(f"ERROR: not connected to rosbridge at {args.url}.", flush=True)
        return 2
    print(f"connected to rosbridge at {args.url}.", flush=True)

    receiver = StateReceiver(client, want_qpos=(not args.dry_run))
    receiver.subscribe()
    try:
        return run(env, model, norm_obs, client, receiver, args)
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)
        return 0
    finally:
        receiver.unsubscribe()
        try:
            client.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
