"""Competition / single-balloon episodes through the ROBOT's AUTO path: FSM -> ROS -> real control -> sim.

What runs is what the robot runs in AUTO, minus the camera:

    GT detections (tools/competition_eval: FOV, range, perception rate, dropout, FP, recall curve)
      -> BalloonDetection msg -> AutoTargetGenerator._to_detection     (umiusi_autonomy, imported)
      -> live_detections (detections_timeout_s)                         (imported)
      -> BalloonBehavior.step(dets, yaw_rate = IMU z (REP-103) * yaw_rate_sign, heading=0, dt=measured)
         with fsm.* from config/competition.yaml applied by fsm_params  (imported)
      -> to_control_setpoint -> /cmd/target (velocity.x/z) + /cmd/attitude_target (level, yaw_rate)
      -> sinsei_umiusi_control (ros2_control_node + gate / attitude / thruster x4, UNMODIFIED, use_sim_time)
      -> umiusi_sim_bridge/MujocoSystem -> Unix socket -> THIS process: tools.sim_server.SimServer
         (REP-103 state, body gyro, servo rad) on the competition scene (balloons, pin)

The lockstep clock (one /clock tick per served control cycle, physics time == controller time) is
the one from tools/closed_loop_replay.py (see docs/closed_loop_replay_20261003.md §5). The plant's
duty cap is lifted (--plant-max-duty 1.0) so only the control's max_duty (0.5) acts.

Scoring is competition_eval's: scn.popped (POP_ANGLE_TOL_DEG, MIN_POP_SPEED), scn.entanglement, the
same layouts for the same seed (ep seed = 1000 + seed + e), the same summary lines.

Usage (umiusi_sim; source /opt/ros/jazzy and ros2_ws/install first):

    nice -n 10 uv run python -m tools.competition_ros --layout single --single-colour red --episodes 4 --minutes 2
    nice -n 10 uv run python -m tools.competition_ros --layout field --episodes 2 --minutes 3 \
        [--fsm ram_surge=0.4] [--param attitude_controller.feedback.kp_roll=1.0] [--yaw-rate-scale -1]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from tools.closed_loop_replay import CONTROLLERS, FrameReader, build_param_file, xacro_urdf
from tools.competition_eval import START, SURFACE_BAND, summarize
from tools.ram_eval import CAM_W, degrade_projection, false_positive, make_detection, project_balloon
from tools.sim_server import _REPLY_FIXED, _REPLY_HEAD, _REQ, SimServer, _send_msg
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator

COMPETITION_YAML = Path(
    "/home/satoimo/mujoco_ws/ros2_ws/src/sinsei_UMIUSI_autonomy/umiusi_autonomy/config/competition.yaml"
)


# -- detections (competition_eval.run_episode's detector tick, same RNG order) ----------------
def episode_layout(rng, args):
    """competition_eval.run_episode's layout draw (same RNG calls, so the same seed = the same field)."""
    if args.layout == "single":
        rr = rng.uniform(*args.single_range)
        th = rng.uniform(-np.pi, np.pi)
        return [
            (
                f"balloon_{args.single_colour}_1",
                args.single_colour,
                START[0] + rr * np.cos(th),
                START[2] + rr * np.sin(th),
            )
        ]
    return scn.sample_layout(rng)


def episode_phantoms(rng, args):
    out = []
    for i in range(args.phantoms):
        y = scn.POOL_DEPTH - 0.1 if i % 2 == 0 else scn.FLOOR_Y + 0.1
        out.append(np.array([rng.uniform(0.5, 5.0), y, rng.uniform(-2.0, 2.0)]))
    return out


def detector_frame(sim, cam_id, balloons, popped, phantoms, rng, args, recall_curve):
    """One detector frame of GT detections (mirror of the block in competition_eval.run_episode)."""
    R = sim.data.xmat[sim.base_id].reshape(3, 3)
    cam_pos = sim.data.cam_xpos[cam_id]
    dets = []
    for b in balloons:
        if b["name"] in popped:
            continue
        proj = project_balloon(R.T @ (b["pos"] - cam_pos))
        if proj is None:
            continue
        dproj = degrade_projection(proj, rng, args)
        if dproj is None:
            continue
        d = make_detection(*dproj, b["colour"], b["points"])
        if d is not None and recall_curve:
            frac = (d.bbox[2] - d.bbox[0]) / CAM_W
            rc = next((r for f, r in recall_curve if frac < f), recall_curve[-1][1])
            if rng.random() >= rc:
                d = None
        if d is not None:
            dets.append(d)
    for _ in range(int(args.fp_per_frame) + (rng.random() < args.fp_per_frame % 1)):
        fp = false_positive(rng)
        if fp is not None:
            dets.append(fp)
    if args.fp_rate > 0 and rng.random() < args.fp_rate:
        fp = false_positive(rng)
        if fp is not None:
            dets.append(fp)
    for ph in phantoms:
        if rng.random() >= args.phantom_p:
            continue
        proj = project_balloon(R.T @ (ph - cam_pos))
        dproj = degrade_projection(proj, rng, args) if proj is not None else None
        d = make_detection(*dproj, args.phantom_colour, 0) if dproj is not None else None
        if d is not None:
            dets.append(d)
    return dets


def to_msg(dets):
    """Detections -> BalloonDetectionArray, as perception_node would publish them (float32 fields)."""
    from umiusi_autonomy_msgs.msg import BalloonDetection, BalloonDetectionArray

    m = BalloonDetectionArray()
    for d in dets:
        m.detections.append(
            BalloonDetection(
                colour=d.colour,
                points=int(d.points),
                azimuth=float(d.bearing[0]),
                elevation=float(d.bearing[1]),
                range_m=float(d.range_m),
                confidence=float(d.confidence),
                bbox=[int(v) for v in d.bbox],
                centroid=[float(d.centroid[0]), float(d.centroid[1])],
                area_px=int(d.area_px),
            )
        )
    return m


# -- the FSM as auto_target_generator runs it -------------------------------------------------
class AutoFsm:
    """auto_target_generator._tick without the node: same imports, same conversions."""

    def __init__(self, gen_params, fsm_overrides, use_yaml_fsm=True):
        from umiusi_autonomy import auto_target_generator as atg
        from umiusi_autonomy import fsm_params
        from umiusi_perception.autonomy import BalloonBehavior
        from umiusi_perception.balloon_detector import Detection

        self.atg, self.fp = atg, fsm_params
        p = gen_params
        self.nominal_dt = 1.0 / float(p["control_hz"])
        self.surge_sign = float(p["surge_sign"])
        self.yaw_rate_scale = float(p["yaw_rate_scale"])
        self.timeout = float(p["detections_timeout_s"])
        self.yaw_rate_sign = float(p.get("yaw_rate_sign", 1.0))
        assert str(p.get("yaw_rate_axis", "z")).lower() == "z", "only yaw_rate_axis z is modelled"
        self.b = BalloonBehavior(
            frame_h=int(p["frame_h"]), frame_w=int(p["frame_w"]), fovy_deg=float(p["fovy_deg"]), dt=self.nominal_dt
        )
        # _declare_fsm_params: default = what the FSM holds now, yaml wins. Module constants are
        # process-global, so the module is restored to its import-time defaults first (episodes and
        # --fsm-defaults must not inherit a previous episode's values).
        absent = set(fsm_params.missing(self.b))
        if absent:
            print(f"[ros] WARN FSM lacks tunables (old umiusi_perception): {sorted(absent)}")
        yaml_fsm = p.get("fsm", {}) if use_yaml_fsm else {}
        for fp in fsm_params.FSM_PARAMS:
            if fp.name in absent:
                continue
            fsm_params.apply(
                self.b, fp, yaml_fsm.get(fp.name, _DEFAULTS.setdefault(fp.name, fsm_params.current_value(self.b, fp)))
            )
        for kv in fsm_overrides:
            k, _, v = kv.partition("=")
            fp = fsm_params.by_ros_name(fsm_params.PREFIX + k)
            if fp is None:
                raise SystemExit(f"unknown --fsm {k}")
            fsm_params.apply(self.b, fp, yaml.safe_load(v))
        self._detection_cls = SimpleNamespace(_Detection=Detection)
        self.dets, self.new_dets, self.last_rx, self.last_tick = [], False, None, None

    def on_detections(self, msg, now):
        self.dets = [self.atg.AutoTargetGenerator._to_detection(self._detection_cls, d) for d in msg.detections]
        self.new_dets = True
        self.last_rx = now

    def tick(self, now, gyro_z_imu):
        dt = self.fp.measured_dt(self.last_tick, now, self.nominal_dt)
        self.last_tick = now
        self.b.dt = dt
        fresh, self.new_dets = self.new_dets, False
        dets = self.atg.live_detections(self.dets, self.last_rx, now, self.timeout)
        cmd, info = self.b.step(dets, self.yaw_rate_sign * gyro_z_imu, heading=0.0, dt=dt, fresh=fresh)
        return cmd, self.atg.to_control_setpoint(cmd, self.surge_sign, self.yaw_rate_scale)


_DEFAULTS = {}  # FSM tunables at import time (filled on first AutoFsm)


# -- one episode: real control stack in lockstep with the scene ------------------------------
class Episode:
    def __init__(self, rng, args, ros, xml_path, ep_seed):
        self.args, self.ros, self.rng = args, ros, rng
        layout = episode_layout(rng, args)
        self.phantoms = episode_phantoms(rng, args)
        xml_path.write_text(scn.build_spec(layout=layout).to_xml())
        sim = UmiusiSimulator(model_path=xml_path)
        sim.max_duty = float(args.plant_max_duty)
        sim.water_surface_y = scn.POOL_DEPTH  # float at the surface (as competition_eval), not rise out of the pool
        sim.floor_y = scn.FLOOR_Y             # rest on the floor, not sink through it
        for kv in args.plant:  # isolation knobs (plant only; control unchanged)
            k, _, v = kv.partition("=")
            if k == "vertical_eff":
                sim.thrust_vertical_eff = float(v)
            elif k == "net_buoy":
                sim.set_net_buoyancy(float(v))
            else:
                raise SystemExit(f"unknown --plant {k}")
        self.server = SimServer(sim, servo_unit="rad")
        self.sim = sim
        self.start_pos = (START[0], args.start_height, START[2])
        sim.reset(pos=self.start_pos)
        self.balloons = scn.balloon_table(layout=layout)
        self.positive = {b["name"] for b in self.balloons if b["points"] > 0}
        self.cam_id = sim.model.camera("front_cam").id
        self.pin_sid = sim.model.site("pin_tip").id
        self.recall_curve = (
            [tuple(float(v) for v in kv.split(":")) for kv in args.recall_curve.split(",")]
            if args.recall_curve
            else None
        )
        self.fsm = AutoFsm(ros.gen_params, args.fsm, use_yaml_fsm=not args.fsm_defaults)
        self.ep_seed = ep_seed
        self.layout = layout
        self.record_video = False

    def reset_plant(self):
        self.sim.reset(pos=self.start_pos)
        self.server._prev_lin_world = np.zeros(3)
        self.server._have_prev = False

    def serve(self, conn, payload, dt):
        f = _REQ.unpack(payload)
        reply = self.server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], dt)
        _send_msg(conn, reply)
        return f, reply

    # ---- per served control cycle, after physics ----
    def begin(self):
        a = self.args
        self.t = 0.0
        self.next_det, self.next_fsm, self.next_arm = 0.0, 0.0, 0.0
        self.pending = []
        self.popped, self.score, self.prev_pin = set(), 0, None
        self.snag_prev, self.wire, self.t_clear, self.first_pop = set(), 0, None, None
        self.heights, self.occ, self.trace = [], {}, []
        self.det_period = 1.0 / a.perception_hz
        self.fsm_period = 1.0 / a.fsm_hz if a.fsm_hz > 0 else self.fsm.nominal_dt
        self.last_cmd = ({"surge": 0.0, "heave": 0.0, "yaw": 0.0}, (0.0, 0.0, 0.0))
        self.frames, self.next_frame = [], 0.0
        self.entries, self._inside = [], {}

    def after_cycle(self, dt, req, reply):
        a, sim, ros = self.args, self.sim, self.ros
        self.t += dt
        t = self.t
        fixed = _REPLY_FIXED.unpack_from(reply, _REPLY_HEAD.size)
        gyro_imu = fixed[4:7]
        if t >= self.next_arm:  # core's AUTO keeps the runnable flag up
            ros.publish_runnable(True)
            self.next_arm += 0.5
        # pop / wire / height (competition_eval, every physics return instead of every 20 ms)
        d = sim.data
        pin_tip = d.site_xpos[self.pin_sid].copy()
        axis = d.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0, 0])
        vel = (pin_tip - self.prev_pin) / dt if self.prev_pin is not None else np.zeros(3)
        self.prev_pin = pin_tip
        r_pop = scn.BALLOON_RADIUS + scn.POP_MARGIN
        for b in self.balloons:  # pop-gate values when the pin tip ENTERS a balloon (why no pop)
            if b["name"] in self.popped:
                continue
            delta = b["pos"] - pin_tip
            dist = float(np.linalg.norm(delta))
            inside = dist < r_pop
            if inside and not self._inside.get(b["name"], False):
                u = delta / max(dist, 1e-6)
                self.entries.append(
                    (
                        b["name"],
                        round(t, 2),
                        round(float(np.dot(vel, u)), 3),
                        round(math.degrees(math.acos(float(np.clip(np.dot(axis, u), -1, 1)))), 1),
                        self.fsm.b.state,
                    )
                )
            self._inside[b["name"]] = inside
        for b in self.balloons:
            if b["name"] not in self.popped and scn.popped(
                pin_tip, b["pos"], axis, vel, min_speed=a.min_pop_speed, angle_tol_deg=a.pop_angle_tol
            ):
                self.popped.add(b["name"])
                self.score += b["points"]
                if self.first_pop is None:
                    self.first_pop = t
                if a.verbose:
                    print(f"    t={t:6.1f}s POP {b['name']} (state {self.fsm.b.state})", flush=True)
        snag = set(scn.entanglement(d.xpos[sim.base_id], self.balloons, self.popped))
        self.wire += len(snag - self.snag_prev)
        self.snag_prev = snag
        self.heights.append(float(d.xpos[sim.base_id][1]))
        # detector at perception_hz, delivered after the latency
        if t >= self.next_det - 1e-9:
            self.next_det += self.det_period
            dets = detector_frame(
                sim, self.cam_id, self.balloons, self.popped, self.phantoms, self.rng, a, self.recall_curve
            )
            self.pending.append((t + a.perception_latency, to_msg(dets)))
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.fsm.on_detections(self.pending.pop(0)[1], t)
        # FSM at the auto_target_generator rate
        if t >= self.next_fsm - 1e-9:
            self.next_fsm += self.fsm_period
            cmd, (vx, vz, yaw_rate) = self.fsm.tick(t, gyro_imu[2])
            ros.publish_setpoint(vx, vz, yaw_rate)
            st = self.fsm.b.state
            self.occ[st] = self.occ.get(st, 0) + 1
            R = d.xmat[sim.base_id].reshape(3, 3)
            fwd = R @ np.array([1.0, 0, 0])
            trk = self.fsm.b.trk
            self.trace.append(
                [
                    t,
                    *d.xpos[sim.base_id],
                    math.atan2(-fwd[2], fwd[0]),
                    gyro_imu[2],
                    cmd["surge"],
                    cmd["heave"],
                    cmd["yaw"],
                    vx,
                    vz,
                    yaw_rate,
                    trk.az if trk.colour else np.nan,
                    trk.el if trk.colour else np.nan,
                    len(self.fsm.dets),
                    *req[4:8],
                    *req[0:4],
                    STATES.index(st) if st in STATES else -1,
                ]
            )
        if self.record_video and t >= self.next_frame - 1e-9:  # snapshot only; drawn after the episode
            self.next_frame += 1.0 / a.video_fps
            self.frames.append(
                (t, d.qpos.copy(), self.fsm.b.state, self.score, sorted(self.popped), list(self.fsm.dets))
            )
        if self.positive <= self.popped:
            self.t_clear = t
            return False
        return t < a.minutes * 60.0

    def result(self):
        b = self.fsm.b
        h = np.asarray(self.heights)
        return {
            "cleared": self.t_clear is not None,
            "t_clear": self.t_clear,
            "score": self.score,
            "blue_popped": sum(1 for x in self.balloons if x["name"] in self.popped and x["points"] < 0),
            "wire": self.wire,
            "n_positive": len(self.positive),
            "n_pos_popped": len(self.positive & self.popped),
            "first_pop": self.first_pop,
            "n_red": sum(1 for x in self.balloons if x["name"] in self.popped and x["colour"] == "red"),
            "surface_frac": float(np.mean(h > scn.POOL_DEPTH - SURFACE_BAND)),
            "floor_frac": float(np.mean(h < scn.FLOOR_Y + SURFACE_BAND)),
            "h_min": float(h.min()),
            "h_max": float(h.max()),
            "occ": self.occ,
            "n_ram": b.n_ram,
            "n_miss": b.n_miss,
            "n_recover": b.n_recover,
            "n_abandon": b.n_abandon,
            # pin tip entering a balloon's pop sphere: (name, t, closing [m/s], pin-axis angle [deg], FSM state)
            "entries": self.entries,
        }


STATES = ("SEARCH", "APPROACH", "ALIGN", "RAM", "RECOVER", "CONFIRM", "AVOID", "STANDBY")
TRACE_COLS = (
    "t,x,y,z,heading,gyro_z_imu,fsm_surge,fsm_heave,fsm_yaw,vx,vz,yaw_rate,trk_az,trk_el,n_dets,"
    "duty_lf,duty_lb,duty_rb,duty_rf,servo_lf,servo_lb,servo_rb,servo_rf,state"
)


class Ros:
    """One rclpy node for the whole run: /clock, /robot_description, the AUTO setpoint topics."""

    def __init__(self, args):
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sinsei_umiusi_msgs.msg import AttitudeTarget, Target, ThrusterRunnableAll
        from std_msgs.msg import String
        from ament_index_python.packages import get_package_share_directory

        self.rclpy = rclpy
        self.Clock, self.Target, self.TRA = Clock, Target, ThrusterRunnableAll
        from umiusi_autonomy.auto_target_generator import neutral_attitude_target

        self.neutral = neutral_attitude_target
        self.control_share = get_package_share_directory("sinsei_umiusi_control")
        self.bridge_share = get_package_share_directory("umiusi_sim_bridge")
        gp = yaml.safe_load(Path(args.competition_yaml).read_text())["auto_target_generator"]["ros__parameters"]
        if args.surge_sign is not None:
            gp["surge_sign"] = args.surge_sign
        if args.yaw_rate_scale is not None:
            gp["yaw_rate_scale"] = args.yaw_rate_scale
        self.gen_params = gp
        rclpy.init()
        self.node = rclpy.create_node("competition_ros")
        clock_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", clock_qos)
        # auto_target_generator publishes with the default depth-10 (reliable, volatile) profile
        self.pub_target = self.node.create_publisher(Target, gp.get("target_topic", "/cmd/target"), 10)
        self.pub_att = self.node.create_publisher(
            AttitudeTarget, gp.get("attitude_target_topic", "/cmd/attitude_target"), 10
        )
        self.pub_run = self.node.create_publisher(ThrusterRunnableAll, "/cmd/thruster_runnable_all", 10)
        urdf_pub = self.node.create_publisher(
            String,
            "/robot_description",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        urdf_pub.publish(String(data=xacro_urdf(self.bridge_share)))
        self.t_clock = 0.0

    def publish_clock(self, t):
        self.t_clock = t
        self.clock_pub.publish(self.Clock(clock=self.rclpy.time.Time(nanoseconds=int(round(t * 1e9))).to_msg()))

    def publish_runnable(self, on):
        m = self.TRA()
        for k in ("lf", "lb", "rb", "rf"):
            getattr(m, k).esc = bool(on)
            getattr(m, k).servo = bool(on)
        self.pub_run.publish(m)

    def publish_setpoint(self, vx, vz, yaw_rate):
        msg = self.Target()
        msg.velocity.x = vx
        msg.velocity.z = vz
        self.pub_target.publish(msg)
        att = self.neutral()
        att.header.stamp = self.rclpy.time.Time(nanoseconds=int(round(self.t_clock * 1e9))).to_msg()
        att.yaw_rate = yaw_rate
        self.pub_att.publish(att)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def run_episode(ep, args, ros, params_file):
    """Launch the control stack on a fresh socket, spawn controllers, then serve the episode in lockstep."""
    tmp = Path(tempfile.mkdtemp(prefix="cros_"))
    sock_path = str(tmp / "sim.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    srv.settimeout(1.0)
    env = dict(os.environ, UMIUSI_SIM_SOCK=sock_path, ROS_DOMAIN_ID=str(args.domain))
    logf = open(tmp / "control.log", "w")
    nice = ["nice", "-n", str(args.nice)]
    cm = subprocess.Popen(
        nice
        + ["ros2", "run", "controller_manager", "ros2_control_node", "--ros-args", "--params-file", str(params_file)],
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    spawner = subprocess.Popen(
        nice
        + [
            "ros2",
            "run",
            "controller_manager",
            "spawner",
            *CONTROLLERS,
            "--param-file",
            str(params_file),
            "--controller-manager-timeout",
            "120",
        ],
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    procs = [cm, spawner]
    # phase A: wall-paced /clock while the controllers spawn (nothing armed; plant reset at the jump)
    t_a = [ros.t_clock + 10.0]
    lockstep = threading.Event()

    def clock_thread():
        while not lockstep.is_set():
            ros.publish_clock(t_a[0])
            t_a[0] += 0.01
            time.sleep(0.01)

    th = threading.Thread(target=clock_thread, daemon=True)
    th.start()
    conn = None
    stats = {}
    try:
        t_deadline = time.time() + 120
        while conn is None:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if cm.poll() is not None or time.time() > t_deadline:
                    raise RuntimeError(f"controller_manager died / no bridge connection; see {tmp}/control.log")
        reader = FrameReader(conn)
        while spawner.poll() is None:
            payload = reader.next()
            if payload is None:
                raise RuntimeError("bridge disconnected during startup")
            ep.serve(conn, payload, _REQ.unpack(payload)[16])
        if spawner.returncode != 0:
            raise RuntimeError(f"spawner failed (rc {spawner.returncode}); see {tmp}/control.log")
        for _ in range(50):
            payload = reader.next()
            ep.serve(conn, payload, _REQ.unpack(payload)[16])
        lockstep.set()
        th.join()

        # phase B: lockstep (closed_loop_replay.run: jump cycle pins t0, then one request per tick)
        dt = args.control_dt
        t0 = ros.t_clock + 1000.0
        t_sim = t_pub = t0
        ros.publish_clock(t_sim)
        jumped, retries, stall, periods = False, 0, 0, []
        conn.settimeout(args.tick_timeout)
        wall0 = time.perf_counter()
        running = True
        while running:
            while True:
                try:
                    payload = reader.next()
                    break
                except socket.timeout:
                    retries += 1
                    stall += 1
                    if stall > 2000:
                        print(f"[ros] ABORT: controller_manager stopped cycling at t={t_sim - t0:.1f} s")
                        payload = None
                        break
                    t_pub += dt
                    ros.publish_clock(t_pub)
            if payload is None:
                break
            stall = 0
            period = _REQ.unpack(payload)[16]
            if period > 0.1:  # the jump: CM time == t0; start the episode here
                jumped = True
                t_sim = t_pub
                ep.reset_plant()
                ep.begin()
                f, reply = ep.serve(conn, payload, dt)
                periods.append(dt)
                running = ep.after_cycle(dt, f, reply)
            elif not jumped:  # straggling phase-A cycle
                ep.serve(conn, payload, dt)
                continue
            else:
                f, reply = ep.serve(conn, payload, period)
                t_sim += period
                periods.append(period)
                running = ep.after_cycle(period, f, reply)
            time.sleep(args.tick_delay)
            t_pub = t_sim + dt
            ros.publish_clock(t_pub)
        wall = time.perf_counter() - wall0
        periods = np.array(periods) if periods else np.array([np.nan])
        stats = dict(
            sim_s=float(t_sim - t0),
            wall_s=wall,
            rt=float((t_sim - t0) / max(wall, 1e-9)),
            period_ms=float(np.nanmean(periods) * 1e3),
            period_max_ms=float(np.nanmax(periods) * 1e3),
            retries=retries,
            log=str(tmp / "control.log"),
        )
        ros.publish_runnable(False)
        ros.publish_setpoint(0.0, 0.0, 0.0)
    finally:
        lockstep.set()
        if conn is not None:
            conn.close()
        srv.close()
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        logf.close()
    return stats


def render_video(ep, path, fps, size=(360, 480)):
    """Draw the recorded snapshots OUTSIDE the lockstep: onboard camera (degraded, with the detections
    the FSM held, state / time / score banner — tools/autonomy_run.annotate_onboard) | third-person
    "track" camera. Uses autonomy_run's drawing only; its control path (old feed-forward) is not used."""
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    from tools.autonomy_run import annotate_onboard, build_perception_model

    xml = Path(tempfile.mkdtemp(prefix="cros_vid_")) / "scene_vis.xml"
    xml.write_text(build_perception_model(ep.layout).to_xml())
    vis = UmiusiSimulator(model_path=xml)  # onboard look: the detector's training appearance
    if vis.model.nq != ep.sim.model.nq:  # appearance changed the joint layout: fall back to the physics scene
        vis = ep.sim
    # third person from the plain physics scene (the appearance's pool box/walls hide the "track" camera)
    tp_sim = ep.sim
    H, W = size
    third = mujoco.Renderer(tp_sim.model, height=H, width=W)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=8)
    hidden = set()
    n_bal = len(ep.balloons)
    try:
        for t, qpos, state, score, popped, dets in ep.frames:
            for m in {id(vis): vis, id(tp_sim): tp_sim}.values():
                m.data.qpos[:] = qpos
                m.data.qvel[:] = 0.0
                mujoco.mj_forward(m.model, m.data)
            for name in popped:
                if name not in hidden:
                    scn.hide_balloon(vis.model, name)
                    if tp_sim is not vis:
                        scn.hide_balloon(tp_sim.model, name)
                    hidden.add(name)
            cam = vis.render_camera("front_cam", width=320, height=240, degrade=True)
            cam = annotate_onboard(cam, dets, state, score, len(popped), n_bal, t)
            cam = np.asarray(Image.fromarray(cam).resize((W, H), Image.BILINEAR))
            third.update_scene(tp_sim.data, camera="track")
            tp = Image.fromarray(third.render())
            dr = ImageDraw.Draw(tp)
            dr.rectangle([0, 0, W, 13], fill=(0, 0, 0))
            dr.text(
                (3, 2),
                f"ROS AUTO path (FSM -> control -> sim)  t={t:5.1f}s  {state}  score {score:+d}",
                fill=(255, 255, 255),
            )
            writer.append_data(np.concatenate([cam, np.asarray(tp)], axis=1))
    finally:
        writer.close()
        third.close()
    print(f"  wrote {path} ({len(ep.frames)} frames @ {fps} fps)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", choices=("field", "single"), default="field")
    ap.add_argument("--single-colour", default="red", choices=("red", "yellow", "blue"))
    ap.add_argument("--single-range", default="1.5,4.0", help="min,max distance of the single balloon [m]")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="ep seed = 1000 + seed + e (as competition_eval)")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--param", action="append", default=[], help="control yaml override node.dotted=value")
    ap.add_argument("--fsm", action="append", default=[], help="fsm.* override name=value (competition.yaml units)")
    ap.add_argument(
        "--fsm-defaults",
        action="store_true",
        help="ignore competition.yaml fsm.* (behavior.py defaults, as competition_eval runs)",
    )
    ap.add_argument("--competition-yaml", default=str(COMPETITION_YAML))
    ap.add_argument("--surge-sign", type=float, default=None, help="override auto_target_generator surge_sign")
    ap.add_argument("--yaw-rate-scale", type=float, default=None, help="override auto_target_generator yaw_rate_scale")
    ap.add_argument(
        "--fsm-hz",
        type=float,
        default=0.0,
        help="FSM tick rate (0 = competition.yaml control_hz, 50). Robot 10/03 AUTO measured 26 Hz",
    )
    ap.add_argument("--perception-hz", type=float, default=5.0, help="detector rate (robot full stack ~5 Hz)")
    ap.add_argument("--perception-latency", type=float, default=0.0, help="detection age on arrival [s]")
    ap.add_argument("--bearing-noise-deg", type=float, default=0.0)
    ap.add_argument("--range-noise", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fp-rate", type=float, default=0.0)
    ap.add_argument("--fp-per-frame", type=float, default=0.0)
    ap.add_argument("--recall-curve", default=None)
    ap.add_argument("--phantoms", type=int, default=0)
    ap.add_argument("--phantom-p", type=float, default=0.7)
    ap.add_argument("--phantom-colour", default="red")
    ap.add_argument("--min-pop-speed", type=float, default=scn.MIN_POP_SPEED)
    ap.add_argument("--pop-angle-tol", type=float, default=scn.POP_ANGLE_TOL_DEG)
    ap.add_argument("--start-height", type=float, default=START[1])
    ap.add_argument("--plant-max-duty", type=float, default=1.0, help="plant duty cap (control clamps at 0.5)")
    ap.add_argument(
        "--plant",
        action="append",
        default=[],
        help="plant knob k=v: vertical_eff=<0..1> (configs: 1.0) | net_buoy=<N> (configs: +1.14)",
    )
    ap.add_argument("--control-dt", type=float, default=0.01)
    ap.add_argument("--tick-timeout", type=float, default=0.005)
    ap.add_argument("--tick-delay", type=float, default=0.002)
    ap.add_argument("--domain", type=int, default=78, help="ROS_DOMAIN_ID (closed_loop_replay uses 77)")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--out", default=None, help="directory for per-episode trace npz + summary json")
    ap.add_argument(
        "--video",
        default=None,
        help="mp4 of one episode (onboard cam + third person, drawn afterwards; needs MUJOCO_GL=egl headless)",
    )
    ap.add_argument("--video-episode", type=int, default=0, help="which episode --video records")
    ap.add_argument("--video-fps", type=float, default=12.5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.single_range = tuple(float(v) for v in args.single_range.split(","))
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    ros = Ros(args)
    tmp = Path(tempfile.mkdtemp(prefix="cros_params_"))
    params_file = build_param_file(ros.control_share, args.param, tmp / "params.yaml")
    xml_path = tmp / "scene.xml"
    args._xml_path = xml_path
    gp = ros.gen_params
    print(
        f"competition_ros: {args.episodes} x {args.layout} ({args.single_colour if args.layout == 'single' else ''})"
        f" {args.minutes} min  control={ros.control_share}  params={args.param}"
    )
    print(
        f"  auto_target_generator: surge_sign={gp['surge_sign']} yaw_rate_scale={gp['yaw_rate_scale']} "
        f"control_hz={gp['control_hz']} fsm_hz={args.fsm_hz or gp['control_hz']} "
        f"fsm={'behavior.py defaults' if args.fsm_defaults else args.competition_yaml} overrides={args.fsm}"
    )
    print(
        f"  perception_hz={args.perception_hz} latency={args.perception_latency} dropout={args.dropout} "
        f"fp_rate={args.fp_rate} min_pop_speed={args.min_pop_speed} plant_max_duty={args.plant_max_duty} "
        f"plant={args.plant}"
    )
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for e in range(args.episodes):
            seed = 1000 + args.seed + e
            ep = Episode(np.random.default_rng(seed), args, ros, xml_path, seed)
            if args.verbose:
                print(
                    f"  ep {e} seed {seed}: layout "
                    + ", ".join(f"{n}@({x:+.2f},{z:+.2f})" for n, _, x, z in ep.layout),
                    flush=True,
                )
            ep.record_video = bool(args.video) and e == args.video_episode
            stats = run_episode(ep, args, ros, params_file)
            r = ep.result()
            r["stats"] = stats
            rows.append(r)
            tc = f"{r['t_clear']:.1f}s" if r["cleared"] else "TIMEOUT"
            fp = f"{r['first_pop']:.1f}s" if r["first_pop"] is not None else "-"
            print(
                f"  ep {e:2d}: {'CLEAR' if r['cleared'] else 'fail '} {tc:>8s} first {fp:>7s} "
                f"pos {r['n_pos_popped']}/{r['n_positive']} score {r['score']:+d} blue {r['blue_popped']} "
                f"wire {r['wire']} h[{r['h_min']:.2f},{r['h_max']:.2f}] surf {r['surface_frac']:.0%} "
                f"floor {r['floor_frac']:.0%} | ram {r['n_ram']} miss {r['n_miss']} | "
                f"x{stats.get('rt', float('nan')):.2f} rt, period {stats.get('period_ms', float('nan')):.2f} ms",
                flush=True,
            )
            if args.verbose and r["entries"]:
                print(
                    f"        pin-tip entries (gate: closing >= {args.min_pop_speed} m/s, angle <= "
                    f"{args.pop_angle_tol} deg): "
                    + "; ".join(
                        f"{n.replace('balloon_', '')}@{tt:.1f}s {c:+.2f}m/s {a:.0f}deg {st}"
                        for n, tt, c, a, st in r["entries"][:12]
                    ),
                    flush=True,
                )
            if out:
                np.savez(
                    out / f"ep{e:02d}_seed{seed}.npz",
                    trace=np.array(ep.trace, float),
                    cols=TRACE_COLS,
                    layout=json.dumps([(n, c, float(x), float(z)) for n, c, x, z in ep.layout]),
                    balloons=json.dumps([{**b, "pos": [float(v) for v in b["pos"]]} for b in ep.balloons]),
                )
            if ep.record_video:
                render_video(ep, args.video, args.video_fps)
    finally:
        ros.close()
    if rows:
        summarize(rows, args)
        if out:
            (out / "summary.json").write_text(
                json.dumps(dict(args={k: v for k, v in vars(args).items()}, rows=rows), indent=1, default=str)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
