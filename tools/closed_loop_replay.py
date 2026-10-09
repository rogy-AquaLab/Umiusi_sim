"""Closed-loop replay of a pool-test rosbag through the REAL control stack + the sim plant.

The recorded operator / autonomy inputs (``/cmd/attitude_target``, ``/cmd/target``,
``/cmd/thruster_runnable_all``) are published at their recorded times into an unmodified
``sinsei_umiusi_control`` stack (ros2_control_node + gate / attitude / thruster x4) whose
hardware plugin is ``umiusi_sim_bridge/MujocoSystem``; the plugin's socket peer is THIS process,
which steps ``UmiusiSimulator`` and records what the controllers commanded (duty / servo angle)
and what the plant did (IMU quaternion / gyro). The result is compared with the bag.

Why this is its own server instead of ``tools/sim_server.py``:
  * lockstep time: the controller_manager runs with ``use_sim_time`` and this process publishes
    ``/clock`` AFTER each control cycle has been served, so one request == one 10 ms tick no
    matter how loaded the CPU is (no real-time requirement, bit-reproducible);
  * frame: the controllers assume the BNO055 frame (REP-103, z up); the sim body/world is the
    CAD frame (+Y up, +Z starboard). State goes out as R_imu = P^T R_cad P, w_imu = P^T w_cad
    (P = tools.bag_replay.FRAME_P), initial attitude comes in the other way;
  * unit: the ``thrusterN/servo/angle`` command interface carries RADIANS (the mixer clamps to
    +/- pi/2 and VescModel::make_servo_angle_frame takes rad). The bridge/sim_server label it
    degrees and call np.radians() on it — see --servo-unit;
  * plant duty cap: the stock plant clips |duty| at configs max_duty (0.25); the real ESC only
    sees the thruster_controller's max_duty (0.5), so the plant cap is lifted (--plant-max-duty);
  * segment re-initialisation: every --segment seconds the plant pose/velocity is reset to the
    bag's IMU sample (controllers keep their state) so closed-loop drift does not accumulate.

Usage (from the umiusi_sim repo; source /opt/ros/jazzy + ros2_ws/install first, so rclpy and
the sinsei_umiusi_* packages resolve inside the uv venv):

    nice -n 10 uv run python -m tools.closed_loop_replay run \
        --bag ../data/20261003-pool/rosbag2_2026_10_03-13_45_43 \
        --npz ../data/20261003-pool/npz/rosbag2_2026_10_03-13_45_43.npz \
        --start 40 --end 90 --segment 15 --out out/closed_loop/manual_1345_after.npz \
        [--param attitude_controller.feedback.kp_roll=1.0 ...]

    uv run python -m tools.closed_loop_replay compare --sim out/closed_loop/x.npz --npz ...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from tools.bag_replay import FRAME_P, POS
from tools.sim_server import _REPLY_FIXED, _REPLY_HEAD, _REQ, SimServer, _send_msg
from umiusi_sim.simulator import UmiusiSimulator

CMD_TOPICS = ("/cmd/attitude_target", "/cmd/target", "/cmd/thruster_runnable_all")
IMU_TOPIC = "/state/imu"
THR_TOPIC = "/state/thruster_state_all"
CONTROLLERS = ("thruster_controller_lf", "thruster_controller_lb", "thruster_controller_rb",
               "thruster_controller_rf", "attitude_controller", "gate_controller")


# -- frame helpers ----------------------------------------------------------------------------
def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def mat_to_quat(M):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(M, dtype=float).flatten())
    return q


def imu_to_cad(quat_imu, gyro_imu, P=FRAME_P):
    """IMU-frame (REP-103, z up) orientation/body-rate -> sim CAD frame (+Y up)."""
    return mat_to_quat(P @ quat_to_mat(quat_imu) @ P.T), P @ np.asarray(gyro_imu, float)


def cad_to_imu(quat_cad, vec_cad, P=FRAME_P):
    return mat_to_quat(P.T @ quat_to_mat(quat_cad) @ P), P.T @ np.asarray(vec_cad, float)


def up_world(quat):
    """Body +z expressed in the world frame (what AttitudeFeedback::body_up_in_world uses)."""
    return quat_to_mat(quat)[:, 2]


# -- bag reading ------------------------------------------------------------------------------
def read_bag(bag_dir):
    """Return {topic: [(t_sec, raw_bytes, msg)]} for the cmd topics and the IMU state."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Imu
    from sinsei_umiusi_msgs.msg import AttitudeTarget, Target, ThrusterRunnableAll

    types = {"/cmd/attitude_target": AttitudeTarget, "/cmd/target": Target,
             "/cmd/thruster_runnable_all": ThrusterRunnableAll, IMU_TOPIC: Imu}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = {k: [] for k in types}
    t0 = None
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if t0 is None:
            t0 = ts
        if topic in types:
            try:
                msg = deserialize_message(data, types[topic])
            except Exception:
                # Bags before sinsei_umiusi_msgs dcd3642 (hold_yaw added, 2026-10-03 morning) carry a
                # 60-byte AttitudeTarget; one trailing zero byte makes it the current layout
                # (hold_yaw = False). Re-serialize so the republished bytes match the current type.
                if types[topic] is not AttitudeTarget:
                    raise
                msg = deserialize_message(bytes(data) + b"\x00", AttitudeTarget)
                data = serialize_message(msg)
            out[topic].append((ts * 1e-9, data, msg))
    return out, t0 * 1e-9


class FrameReader:
    """Length-prefixed frames from a stream socket, keeping partial data across recv timeouts.

    sim_server._recv_exactly discards what it has buffered when recv() times out; the bridge
    sends the uint32 length and the 80-byte payload as two send() calls, so a timeout between
    them desynchronised the stream and both sides waited forever (seen at ~80 s of replay).
    """

    def __init__(self, conn):
        self.conn = conn
        self.buf = bytearray()

    def next(self):
        """One payload, None on EOF; raises socket.timeout if no complete frame arrived in time."""
        while True:
            if len(self.buf) >= 4:
                (n,) = struct.unpack_from("<I", self.buf, 0)
                if len(self.buf) >= 4 + n:
                    payload = bytes(self.buf[4:4 + n])
                    del self.buf[:4 + n]
                    return payload
            chunk = self.conn.recv(65536)
            if not chunk:
                return None
            self.buf.extend(chunk)


# -- the lockstep sim server ------------------------------------------------------------------
class ReplayServer(SimServer):
    def __init__(self, servo_unit="rad", plant_max_duty=1.0, thrust_scale=1.0, plant=()):
        super().__init__(UmiusiSimulator())
        self.servo_unit = servo_unit
        self.sim.max_duty = float(plant_max_duty)
        self.sim.thrust_per_cmd *= float(thrust_scale)
        # --plant knobs from the 10/03 roll/pitch calibration (tools/bag_replay grid):
        #   vertical_eff=0.25  rp_drag=4  rp_inertia=3   (roll/pitch = CAD x/z; yaw = y untouched)
        for kv in plant:
            k, v = kv.split("=", 1)
            v = float(v)
            if k == "vertical_eff":
                self.sim.thrust_vertical_eff = v
            elif k == "rp_drag":
                for i in (3, 5):
                    self.sim.drag_lin[i] *= v
                    self.sim.drag_quad[i] *= v
            elif k == "rp_inertia":
                for i in (3, 5):
                    self.sim.added_mass_diag[i] *= v
            else:
                raise SystemExit(f"unknown --plant key {k}")
        self.rec = []          # per served cycle: [t_sim, dt, wall, servo_rad*4, duty*4, allowed*4, quat_imu*4, gyro_imu*3]
        self.t_sim = 0.0
        self.recording = False

    def reinit_pose(self, quat_imu, gyro_imu):
        """Reset the base pose/velocity to a bag IMU sample; keep servo/ESC state (controllers keep theirs)."""
        q_cad, w_cad = imu_to_cad(quat_imu / np.linalg.norm(quat_imu), gyro_imu)
        d = self.sim.data
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = q_cad
        d.qvel[0:3] = 0.0           # [assumed] translational velocity unknown from the bag
        d.qvel[3:6] = w_cad         # free-joint angular velocity is body-frame in MuJoCo
        self.sim.prev_vel_body[:] = 0.0
        self._have_prev = False
        mujoco.mj_forward(self.sim.model, d)

    def step_command(self, servo_in, esc_duty, servo_allowed, esc_allowed, dt):
        sim = self.sim
        srange = sim.servo_range_rad
        servo_rad = np.array([np.radians(v) if self.servo_unit == "deg" else v for v in servo_in], float)
        action = np.zeros(8)
        for k in range(4):
            if servo_allowed[k]:
                action[k] = float(np.clip(servo_rad[k], -srange, srange)) / srange
            action[4 + k] = float(np.clip(esc_duty[k], -1.0, 1.0)) if esc_allowed[k] else 0.0
        ctrl_dt = max(1e-6, float(dt))
        sim.substeps = int(min(100, max(1, round(ctrl_dt / sim.dt))))
        t_wall = time.perf_counter()
        sim.step(action)
        reply, quat_imu, gyro_imu = self._encode_state_imu(ctrl_dt)
        if self.recording:
            self.rec.append(np.concatenate([[self.t_sim, ctrl_dt, time.perf_counter() - t_wall],
                                            servo_rad, np.asarray(esc_duty, float),
                                            [float(a and b) for a, b in zip(servo_allowed, esc_allowed)],
                                            quat_imu, gyro_imu]))
        return reply

    def _encode_state_imu(self, ctrl_dt):
        sim = self.sim
        d = sim.data
        base = sim.base_id
        quat_cad = d.xquat[base].copy()
        R = d.xmat[base].reshape(3, 3)
        # Body-frame angular velocity = R^T * world rate (as tools/bag_replay.py does). NOTE:
        # mj_objectVelocity(mjOBJ_BODY, flg_local=1) — what sim_server.py sends — is expressed in
        # the body's INERTIAL frame (ximat), which for this hull is a permutation of the body
        # axes (checked 2026-10-07: all-horizontal thrust yawed about "x"). Not used here.
        vglob = np.zeros(6)
        mujoco.mj_objectVelocity(sim.model, d, mujoco.mjtObj.mjOBJ_BODY, base, vglob, 0)
        gyro_cad = R.T @ vglob[:3]
        lin_world = d.subtree_linvel[base].copy()
        acc_world = (lin_world - self._prev_lin_world) / ctrl_dt if self._have_prev else np.zeros(3)
        accel_cad = R.T @ acc_world - R.T @ sim.gravity
        self._prev_lin_world = lin_world
        self._have_prev = True
        quat_imu, gyro_imu = cad_to_imu(quat_cad, gyro_cad)
        accel_imu = FRAME_P.T @ accel_cad
        servo = np.array([d.qpos[a] for a in sim.servo_qadr], dtype=float)
        esc_rpm = sim.esc_current * 1000.0
        qpos = np.asarray(d.qpos, dtype=float)
        fixed = _REPLY_FIXED.pack(*quat_imu, *gyro_imu, *accel_imu, *servo, *esc_rpm)
        return _REPLY_HEAD.pack(qpos.size) + fixed + qpos.astype("<f8").tobytes(), quat_imu, gyro_imu


# -- ROS side ----------------------------------------------------------------------------------
def build_param_file(control_share, overrides, out_path):
    """controllers.yaml + use_sim_time + overrides -> one params file (CM + spawner)."""
    src = yaml.safe_load(Path(control_share, "params", "controllers.yaml").read_text())
    tree = src["/**"]
    for node_name, node in tree.items():
        node.setdefault("ros__parameters", {})["use_sim_time"] = True
    for item in overrides:
        key, _, raw = item.partition("=")
        node_name, _, dotted = key.partition(".")
        val = yaml.safe_load(raw)
        cur = tree.setdefault(node_name, {}).setdefault("ros__parameters", {})
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    Path(out_path).write_text(yaml.safe_dump(src, allow_unicode=True, sort_keys=False))
    return out_path


def xacro_urdf(bridge_share):
    urdf = Path(bridge_share, "urdf", "umiusi_sim.urdf.xacro")
    model = Path(bridge_share, "model", "umiusi.xml")
    return subprocess.check_output(["xacro", str(urdf), f"model_path:={model}"], text=True)


def latest_before(entries, t):
    """Last (t_i, ...) entry with t_i <= t, or None."""
    lo, hi = 0, len(entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return entries[lo - 1] if lo > 0 else None


def run(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rosgraph_msgs.msg import Clock
    from sinsei_umiusi_msgs.msg import AttitudeTarget, Target, ThrusterRunnableAll
    from ament_index_python.packages import get_package_share_directory

    os.environ.setdefault("ROS_DOMAIN_ID", str(args.domain))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # so `timeout` still runs the cleanup
    control_share = get_package_share_directory("sinsei_umiusi_control")
    bridge_share = get_package_share_directory("umiusi_sim_bridge")
    print(f"[replay] control share: {control_share}")

    bag, t_bag0 = read_bag(args.bag)
    imu = bag[IMU_TOPIC]
    t_start = t_bag0 + args.start
    t_end = t_bag0 + args.end if args.end is not None else imu[-1][0]
    arm_times = [(e[0] - t_bag0) for e in bag["/cmd/thruster_runnable_all"] if e[2].lf.esc]
    print(f"[replay] bag {Path(args.bag).name}: {len(imu)} imu, "
          f"{len(bag['/cmd/attitude_target'])} attitude_target, {len(bag['/cmd/target'])} target; "
          f"first esc-runnable at {arm_times[0]:.1f} s" if arm_times else "[replay] never armed")
    print(f"[replay] replaying bag time {args.start:.1f} .. {t_end - t_bag0:.1f} s, segment {args.segment} s")

    # -- files / sockets ------------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="clr_"))
    sock_path = str(tmp / "sim.sock")
    params_file = build_param_file(control_share, args.param, tmp / "params.yaml")
    server = ReplayServer(args.servo_unit, args.plant_max_duty, args.thrust_scale, args.plant)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    srv.settimeout(1.0)

    # -- ROS node + publishers ---------------------------------------------------------------
    rclpy.init()
    node = rclpy.create_node("closed_loop_replay")
    clock_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    clock_pub = node.create_publisher(Clock, "/clock", clock_qos)
    cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    pubs = {"/cmd/attitude_target": node.create_publisher(AttitudeTarget, "/cmd/attitude_target", cmd_qos),
            "/cmd/target": node.create_publisher(Target, "/cmd/target", cmd_qos),
            "/cmd/thruster_runnable_all": node.create_publisher(ThrusterRunnableAll, "/cmd/thruster_runnable_all", cmd_qos)}

    # The CM (Jazzy) takes the URDF from the latched /robot_description topic.
    from std_msgs.msg import String
    urdf_pub = node.create_publisher(String, "/robot_description",
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                durability=DurabilityPolicy.TRANSIENT_LOCAL))
    urdf_pub.publish(String(data=xacro_urdf(bridge_share)))

    def publish_clock(t):
        clock_pub.publish(Clock(clock=rclpy.time.Time(nanoseconds=int(round(t * 1e9))).to_msg()))

    # -- launch the real control stack ----------------------------------------------------
    env = dict(os.environ, UMIUSI_SIM_SOCK=sock_path, ROS_DOMAIN_ID=str(args.domain))
    logf = open(tmp / "control.log", "w")
    nice = ["nice", "-n", str(args.nice)]
    cm = subprocess.Popen(nice + ["ros2", "run", "controller_manager", "ros2_control_node", "--ros-args",
                                  "--params-file", str(params_file)],
                          env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    spawner = subprocess.Popen(nice + ["ros2", "run", "controller_manager", "spawner", *CONTROLLERS,
                                       "--param-file", str(params_file), "--controller-manager-timeout", "120"],
                               env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    procs = [cm, spawner]

    def shutdown():
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

    # Phase A: free-running /clock (wall-paced) so the CM loop runs while controllers spawn.
    t_phase_a = [t_start - 300.0]
    lockstep = threading.Event()

    def clock_thread():
        while not lockstep.is_set():
            publish_clock(t_phase_a[0])
            t_phase_a[0] += 0.01
            time.sleep(0.01)
    th = threading.Thread(target=clock_thread, daemon=True)
    th.start()

    conn = None
    try:
        t_deadline = time.time() + 120
        while conn is None:
            try:
                conn, _ = srv.accept()
                reader = FrameReader(conn)
            except socket.timeout:
                if cm.poll() is not None or time.time() > t_deadline:
                    raise RuntimeError(f"controller_manager died / no bridge connection; see {tmp}/control.log")
        print(f"[replay] bridge connected ({sock_path})")

        # serve phase A cycles until the spawner reports every controller active
        cycles_a = 0
        while spawner.poll() is None:
            payload = reader.next()
            if payload is None:
                raise RuntimeError("bridge disconnected during startup")
            f = _REQ.unpack(payload)
            _send_msg(conn, server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], f[16]))
            cycles_a += 1
        if spawner.returncode != 0:
            raise RuntimeError(f"spawner failed (rc {spawner.returncode}); see {tmp}/control.log")
        for _ in range(50):     # a few more cycles with everything active
            payload = reader.next()
            f = _REQ.unpack(payload)
            _send_msg(conn, server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], f[16]))
        lockstep.set()
        th.join()
        try:   # which control binary actually got loaded (matters when two installs are sourced);
            # `ros2 run` is a python wrapper, the node is its child
            kids = subprocess.run(["pgrep", "-P", str(cm.pid)], capture_output=True, text=True).stdout.split()
            libs = set()
            for pid in kids + [str(cm.pid)]:
                libs |= {ln.split()[-1] for ln in open(f"/proc/{pid}/maps") if "sinsei_umiusi_control_controller" in ln}
            print("[replay] control libs:", *sorted(libs))
        except OSError:
            pass
        print(f"[replay] controllers active after {cycles_a} startup cycles; entering lockstep")

        # Phase B: lockstep. Catch-up (ZOH) of the latest commands before t_start, plant init.
        t_sim = t_start
        cursor ={topic: sum(1 for e in bag[topic] if e[0] <= t_start) for topic in CMD_TOPICS}
        imu0 = latest_before(imu, t_start) or imu[0]
        q0 = np.array([imu0[2].orientation.w, imu0[2].orientation.x, imu0[2].orientation.y, imu0[2].orientation.z])
        w0 = np.array([imu0[2].angular_velocity.x, imu0[2].angular_velocity.y, imu0[2].angular_velocity.z])
        server.reinit_pose(q0, w0)
        server.t_sim = t_sim
        server.recording = True
        next_reinit = t_start + args.segment if args.segment > 0 else float("inf")
        n_reinit = 0
        wall0 = time.perf_counter()
        periods = []
        dt = args.control_dt
        # The clock jumps from the phase-A value to t_start: serve that one cycle (huge measured
        # period, nothing armed yet) before the catch-up commands go out, then tick normally.
        # Lockstep accounting: t_sim == the CM's current_time at the request just served, kept
        # in step with the CM's own measured periods (sum of request dt) so physics time ==
        # controller time even if a cycle spans two ticks. The clock jumps from the phase-A value
        # to t_start; the request whose measured period is that jump (> 0.1 s) pins t_sim =
        # t_start exactly (a straggling phase-A request may arrive first and must not count —
        # being off by one dt here locks every later cycle into a 2*dt period). The CM sleeps
        # until cycle_end + period, so a tick that lands while it is still inside read() does
        # not wake it: wait --tick-delay after the reply, and if no request shows up within the
        # recv timeout, publish the next tick (that cycle then measures 2*dt, physics follows).
        publish_clock(t_sim)
        jumped = False
        conn.settimeout(args.tick_timeout)
        retries = 0
        stall = 0
        t_pub = t_sim
        while t_sim < t_end:
            while True:
                try:
                    payload = reader.next()
                    break
                except socket.timeout:
                    retries += 1
                    stall += 1
                    if stall == 200:    # ~2 s without a request: dump the CM thread states once
                        try:
                            kids = subprocess.run(["pgrep", "-P", str(cm.pid)], capture_output=True, text=True).stdout.split()
                            for pid in kids:
                                for tdir in Path(f"/proc/{pid}/task").iterdir():
                                    st = (tdir / "stat").read_text().split()
                                    print(f"[replay] stall at bag {t_sim - t_bag0:.1f} s: thread {st[1]} state {st[2]} "
                                          f"wchan {(tdir / 'wchan').read_text().strip()}")
                        except OSError as e:
                            print("[replay] stall diag failed:", e)
                    if stall > 2000:
                        print(f"[replay] ABORT: controller_manager stopped cycling at bag {t_sim - t_bag0:.1f} s "
                              f"(20 s of clock retries); keeping the partial record")
                        payload = None
                        break
                    t_pub += dt
                    publish_clock(t_pub)
            if payload is None:
                break
            stall = 0
            f = _REQ.unpack(payload)
            if f[16] > 0.1:                 # the jump cycle: CM time is exactly t_start now
                jumped = True
                p = dt
                t_sim = t_pub               # == t_start unless a retry tick went out first
                _send_msg(conn, server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], p))
                for topic in CMD_TOPICS:    # catch-up: latest command before t_start (ZOH)
                    e = latest_before(bag[topic], t_start)
                    if e is not None:
                        pubs[topic].publish(e[1])
            elif not jumped:                # straggling phase-A cycle: serve, do not advance
                _send_msg(conn, server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], dt))
                continue
            else:
                p = f[16]
                _send_msg(conn, server.step_command(f[0:4], f[4:8], f[8:12], f[12:16], p))
                t_sim = t_sim + p
            periods.append(p)
            server.t_sim = t_sim
            if t_sim >= next_reinit:
                e = latest_before(imu, t_sim)
                q = np.array([e[2].orientation.w, e[2].orientation.x, e[2].orientation.y, e[2].orientation.z])
                w = np.array([e[2].angular_velocity.x, e[2].angular_velocity.y, e[2].angular_velocity.z])
                server.reinit_pose(q, w)
                n_reinit += 1
                next_reinit += args.segment
            for topic in CMD_TOPICS:
                lst, i = bag[topic], cursor[topic]
                while i < len(lst) and lst[i][0] <= t_sim:
                    pubs[topic].publish(lst[i][1])
                    i += 1
                cursor[topic] = i
            time.sleep(args.tick_delay)
            t_pub = t_sim + dt
            publish_clock(t_pub)
        wall = time.perf_counter() - wall0
        periods = np.array(periods)
        print(f"[replay] done: {len(periods)} cycles, sim {t_sim - t_start:.1f} s in {wall:.1f} s wall "
              f"(x{(t_sim - t_start) / wall:.2f} realtime); control period mean {periods.mean() * 1e3:.2f} ms "
              f"min {periods.min() * 1e3:.2f} max {periods.max() * 1e3:.2f}; {n_reinit} plant re-inits; "
              f"{retries} clock retries")
    finally:
        lockstep.set()
        if conn is not None:
            conn.close()
        srv.close()
        shutdown()
        node.destroy_node()
        rclpy.shutdown()

    rec = np.array(server.rec)
    at = bag["/cmd/attitude_target"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=rec[:, 0], dt=rec[:, 1], step_wall=rec[:, 2], servo_rad=rec[:, 3:7], duty=rec[:, 7:11],
             allowed=rec[:, 11:15], quat=rec[:, 15:19], gyro=rec[:, 19:22],
             cmd_t=np.array([e[0] for e in at]), cmd_yaw_rate=np.array([e[2].yaw_rate for e in at]),
             cmd_hold_yaw=np.array([e[2].hold_yaw for e in at]),
             cmd_quat=np.array([[e[2].attitude.w, e[2].attitude.x, e[2].attitude.y, e[2].attitude.z] for e in at]),
             t_bag0=t_bag0, t_start=t_start, t_end=t_end, segment=args.segment,
             meta=json.dumps(dict(bag=str(args.bag), control_share=control_share, params=args.param,
                                  servo_unit=args.servo_unit, plant_max_duty=args.plant_max_duty,
                                  thrust_scale=args.thrust_scale, plant=args.plant, tmp=str(tmp))))
    print(f"[replay] saved {out}  (control log: {tmp}/control.log)")
    if args.npz:
        compare(argparse.Namespace(sim=str(out), npz=args.npz, segment=args.segment, json=None))


# -- comparison --------------------------------------------------------------------------------
def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def zoh(ts, vals, grid):
    idx = np.searchsorted(ts, grid, side="right") - 1
    return vals[np.clip(idx, 0, len(ts) - 1)]


def compare(args):
    s = np.load(args.sim, allow_pickle=False)
    d = np.load(args.npz)
    t_start, t_end = float(s["t_start"]), float(s["t_end"])
    grid = d["cmd_lf_t"]
    m = (grid >= t_start) & (grid <= min(t_end, s["t"][-1]))
    grid = grid[m]
    sim_duty = zoh(s["t"], s["duty"], grid)
    sim_ang = np.degrees(zoh(s["t"], s["servo_rad"], grid))
    sim_gyro = zoh(s["t"], s["gyro"], grid)
    sim_quat = zoh(s["t"], s["quat"], grid)
    bag_duty = np.stack([d[f"cmd_{p}_duty"][m] for p in POS], 1)
    bag_ang = np.stack([d[f"cmd_{p}_angle"][m] for p in POS], 1)
    gi = np.clip(np.searchsorted(d["imu_t"], grid), 0, len(d["imu_t"]) - 1)
    bag_gyro = d["imu_gyro"][gi]
    bag_quat = d["imu_quat"][gi]
    bag_quat = bag_quat / np.linalg.norm(bag_quat, axis=1, keepdims=True)
    cmd_yaw = zoh(s["cmd_t"], s["cmd_yaw_rate"], grid)
    sim_up = np.array([up_world(q) for q in sim_quat])
    bag_up = np.array([up_world(q) for q in bag_quat])
    tilt = np.degrees(np.arccos(np.clip(np.sum(sim_up * bag_up, axis=1), -1, 1)))
    bag_tilt = np.degrees(np.arccos(np.clip(bag_up[:, 2], -1, 1)))
    sim_tilt = np.degrees(np.arccos(np.clip(sim_up[:, 2], -1, 1)))

    seg = float(s["segment"]) if float(s["segment"]) > 0 else (grid[-1] - grid[0])
    rows = []
    edges = np.arange(t_start, grid[-1] + seg, seg)
    for a, b in list(zip(edges[:-1], edges[1:])) + [(t_start, grid[-1] + 1e-9)]:
        k = (grid >= a) & (grid < b)
        if k.sum() < 25:
            continue
        label = f"{a - float(s['t_bag0']):.0f}-{min(b, grid[-1]) - float(s['t_bag0']):.0f}s" if b - a <= seg + 1e-6 else "ALL"
        row = dict(seg=label, n=int(k.sum()))
        # (angle, duty) is a folded representation: (a, d) and (-a, -d) are the same thrust at
        # +/-90 deg, so the physically meaningful comparison is the per-thruster thrust vector
        # horizontal = cos(a) d (surge/sway/yaw), vertical = sin(a) d (heave/roll/pitch).
        sim_h, sim_v = np.cos(np.radians(sim_ang)) * sim_duty, np.sin(np.radians(sim_ang)) * sim_duty
        bag_h, bag_v = np.cos(np.radians(bag_ang)) * bag_duty, np.sin(np.radians(bag_ang)) * bag_duty
        for i, p in enumerate(POS):
            row[f"duty_{p}"] = (_rmse(sim_duty[k, i], bag_duty[k, i]), _corr(sim_duty[k, i], bag_duty[k, i]))
            row[f"ang_{p}"] = (_rmse(sim_ang[k, i], bag_ang[k, i]), _corr(sim_ang[k, i], bag_ang[k, i]))
            row[f"h_{p}"] = (_rmse(sim_h[k, i], bag_h[k, i]), _corr(sim_h[k, i], bag_h[k, i]))
            row[f"v_{p}"] = (_rmse(sim_v[k, i], bag_v[k, i]), _corr(sim_v[k, i], bag_v[k, i]))
        row["absduty"] = (_rmse(np.abs(sim_duty[k]).ravel(), np.abs(bag_duty[k]).ravel()),
                          _corr(np.abs(sim_duty[k]).ravel(), np.abs(bag_duty[k]).ravel()))
        row["sat_bag"] = float(np.mean(np.abs(bag_duty[k]) > 0.49))
        row["sat_sim"] = float(np.mean(np.abs(sim_duty[k]) > 0.49))
        for i, ax in enumerate("xyz"):
            row[f"gyro_{ax}"] = (_rmse(sim_gyro[k, i], bag_gyro[k, i]), _corr(sim_gyro[k, i], bag_gyro[k, i]))
        row["yaw_cmd_bag"] = (_rmse(cmd_yaw[k], bag_gyro[k, 2]), _corr(cmd_yaw[k], bag_gyro[k, 2]))
        row["yaw_cmd_sim"] = (_rmse(cmd_yaw[k], sim_gyro[k, 2]), _corr(cmd_yaw[k], sim_gyro[k, 2]))
        row["tilt_err_deg"] = (float(np.sqrt(np.mean(tilt[k] ** 2))), _corr(sim_tilt[k], bag_tilt[k]))
        row["duty_abs_bag"] = float(np.mean(np.abs(bag_duty[k])))
        row["duty_abs_sim"] = float(np.mean(np.abs(sim_duty[k])))
        rows.append(row)

    def fmt(v):
        return f"{v[0]:.3f}/{v[1]:+.2f}"
    print(f"\ncompare: {Path(args.sim).name} vs {Path(args.npz).name}  (RMSE/corr; duty [-], gyro [rad/s], tilt [deg])")
    hdr = ["seg", "n"] + [f"duty_{p}" for p in POS] + [f"vert_{p}" for p in POS] + [f"horiz_{p}" for p in POS] + \
          ["|duty|", "sat bag/sim", "gyro_x", "gyro_y", "gyro_z", "yaw cmd~bag", "yaw cmd~sim", "tilt sim-bag",
           "mean|duty| bag/sim"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        cells = [r["seg"], str(r["n"])] + [fmt(r[f"duty_{p}"]) for p in POS] + [fmt(r[f"v_{p}"]) for p in POS] + \
                [fmt(r[f"h_{p}"]) for p in POS] + [fmt(r["absduty"]), f"{r['sat_bag']:.2f}/{r['sat_sim']:.2f}"] + \
                [fmt(r[f"gyro_{a}"]) for a in "xyz"] + [fmt(r["yaw_cmd_bag"]), fmt(r["yaw_cmd_sim"]),
                fmt(r["tilt_err_deg"]), f"{r['duty_abs_bag']:.3f}/{r['duty_abs_sim']:.3f}"]
        print("| " + " | ".join(cells) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run", help="bag inputs -> real controllers -> sim plant; record and compare")
    p.add_argument("--bag", required=True, help="rosbag2 directory (mcap)")
    p.add_argument("--npz", default=None, help="tools/bag_export_npz.py export of the same bag (for compare)")
    p.add_argument("--out", required=True, help="output npz of the sim run")
    p.add_argument("--start", type=float, default=0.0, help="bag-relative start [s]")
    p.add_argument("--end", type=float, default=None, help="bag-relative end [s] (default: bag end)")
    p.add_argument("--segment", type=float, default=15.0, help="re-init plant pose every N s (0 = never)")
    p.add_argument("--param", action="append", default=[],
                   help="controller param override node.dotted.name=value (repeatable)")
    p.add_argument("--servo-unit", choices=("rad", "deg"), default="rad",
                   help="unit of thrusterN/servo/angle on the wire (control >= 2026-09-14: rad)")
    p.add_argument("--plant-max-duty", type=float, default=1.0, help="sim plant duty cap (control clamps at max_duty)")
    p.add_argument("--thrust-scale", type=float, default=1.0, help="multiply thrust_per_cmd")
    p.add_argument("--plant", action="append", default=[],
                   help="plant knob k=v: vertical_eff=<0..1> | rp_drag=<scale> | rp_inertia=<scale>")
    p.add_argument("--control-dt", type=float, default=0.01, help="lockstep tick = 1 / controller_manager update_rate")
    p.add_argument("--tick-timeout", type=float, default=0.005,
                   help="wall seconds to wait for the CM's request before publishing the next tick (then that "
                        "cycle measures 2*dt; the controllers run at 50 Hz either way and physics follows the CM)")
    p.add_argument("--tick-delay", type=float, default=0.002,
                   help="wall seconds between the reply and the next /clock tick (lets the CM finish update/write)")
    p.add_argument("--domain", type=int, default=77, help="ROS_DOMAIN_ID for the replay stack")
    p.add_argument("--nice", type=int, default=10)
    c = sub.add_parser("compare", help="score a saved run against the bag npz")
    c.add_argument("--sim", required=True)
    c.add_argument("--npz", required=True)
    c.add_argument("--json", default=None)
    args = ap.parse_args()
    (run if args.mode == "run" else compare)(args)


if __name__ == "__main__":
    main()
