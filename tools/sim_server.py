"""Python sim server — the SINGLE physics implementation, exposed over local IPC.

The ROS 2 bridge (``umiusi_sim_bridge``, C++) is a thin relay: it marshals one control
cycle's command to this server and reads back the state. All physics — buoyancy, drag,
LIFT, CoP moment, thrust — lives in ``UmiusiSimulator`` (``packages/sim/src/umiusi_sim``) and is reused
here verbatim, so there is exactly one place fidelity is authored.

Transport
---------
A Unix-domain stream socket (dependency-free, sub-millisecond round-trip). Default path
``/tmp/umiusi_sim.sock`` (override with ``--sock`` or ``$UMIUSI_SIM_SOCK``). One client at a
time (the control loop is single-threaded); a client disconnect resets the sim and the
server waits for the next connection.

Wire protocol (little-endian, every message length-prefixed with a uint32 byte count):

  request  (relay -> server), 80-byte payload ``<8d8Bd``:
      servo_angle_deg[0..3]  (4 x float64, degrees, as the ROS command interface carries)
      esc_duty[0..3]         (4 x float64, [-1, 1])
      servo_allowed[0..3]    (4 x uint8, gate bit)
      esc_allowed[0..3]      (4 x uint8, gate bit)
      control_dt             (float64, seconds; substeps = round(dt / physics_dt))

  reply    (server -> relay) payload:
      nq                     (uint32)
      quat[w,x,y,z]          (4 x float64, MuJoCo order)
      gyro[x,y,z]            (3 x float64, body frame)
      accel[x,y,z]           (3 x float64, specific force a - R^T g, body frame)
      servo_angle[0..3]      (4 x float64, radians)
      esc_rpm[0..3]          (4 x float64)
      qpos[0..nq-1]          (nq x float64, full MuJoCo qpos, for the viewer)

The command decode (deg->normalized, ``allowed`` gating, clamp) and the state encode
(quaternion order, body-frame gyro, specific-force accel, ESC rpm) reproduce EXACTLY what
the old C++ ``MujocoSystem::read()/write()`` produced, so the controllers see identical
state whether the physics runs in C++ (old) or here (now).

Usage
-----
    uv run python -m tools.sim_server                 # serve until Ctrl-C
    uv run python -m tools.sim_server --selftest      # standalone client round-trip check
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time

import mujoco
import numpy as np

from umiusi_sim.simulator import UmiusiSimulator

DEFAULT_SOCK = os.environ.get("UMIUSI_SIM_SOCK", "/tmp/umiusi_sim.sock")

# Fixed request payload: 4 servo angles (deg) + 4 esc duties, 4+4 allowed bytes, 1 dt.
_REQ = struct.Struct("<8d8Bd")
REQ_SIZE = _REQ.size  # 80
# Reply fixed part after the uint32 nq: quat(4) gyro(3) accel(3) servo(4) esc_rpm(4) = 18 doubles.
_REPLY_HEAD = struct.Struct("<I")
_REPLY_FIXED = struct.Struct("<18d")
_LEN = struct.Struct("<I")


# -- framing helpers ----------------------------------------------------------
def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes; return None on a clean EOF (peer closed)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> bytes | None:
    """Read one length-prefixed message payload; None on clean EOF."""
    head = _recv_exactly(conn, _LEN.size)
    if head is None:
        return None
    (length,) = _LEN.unpack(head)
    return _recv_exactly(conn, length)


def _send_msg(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(_LEN.pack(len(payload)) + payload)


# -- server -------------------------------------------------------------------
class SimServer:
    """Wraps ONE ``UmiusiSimulator`` and steps it one control period per request."""

    def __init__(self, sim: UmiusiSimulator | None = None):
        self.sim = sim if sim is not None else UmiusiSimulator()
        self._reset_integration()

    def _reset_integration(self) -> None:
        self.sim.reset()
        self._prev_lin_world = np.zeros(3)
        self._have_prev = False

    def step_command(self, servo_deg, esc_duty, servo_allowed, esc_allowed, dt) -> bytes:
        """Decode the command, advance the sim one control period, encode the reply payload.

        Mirrors the old C++ write()+read(): ``allowed`` gates each channel to zero, servo angle
        (deg) is clamped to +/- range and normalized, ESC duty is clamped to [-1, 1]; substeps =
        round(dt / physics_dt) so the physics rate matches the ROS update_rate (100 Hz -> 5).
        """
        sim = self.sim
        srange = sim.servo_range_rad

        action = np.zeros(8)
        for k in range(4):
            if servo_allowed[k]:
                ang = float(np.clip(np.radians(servo_deg[k]), -srange, srange))
                action[k] = ang / srange if srange > 0.0 else 0.0
            else:
                action[k] = 0.0
            action[4 + k] = float(np.clip(esc_duty[k], -1.0, 1.0)) if esc_allowed[k] else 0.0

        # Substep count from the control dt (matches the old C++ read(): round(period/dt)).
        ctrl_dt = max(1e-6, float(dt))
        sim.substeps = int(min(100, max(1, round(ctrl_dt / sim.dt))))
        sim.step(action)

        return self._encode_state(ctrl_dt)

    def _encode_state(self, ctrl_dt: float) -> bytes:
        sim = self.sim
        d = sim.data
        base = sim.base_id

        # Quaternion — MuJoCo order [w, x, y, z] (unchanged from the C++ read()).
        quat = d.xquat[base].copy()

        # Gyro — angular velocity in the BODY frame (mj_objectVelocity local flag).
        vloc = np.zeros(6)
        mujoco.mj_objectVelocity(sim.model, d, mujoco.mjtObj.mjOBJ_BODY, base, vloc, 1)
        gyro = vloc[:3].copy()

        # Accel — specific force f = a_body - R^T g. a is the finite difference of the world CoM
        # velocity (subtree_linvel, left over from the last apply_external_forces, exactly as the
        # C++ read() sampled it), rotated into the body frame; then subtract body-frame gravity.
        R = d.xmat[base].reshape(3, 3)
        lin_world = d.subtree_linvel[base].copy()
        if self._have_prev:
            acc_world = (lin_world - self._prev_lin_world) / ctrl_dt
        else:
            acc_world = np.zeros(3)
        acc_body = R.T @ acc_world
        g_body = R.T @ sim.gravity
        accel = acc_body - g_body
        self._prev_lin_world = lin_world
        self._have_prev = True

        servo = np.array([d.qpos[a] for a in sim.servo_qadr], dtype=float)  # radians
        esc_rpm = sim.esc_current * 1000.0
        qpos = np.asarray(d.qpos, dtype=float)

        head = _REPLY_HEAD.pack(qpos.size)
        fixed = _REPLY_FIXED.pack(
            quat[0], quat[1], quat[2], quat[3],
            gyro[0], gyro[1], gyro[2],
            accel[0], accel[1], accel[2],
            servo[0], servo[1], servo[2], servo[3],
            esc_rpm[0], esc_rpm[1], esc_rpm[2], esc_rpm[3],
        )
        return head + fixed + qpos.astype("<f8").tobytes()

    def _serve_conn(self, conn: socket.socket, log) -> None:
        self._reset_integration()
        while True:
            payload = _recv_msg(conn)
            if payload is None:
                log("client disconnected; resetting sim, waiting for next client")
                return
            if len(payload) != REQ_SIZE:
                log(f"bad request size {len(payload)} (want {REQ_SIZE}); dropping client")
                return
            fields = _REQ.unpack(payload)
            reply = self.step_command(fields[0:4], fields[4:8], fields[8:12], fields[12:16], fields[16])
            _send_msg(conn, reply)

    def serve_forever(self, sock_path: str, log=print) -> None:
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(1)
        log(f"umiusi sim server listening on {sock_path}  "
            f"(physics_dt={self.sim.dt}, substeps/cycle auto from control dt)")
        try:
            while True:
                conn, _ = srv.accept()
                log("client connected")
                with conn:
                    try:
                        self._serve_conn(conn, log)
                    except (ConnectionResetError, BrokenPipeError):
                        log("client connection reset; waiting for next client")
        finally:
            srv.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)


# -- client (used by --selftest; the real client is the C++ relay) ------------
class SimClient:
    """Minimal Python client: one round-trip per control cycle over the socket."""

    def __init__(self, sock_path: str = DEFAULT_SOCK):
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.connect(sock_path)

    def step(self, servo_deg, esc_duty, servo_allowed, esc_allowed, dt):
        req = _REQ.pack(
            servo_deg[0], servo_deg[1], servo_deg[2], servo_deg[3],
            esc_duty[0], esc_duty[1], esc_duty[2], esc_duty[3],
            int(servo_allowed[0]), int(servo_allowed[1]), int(servo_allowed[2]), int(servo_allowed[3]),
            int(esc_allowed[0]), int(esc_allowed[1]), int(esc_allowed[2]), int(esc_allowed[3]),
            float(dt),
        )
        _send_msg(self.conn, req)
        payload = _recv_msg(self.conn)
        if payload is None:
            raise ConnectionError("server closed the connection")
        (nq,) = _REPLY_HEAD.unpack_from(payload, 0)
        fixed = _REPLY_FIXED.unpack_from(payload, _REPLY_HEAD.size)
        off = _REPLY_HEAD.size + _REPLY_FIXED.size
        qpos = np.frombuffer(payload, dtype="<f8", count=nq, offset=off).copy()
        return {
            "quat": np.array(fixed[0:4]),
            "gyro": np.array(fixed[4:7]),
            "accel": np.array(fixed[7:10]),
            "servo": np.array(fixed[10:14]),
            "esc_rpm": np.array(fixed[14:18]),
            "qpos": qpos,
        }

    def close(self):
        self.conn.close()


# -- self-test ----------------------------------------------------------------
def _selftest(sock_path: str) -> int:
    """Spawn the server in a thread, drive it with a small client, check the physics + latency."""
    import threading

    server = SimServer()
    stop = threading.Event()

    def run():
        # Serve exactly the self-test's two clients, then exit when told.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv.bind(sock_path)
        srv.listen(1)
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    server._serve_conn(conn, log=lambda *_: None)
                except (ConnectionResetError, BrokenPipeError):
                    pass
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.2)

    dt = 0.01  # 100 Hz control period, as the ROS loop runs
    allowed = (1, 1, 1, 1)
    ok = True

    # (1) Zero command: the vehicle should float up (+Y) and stay ~level.
    client = SimClient(sock_path)
    latencies = []
    st = None
    for _ in range(300):
        t0 = time.perf_counter()
        st = client.step((0, 0, 0, 0), (0, 0, 0, 0), allowed, allowed, dt)
        latencies.append((time.perf_counter() - t0) * 1e3)
    # qpos layout: [x, y, z, qw, qx, qy, qz, servo1..4]
    up = st["qpos"][1]
    quat = st["quat"]
    level = abs(quat[1]) < 0.1 and abs(quat[3]) < 0.1  # small roll/pitch components
    print(f"[zero cmd]    rose to y={up:+.3f} m, quat={np.round(quat, 3)}  level={level}")
    ok &= up > 0.0 and level
    client.close()

    # (2) Reconnect (proves clean disconnect handling) + forward command -> cruise.
    client = SimClient(sock_path)
    # Forward surge via the feed-forward allocation (servo ~0, all ESC forward).
    from umiusi_perception.control import feedforward_allocation
    act = feedforward_allocation([0, 0, 0], [1, 0, 0])
    servo_deg = list(np.degrees(act[:4] * server.sim.servo_range_rad))
    esc = list(act[4:8])
    prev_y = None
    speed = 0.0
    for i in range(400):
        t0 = time.perf_counter()
        st = client.step(servo_deg, esc, allowed, allowed, dt)
        latencies.append((time.perf_counter() - t0) * 1e3)
        if prev_y is not None:
            speed = abs(st["qpos"][0] - prev_y) / dt
        prev_y = st["qpos"][0]
    horiz = np.array([st["qpos"][0], st["qpos"][2]])
    print(f"[forward cmd] cruise speed ~{speed:.3f} m/s (last-step), horiz pos={np.round(horiz, 2)} m")
    ok &= speed > 0.3  # clearly cruising
    client.close()

    lat = np.array(latencies)
    print(f"[latency]     round-trip mean={lat.mean():.3f} ms  p99={np.percentile(lat, 99):.3f} ms  "
          f"max={lat.max():.3f} ms  (n={lat.size})")
    ok &= lat.mean() < 10.0

    stop.set()
    t.join(timeout=1.0)
    print("SELFTEST:", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sock", default=DEFAULT_SOCK, help=f"Unix socket path (default {DEFAULT_SOCK})")
    ap.add_argument("--selftest", action="store_true", help="run a standalone client round-trip check")
    args = ap.parse_args()
    if args.selftest:
        return _selftest(args.sock)
    SimServer().serve_forever(args.sock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
