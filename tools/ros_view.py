"""Decoupled live viewer for the ROS-driven MuJoCo sim (rviz-style, over rosbridge).

This is a SEPARATE PROCESS from the sim. The C++ ros2_control hardware plugin
(`umiusi_sim_bridge::MujocoSystem`) runs the real MuJoCo physics at 100 Hz inside the ROS
control loop and publishes the full MuJoCo `qpos` (base free-joint pose + servo hinge angles)
on `/umiusi_sim/qpos` (`std_msgs/Float64MultiArray`). This tool attaches to that running sim
over **rosbridge** (ws://localhost:9090) using **roslibpy** — no rclpy / ROS install is needed
in this venv — loads its OWN copy of the model, and on every incoming message sets
`data.qpos[:] = msg.data`, runs `mj_forward`, and renders through the shared `UmiusiViewer`.
So the window shows exactly what the ROS control loop is simulating, in a decoupled GUI you can
"launch and watch" independently of the sim, exactly like rviz attaches to a running robot.

Usage (needs a display for the GUI window; cannot open headless):
    # 1. start the ROS-driven sim (control_node + rosbridge on :9090) in ros2_ws, then:
    uv run --extra viz python -m tools.ros_view
    uv run --extra viz python -m tools.ros_view --scenario competition_balloon
    uv run --extra viz python -m tools.ros_view --url ws://localhost:9090

Drive it (so the pose actually moves) from the web UI, or e.g. (confirm the exact topic with
`ros2 topic list | grep cmd`; the verified working topics are the global /cmd/... ones):
    ros2 topic pub -1 /cmd/thruster_runnable_all \\
        sinsei_umiusi_msgs/msg/ThrusterRunnableAll \\
        '{lf: {esc: true, servo: true}, lb: {esc: true, servo: true}, rb: {esc: true, servo: true}, rf: {esc: true, servo: true}}'
    ros2 topic pub /cmd/target sinsei_umiusi_msgs/msg/Target '{velocity: {x: 1.0, y: 0.0, z: 0.0}}'

Headless self-test (no GUI): prove the rosbridge -> python data path end-to-end.
    uv run --extra viz python -m tools.ros_view --selftest --selftest-secs 5
It connects, listens for `/umiusi_sim/qpos`, and reports how many messages arrived plus the
latest qpos, WITHOUT opening the window.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import mujoco
import numpy as np
import roslibpy

QPOS_TOPIC = "/umiusi_sim/qpos"
QPOS_MSG_TYPE = "std_msgs/Float64MultiArray"


def _load_model(scenario: str):
    """Return a compiled ``mujoco.MjModel`` for the requested scene.

    ``bare`` loads the untouched robot description; ``default_world`` / ``competition_balloon``
    compose a grounded, legible scene around it (so up/forward/motion are readable in the GUI).
    """
    if scenario == "bare":
        from pathlib import Path

        import umiusi_sim.description as desc

        xml = Path(desc.__file__).resolve().parent / "umiusi.xml"
        return mujoco.MjModel.from_xml_path(str(xml))
    if scenario == "default_world":
        from umiusi_sim.description.scenarios import default_world as scn
        return scn.build_model()
    if scenario == "competition_balloon":
        from umiusi_sim.description.scenarios import competition_balloon as scn
        return scn.build_model()
    raise ValueError(f"unknown scenario {scenario!r}")


class QposReceiver:
    """Subscribes to /umiusi_sim/qpos over rosbridge and keeps the latest qpos vector.

    Thread-safe: roslibpy delivers messages on its own network thread; the render/main thread
    reads :attr:`latest` under a lock. :attr:`count` tracks how many messages have arrived.
    """

    def __init__(self, client: roslibpy.Ros):
        self._client = client
        self._lock = threading.Lock()
        self.latest: np.ndarray | None = None
        self.count = 0
        self._topic = roslibpy.Topic(
            client, QPOS_TOPIC, QPOS_MSG_TYPE,
            # best-effort-ish: only the freshest pose matters for a live view.
            throttle_rate=0, queue_length=1,
        )

    def subscribe(self):
        self._topic.subscribe(self._on_msg)

    def unsubscribe(self):
        try:
            self._topic.unsubscribe()
        except Exception:
            pass

    def _on_msg(self, message):
        # std_msgs/Float64MultiArray serializes as {"layout": {...}, "data": [...]} over rosbridge.
        data = message.get("data")
        if not data:
            return
        arr = np.asarray(data, dtype=float)
        with self._lock:
            self.latest = arr
            self.count += 1

    def get(self):
        with self._lock:
            return None if self.latest is None else self.latest.copy(), self.count


def _run_selftest(client: roslibpy.Ros, receiver: QposReceiver, secs: float) -> int:
    """Headless proof of the rosbridge->python path: listen, then report count + latest qpos."""
    print(f"[selftest] connected={client.is_connected}; listening on {QPOS_TOPIC} "
          f"for {secs:.1f}s ...", flush=True)
    deadline = time.time() + secs
    while time.time() < deadline and client.is_connected:
        time.sleep(0.1)
    latest, count = receiver.get()
    print(f"[selftest] received {count} message(s) on {QPOS_TOPIC}", flush=True)
    if latest is not None:
        np.set_printoptions(precision=4, suppress=True)
        print(f"[selftest] latest qpos ({latest.size} elems): {latest}", flush=True)
    if count == 0:
        print("[selftest] NO messages received — is the sim launched and publishing? "
              "(check control_node + rosbridge on the given --url)", flush=True)
        return 1
    print("[selftest] OK: rosbridge -> python data path verified.", flush=True)
    return 0


def _run_viewer(model, receiver: QposReceiver, client: roslibpy.Ros) -> int:
    """Live GUI: replay each incoming qpos into our own model/data and render (needs a display)."""
    from umiusi_sim.viewer import UmiusiViewer

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"waiting for {QPOS_TOPIC} … drive it with the web UI or `ros2 topic pub`.", flush=True)
    viewer = UmiusiViewer(model, data, base_id=model.body("base_link").id, cam="track").launch()
    nq = model.nq
    warned = False
    while viewer.is_running() and client.is_connected:
        tic = time.time()
        latest, count = receiver.get()
        if latest is not None:
            if latest.size == nq:
                data.qpos[:] = latest
            elif not warned:
                warned = True
                print(f"WARNING: qpos length {latest.size} != model nq {nq}; "
                      f"copying the overlapping prefix. Is the scenario the same robot?",
                      flush=True)
                n = min(latest.size, nq)
                data.qpos[:n] = latest[:n]
            else:
                n = min(latest.size, nq)
                data.qpos[:n] = latest[:n]
            mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(max(0.0, 1.0 / 50.0 - (time.time() - tic)))
    viewer.close()
    print("viewer closed.", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="ws://localhost:9090",
                        help="rosbridge websocket URL (default ws://localhost:9090)")
    parser.add_argument("--scenario", default="default_world",
                        choices=["default_world", "bare", "competition_balloon"],
                        help="scene to load for rendering (default: default_world)")
    parser.add_argument("--selftest", action="store_true",
                        help="headless: connect, count /umiusi_sim/qpos messages, print latest, exit")
    parser.add_argument("--selftest-secs", type=float, default=5.0,
                        help="how long the --selftest listens (seconds)")
    args = parser.parse_args(argv)

    # ws://host:port -> roslibpy Ros(host, port).
    from urllib.parse import urlparse

    parsed = urlparse(args.url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9090
    is_secure = parsed.scheme in ("wss", "https")

    client = roslibpy.Ros(host=host, port=port, is_secure=is_secure)
    receiver = QposReceiver(client)

    print(f"connecting to rosbridge at {args.url} …", flush=True)
    try:
        client.run(timeout=10)
    except Exception as e:  # noqa: BLE001 — surface any connection failure clearly
        print(f"ERROR: could not connect to rosbridge at {args.url}: {e}\n"
              "Is the sim launched (rosbridge_server on that port)?", flush=True)
        return 2
    if not client.is_connected:
        print(f"ERROR: not connected to rosbridge at {args.url} "
              "(is the sim launched with rosbridge on that port?).", flush=True)
        return 2
    print(f"connected to rosbridge at {args.url}.", flush=True)

    receiver.subscribe()
    try:
        if args.selftest:
            return _run_selftest(client, receiver, args.selftest_secs)
        model = _load_model(args.scenario)
        return _run_viewer(model, receiver, client)
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
