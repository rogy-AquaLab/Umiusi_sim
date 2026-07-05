# umiusi_sim — runtime architecture (sim ⇄ real)

How the pieces fit together across **development/simulation** and **real-robot deployment**, and the
single rule that keeps them from drifting: **physics is authored once (Python); everything else is
one library reused in both worlds.**

## The three concerns

| concern | what it is | where it lives |
|---|---|---|
| **Physics / sim** | buoyancy · drag · lift · thruster forces · MuJoCo step | **`umiusi_sim` (Python) — the single implementation** |
| **Perception + navigator** (high level) | camera → learned detector → detections → behaviour FSM → velocity/attitude command | **one Python library** (`umiusi_sim.perception` + `tools/behavior.py`), reused in sim **and** on the robot |
| **Low-level control** | Gate → Attitude → Thruster (or an RL policy) → per-thruster servo/ESC | ROS 2 `sinsei_umiusi_control` controllers (unchanged) |
| **Learning** | RL (cruise/attitude) + detector training | **Python, offline** → emits models (`.zip`, `.pt`/`.onnx`) |

## Single source of physics

There must be **exactly one** physics implementation: `src/umiusi_sim/simulator.py` +
`physics/` (analytical hydro) + the MJCF. Two copies drift — the earlier C++ hydro port already
lagged (no lift/CoP). **The sim only ever runs in dev/test, never on the real robot** (the robot is
real hardware), so the C++ port's only justification — "Pi parity" — does not apply. Therefore:

- **The ROS bridge does NOT re-implement physics.** It is a *thin relay* to the running Python sim
  (see "IPC bridge" below). Delete/deprecate the C++ hydro.
- Fidelity work (lift, CoP moment, calibration) happens in **one place**: the Python sim.

## Runtime configurations

```
(1) DEV — pure Python (fast iteration)
    tools/autonomy_run.py :  perception + navigator FSM  ──drive──▶  Python sim (umiusi_sim)
    tools/drive.py / umiusi_rl.eval : RL policy / manual ──────────▶  Python sim

(2) DEV — ROS-in-the-loop (test the REAL control stack against the sim)
    sinsei Gate→Attitude→Thruster (ros2_control)
        │ command interfaces (esc duty, servo angle)
        ▼
    umiusi_sim_bridge  ── thin IPC relay (NO physics) ──▶  Python sim server (umiusi_sim)
        ▲ state interfaces (imu quat/gyro/accel, thruster servo/esc)
    (same perception_node + navigator_node as (3) can run on top)

(3) REAL ROBOT — NO sim
    [camera] ─▶ perception_node (loads examples/balloon_detector) ─▶ /detections
    /detections + /state/imu ─▶ navigator_node (tools/behavior FSM) ─▶ /cmd/target (or /cmd/direct)
    /cmd/target ─▶ sinsei Gate→Attitude→Thruster ─▶ CAN plugin ─▶ real thrusters
    (optional: RL policy node replaces AttitudeController via /cmd/direct/…)
```

Sim ↔ real is a **swap of the ros2_control hardware plugin** (IPC-bridge-to-Python ↔ CAN). The
controllers, launch, parameters, perception_node, and navigator_node are **identical** in (2) and (3).

## IPC bridge (config 2) — "just the connective part"

The ros2_control hardware component must be C++, but it holds **no physics** — it marshals one
cycle's command/state to/from the Python sim:

- **Python sim server** (`umiusi_sim`): wraps `UmiusiSimulator`, listens on a local IPC channel
  (Unix domain socket / ZMQ). Per request: receive the 8-D command (per-thruster esc duty + servo
  angle + allowed bits), step the sim one control period, reply with the state (IMU quaternion
  `[w,x,y,z]`, gyro, accel = specific force, per-thruster servo angle + esc rpm, and qpos for the
  viewer). The sim — and thus buoyancy/drag/**lift/CoP**/thrust — is the one Python implementation.
- **C++ relay** (`umiusi_sim_bridge`): the `SystemInterface` connects to the socket in
  `on_activate`; `write()` sends the command, `read()` receives the state and fills the interface
  handles (cached, no per-cycle alloc). Same interface names as the CAN hardware, so the six
  controllers spawn unchanged.
- 100 Hz is fine over a local socket (sub-ms round-trip); the sim runs on the dev machine (Python
  available), so there is no embedded-target constraint on it.

## Perception + navigator = one library, two front-ends

`umiusi_sim.perception` (detector) and `tools/behavior.py` (the balloon FSM) are **plain Python
library code with no ROS and no sim dependency** — they take detections / state and return commands.

- **Sim front-end**: `tools/autonomy_run.py` feeds them the degraded sim camera + drives the Python sim.
- **Robot front-end**: thin `rclpy` nodes — `perception_node` (camera topic → detector → detections)
  and `navigator_node` (detections + IMU → FSM → `/cmd/target`) — wrap the **same** functions.

So the detector and the navigator are written **once**, verified in sim, and deployed on the robot
unchanged. Inference on the Pi uses the ONNX/int8 export for fps (~12–30 fps @320, benched).

## Learning stays in Python

RL (`umiusi_rl`, PPO on the Python sim) and detector training (`tools/perception_train`) run
**offline in Python** and emit models. The robot and the ROS bridge only ever **load** those models
(`examples/cruise_policy/`, `examples/balloon_detector/`) — they never train. No learning on the robot.

## Consequences / decisions

- **Do not** port lift/CoP (or any physics) to C++ — the bridge relays to Python instead.
- Keep `tools/behavior.py` + `umiusi_sim.perception` free of ROS/sim imports so both front-ends reuse them.
- The real robot runs three things (perception_node, navigator_node, low-level controllers) — **no sim process**.
