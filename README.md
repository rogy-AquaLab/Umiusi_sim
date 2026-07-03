# umiusi_sim

MuJoCo simulation + reinforcement learning for the **UMIUSI** azimuth-thruster underwater robot.

It simulates each thruster's **servo drive** (azimuth angle) and **thruster output** (ESC command → thrust) with an
explicit **analytical hydrodynamic model** (buoyancy, drag, added mass). It trains policies for attitude hold and
attitude + direction cruise, and — toward the target competition — hosts a **balloon-popping scenario** that runs
end-to-end (drive the world, pop balloons, score) with onboard cameras and an analytical feed-forward controller.

- **Training** depends only on Python (MuJoCo + Gymnasium + Stable-Baselines3 + PyTorch) — **no ROS 2 required**.
- A thin **ROS 2 bridge** (under `ros2_ws/`) is planned only for evaluation / sim-to-real, reusing the existing
  `sinsei_umiusi_control` interfaces.

Design rationale and architecture notes are maintained as separate working docs outside this
repository; this README plus code comments are the in-repo reference.

![UMIUSI model](media/umiusi_iso.png)

---

## Status

| Phase | What | State |
| ----- | ---- | ----- |
| 0 | Project spec, architecture, ROS metapackage | ✅ done |
| 1 | `umiusi_sim/` — MJCF model + analytical physics + simulator API | ✅ done |
| 2 | `tools/validate_sim.py` — simulator validation (gate before RL) | ✅ done (8/8) |
| 3 | `umiusi_rl/` — Gymnasium env + PPO training + eval | ✅ done — attitude hold + direction cruise + disturbance/sim2real robustness |
| 5a | Competition sim: balloon world + cameras + analytical FF driver, runnable & scoring (no RL) | ✅ done |
| 5b | Perception (camera → balloon detect) + behavior FSM (autonomy), Pi 4 deploy | 🟡 next |
| 4 | `ros2_ws/` — ROS 2 bridge: a custom MuJoCo `ros2_control` hardware plugin (integration phase) | ⬜ planned |

---

## Repository layout

```
mujoco_ws/                # workspace container (NOT version-controlled)
  umiusi_sim/             # ← THIS git repo (monorepo: two installable packages under src/)
    src/
      umiusi_sim/           # PACKAGE 1: reusable simulator (no ROS, no RL) — usable by other robots/tasks
        simulator.py        #   UmiusiSimulator: reset() / step(action) / render_camera()
        control.py          #   analytical feed-forward allocation (AttitudeController port, no learning)
        physics/            #   analytical hydrodynamics + thruster model
        description/        #   robot description: umiusi.xml (MJCF) + onboard cameras
          scenarios/        #     composed worlds (MjSpec) — competition_balloon.py (pool+balloons+pin)
      umiusi_rl/            # PACKAGE 2: RL experiments (depends on umiusi_sim)
        envs/umiusi_pose_env.py #   UmiusiPoseEnv: attitude / depth / pose / attitude_velocity
        train.py  eval.py   #   PPO default (--algo sac/td3); models/ is gitignored
    configs/                # umiusi.yaml (physics) + train_ppo.yaml (env/reward/algo)
    tools/                  # validate_sim, view, snapshot, camera_demo, scenario_demo, competition_run, analyze_steady
    media/                  # rendered placement screenshots
    pyproject.toml  uv.lock # uv-managed deps (CPU torch pinned); reproducible via `uv sync`
    # local-only (gitignored / outside the repo): models/ (trained policies), umiusi_model/ (CAD provenance),
    # and the ~/mujoco_ws/ai/ working docs (project_spec, architecture, sim_spec) kept beside the repo.
  ros2_ws/                 # separate ROS 2 workspace (bridge, later phases; its own repo when needed)
```

The split keeps the **reusable simulation** (`umiusi_sim`) independent of the **RL experiments**
(`umiusi_rl`): `umiusi_rl` imports `umiusi_sim` as a library. If a second consumer appears (e.g. a
perception / Pi 4 stack), `umiusi_sim` can be promoted to its own repo consumed via a normal
`pip install` git dependency with no churn to the import paths.

---

## Setup

WSL Ubuntu 24.04, Python ≥3.10, CPU-only. ROS 2 is **not** needed for the simulation/RL.

The project uses [**uv**](https://docs.astral.sh/uv/) — one command builds a reproducible virtual
environment from `pyproject.toml` + `uv.lock` (CPU-only torch is pinned, so no CUDA download).

```bash
# 1. system deps (MuJoCo GUI + video encoding)
sudo apt update && sudo apt install -y build-essential libglfw3 libglfw3-dev ffmpeg

# 2. install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. create the environment (installs both packages editable + all deps)
cd ~/mujoco_ws/umiusi_sim
uv sync --extra dev
```

Run any command with `uv run` (it uses the managed `.venv` automatically), e.g.
`uv run python -m tools.validate_sim`. All commands below assume the repo root
(`~/mujoco_ws/umiusi_sim`); drop the `uv run` prefix if you `source .venv/bin/activate` first.

---

## Usage

### Interactive GUI (watch the simulation live)

Requires a display — WSLg provides one on Windows 11. **Do not** set `MUJOCO_GL` for the GUI.

```bash
python -m tools.view          # float freely and self-level (zero action)
python -m tools.view --demo   # sweep the azimuth servos and pulse the thrusters
python -m tools.view --free   # start on the mouse-controllable free camera
```

The window opens on the upright, well-framed **iso** fixed camera. The mouse (left-drag orbit, right-drag pan, scroll
zoom) only drives the **free** camera; press `[` / `]` in the window to cycle cameras (free ↔ `iso` ↔ `top`), or start
with `--free`. The free camera assumes Z-up while the model is Y-up, so it looks tilted — that's why the default and the
snapshots use the fixed cameras.

### Placement screenshots (headless)

```bash
MUJOCO_GL=egl python -m tools.snapshot     # writes media/umiusi_{iso,top,front,corner}.png
```

`MUJOCO_GL=egl` selects offscreen rendering (needed when there is no display).

### Validate the simulator (gate before RL)

```bash
python -m tools.validate_sim        # PASS/FAIL gate; exit 1 if any check fails
python -m tools.validate_sim -v     # also print per-thruster detail + calibration numbers
```

Checks buoyancy float/sink, self-leveling, drag opposing velocity, per-thruster thrust direction,
servo tracking + tilt, and open-loop stability. The `-v` calibration section reports the numbers to
tune the `PLACEHOLDER` values (buoyancy `displaced_volume`, drag, thrust map). Runs headless.

### Train / evaluate a policy (RL)

Three selectable **tasks** (`--task`), each matched to a realistic (cheap) sensor suite. The reward
and success are always computed from the true state, so a limited sensor set just leaves part of the
task unobservable. Training is CPU-only and scales with the number of parallel environments (not a
GPU — the MuJoCo sim is the bottleneck and runs on CPU).

| `--task` | goal | sensor suite (default `obs_mode`) | notes |
| --- | --- | --- | --- |
| `attitude` | track a random target **orientation** | AHRS, e.g. BNO055 (`imu`) | horizontal & depth drift (unobserved) |
| `attitude_depth` | random orientation **+ depth** | AHRS + pressure/depth (`imu_depth`) | horizontal drifts |
| `attitude_velocity` | hold orientation **+ cruise in a commanded direction** | AHRS + body-frame velocity command (`imu`) | **direction-only** (speed magnitude unobservable without a DVL) |
| `pose` | go-to-pose: random **position** (upright) | AHRS + depth + DVL + position (`full`) | needs a position reference |

```bash
python -m umiusi_rl.train --task attitude          --run-name att      --n-envs 12   # AHRS only
python -m umiusi_rl.train --task attitude_velocity --run-name cruise   --n-envs 12   # hold + direction cruise (auto-curriculum)
python -m umiusi_rl.train --task attitude_velocity --run-name cruise_dr --disturb --domain-rand   # + water current/impulses + sim2real DR
python -m umiusi_rl.train --task pose --obs-mode imu_depth_dvl --run-name pose_dvl   # DVL velocity, no abs. XZ
python -m umiusi_rl.train --algo sac --task attitude                                 # switch algorithm

tensorboard --logdir models/cruise/tb                             # watch training curves
python -m umiusi_rl.eval --model models/cruise/final.zip                  # headless metrics (task/flags auto-loaded from meta)
python -m umiusi_rl.eval --model models/cruise/final.zip --no-disturb     # isolate the policy's own steadiness (no current/impulses)
python -m umiusi_rl.eval --model models/cruise/final.zip --domain-rand    # stress-test under model mismatch (sim2real)
MUJOCO_GL=egl python -m umiusi_rl.eval --model models/cruise/final.zip --record out.mp4   # headless video
```

Flags: `--disturb` adds a per-episode water current + random force impulses; `--domain-rand` randomizes
buoyancy/thrust/drag + adds observation noise + a 1-step control→actuation latency (sim2real robustness).
Both are recorded in `meta.yaml` so `eval` reproduces the training condition; `eval --no-disturb` /
`--domain-rand` override it. `eval` also reports **mean angular velocity** (cruise wobble) alongside
speed-along-command, sideways drift, thrust use, and servo motion.

An **RGB target marker** (red=X, green=Y, blue=Z axis triad) shows the commanded pose next to the
vehicle so you can see it tracking the target orientation. `--render`/`--record` use a **tracking
camera** that follows the vehicle, so it stays in frame
even when it drifts (attitude/attitude_depth tasks don't control horizontal position, so the
vehicle holds its commanded attitude while floating away — that drift is the expected,
sensor-limited behavior). `--record` renders offscreen, so run it with `MUJOCO_GL=egl`.

`eval` reads the task + sensor suite from the run's `meta.yaml`, so it always matches training.
Reward weights, target/workspace ranges, tolerances, domain-randomization hooks (default off), and
the algorithm hyperparameters live in [`configs/train_ppo.yaml`](configs/train_ppo.yaml).
Checkpoints, tensorboard logs, and the final policy go to the gitignored `models/<run-name>/`.

**Sensor note:** underwater there is no GPS, so horizontal position (X, Z) is only observable in
`full`. A 9-DOF AHRS (BNO055) gives absolute orientation incl. magnetometer heading (cheap → attitude
tasks are well-posed); a pressure sensor gives depth; a DVL gives body velocity (enables drift/current
rejection without an absolute position). `imu`/`imu_depth` therefore cannot hold horizontal position.

### Onboard cameras (perception groundwork)

The model carries two fixed cameras that move with the vehicle: **`front_cam`** (looks along +X, the
cruise/forward axis) and **`down_cam`** (nadir, −Y). `UmiusiSimulator.render_camera(camera, w, h)` returns
an `(H, W, 3)` uint8 RGB frame (offscreen, so run headless with `MUJOCO_GL=egl`).

```bash
MUJOCO_GL=egl python -m tools.camera_demo [out.png]   # step, capture a front_cam frame (default ./front_cam.png)
```

### Competition simulation (balloon-popping, no RL required)

Toward the target competition (fully-autonomous underwater **balloon-popping**), a composed scenario adds a
3.3 m pool, colour-coded tethered balloons (**red @0.5 m +30, yellow @1.5 m +10, blue @0.7 m −10 decoy**),
and a pin on the vehicle. It runs **end-to-end without any learning**, driven by an analytical feed-forward
controller (`umiusi_sim.control`, a port of the real `sinsei_umiusi_control` AttitudeController allocation).

```bash
MUJOCO_GL=egl python -m tools.scenario_demo                          # render the competition world (front/down cams)
MUJOCO_GL=egl python -m tools.competition_run --seconds 40           # drive, pop balloons, score, write an mp4
#   options: --record <path.mp4>  --render (GUI viewer)  --seed <N>
python -m umiusi_sim.control                                         # feed-forward allocation self-test
```

`competition_run` uses a ground-truth greedy driver (seek the nearest positive-value balloon, avoid blue),
geometric pop detection (pin tip vs balloon), and prints a pop timeline + final score (typically 80: both
reds + both yellows, blue avoided). The composition uses `mujoco.MjSpec` (`description/scenarios/`) and does
**not** modify the base robot model, so `validate_sim` stays 8/8. Perception (camera → balloon detection)
and a behavior FSM replace the ground-truth driver in the next phase.

> **Note:** the feed-forward allocation matrix comes from the real controller, whose axes do **not** line up
> 1:1 with this MuJoCo model (empirically `Vx → −X`, `Vz → +Y`, `Vy → yaw couple`). This frame mapping is
> documented in `control.py` and must be reconciled before the real `ros2_control` controllers can drive the sim.

### Drive the simulator from Python

```python
import numpy as np
from umiusi_sim.simulator import UmiusiSimulator

sim = UmiusiSimulator()                 # loads src/umiusi_sim/description/umiusi.xml + configs/umiusi.yaml
sim.reset(pos=(0.0, 0.5, 0.0))          # +Y is up

# action = [servo_1..4, esc_1..4], each in [-1, 1]
#   servo_k -> target azimuth angle (rate-limited)
#   esc_k   -> thrust = esc_k * thrust_per_cmd  [N]
for _ in range(100):
    state = sim.step(np.array([0, 0, 0, 0,  0.5, 0.5, 0.5, 0.5]))

print(state["pos"], state["quat"], state["lin_vel"], state["servo"], state["thrust"])
```

Quick smoke test of the physics loop:

```bash
uv run python -m umiusi_sim.simulator
```

---

## Configuration

All physical parameters live in [`configs/umiusi.yaml`](configs/umiusi.yaml): water density / displaced volume /
buoyancy offset, diagonal linear+quadratic drag, added mass (off by default), thrust map, servo range/slew, and the
measured hull mass/CoM/inertia. Values marked `PLACEHOLDER` (drag, thrust, buoyancy volume) still need calibration in
phase-2 validation.

The robot geometry lives in [`src/umiusi_sim/description/umiusi.xml`](src/umiusi_sim/description/umiusi.xml). It is intentionally coarse
(octagonal-prism hull + T-shaped azimuth thrusters + two onboard cameras) for speed; **dynamics use the
measured mass/inertia**, not the coarse shapes.

**Coordinate frame:** the CAD frame with **+Y up** (the 4 thrusters lie in the X-Z plane). Gravity is `(0, -9.81, 0)`.

**Assumptions to verify against hardware** (see the notes in `configs/umiusi.yaml`): servo rotation axis = +Y, thrust
direction = each thruster's local +X, mounting neutral angles, and the id↔`lf/lb/rb/rf` mapping.

---

## Notes for regenerating the model from CAD

The measured data is in `umiusi_model/` (Fusion 360 export: `STL/` + `description/` mass & placement notes). The raw
`base_link.stl` has ~1.08M triangles (over MuJoCo's 200k limit), so the model uses primitives instead of the mesh.
`tools/decimate_mesh.py` can produce a low-poly STL if you later want a mesh visual.
