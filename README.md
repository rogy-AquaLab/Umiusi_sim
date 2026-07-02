# umiusi_sim

MuJoCo simulation + reinforcement learning for the **UMIUSI** azimuth-thruster underwater robot.

It simulates each thruster's **servo drive** (azimuth angle) and **thruster output** (ESC command → thrust) with an
explicit **analytical hydrodynamic model** (buoyancy, drag, added mass), and is built toward training a policy to reach
and hold a target pose (go-to-pose / station-keeping).

- **Training** depends only on Python (MuJoCo + Gymnasium + Stable-Baselines3 + PyTorch) — **no ROS 2 required**.
- A thin **ROS 2 bridge** (under `ros2_ws/`) is planned only for evaluation / sim-to-real, reusing the existing
  `sinsei_umiusi_control` interfaces.

Design docs: [`ai/project_spec.yaml`](ai/project_spec.yaml) and [`ai/architecture.md`](ai/architecture.md).

![UMIUSI model](media/umiusi_iso.png)

---

## Status

| Phase | What | State |
| ----- | ---- | ----- |
| 0 | Project spec, architecture, ROS metapackage | ✅ done |
| 1 | `sim/` — MJCF model + analytical physics + simulator API | ✅ done |
| 2 | `tools/validate_sim.py` — simulator validation (gate before RL) | ✅ done |
| 3 | `rl/` — Gymnasium env + PPO training + eval | ⏳ next |
| 4 | `ros2_ws/` — ROS 2 policy bridge (optional) | ⬜ planned |

---

## Repository layout

```
mujoco_ws/                # workspace container (NOT version-controlled)
  umiusi_sim/             # ← THIS git repo: the standalone Python project
    sim/                    # simulator core (standalone Python, no ROS)
      assets/umiusi.xml     #   MJCF model: free base + 4 azimuth servos + 4 thrust sites
      physics/              #   analytical hydrodynamics + thruster model
      simulator.py          #   UmiusiSimulator: reset() / step(action) / get_state()
    configs/umiusi.yaml     # all physical parameters (mass, drag, thrust, servo, buoyancy)
    tools/                  # view (GUI), snapshot, mesh decimation
    media/                  # rendered placement screenshots
    umiusi_model/           # measured CAD data (STL + mass/placement notes)
    ai/                     # project_spec.yaml, architecture.md, review_policy.yaml
  ros2_ws/                 # separate ROS 2 workspace (bridge, later phases; its own repo when needed)
```

---

## Setup

WSL Ubuntu 24.04, Python 3.12, CPU-only. ROS 2 is **not** needed for the simulation/RL.

```bash
# system deps (MuJoCo GUI + build)
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential libglfw3 libglfw3-dev

# python env
cd ~/mujoco_ws/umiusi_sim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel (no GPU)
pip install "mujoco>=3.2" "gymnasium>=0.29" "stable-baselines3[extra]>=2.3" \
            numpy pyyaml tensorboard glfw "imageio[ffmpeg]"
```

Everything below assumes `source .venv/bin/activate` and running from the repo root (`~/mujoco_ws/umiusi_sim`).

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

### Drive the simulator from Python

```python
import numpy as np
from sim.simulator import UmiusiSimulator

sim = UmiusiSimulator()                 # loads sim/assets/umiusi.xml + configs/umiusi.yaml
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
python -m sim.simulator
```

---

## Configuration

All physical parameters live in [`configs/umiusi.yaml`](configs/umiusi.yaml): water density / displaced volume /
buoyancy offset, diagonal linear+quadratic drag, added mass (off by default), thrust map, servo range/slew, and the
measured hull mass/CoM/inertia. Values marked `PLACEHOLDER` (drag, thrust, buoyancy volume) still need calibration in
phase-2 validation.

The robot geometry lives in [`sim/assets/umiusi.xml`](sim/assets/umiusi.xml). It is intentionally coarse (octahedron
hull + cylinder thrusters) for speed; **dynamics use the measured mass/inertia**, not the coarse shapes.

**Coordinate frame:** the CAD frame with **+Y up** (the 4 thrusters lie in the X-Z plane). Gravity is `(0, -9.81, 0)`.

**Assumptions to verify against hardware** (see the notes in `configs/umiusi.yaml`): servo rotation axis = +Y, thrust
direction = each thruster's local +X, mounting neutral angles, and the id↔`lf/lb/rb/rf` mapping.

---

## Notes for regenerating the model from CAD

The measured data is in `umiusi_model/` (Fusion 360 export: `STL/` + `description/` mass & placement notes). The raw
`base_link.stl` has ~1.08M triangles (over MuJoCo's 200k limit), so the model uses primitives instead of the mesh.
`tools/decimate_mesh.py` can produce a low-poly STL if you later want a mesh visual.
