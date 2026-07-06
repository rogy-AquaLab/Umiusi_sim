# tools/ — repo-level CLI entry points

These scripts are the **operational + dev entry points** over the three packages (`umiusi_sim`,
`umiusi_perception`, `umiusi_rl`). They are **part of the repo, not the wheel** — they are *not*
installed by `pip install umiusi_sim`; you run them from a checkout against the installed packages.
That is deliberate: the packages are the reusable library; `tools/` is how *this project* drives them.

## How to run
From the repo root (`~/mujoco_ws/umiusi_sim`), as a module so `tools.*` and the `src/` packages resolve:
```bash
uv run python -m tools.<name> [args]      # e.g. uv run python -m tools.validate_sim
```
`uv sync --extra dev` covers every tool below **except** the two `[learn]` and two `[viz]` ones
(add `--extra learn` / `--extra viz` when you need those). If you `source .venv/bin/activate`, drop
the `uv run` prefix.

## Catalogue

### Simulator / ops — needs `[sim]` (mujoco)
| tool | what it does | extra |
|---|---|---|
| `validate_sim` | headless physics/interface self-check (the sim's test) | `[sim]` |
| `drive` | scripted/interactive drive; can load an RL policy | `[rl]` |
| `snapshot` | render a single framed screenshot | `[dev]` (imageio) |
| `camera_demo` | onboard-camera preview (what the detector sees) | `[dev]` |
| `scenario_demo` | render the composed MjSpec worlds | `[dev]` |
| `competition_run` | run + record the competition scenario | `[dev]` |
| `analyze_steady` | steady-state / trim analysis of a policy | `[rl]` |
| `sim_server` | IPC sim backend for the ROS bridge (`ros2_ws/umiusi_sim_bridge`) | `[sim]` |

### Perception — needs `[perception]` (torch)
| tool | what it does | extra |
|---|---|---|
| `perception_demo` | run the detector on rendered onboard frames | `[dev]` |
| `perception_eval` | IoU/PR eval of the classical detector (shim → `umiusi_perception.eval`) | `[perception]` |
| `perception_eval_learned` | same harness for the learned detector | `[perception]` |
| `perception_pseudolabel` | pseudo-label unlabelled frames for training | `[perception]` |
| `underwater_correct` | underwater colour-restoration demo | `[perception]` |
| `gen_sim_dataset` | synthesize a labelled balloon dataset from the sim | `[dev]` (sim+perception+imageio) |
| `perception_train` | train the learned TinyBalloonNet detector | `[perception] [learn]` |
| `perception_bench` | benchmark + ONNX/int8 export (Pi 4 projection) | `[perception] [learn]` |

### Autonomy (perception-in-loop, no RL) — needs `[sim] [perception]`
| tool | what it does | extra |
|---|---|---|
| `autonomy_run` | full balloon-pop run: detector → `BalloonBehavior` FSM → thrusters (headless mp4 / live) | `[dev]` |

### ROS bridge (rosbridge / roslibpy — no rclpy) — needs `[viz]`
| tool | what it does | extra |
|---|---|---|
| `ros_view` | attach to a ROS-driven sim over `ws://localhost:9090` and render it live | `[sim] [viz]` |
| `ros_policy` | run an RL policy and publish `/cmd/direct/...` over rosbridge | `[rl] [viz]` |

### Dev tooling
| tool | what it does | extra |
|---|---|---|
| `decimate_mesh` | decimate CAD meshes (fast-simplification) for the MJCF | `[dev]` |

## Not to be confused with the ROS **deploy** nodes
`tools/autonomy_run.py` and `tools/ros_policy.py` drive the sim from Python. The **on-robot** ROS 2
nodes live separately in `../ros2_ws/src/umiusi_autonomy/` (`perception_node` + `navigator_node`);
they import the **same** `umiusi_perception` + `umiusi_sim.control` code, installed via the
mujoco-free `[perception]` extra. See that package's README.
