# tools/ — repo-level CLI entry points

These scripts are the **operational + dev entry points** and the **cross-cutting glue** over the repo's
two wheels (`umiusi_perception`, `umiusi_sim`+`umiusi_rl`). They are **part of the repo, not any wheel** —
they are *not* installed by `pip install`; you run them from a checkout against the installed packages.
That is deliberate: the wheels are the reusable libraries; `tools/` is how *this project* drives them, and
several tools (e.g. `autonomy_run`) legitimately import from **both** wheels — which is exactly why they
live at the repo root rather than inside either wheel.

## How to run
From the repo root (`~/mujoco_ws/umiusi_sim`), as a module so `tools.*` and the workspace packages resolve:
```bash
uv run python -m tools.<name> [args]      # e.g. uv run python -m tools.validate_sim
```
A plain **`uv sync`** installs the full workspace (both wheels + mujoco + rl + imageio/mesh/lint tooling),
which covers every tool below **except** the `learn` and `viz` ones — add `uv sync --extra learn` /
`--extra viz` for those. If you `source .venv/bin/activate`, drop the `uv run` prefix.

The **needs** column names the capability a tool exercises (which wheel / optional extra); all of it is
present after `uv sync` except where `--extra learn` / `--extra viz` is called out.

## Catalogue

### Simulator / ops
| tool | what it does | needs |
|---|---|---|
| `validate_sim` | headless physics/interface self-check (the sim's test) | sim |
| `drive` | scripted/interactive drive; can load an RL policy | sim + rl |
| `snapshot` | render a single framed screenshot | sim (imageio) |
| `camera_demo` | onboard-camera preview (what the detector sees) | sim |
| `scenario_demo` | render the composed MjSpec worlds | sim |
| `competition_run` | run + record the competition scenario (FF driver, no RL) | sim + perception |
| `analyze_steady` | steady-state / trim analysis of a policy | sim + rl |
| `sim_server` | IPC sim backend for the ROS bridge (`ros2_ws/umiusi_sim_bridge`) | sim + perception |

### Perception
| tool | what it does | needs |
|---|---|---|
| `perception_demo` | run the detector on rendered onboard frames | sim + perception |
| `perception_eval` | IoU/PR eval of the classical detector (shim → `umiusi_perception.eval`) | perception |
| `perception_eval_learned` | same harness for the learned detector | perception |
| `perception_pseudolabel` | pseudo-label unlabelled frames for training | perception |
| `underwater_correct` | underwater colour-restoration demo | perception |
| `gen_sim_dataset` | synthesize a labelled balloon dataset from the sim | sim + perception |
| `perception_train` | train the learned TinyBalloonNet detector | perception + `--extra learn` |
| `perception_bench` | benchmark + ONNX/int8 export (Pi 4 projection) | perception + `--extra learn` |

### Autonomy (perception-in-loop, no RL)
| tool | what it does | needs |
|---|---|---|
| `autonomy_run` | full balloon-pop run: detector → `BalloonBehavior` FSM → thrusters (headless mp4 / live) | sim + perception |
| `competition_eval` | headless full-FIELD success/time metric: runs the FSM over the sampled field (GT/degraded detections) until all positives pop or timeout; reports **success rate + time-to-clear** over many episodes. Supports the pin study + perception model. For a rendered single run use `autonomy_run`. | sim + perception |
| `ram_eval` | control-isolated RAM failure-mode diagnostics: drives the real FSM with **ground-truth** detections over many DR single-balloon trials and classifies why each ram fails (incl. wire-touch as a failure). A **perception model** (`--perception-hz` detector rate + `--dropout`/`--bearing-noise-deg`/`--range-noise`/`--fp-rate`) sweeps how much detector quality/rate the ram needs, and a **pin study** (`--pin-tip`/`--pin-base` mount + `--pin-aware` geometry-solving aim) tunes the popping-pin. No GL/camera. | sim + perception |

### Calibration / sim fidelity
| tool | 用途 | needs |
|---|---|---|
| `bag_replay` | 実機 bag(npz 化)への**開ループ / ポリシー再生**とパラメータフィット(浮力オフセット・推力曲線はこれで較正した) | sim + rl |
| `estimate_hydro` | CAD シルエット × BlueROV2 実測係数から **drag / added mass を推定**(再現可能な導出) | sim |
| `gen_dynamics_dataset` | world model / 残差流体モデル用の (state, action) → next state **データセット生成** | sim |

### Policy deployment / evaluation
| tool | 用途 | needs |
|---|---|---|
| `export_policy` | ポリシーを SB3 非依存の**素 torch bundle** へ書き出し、SB3 と出力一致を検証 | sim + rl |
| `convert_policy_frame` | 学習済みポリシーの **`obs_frame` を厳密変換**(sim ⇄ rep103 ⇄ ned、再学習不要) | sim + rl |
| `preflight_policy` | 配備前検証: **golden ベクタの生成 / 検証**(実機ロード後にビット一致を証明)+ サニティ電池 | sim + rl |
| `policy_restore_test` | 傾けた状態から回した閉ループでの**復元性 / 発散有無**の検証 | sim + rl |
| `vectoring_eval` | 速度指令ポリシーの受け入れ試験: **41 セル**の方向グリッド采点(dir_err / v_along) | sim + rl |
| `mode_switch_eval` | 深度しきい値による**水平 ⇄ 鉛直モード切替のリハーサル**(実機スーパバイザ設計の閉ループ検証) | sim + rl |

### ROS bridge (rosbridge / roslibpy — no rclpy)
| tool | what it does | needs |
|---|---|---|
| `ros_view` | attach to a ROS-driven sim over `ws://localhost:9090` and render it live | sim + `--extra viz` |
| `ros_policy` | run an RL policy and publish `/cmd/direct/...` over rosbridge | rl + `--extra viz` |

### Dev tooling
| tool | what it does | needs |
|---|---|---|
| `decimate_mesh` | decimate CAD meshes (fast-simplification) for the MJCF | (base) |

## Not to be confused with the ROS **deploy** nodes
`tools/autonomy_run.py` and `tools/ros_policy.py` drive the sim from Python. The **on-robot** ROS 2
nodes live separately in `../ros2_ws/src/umiusi_autonomy/` (`perception_node` + `navigator_node`);
they import the **same** `umiusi_perception` code (detector + FSM + `umiusi_perception.control`
allocation), installed from the sim/rl/mujoco-free `packages/perception` wheel. See that package's README.
