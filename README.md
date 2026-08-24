# umiusi_sim

MuJoCo simulation + reinforcement learning for the **UMIUSI** azimuth-thruster underwater robot.

It simulates each thruster's **servo drive** (azimuth angle) and **thruster output** (ESC command → thrust) with an
explicit **analytical hydrodynamic model** (buoyancy, drag, lift, added mass, and a CoP-offset
translation moment; see [`docs/physics.md`](docs/physics.md)).
It trains policies for attitude hold and
attitude + direction cruise, and — toward the target competition — hosts a **balloon-popping scenario** that runs
end-to-end (drive the world, pop balloons, score) with onboard cameras — either with a ground-truth
analytical feed-forward driver, or fully perception-driven (a learned onboard detector feeding a
rule-based behaviour FSM).
The sim ⇄ real runtime split (single Python physics source, ROS relays to it, perception+navigator
reused on the robot) is in [`docs/architecture.md`](docs/architecture.md).

- **Training** depends only on Python (MuJoCo + Gymnasium + Stable-Baselines3 + PyTorch) — **no ROS 2 required**.
- A **ROS 2 bridge** (under `ros2_ws/`, **done**) lets the real `sinsei_umiusi_control` controllers — or a
  trained RL policy — drive the sim unchanged, for evaluation / sim-to-real. See
  [`ros2_ws/src/umiusi_sim_bridge/README.md`](../ros2_ws/src/umiusi_sim_bridge/README.md).

Design rationale and architecture notes are maintained as separate working docs outside this
repository; this README plus code comments are the in-repo reference.

![UMIUSI model](media/umiusi_iso.png)

---

## Status

| Phase | What | State |
| ----- | ---- | ----- |
| 0 | Project spec, architecture, ROS metapackage | ✅ done |
| 1 | `umiusi_sim/` — MJCF model + analytical physics + simulator API | ✅ done |
| 2 | `tools/validate_sim.py` — simulator validation (gate before RL) | ✅ done (10/10) |
| 3 | `umiusi_rl/` — Gymnasium env + PPO training + eval | ✅ done — attitude hold + direction cruise + disturbance/sim2real robustness |
| 5a | Competition sim: balloon world + cameras + analytical FF driver, runnable & scoring (no RL) | ✅ done |
| 5b | Perception (learned detector + underwater synth-data pipeline + eval) + behavior FSM (perception-in-loop autonomy) | ✅ done (runnable end-to-end); 🟡 sim-to-real tuning + Pi 4 deploy pending |
| 4 | `ros2_ws/` — ROS 2 bridge: custom MuJoCo `ros2_control` hardware plugin; the real controllers + an RL policy drive the sim | ✅ done (in the sibling `ros2_ws/`) |
| 6 | Sim-to-real deployment: bag-calibrated physics, REP-103 policy bundles + golden verification, RL running on the robot | ✅ first pool test 2026-08-21 (`rl_attitude_node`); 🟡 next: in-water calibration campaign → narrow DR → retrain |

---

## Repository layout

```
mujoco_ws/                # workspace container (NOT version-controlled)
  umiusi_sim/             # ← THIS git repo. uv WORKSPACE: TWO installable wheels under packages/
    packages/
      perception/         # ── wheel "umiusi_perception" = the ON-ROBOT execution library ──
        src/umiusi_perception/
          balloon_detector.py #   learned/HSV/Hough detectors + tracker.py + underwater.py restoration
          learned_detector.py #   TinyBalloonNet · eval.py (IoU harness)
          autonomy/behavior.py #   BalloonBehavior search→approach→align→ram→confirm FSM
          control.py          #   feed-forward thruster allocation (AttitudeController port) — PURE numpy
        pyproject.toml       #   deps: numpy + torch + scipy + opencv.  NO mujoco / sb3 / gymnasium / sim.
      sim/                # ── wheel "umiusi_sim" = the DEV / TRAIN library (never on the robot) ──
        src/umiusi_sim/       #   PACKAGE: simulator. deps numpy+pyyaml, + mujoco via extra [sim]
          simulator.py        #     UmiusiSimulator: reset() / step(action) / render_camera()
          physics/            #     analytical hydrodynamics + thruster model
          rendering/          #     underwater_sim.py — onboard-camera degradation forward model
          description/        #     umiusi.xml (MJCF) + appearance.py + scenarios/ (composed MjSpec worlds)
        src/umiusi_rl/        #   PACKAGE: RL. extra [rl] (implies [sim] — imports umiusi_sim.simulator)
          envs/umiusi_pose_env.py #  UmiusiPoseEnv: attitude / depth / pose / attitude_velocity
          train.py  eval.py   #     PPO default (--algo sac/td3)
        pyproject.toml       #   deps numpy+pyyaml; extras [sim]=mujoco, [rl]=gymnasium/sb3/tensorboard/torch
    configs/                # umiusi.yaml (physics) + train_ppo.yaml (env/reward/algo) — at repo root
    tools/                  # cross-cutting CLI glue over BOTH wheels (repo root, NOT in any wheel) —
                            # run as `uv run python -m tools.<name>`. Catalogued in tools/README.md:
                            #   sim: validate_sim, drive, snapshot, camera_demo, scenario_demo, competition_run
                            #   perception: perception_demo/train/eval/eval_learned/bench/pseudolabel, gen_sim_dataset
                            #   autonomy: autonomy_run (perception-in-loop balloon-pop, no RL)
                            #   calibration: bag_replay, estimate_hydro, gen_dynamics_dataset
                            #   policy deploy/eval: export_policy, convert_policy_frame, preflight_policy,
                            #                       vectoring_eval, mode_switch_eval
                            #   ROS (rosbridge): ros_view (viewer), ros_policy (RL -> cmd/direct); sim_server (IPC backend)
    tools/README.md         # ← catalogue: what each tool does + which wheel/extra it needs + how to run
    examples/               # shipped example models — cruise_policy/ (RL) + balloon_detector/ (learned
                            #   detector) — so eval / drive / autonomy run out of the box
    media/                  # rendered placement screenshots
    pyproject.toml  uv.lock # uv workspace root (virtual, not published) + CPU-torch pin; `uv sync` = full dev
  ros2_ws/                 # separate ROS 2 workspace: umiusi_sim_bridge (IPC relay) + umiusi_autonomy (deploy nodes)
```

**Why one repo, two wheels (deliberate monorepo).** The code co-evolves and shares one config / robot
description, so it lives in one git repo — but it splits into **two independently-installable wheels along
the two deployment targets**, so the code that lands on each target is exactly what that target runs:

- `umiusi_perception` — the **on-robot** wheel: detector + tracker + autonomy FSM + thruster allocation
  (`umiusi_perception.control`, pure numpy). It imports **neither** sibling, so `pip install
  ./packages/perception` puts execution-only code on the Pi — **no simulator, no training source, no
  mujoco**. This is the whole point of the split (an aggregated single-wheel install would copy the sim +
  training source onto the robot even when their heavy deps are extra-gated).
- `umiusi_sim` — the **dev/train** wheel: `umiusi_sim` (simulator, `[sim]`=mujoco) + `umiusi_rl` (training,
  `[rl]`). It also imports nothing from perception; the two wheels are independent siblings, glued only by
  `tools/` at the repo root running in the full `uv sync` env.

Detector + FSM behaviour is therefore **bit-identical** in sim (`tools/autonomy_run.py`) and on the robot
(`ros2_ws/src/umiusi_autonomy`), because both import the same `umiusi_perception` wheel.

---

## Setup

WSL Ubuntu 24.04, Python ≥3.10, CPU-only. ROS 2 is **not** needed for the simulation/RL.

The repo is a **uv workspace** of **two wheels** (`packages/perception`, `packages/sim`); you install
the one your machine needs. Pick the row that matches your target:

| I want to… | Install | Pulls | Notes |
|---|---|---|---|
| **work on the repo** (daily driver) | `uv sync` | mujoco + torch + gym/SB3 + tooling | both wheels editable — sim + perception + rl |
| **deploy on the robot / Pi** | `pip install ./packages/perception` | torch + scipy + opencv | **sim/rl/mujoco-FREE** — detector + FSM + `umiusi_perception.control` only |
| **train / eval RL** (train box) | `pip install './packages/sim[rl]'` | mujoco + torch + gym/SB3 | implies `[sim]`; perception not needed |
| run the **core simulator** only | `pip install './packages/sim[sim]'` | mujoco (no torch) | sim + `tools/validate_sim`, `drive`, … |
| add ONNX/YOLO export, ROS viewer | append `uv sync --extra learn` / `--extra viz` | ultralytics/onnx… / roslibpy | on top of `uv sync` |

The **deploy row is the point of the split**: `./packages/perception` is a wheel that contains *only*
execution code — no simulator or training source lands on the robot at all, and it never drags MuJoCo
onto aarch64 (the allocation the deploy nodes use, `umiusi_perception.control`, is pure numpy). See
`ros2_ws/src/umiusi_autonomy/README.md`.

```bash
# 1. system deps for the SIMULATOR (MuJoCo GUI + video encoding) — NOT needed on the deploy target
sudo apt update && sudo apt install -y build-essential libglfw3 libglfw3-dev ffmpeg
```

**Recommended — [uv](https://docs.astral.sh/uv/)** (one command, reproducible from `uv.lock`; CPU
torch pinned, so no CUDA download):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # install uv once
cd ~/mujoco_ws/umiusi_sim
uv sync                                    # ← the one-command bootstrap: BOTH wheels + mujoco + rl + tooling
# heavier: add `--extra learn` (detector training) and/or `--extra viz` (ROS viewer)
```
> **Note:** bare `uv sync` installs the *full* dev workspace (both members editable). The workspace root
> is a virtual project (`package = false`) — it is not built or published; the distributables are the two
> `packages/*` wheels.

Run any command with `uv run` (uses the managed `.venv`); the synced env already has every runtime dep,
so no per-command `--extra` is needed.

**Fallback — plain venv + pip** (no uv, e.g. on the robot):
```bash
cd ~/mujoco_ws/umiusi_sim                   # or wherever the checkout lives
python -m venv .venv && . .venv/bin/activate
pip install ./packages/perception          # mujoco-free deploy; or './packages/sim[rl]' for a train box
```

All commands below assume the repo root (`~/mujoco_ws/umiusi_sim`); drop the `uv run` prefix if you
`source .venv/bin/activate` first.

---

## Usage

All commands run from the repo root (`~/mujoco_ws/umiusi_sim`) through **uv**: either prefix each with
`uv run` (e.g. `uv run python -m tools.validate_sim`), or `source .venv/bin/activate` once and drop it.
The examples below omit the `uv run` prefix for brevity.

### Display & rendering (read once — the whole rule)

MuJoCo renders one of two ways; choose per command:

| Mode | Use it for | How | Applies to |
| --- | --- | --- | --- |
| **Interactive GUI** (live window) | watching it move | needs a display (WSLg on Win 11); **do NOT set `MUJOCO_GL`** | `tools.drive`, any `--render` |
| **Headless / offscreen** (save PNG/MP4, or no display) | saving images/video, headless boxes | prefix the command with **`MUJOCO_GL=egl`** | `tools.snapshot`, `camera_demo`, `scenario_demo`, any `--record` |

That's it: `MUJOCO_GL=egl` is **only** for offscreen capture — setting it breaks the live viewer, and
omitting it on a headless machine makes offscreen rendering fail. Every command below follows this rule.

**Free camera (GUI only):** the viewer opens on a fixed, well-framed camera. Press `[` / `]` in the
window to cycle cameras; the **free** camera is the only mouse-controllable one (left-drag orbit,
right-drag pan, scroll zoom) — start on it with `--free`. It assumes Z-up while the model is Y-up, so it
looks tilted; that's why the defaults and snapshots use the fixed cameras.

---

### Simulator — `umiusi_sim`

**Validate** (physics gate — run before trusting anything; keep it green):
```bash
python -m tools.validate_sim        # PASS/FAIL gate (10 checks); exit 1 on any failure
python -m tools.validate_sim -v     # + per-thruster detail and calibration numbers
```
Covers buoyancy float/sink, self-leveling, drag, per-thruster thrust direction, servo tracking + tilt,
feed-forward allocation (heave/surge decoupling), and open-loop stability. Headless.

The interactive GUI is **`tools.drive`** — load a trained policy and steer it from the keyboard (like a
vehicle) in the legible default world (checker floor grid + axis triad). It and the `--render` flags all
share one viewer module (`umiusi_sim.viewer`): default fixed **track** camera that follows the vehicle;
`[` / `]` cycle cameras; the free camera is mouse-controllable but looks tilted (the model is +Y-up — the
fixed cameras are correct). GUI needs a display, no `MUJOCO_GL`.

```bash
python -m tools.drive --model examples/cruise_policy/final.zip           # W/S fwd/back, A/D strafe, Q/E turn, R/F tilt, Space stop
python -m tools.drive --model examples/cruise_policy/final.zip --headless 150   # headless self-test (forward → cosine ~1)
```
A pretrained cruise policy ships in [`examples/`](examples/) so `drive` / `eval` work right after cloning
(no training needed). Press **Space** to stop the cruise command and just watch it hold in place.
`examples/cruise_policy` is a **pre-calibration demo artifact** (a copy of `av_curr4`: pre-bag-calibration
physics, sim-frame observations, legacy 25-D proprio) — it is there so the tools run out of the box, and
is **not** a deployment artifact; what goes on the robot are the REP-103 bundles (see the RL section).

To watch a *specific* automated run live instead, pass `--render` to it (`eval --render`,
`competition_run --render`); training is headless (watch curves in tensorboard, then `eval --render`).
A standalone rviz-style viewer that attaches to a separately-running sim is a possible future addition
(it would need an IPC layer since the sim isn't ROS-based) — see the Roadmap.

**Onboard cameras** — two fixed cameras move with the vehicle: `front_cam` (+X, forward) and `down_cam`
(nadir, −Y). `UmiusiSimulator.render_camera(camera, w, h, degrade=True)` returns an `(H, W, 3)` uint8
RGB frame (`degrade=True` applies the underwater degradation the balloon detector was trained on):
```bash
MUJOCO_GL=egl python -m tools.snapshot          # media/umiusi_{iso,top,front,corner}.png
MUJOCO_GL=egl python -m tools.camera_demo [out] # capture a front_cam frame (default ./front_cam.png)
```

**Competition simulation** (balloon-popping, no RL) — a composed world (3.3 m pool + tethered balloons
**red @0.5 m +30 / yellow @1.5 m +10 / blue @0.7 m −10 decoy** + a pin), driven by the analytical
feed-forward controller (`umiusi_perception.control`, a port of the real AttitudeController allocation):
```bash
MUJOCO_GL=egl python -m tools.scenario_demo                 # render the world (front/down cams)
MUJOCO_GL=egl python -m tools.competition_run --seconds 40  # drive, pop, score, write an mp4 (--record <p> --seed <n>)
python -m tools.competition_run --render                    # or watch it live (GUI, no MUJOCO_GL)
python -m umiusi_perception.control                         # feed-forward allocation self-test
```
It seeks the nearest positive-value balloon (avoids targeting blue), detects pops geometrically (pin tip
vs balloon), and prints a pop timeline + final score (typically 80). The world is composed with
`mujoco.MjSpec` and does **not** touch the base model, so `validate_sim` stays green. Perception + a
behavior FSM replace the ground-truth driver in the next phase.
> The feed-forward allocation's axes don't line up 1:1 with the sim (empirically `Vx→−X`, `Vz→+Y`,
> `Vy→yaw couple`) — documented in `control.py`; reconcile before driving the sim from real `ros2_control`.

**Perception-in-the-loop autonomy** (vision replaces the ground-truth driver, still no RL) — the robot
detects balloons from its OWN underwater-degraded `front_cam` with the learned detector, and a
**rule-based behaviour FSM** (`umiusi_perception.autonomy.BalloonBehavior`: search → approach → align
→ ram → camera-confirm, with multi-frame track voting) drives it through the same feed-forward
allocation. A ~4.5 m detection range gate drops far false positives, selection is the **nearest
reachable non-blue** track (maximise throughput / clear the field near-to-far, which also avoids
under-passing un-popped balloons — the wire-entanglement lever), and pops are confirmed from the
camera alone (balloons vanish when popped):
```bash
MUJOCO_GL=egl uv run python -m tools.autonomy_run --headless --seed 1  # short self-test (mp4)
MUJOCO_GL=egl uv run python -m tools.autonomy_run --full-run           # record the FULL competition to mp4
DISPLAY=:0    uv run python -m tools.autonomy_run --render              # watch live (passive viewer)
```
The learned detector ships in [`examples/balloon_detector/`](examples/balloon_detector/) (the default
`--model`), so this runs right after cloning; pass `--model` to use your own trained checkpoint.

**From Python:**
```python
import numpy as np
from umiusi_sim.simulator import UmiusiSimulator

sim = UmiusiSimulator()                 # loads description/umiusi.xml + configs/umiusi.yaml
sim.reset(pos=(0.0, 0.5, 0.0))          # +Y is up
for _ in range(100):                    # action = [servo_1..4, esc_1..4], each in [-1, 1]
    state = sim.step(np.array([0, 0, 0, 0,  0.5, 0.5, 0.5, 0.5]))
print(state["pos"], state["quat"], state["lin_vel"], state["servo"], state["thrust"])
```
Physics-loop smoke test: `python -m umiusi_sim.simulator`.

---

### Reinforcement learning — `umiusi_rl`

Four selectable **tasks** (`--task`), each matched to a realistic (cheap) sensor suite. Reward and
success always use the true state, so a limited sensor suite just leaves part of the task unobservable.
Training is CPU-only and scales with the number of parallel environments (`--n-envs`).

| `--task` | goal | sensor suite (default `obs_mode`) | notes |
| --- | --- | --- | --- |
| `attitude` | track a random target **orientation** | AHRS, e.g. BNO055 (`imu`) | horizontal & depth drift (unobserved) |
| `attitude_depth` | random orientation **+ depth** | AHRS + pressure/depth (`imu_depth`) | horizontal drifts |
| `attitude_velocity` | hold orientation **+ cruise in a commanded direction** | AHRS + body-frame velocity command (`imu`) | **direction-only** (speed magnitude unobservable without a DVL) |
| `pose` | go-to-pose: random **position** (upright) | AHRS + depth + DVL + position (`full`) | needs a position reference |

**Train:**
```bash
python -m umiusi_rl.train --task attitude          --run-name att       --n-envs 12   # AHRS only
python -m umiusi_rl.train --task attitude_velocity --run-name cruise    --n-envs 12   # hold + direction cruise (auto-curriculum)
python -m umiusi_rl.train --task attitude_velocity --run-name cruise_dr --disturb --domain-rand   # + disturbances + sim2real DR
python -m umiusi_rl.train --task pose --obs-mode imu_depth_dvl --run-name pose_dvl    # DVL velocity, no abs. XZ
python -m umiusi_rl.train --algo sac --task attitude                                  # switch algorithm
```
`--disturb` = per-episode water current + force impulses; `--domain-rand` = randomize buoyancy/thrust/drag
+ observation noise + a 1-step control→actuation latency (sim2real). Both are recorded in `meta.yaml`.

**Evaluate / watch** (`eval` reloads task + sensor suite + disturbance/DR from `meta.yaml`, so it matches training):
```bash
tensorboard --logdir models/cruise/tb                                                     # training curves
python -m umiusi_rl.eval --model models/cruise/final.zip                                   # headless metrics
python -m umiusi_rl.eval --model models/cruise/final.zip --render                          # watch live (GUI, no MUJOCO_GL)
MUJOCO_GL=egl python -m umiusi_rl.eval --model models/cruise/final.zip --record out.mp4    # save a video
python -m umiusi_rl.eval --model models/cruise/final.zip --no-disturb                      # isolate the policy's own steadiness
python -m umiusi_rl.eval --model models/cruise/final.zip --domain-rand                     # stress-test under model mismatch
```
`eval` reports mean angular velocity (cruise wobble), speed-along-command, sideways drift, thrust use, and
servo motion. An RGB axis-triad marker shows the commanded orientation; `--render`/`--record` use a
tracking camera that follows the vehicle (attitude/attitude_velocity tasks don't hold horizontal position,
so it holds its commanded attitude while drifting — expected, sensor-limited behavior).

Reward weights, ranges, tolerances, and PPO hyperparameters live in
[`configs/train_ppo.yaml`](configs/train_ppo.yaml); checkpoints, tb logs, and the final policy go to
`models/<run-name>/` (a local run-output directory).

**Observation contract (what the robot actually consumes).** Proprioception defaults to
`proprio_mode: action` — only `prev_action` (8), *not* the servo/thrust telemetry, because on the robot
that telemetry is a command echo rather than an independent measurement. So the deployed observation is
**17-D** for `attitude_velocity` (`ori_err` 3 + gyro 3 + `v_cmd` 3 + `prev_action` 8) and **14-D** for
`attitude` (`proprio_mode: full` = the legacy 16-D proprio, kept only for old runs). Orthogonally,
`obs_frame` selects how the 3-vectors are expressed — `sim` (CAD, +Y up), **`rep103`** (x fwd / y left /
z up, **the deployment contract**), or `ned`. Train in any frame, then hand the robot a rep103 bundle:
`tools/convert_policy_frame.py` converts a trained policy exactly (no retraining — a signed permutation
of the input weights + VecNormalize stats), and `tools/preflight_policy.py` generates/verifies
`golden.npz` so a loaded-on-the-robot policy is provably the one the sim validated.

**Sensor note:** underwater there is no GPS, so horizontal position (X, Z) is only observable in `full`.
A 9-DOF AHRS (BNO055) gives absolute orientation incl. magnetometer heading (cheap → attitude tasks are
well-posed); a pressure sensor gives depth; a DVL gives body velocity (drift/current rejection without an
absolute position). `imu`/`imu_depth` therefore cannot hold horizontal position.

---

## Roadmap & current limitations

**What works today** (standalone Python, CPU): the analytical simulator + validation gate, the four RL
tasks with trained attitude-hold and attitude+direction-cruise policies (incl. disturbance + light
sim2real domain randomization), onboard cameras, the competition balloon-popping scenario running
end-to-end with the analytical feed-forward driver + scoring, a unified live viewer with a legible
default world, an interactive drive tool (steer a trained policy from the keyboard), and — for the
balloon competition — a learned onboard balloon detector (backed by an underwater sim2real
synthetic-data pipeline) feeding a rule-based behaviour FSM that pops balloons perception-in-the-loop
end-to-end (`tools/autonomy_run`, no RL).

**Not supported yet:**
- **ROS 2 integration — done** (in the sibling `ros2_ws/`, not this repo). Physics lives in **one**
  place (the Python sim); the `ros2_control` hardware plugin (`umiusi_sim_bridge`) is a **thin IPC
  relay** — it marshals each control cycle's command/state over a Unix socket to the Python sim server
  (`tools/sim_server.py`), holding no physics itself, so the REAL `sinsei_umiusi_control` controllers
  drive the sim unchanged (swap the plugin for CAN to deploy). A trained RL policy can also drive it
  over the thruster direct-override path (`tools/ros_policy.py`), and `tools/ros_view.py` renders it
  live (both over rosbridge). **Deploy nodes** (`ros2_ws/src/umiusi_autonomy`): `perception_node` +
  `navigator_node` wrap `umiusi_perception` so the same detector + FSM run on the robot. See
  [`ros2_ws/src/umiusi_sim_bridge/README.md`](../ros2_ws/src/umiusi_sim_bridge/README.md) and the
  `umiusi_autonomy` README. Remaining: aarch64 MuJoCo for the Pi; FF-frame sign reconcile.
- **3-D depth-holding locomotion — a single 3-D policy is not practical yet.** The env supports a full
  3-D velocity command (`vel_cmd_horizontal: false`, with an elevation curriculum + a horizontal-episode
  floor), and the family was trained out to `av_cal5_3d` — but the best 3-D policy still **fails 29 of
  the 41 vectoring cells** (`tools/vectoring_eval`): oblique directions are unreliable and commanded
  ascent is *slower* than the passive buoyant drift. The multimodality (vertical "drone mode" is far
  easier than tangential cruise) keeps collapsing one mode. The deployment answer is therefore
  **depth-threshold mode switching on the robot** — a supervisor picks the horizontal policy or the
  vertical one from the pressure sensor (`sinsei_UMIUSI_autonomy` PR #17), rehearsed closed-loop in sim
  by `tools/mode_switch_eval.py`. `av_cal5_3d_rep103` ships as a **descent-only, EXPERIMENTAL** bundle.
- **Perception.** On the **sim** scene, classical colour detection is done (`tools.perception_demo`,
  `umiusi_perception`, installed by `uv sync`: colour + bearing + range, ~100%). On **real
  underwater images** classical CV hits a recall wall (red attenuates to near-invisible, blue ≈ pool
  water); colour + a Hough-circle shape pass cut the false-positive flood but can't recover red. A
  **learned tiny-CNN detector** (`umiusi_perception.learned_detector`, `tools.perception_train` /
  `perception_bench`, `uv sync --extra learn`) breaks that wall — a 40-image baseline lifts red recall
  0.00→0.82, blue→0.63 — and is **Pi-4-safe** (int8 ONNX, ~12–30 fps @320px projected). An
  **underwater colour-restoration** preprocess (`umiusi_perception.underwater`, `tools.underwater_correct`:
  red-channel compensation + white balance + CLAHE) recovers red-vs-blue — the hardest real-world
  ambiguity, since deep red attenuates to look blue — for both labelling and detector input. Synthetic
  data: `tools.gen_sim_dataset` renders the balloon scene, applies a physically-based underwater
  degradation (`umiusi_sim.rendering.underwater_sim`: depth-based colour attenuation + backscatter haze +
  turbidity + surface reflection, domain-randomised), and auto-labels balloons from the segmentation
  buffer — free perfectly-labelled data to pretrain on + a hard, difficulty-dialable eval set.
  Detectors are compared head-to-head (classical vs learned, per-colour P/R/F1) with
  `tools.perception_eval` / `perception_eval_learned`, and `tools.perception_pseudolabel` auto-drafts
  COCO labels for fast human correction. **Sim-to-real** (see `ai/balloon/campaign_results.md`):
  real-data-only is currently best on real footage; the sim's value is a large, condition-tagged
  stress-eval set, and closing the sim↔real **colour-cast gap** (domain-randomised water colour + a
  strong colour/resolution training aug) is the lever for making synthetic data help. Next: more
  labelled real frames, then the final int8 export + a real Pi 4 benchmark.
- **Autonomy (behavior FSM) — done.** A rule-based behaviour FSM (`umiusi_perception.autonomy`: search →
  approach → align → ram → camera-confirm, with multi-frame track voting, a ~4.5 m detection range gate,
  nearest-reachable-non-blue target selection + a wire-avoidance path guard, and blue avoidance) consumes
  the learned detector's output to replace `competition_run`'s ground-truth driver, driving
  perception-in-the-loop via the analytical feed-forward allocation (`tools/autonomy_run.py`;
  `--full-run` records the whole competition, `--render` watches live). Deployed on the robot by the
  `ros2_ws/src/umiusi_autonomy` nodes (same FSM). Remaining: ram/pop reliability (the ~15% hit-rate
  control lever), sim-to-real detector quality + the feed-forward frame-sign reconcile (see `control.py`).
- **Decoupled viewer — done (ROS path).** For the standalone Python sim, `tools.drive` / `--render`
  each launch their own in-process viewer. An rviz-style viewer that attaches to a *separately-running*
  sim now exists once the ROS bridge is up: the C++ `MujocoSystem` plugin publishes the MuJoCo `qpos`,
  and `tools/ros_view.py` (`uv sync --extra viz`) renders it over **rosbridge** via `roslibpy` (no rclpy).
- **Sim-to-real + Pi 4 deploy.** Domain-randomization hooks exist; on-hardware tuning is future.

**Planned order:** perception + behavior FSM (phase 5b) is runnable end-to-end; next is sim-to-real
detector tuning + the final int8 export / Pi 4 benchmark → the **in-water calibration campaign**
([`docs/calibration_plan.md`](docs/calibration_plan.md): thrust-map scale, buoyancy/restoring, drag,
added mass) → **narrow the domain randomization** to the measured error bars → **retrain** the deployed
policies on the tightened physics, running the `max_duty` 0.2 → 0.4 protocol (0.4 is required for the
depth/vertical mode) → on-hardware sim-to-real (aarch64 MuJoCo). ROS 2 bridge (phase 4) is done. See the
Status table above.

---

## Configuration

All physical parameters live in [`configs/umiusi.yaml`](configs/umiusi.yaml): water density / displaced volume /
buoyancy offset, diagonal linear+quadratic drag, lift + a CoP-offset translation moment (both on by default;
optional off-diagonal damping off), added mass (ON — estimated), thrust map, servo range/slew + a converging
servo tracking model (`servo_tau_s`), and the measured hull mass/CoM/inertia. Drag and added mass are
**derived estimates** (CAD silhouette areas x BlueROV2-identified effective coefficients — reproduce with
`python -m tools.estimate_hydro`); `displaced_volume` and the thrust map are first estimates pending hardware
confirmation. `validate_sim -v` prints calibration numbers, and [`docs/calibration_plan.md`](docs/calibration_plan.md)
lists the in-water experiments that replace the estimates with measurements.

The robot geometry lives in [`packages/sim/src/umiusi_sim/description/umiusi.xml`](packages/sim/src/umiusi_sim/description/umiusi.xml). It is intentionally coarse
(octagonal-prism hull + T-shaped azimuth thrusters + two onboard cameras) for speed; **dynamics use the
measured mass/inertia**, not the coarse shapes.

**Coordinate frame:** the CAD frame with **+Y up** (the 4 thrusters lie in the X-Z plane). Gravity is `(0, -9.81, 0)`.

**Assumptions to verify against hardware** (see the notes in `configs/umiusi.yaml`): servo rotation axis = +Y, thrust
direction = each thruster's local +X, and the mounting neutral angles. The id mapping itself is now settled:
the sim's *geometry* naming is `id1=lf, id2=lb, id3=rf, id4=rb` (+Z = starboard), while the **action channel
order is the separate `action_order: [lf, lb, rb, rf]` name contract** (matching the autonomy `POSITIONS`).
What remains is confirming the physical wiring — which unit actually spins per channel (`sinsei_UMIUSI_autonomy`
issue #18, experiment 1; `docs/calibration_plan.md` §1).

The MJCF hull uses **primitives** (not a CAD mesh) for speed, but the **dynamics use the measured
mass/inertia**, so fidelity is in the numbers, not the shapes.
