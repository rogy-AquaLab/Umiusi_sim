# Contributing

Thanks for helping on UMIUSI. This repo is the standalone Python simulation / perception / RL
project; the ROS 2 integration lives in the sibling `ros2_ws/` workspace.

## Setup

`uv` is the one-command bootstrap (see [README](README.md#setup) for the full install matrix):

```bash
uv sync            # full dev env: both wheels editable + mujoco + rl + tooling
# heavier extras when needed: uv sync --extra learn   (detector training)
#                             uv sync --extra viz     (ROS viewer)
```

Run anything with `uv run` (e.g. `uv run python -m tools.validate_sim`); the synced `.venv`
already has every runtime dep. Offscreen rendering needs a GL backend — `MUJOCO_GL=egl` on a
GPU/EGL box, or `MUJOCO_GL=osmesa` on a plain headless machine.

## Before you open a PR

- **Lint** — CI runs `ruff check`; keep it clean:
  ```bash
  uv run ruff check
  ```
  Line length is **119**. `ruff format` is *not* enforced — match the surrounding style
  (aligned trailing comments etc.); do not blanket-reformat existing files.
- **Tests** — run the suite:
  ```bash
  uv run pytest
  ```
- **Keep diffs minimal and scoped.** Don't refactor unrelated code, and don't rename public
  API — Python package/module names or the ROS 2 topic/service/parameter names in `ros2_ws/`.
- **Keep docs in sync.** A new tool / flag / task / behavior must be reflected in `README.md`,
  `tools/README.md`, or `docs/` in the same change.

## Layout

Three layered packages under `packages/` (details in [`docs/architecture.md`](docs/architecture.md)):

- `umiusi_sim` — core MuJoCo sim + robot description (no torch).
- `umiusi_perception` — the on-robot wheel: detector + tracker + underwater restore + balloon FSM
  + feed-forward allocation. Imported by the ROS deploy nodes.
- `umiusi_rl` — Gymnasium env + SB3 training/eval.

`tools/` holds repo-level CLI entry points (not shipped in the wheels); see
[`tools/README.md`](tools/README.md).
