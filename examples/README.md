# Example policies

Pre-trained policies so you can try `eval` / `drive` without training first.

- **`cruise_policy/`** — an `attitude_velocity` policy (hold orientation + cruise in a commanded
  direction). Trained on the bare model; ~100% success on the cruise task (some servo motion — see
  the Roadmap). Try:

  ```bash
  python -m tools.drive --model examples/cruise_policy/final.zip                # keyboard-drive it (GUI)
  python -m umiusi_rl.eval --model examples/cruise_policy/final.zip --no-disturb   # headless metrics
  ```

Each policy dir has `final.zip` (the SB3 policy), `vecnormalize.pkl` (obs-normalization stats), and
`meta.yaml` (task + sensor suite, so eval/drive reload it automatically).
