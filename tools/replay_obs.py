"""Closed-loop policy probe with NO plant — the bench test golden.npz cannot do.

`preflight_policy.py` replays constructed observations open-loop (obs -> act, one shot). That
misses every defect in a path that carries STATE. A wrench-mode policy carries two:

    m            the mode integrator (m += a * mode_slew_per_s * dt, clipped, never leaks)
    prev_servo   held by the mixer deadband

and the observation itself carries `prev_action`, so the policy is in a feedback loop with its
own output even when the vehicle does not move. On a bench with the motors disconnected the
observation never changes, the integrator has nothing to unwind it, and duty pins at ±max_duty
within ~1 s. That is correct behaviour, not a fault — but it means **duty tells you nothing
about policy health on a static bench**. Drive it with a time-varying observation instead.

Usage:
    python tools/replay_obs.py --model models/av_mode13 --profile sweep
    python tools/replay_obs.py --model models/av_mode13 --profile static --steps 400

Profiles:
    static  fixed attitude error — reproduces the bench rail (expect: rails, then constant)
    step    error steps to +0.3 rad at t=2 s and back at t=6 s
    sweep   attitude error sweeps sinusoidally; the integrator should track, not rail

The deploy side can port this directly: the only sim dependency is ModeMixer, and the contract
values it needs are the ones already in the bundle's meta.json `action_contract`.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))

from umiusi_rl.envs.mode_mixer import MODE_NAMES, ModeMixer  # noqa: E402
from umiusi_sim.simulator import UmiusiSimulator  # noqa: E402


def obs_profile(name, t, dt):
    """-> ori_err(3), gyro(3), v_cmd(3). The vehicle never moves; only the TARGET error varies."""
    if name == "static":
        return np.array([0.02, 0.0, 0.0]), np.zeros(3), np.zeros(3)
    if name == "step":
        e = 0.3 if 2.0 <= t * dt < 6.0 else 0.0
        return np.array([e, 0.0, 0.0]), np.zeros(3), np.zeros(3)
    if name == "sweep":
        w = 2.0 * np.pi * 0.25                      # 0.25 Hz — slow next to the 2.0/s mode slew
        return np.array([0.3 * np.sin(w * t * dt), 0.15 * np.cos(w * t * dt), 0.0]), np.zeros(3), np.zeros(3)
    raise SystemExit(f"unknown profile {name!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="run dir (final.zip + vecnormalize.pkl + export/)")
    ap.add_argument("--profile", default="sweep", choices=["static", "step", "sweep"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-duty", type=float, default=0.25)
    args = ap.parse_args()

    run = Path(args.model)
    contract = json.loads((run / "export" / "meta.json").read_text()).get("action_contract")
    if contract is None:
        raise SystemExit("this probe is for action_mode 'modes' bundles (no action_contract found)")
    slew, dt = contract["mode_slew_per_s"], 1.0 / contract["control_rate_hz"]

    model = PPO.load(str(run / "final.zip"), device="cpu")
    with open(run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    sim = UmiusiSimulator()
    mixer = ModeMixer(sim.unit_names, sim.thrust_axes, sim.unit_pivots, sim.servo_range_rad,
                      sim.thrust_per_cmd, sim.thrust_curve_exp)

    m = np.zeros(len(MODE_NAMES))
    prev_servo, prev_action = np.zeros(4), np.zeros(8)
    railed_steps = 0
    print(f"{run.name}  profile={args.profile}  slew={slew}/s  dt={dt:.3f}s  max_duty={args.max_duty}")
    print(f"{'t[s]':>6} {'ori_err_x':>10} {'|m|max':>7} {'railed':>7} {'esc':>28} {'servo[deg]':>28}")
    for t in range(args.steps):
        ori, gyro, vcmd = obs_profile(args.profile, t, dt)
        obs = np.concatenate([ori, gyro, vcmd, prev_action, [args.max_duty]])
        o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
        a, _ = model.predict(o, deterministic=True)
        m = np.clip(m + a * slew * dt, -1.0, 1.0)
        act = mixer.mix(m, args.max_duty, prev_servo)
        prev_servo, prev_action = act[:4].copy(), act.copy()
        n_rail = int(np.sum(np.abs(m) >= 0.999))
        railed_steps += n_rail > 0
        if t % max(1, args.steps // 12) == 0 or t == args.steps - 1:
            print(f"{t * dt:6.2f} {ori[0]:10.3f} {np.max(np.abs(m)):7.3f} {n_rail:7d} "
                  f"{np.array2string(act[4:], precision=3, floatmode='fixed'):>28} "
                  f"{np.array2string(act[:4] * 90.0, precision=0, floatmode='fixed'):>28}")

    frac = railed_steps / args.steps * 100.0
    print(f"\nmode integrator at a rail on {frac:.0f}% of steps; final m = "
          f"{dict(zip(MODE_NAMES, np.round(m, 3)))}")
    if args.profile == "static" and frac < 50.0:
        print("UNEXPECTED: a static observation should wind the integrator to a rail and hold it.")
    if args.profile == "sweep" and frac > 90.0:
        print("WARNING: railed for the whole sweep — the policy is not tracking a varying target.")


if __name__ == "__main__":
    main()
