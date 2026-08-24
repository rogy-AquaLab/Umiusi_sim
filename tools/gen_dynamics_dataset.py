"""Generate a DYNAMICS transition dataset from the simulator, for world-model / residual-model work.

Rolls the analytical simulator under diverse excitation and records, at the control rate,
    obs_t      : pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + servo(4) + thrust(4)   (world frame)
    action_t   : the 8-D command [servo x4, esc x4] in [-1, 1] (action_order channels)
    obs_{t+1}  : same layout as obs_t
    params     : per-episode physics parameters (drag/added-mass/thrust/servo scalings)

Two intended consumers:
  * a learned world model of the vehicle+water dynamics (train on (obs, action) -> next obs;
    the per-episode `params` let the model be conditioned on — or marginalize over — the
    physics uncertainty, matching the DR ranges used for policy training);
  * a RESIDUAL hydrodynamics model: the same schema is what the real-robot bags reduce to
    (docs/calibration_plan.md — the random-teleop segments), so a network can be fit to
    real-minus-analytical residuals without any format work.

Excitation mixes three modes per episode (chosen at random): smoothed random walk (broad-band),
constant commands (steady states — the drag/terminal-velocity information), and step changes
(transients — the added-mass information).

Usage:
    python -m tools.gen_dynamics_dataset --episodes 200 --steps 300 --out out/dyn_ds --seed 0
    python -m tools.gen_dynamics_dataset --domain-rand   # also randomize physics per episode

Output: <out>/transitions.npz with arrays obs (N, T+1, 21), act (N, T, 8), params (N, P),
param_names, plus a JSON sidecar describing the layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from umiusi_sim.simulator import UmiusiSimulator

OBS_LAYOUT = "pos(0:3) quat(3:7) lin_vel(7:10) ang_vel(10:13) servo(13:17) thrust_applied(17:21)"
PARAM_NAMES = ["volume", "thrust_per_cmd", "servo_slew_rad", "servo_tau",
               "drag_lin_scale", "drag_quad_scale", "added_mass_scale"]


def _obs(sim):
    s = sim.get_state()
    return np.concatenate([s["pos"], s["quat"], s["lin_vel"], s["ang_vel"], s["servo"], s["thrust"]])


def _excite(rng, steps):
    """One episode's 8-D command sequence: random-walk / constant / step mixture."""
    mode = rng.integers(3)
    if mode == 0:    # smoothed random walk (broad-band excitation)
        raw = rng.uniform(-1, 1, size=(steps, 8))
        out = np.zeros_like(raw)
        a = rng.uniform(0.02, 0.2)   # smoothing pole
        prev = rng.uniform(-1, 1, size=8)
        for t in range(steps):
            prev = (1 - a) * prev + a * raw[t]
            out[t] = prev
        return np.clip(out, -1, 1)
    if mode == 1:    # piecewise-constant (steady states -> drag information)
        out = np.zeros((steps, 8))
        t = 0
        while t < steps:
            hold = int(rng.integers(steps // 4, steps // 2 + 1))
            out[t:t + hold] = rng.uniform(-1, 1, size=8)
            t += hold
        return out
    # steps from rest (transients -> added-mass information)
    out = np.zeros((steps, 8))
    t0 = int(rng.integers(steps // 8, steps // 3))
    out[t0:] = rng.uniform(-1, 1, size=8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300, help="control steps per episode (300 = 6 s)")
    ap.add_argument("--out", default="out/dyn_ds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain-rand", action="store_true",
                    help="randomize physics per episode over the training-DR ranges")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = UmiusiSimulator()
    base = dict(volume=sim.volume, thrust=sim.thrust_per_cmd, slew=sim.servo_slew_rad,
                tau=sim.servo_tau, dlin=sim.drag_lin.copy(), dquad=sim.drag_quad.copy(),
                am=sim.added_mass_diag.copy())

    N, T = args.episodes, args.steps
    obs = np.zeros((N, T + 1, 21), dtype=np.float32)
    act = np.zeros((N, T, 8), dtype=np.float32)
    params = np.zeros((N, len(PARAM_NAMES)), dtype=np.float32)

    for n in range(N):
        dls = dqs = ams = 1.0
        if args.domain_rand:  # mirror configs/train_ppo.yaml domain_rand ranges
            sim.volume = base["volume"] * (1 + rng.uniform(-1, 1) * 0.05)
            sim.thrust_per_cmd = base["thrust"] * (1 + rng.uniform(-1, 1) * 0.15)
            dls, dqs, ams = 1 + rng.uniform(-1, 1) * 0.3, 1 + rng.uniform(-1, 1) * 0.3, 1 + rng.uniform(-1, 1) * 0.4
            sim.drag_lin, sim.drag_quad = base["dlin"] * dls, base["dquad"] * dqs
            sim.added_mass_diag = base["am"] * ams
            sim.servo_slew_rad = np.radians(rng.uniform(100.0, 500.0))
            sim.servo_tau = base["tau"] * (1 + rng.uniform(-1, 1) * 0.5)
        params[n] = [sim.volume, sim.thrust_per_cmd, sim.servo_slew_rad, sim.servo_tau, dls, dqs, ams]

        # random initial attitude/velocity so the transition distribution is not all near-hover
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        if quat[0] < 0:
            quat = -quat
        sim.reset(pos=tuple(rng.uniform(-0.2, 0.2, size=3)), quat=tuple(quat))
        cmds = _excite(rng, T)
        obs[n, 0] = _obs(sim)
        for t in range(T):
            sim.step(cmds[t])
            act[n, t] = cmds[t]
            obs[n, t + 1] = _obs(sim)
        if not np.all(np.isfinite(obs[n])):
            raise RuntimeError(f"non-finite state in episode {n} (physics blew up)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "transitions.npz", obs=obs, act=act, params=params,
                        param_names=np.array(PARAM_NAMES))
    meta = {"obs_layout": OBS_LAYOUT, "act_layout": "[servo x4, esc x4] in [-1,1], action_order",
            "control_rate_hz": sim.cfg["sim"]["control_rate_hz"], "episodes": N, "steps": T,
            "domain_rand": bool(args.domain_rand), "seed": args.seed,
            "param_names": PARAM_NAMES}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    size_mb = (out / "transitions.npz").stat().st_size / 1e6
    print(f"wrote {out}/transitions.npz  ({N} episodes x {T} steps, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
