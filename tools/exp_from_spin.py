"""Thrust-curve exponent from a yaw spin — IMU only, ~1 minute in the pool.

WHY. The attitude loop's stability depends on the exponent (tools/command_tracking, 2026-10-01): if the
real curve is FLATTER than the bundle assumes, the small-signal gain near zero force blows up and the
hull falls into a 25-40 deg roll/pitch limit cycle at hold. The exponent in configs/umiusi.yaml is from
a contaminated fit. This needs no depth sensor and no external reference: the gyro is the best sensor
on the robot.

METHOD. All servos at 0 deg (thrust horizontal, tangential), esc signs for pure yaw, duty u held until
the yaw rate settles, at 3+ duties. In steady state thrust torque = drag torque:
        c * u**exp = lin * w + quad * w**2
so for two duties  exp = log(D(w2) / D(w1)) / log(u2 / u1)  with D the drag model — the unknown gain c
cancels. The drag coefficients are ESTIMATES (configs/umiusi.yaml drag, row rot+Y), so the result
inherits their lin/quad split; the fit over 3 duties shows whether it is consistent.

    uv run python -m tools.exp_from_spin                  # sim self-check: recovers the plant's exponent?
    uv run python -m tools.exp_from_spin --plant-exp 1.4
    uv run python -m tools.exp_from_spin --rates 0.10:0.31,0.20:0.55,0.30:0.74   # pool data u:w[rad/s]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from umiusi_sim.simulator import UmiusiSimulator

DUTIES = (0.10, 0.20, 0.30)


def yaw_signs(sim):
    """esc sign per unit (action order) that makes +yaw torque about CAD +Y with servos at 0."""
    t = np.asarray(sim.thrust_axes, dtype=float)
    r = np.asarray(sim.unit_pivots, dtype=float)
    return np.sign([np.cross(r[k], t[k])[1] for k in range(4)])


def spin(u, plant_exp, hold=12.0):
    sim = UmiusiSimulator()
    sim.set_net_buoyancy(0.0)
    sim.max_duty = 1.0
    if plant_exp is not None:
        sim.thrust_curve_exp = plant_exp
    sim.reset(pos=(0.0, 1.5, 0.0))
    a = np.concatenate([np.zeros(4), u * yaw_signs(sim)])
    w = []
    for _ in range(int(hold / 0.02)):
        st = sim.step(a)
        w.append(abs(float(st["ang_vel"][1])))
    return float(np.mean(w[int(0.6 * len(w)):])), sim


def fit(pairs, lin, quad):
    us = np.array([p[0] for p in pairs])
    d = np.array([lin * p[1] + quad * p[1] ** 2 for p in pairs])
    slope, _ = np.polyfit(np.log(us), np.log(d), 1)
    pair_est = [math.log(d[i + 1] / d[i]) / math.log(us[i + 1] / us[i]) for i in range(len(us) - 1)]
    return float(slope), pair_est


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plant-exp", type=float, default=None)
    ap.add_argument("--rates", default=None, help="pool data: 'u:w,u:w,...' (duty : steady |yaw rate| rad/s)")
    args = ap.parse_args()
    sim = UmiusiSimulator()
    lin, quad = float(sim.drag_lin[4]), float(sim.drag_quad[4])   # rot+Y = yaw (CAD +Y up)
    if args.rates:
        pairs = [tuple(float(x) for x in s.split(":")) for s in args.rates.split(",")]
        src = "pool data"
    else:
        pairs = []
        for u in DUTIES:
            w, s = spin(u, args.plant_exp)
            pairs.append((u, w))
        src = f"sim, true exp = {s.thrust_curve_exp}"
    slope, pair_est = fit(pairs, lin, quad)
    print(f"exp_from_spin ({src}); yaw drag lin {lin} quad {quad}")
    for u, w in pairs:
        print(f"  u {u:.2f} -> |w| {w:.3f} rad/s")
    print(f"  fitted exponent {slope:.2f}   (pairwise {', '.join(f'{e:.2f}' for e in pair_est)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
