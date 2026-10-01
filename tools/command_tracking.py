"""Does the vehicle do what the FSM asks? Step responses through the DEPLOY chain.

For each FSM command (surge / heave / yaw, the [-1, 1] units behavior.py emits) this holds a step for
`--hold` seconds and reports what the hull actually reaches against what the chain INTENDS:

    surge  -> cmd * surge_scale            [m/s]   (navigator setpoint)
    heave  -> cmd * surge_scale            [m/s]
    yaw    -> cmd * yaw_rate_scale         [rad/s] (control D-1 rate loop; + = toward body +Z)

and the cross-talk on the other axes (depth drift while surging, pitch while yawing, ...).

The point is MISMATCH. The controller carries the bundle's plant; the real thrust is uncalibrated
(thrust_per_cmd and the curve exponent come from a contaminated fit, configs/umiusi.yaml). So the same
steps are repeated on plants whose thrust gain / exponent differ from what the controller believes.
The heave axis is open loop (k_v_vert = 0), so it is where a wrong thrust model shows up directly.

Usage:
    uv run python -m tools.command_tracking
    uv run python -m tools.command_tracking --control-hz 33.8
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from tools.deploy_driver import DeployDriver, heading_about_up
from umiusi_sim.simulator import UmiusiSimulator

STEPS = [("surge", 0.3), ("surge", 0.6), ("surge", 1.0), ("heave", 0.5), ("heave", -0.5),
         ("heave", 1.0), ("yaw", 0.5), ("yaw", 1.0)]
PLANTS = [("nominal", 1.0, None), ("thrust x0.6", 0.6, None), ("thrust x1.4", 1.4, None),
          ("exp 1.6", 1.0, 1.6), ("exp 2.4", 1.0, 2.4)]


def run_step(axis, value, gain, exp, args):
    sim = UmiusiSimulator()
    sim.set_net_buoyancy(0.0)
    if exp is not None:
        sim.thrust_curve_exp = exp
    sim.thrust_per_cmd *= gain
    sim.reset(pos=(0.0, 1.5, 0.0))
    drv = DeployDriver(sim, max_duty=args.max_duty)
    dt = 1.0 / float(sim.cfg["sim"]["control_rate_hz"])
    period = 1.0 / args.control_hz if args.control_hz > 0 else dt
    drv.dt, drv.ctl.dt = period, period
    n = int(round(args.hold / dt))
    t_ctl, action = period, np.zeros(8)
    cmd = {axis: value}
    trace, y0 = [], None
    for _ in range(n):
        t_ctl += dt
        if t_ctl >= period - 1e-9:
            t_ctl -= period
            action = drv.step(cmd)
        st = sim.step(action)
        R = sim.data.xmat[sim.base_id].reshape(3, 3)
        v_body = R.T @ st["lin_vel"]
        w_body = R.T @ st["ang_vel"]
        if y0 is None:
            y0 = st["pos"][1]
        pitch = math.degrees(math.asin(float(np.clip((R @ [1.0, 0, 0])[1], -1, 1))))
        roll = math.degrees(math.asin(float(np.clip((R @ [0, 0, 1.0])[1], -1, 1))))
        # REP-103 yaw rate is about CAD +Y; the FSM's +yaw is the NEGATIVE of it (see deploy_driver)
        trace.append((v_body[0], st["lin_vel"][1], -w_body[1], st["pos"][1] - y0, pitch, roll,
                      heading_about_up(R)))
    tr = np.array(trace)
    col = {"surge": 0, "heave": 1, "yaw": 2}[axis]
    scale = drv.yaw_rate_scale if axis == "yaw" else drv.surge_scale
    want = value * scale
    tail = tr[int(0.6 * n):, col]
    got = float(np.mean(tail))
    final = got if abs(got) > 1e-6 else 1e-6
    reach = np.where(np.abs(tr[:, col]) >= 0.63 * abs(final))[0]
    t63 = float(reach[0] * dt) if reach.size else float("nan")
    return {"want": want, "got": got, "ratio": got / want if want else float("nan"), "t63": t63,
            "dz": float(tr[-1, 3]), "pitch": float(np.max(np.abs(tr[:, 4]))),
            "roll": float(np.max(np.abs(tr[:, 5]))), "sat": float(np.mean(np.abs(action[4:]) >= args.max_duty - 1e-6))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hold", type=float, default=8.0, help="seconds per step")
    ap.add_argument("--max-duty", type=float, default=0.3)
    ap.add_argument("--control-hz", type=float, default=0.0)
    args = ap.parse_args()
    print(f"command_tracking: cap={args.max_duty} control_hz={args.control_hz or 50} hold={args.hold}s  "
          "(ratio = reached / intended over the last 40 %; t63 = time to 63 % of the reached value)")
    for pname, gain, exp in PLANTS:
        print(f"--- plant: {pname}")
        for axis, value in STEPS:
            r = run_step(axis, value, gain, exp, args)
            unit = "rad/s" if axis == "yaw" else "m/s"
            print(f"  {axis:5s} {value:+.1f}: want {r['want']:+.3f} {unit}  got {r['got']:+.3f}  "
                  f"ratio {r['ratio']:5.2f}  t63 {r['t63']:4.1f}s  | depth drift {r['dz']:+.2f} m  "
                  f"max pitch {r['pitch']:4.1f}  roll {r['roll']:4.1f} deg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
