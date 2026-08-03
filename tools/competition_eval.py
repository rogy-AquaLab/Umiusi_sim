"""tools/competition_eval.py — headless full-FIELD competition success/time metric.

Runs the deployed behaviour FSM over the sampled balloon FIELD (scn.sample_layout: red/yellow/blue per
configs/umiusi.yaml) with ground-truth (optionally degraded) detections, until every POSITIVE balloon
is popped or the competition timeout. Over many episodes it reports the SUCCESS RATE (cleared all
positives, popped no blue) and the TIME-TO-CLEAR distribution — the "how long to pop them all, and how
often" question. No GL/camera (GT detections isolate control); for the rendered single run use
tools/autonomy_run. Supports the pin study (--pin-tip/--pin-base/--pin-aware) and the perception model.

Usage:
    uv run python -m tools.competition_eval --episodes 24 --minutes 3
    uv run python -m tools.competition_eval --episodes 24 --pin-tip 0.28,0.02,0 --pin-base 0.15,0.02,0 --pin-aware
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

from umiusi_perception.autonomy import BalloonBehavior
from umiusi_perception.control import feedforward_allocation
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator
from tools.ram_eval import (CAM_H, CAM_W, FOVY_DEG, degrade_projection, false_positive,  # noqa: E402
                            make_detection, project_balloon)

START = (0.0, 1.0, 0.0)


def run_episode(rng, args, xml_path):
    """One competition episode; return a result dict (cleared, t_clear, score, blue_popped, wire)."""
    layout = scn.sample_layout(rng)
    pin_kw = {}
    if args.pin_base is not None:
        pin_kw["pin_base"] = args.pin_base
    if args.pin_tip is not None:
        pin_kw["pin_tip"] = args.pin_tip
    xml_path.write_text(scn.build_spec(layout=layout, **pin_kw).to_xml())
    sim = UmiusiSimulator(model_path=xml_path)
    sim.reset(pos=START)
    balloons = scn.balloon_table(layout=layout)
    positive = {b["name"] for b in balloons if b["points"] > 0}
    cam_id = sim.model.camera("front_cam").id
    pin_sid = sim.model.site("pin_tip").id
    rate = float(sim.cfg["sim"]["control_rate_hz"])
    dt = 1.0 / rate
    stride = max(1, round(rate / args.perception_hz)) if args.perception_hz > 0 else 1

    pin_offset = None
    if args.pin_aware:
        cam = sim.model.camera("front_cam").pos
        tip = args.pin_tip if args.pin_tip is not None else scn.PIN_TIP
        pin_offset = (tip[0] - cam[0], tip[1] - cam[1], tip[2] - cam[2])
    fsm = BalloonBehavior(frame_h=CAM_H, frame_w=CAM_W, fovy_deg=FOVY_DEG, dt=dt, pin_offset=pin_offset)

    n_steps = int(round(args.minutes * 60 * rate))
    popped, score, prev_pin, held = set(), 0, None, []
    snag_prev, wire_events, t_clear = set(), 0, None
    for k in range(n_steps):
        st = sim.get_state()
        R = sim.data.xmat[sim.base_id].reshape(3, 3)
        cam_pos = sim.data.cam_xpos[cam_id]
        if k % stride == 0:  # detector tick: GT detections of every un-popped balloon (FOV-gated)
            dets = []
            for b in balloons:
                if b["name"] in popped:
                    continue
                proj = project_balloon(R.T @ (b["pos"] - cam_pos))
                if proj is None:
                    continue
                dproj = degrade_projection(proj, rng, args)
                if dproj is None:
                    continue
                d = make_detection(*dproj, b["colour"], b["points"])
                if d is not None:
                    dets.append(d)
            if args.fp_rate > 0 and rng.random() < args.fp_rate:
                fp = false_positive(rng)
                if fp is not None:
                    dets.append(fp)
            held = dets
            fresh = True
        else:
            dets, fresh = held, False
        heading = float(math.atan2((R @ [1.0, 0, 0])[2], (R @ [1.0, 0, 0])[0]))
        cmd, info = fsm.step(dets, float(st["ang_vel"][1]), heading=heading, dt=dt, fresh=fresh)
        sim.step(feedforward_allocation([0, 0, cmd["yaw"]], [-cmd["surge"], 0, cmd["heave"]]))

        pin_tip = sim.data.site_xpos[pin_sid].copy()
        axis = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0, 0])
        vel = (pin_tip - prev_pin) / dt if prev_pin is not None else np.zeros(3)
        prev_pin = pin_tip
        for b in balloons:
            if b["name"] not in popped and scn.popped(pin_tip, b["pos"], axis, vel):
                popped.add(b["name"])
                score += b["points"]
        snag = set(scn.entanglement(sim.data.xpos[sim.base_id], balloons, popped))
        wire_events += len(snag - snag_prev)
        snag_prev = snag
        if positive <= popped:  # all positive cleared
            t_clear = (k + 1) * dt
            break

    blue_popped = sum(1 for b in balloons if b["name"] in popped and b["points"] < 0)
    return {"cleared": t_clear is not None, "t_clear": t_clear, "score": score,
            "blue_popped": blue_popped, "wire": wire_events,
            "n_positive": len(positive), "n_pos_popped": len(positive & popped)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=3.0, help="competition timeout per episode")
    ap.add_argument("--perception-hz", type=float, default=0.0)
    ap.add_argument("--bearing-noise-deg", type=float, default=0.0)
    ap.add_argument("--range-noise", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fp-rate", type=float, default=0.0)
    ap.add_argument("--pin-tip", type=str, default=None)
    ap.add_argument("--pin-base", type=str, default=None)
    ap.add_argument("--pin-aware", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.pin_tip = tuple(float(v) for v in args.pin_tip.split(",")) if args.pin_tip else None
    args.pin_base = tuple(float(v) for v in args.pin_base.split(",")) if args.pin_base else None
    args.yellow_frac = 0.0  # unused here (field sampled from config), but ram_eval helpers read args

    xml_path = Path(tempfile.gettempdir()) / "umiusi_sim" / "competition_eval.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    pin = "default" if not args.pin_aware and args.pin_tip is None else \
        f"{args.pin_tip} aware={args.pin_aware}"
    print(f"competition_eval: {args.episodes} episodes  timeout={args.minutes}min  pin={pin}  "
          f"perception(hz={args.perception_hz} bearing={args.bearing_noise_deg} dropout={args.dropout})")
    rows = []
    for e in range(args.episodes):
        r = run_episode(np.random.default_rng(1000 + args.seed + e), args, xml_path)
        rows.append(r)
        if args.verbose:
            tc = f"{r['t_clear']:.1f}s" if r["cleared"] else "TIMEOUT"
            print(f"  ep {e:2d}: {'CLEAR' if r['cleared'] else 'fail '} {tc:>8s}  "
                  f"pos {r['n_pos_popped']}/{r['n_positive']}  score {r['score']:+d}  "
                  f"blue {r['blue_popped']}  wire {r['wire']}", flush=True)

    n = len(rows)
    ok = [r for r in rows if r["cleared"] and r["blue_popped"] == 0]
    cleared = [r for r in rows if r["cleared"]]
    times = sorted(r["t_clear"] for r in cleared)
    print("-" * 72)
    print(f"SUCCESS (all positives popped, no blue): {len(ok)}/{n} = {len(ok) / n:.0%}")
    print(f"  cleared all positives (blue ignored):  {len(cleared)}/{n} = {len(cleared) / n:.0%}")
    if times:
        print(f"  time-to-clear (cleared eps): median {np.median(times):.1f}s  "
              f"mean {np.mean(times):.1f}s  p90 {np.percentile(times, 90):.1f}s  "
              f"min {times[0]:.1f}s  max {times[-1]:.1f}s")
    frac_pos = np.mean([r["n_pos_popped"] / max(1, r["n_positive"]) for r in rows])
    mean_score = np.mean([r["score"] for r in rows])
    print(f"  positives popped (all eps): {frac_pos:.0%} mean  |  mean score {mean_score:+.0f}"
          f"  |  blue-pop eps {sum(1 for r in rows if r['blue_popped'])}/{n}"
          f"  |  wire under-passes {np.mean([r['wire'] for r in rows]):.1f}/ep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
