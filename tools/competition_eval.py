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
    uv run python -m tools.competition_eval --episodes 8 --net-buoy -0.5 --cob-fwd 0.01   # one plant variant
    uv run python -m tools.competition_eval --episodes 6 --sweep                          # buoyancy x CoB grid

DRIVER. `--driver deploy` (default since 2026-10-01) runs the robot's chain — navigator setpoint ->
yaw rate (control D-1) -> ClassicalController -> GeneralAllocator with the deploy BUNDLE's gains and
allocator knobs, buoyancy trim OFF (D-3). See tools/deploy_driver.py. `--driver ff` is the old
`feedforward_allocation` path, kept only for comparison: it has no attitude loop, assumes a linear
thrust curve on an exponent-2 plant, and turns the nose AWAY from the FSM's +yaw (measured), which is
why it popped nothing.

PLANT VARIANTS. The hull is ballasted "roughly neutral" and its fore/aft balance is unmeasured, so a
single nominal number would be a guess. `--net-buoy` [N, + floats] and `--cob-fwd` / `--cob-lat` [m]
set the PLANT only; the controller keeps the bundle's plant, as on the robot. `--sweep` runs a grid.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from umiusi_perception.autonomy import BalloonBehavior
from umiusi_perception.control import feedforward_allocation
from tools.deploy_driver import DeployDriver
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator
from tools.ram_eval import (CAM_H, CAM_W, FOVY_DEG, degrade_projection, false_positive,  # noqa: E402
                            make_detection, project_balloon)

START = (0.0, 1.0, 0.0)
SURFACE_BAND = 0.15   # [m] from the water surface / the floor counts as "stuck there"


class _Tracer:
    """JSONL event log for diagnosing RAM misses and wire under-passes (`--trace-events FILE`).

    Events: "state" (every FSM transition), "ram" (one per RAM segment: geometry at entry, closest
    approach of the pin tip to the balloon the FSM is locked on, why it ended), "wire" (an under-pass:
    which balloon, FSM state, whether it was the locked target). Geometry is ground truth in the
    vehicle's body frame (x fwd, y up, z starboard) so a miss reads as "passed 12 cm low / 8 cm right".
    The locked target is identified as the un-popped balloon of the locked colour whose TRUE camera
    bearing is closest to the FSM's track bearing.
    """

    def __init__(self, path, ep_seed):
        self.f = open(path, "a")
        self.ep = ep_seed
        self.prev_state, self.ram = None, None

    def _emit(self, **kw):
        self.f.write(json.dumps({"ep": self.ep, **kw}) + "\n")

    @staticmethod
    def _target(fsm, sim, balloons, popped, cam_id):
        if not fsm.trk.colour:
            return None
        R = sim.data.xmat[sim.base_id].reshape(3, 3)
        cam = sim.data.cam_xpos[cam_id]
        best, best_d = None, float("inf")
        for b in balloons:
            if b["name"] in popped or b["colour"] != fsm.trk.colour:
                continue
            pr = project_balloon(R.T @ (b["pos"] - cam))
            if pr is None:
                continue
            d = math.hypot(pr[0] - fsm.trk.az, pr[1] - fsm.trk.el)
            if d < best_d:
                best, best_d = b, d
        return best

    @staticmethod
    def _body(sim, v):
        return (sim.data.xmat[sim.base_id].reshape(3, 3).T @ np.asarray(v)).round(3).tolist()

    def tick(self, t, fsm, sim, balloons, popped, cam_id, pin_sid, held, fresh):
        st = fsm.state
        if st != self.prev_state:
            tgt = self._target(fsm, sim, balloons, popped, cam_id)
            tip = sim.data.site_xpos[pin_sid]
            self._emit(ev="state", t=round(t, 2), frm=self.prev_state, to=st, colour=fsm.trk.colour,
                       trk=[round(fsm.trk.az, 3), round(fsm.trk.el, 3), round(fsm.trk.range_m, 2),
                            round(fsm.trk.bbox_frac, 3)], misses=fsm.trk.misses, fresh=fresh,
                       tgt=None if tgt is None else tgt["name"],
                       tip_to_tgt=None if tgt is None else self._body(sim, tgt["pos"] - tip))
            if st == "RAM" and self.prev_state != "RAM":
                self.ram = {"t0": t, "tgt": None if tgt is None else tgt["name"], "best": None,
                            "entry": None if tgt is None else self._body(sim, tgt["pos"] - tip),
                            "entry_trk": [round(fsm.trk.az, 3), round(fsm.trk.el, 3), round(fsm.trk.bbox_frac, 3)]}
            if self.prev_state == "RAM" and st != "RAM" and self.ram is not None:
                self._close_ram(t, st, popped)
            self.prev_state = st

    def step(self, t, fsm, sim, balloons, popped, pin_sid, vel, new_snags):
        if self.ram is not None and self.ram["tgt"] is not None:
            b = next(x for x in balloons if x["name"] == self.ram["tgt"])
            tip = sim.data.site_xpos[pin_sid]
            d = float(np.linalg.norm(b["pos"] - tip))
            if self.ram["best"] is None or d < self.ram["best"][0]:
                delta = b["pos"] - tip
                closing = float(np.dot(vel, delta / max(d, 1e-6)))
                axis = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0, 0])
                ang = math.degrees(math.acos(float(np.clip(np.dot(axis, delta / max(d, 1e-6)), -1, 1))))
                self.ram["best"] = (d, self._body(sim, delta), round(closing, 3), round(ang, 1), round(t, 2))
            if self.ram["tgt"] in popped and "popped_t" not in self.ram:
                self.ram["popped_t"] = round(t, 2)
        for name in new_snags:
            b = next(x for x in balloons if x["name"] == name)
            self._emit(ev="wire", t=round(t, 2), balloon=name, colour=b["colour"], state=fsm.state,
                       locked=fsm.trk.colour, rel=self._body(sim, b["pos"] - sim.data.xpos[sim.base_id]))

    def _close_ram(self, t, next_state, popped):
        r = self.ram
        best = r["best"]
        self._emit(ev="ram", t0=round(r["t0"], 2), dur=round(t - r["t0"], 2), tgt=r["tgt"], to=next_state,
                   popped=r["tgt"] in popped if r["tgt"] else False, entry=r["entry"], entry_trk=r["entry_trk"],
                   min_dist=None if best is None else round(best[0], 3),
                   at_min=None if best is None else best[1], closing=None if best is None else best[2],
                   angle=None if best is None else best[3])
        self.ram = None

    def close(self, t, popped):
        if self.ram is not None:
            self._close_ram(t, "END", popped)
        self.f.close()


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
    if args.net_buoy is not None:
        sim.set_net_buoyancy(args.net_buoy)
    if args.cob_fwd or args.cob_lat:
        sim.set_cob_horizontal(args.cob_fwd, args.cob_lat)
    sim.reset(pos=(START[0], args.start_height, START[2]))
    drv = None
    if args.driver == "deploy":
        drv = DeployDriver(sim, max_duty=args.max_duty, yaw_sign=args.yaw_sign, hold_yaw=args.hold_yaw,
                           alloc_overrides={"w_effort": args.w_effort} if args.w_effort is not None else None,
                           gain_overrides={"ki": args.ki} if args.ki is not None else None,
                           servo_aware=args.servo_aware,
                           contract_overrides=({"thrust_curve_exp": args.bundle_exp}
                                               if args.bundle_exp is not None else None))
        if args.plant_exp is not None:
            sim.thrust_curve_exp = args.plant_exp
    balloons = scn.balloon_table(layout=layout)
    positive = {b["name"] for b in balloons if b["points"] > 0}
    cam_id = sim.model.camera("front_cam").id
    pin_sid = sim.model.site("pin_tip").id
    rate = float(sim.cfg["sim"]["control_rate_hz"])
    dt = 1.0 / rate
    stride = max(1, round(rate / args.perception_hz)) if args.perception_hz > 0 else 1
    # CONTROL RATE. The sim steps at 50 Hz, but the robot's Python attitude node measured 33.8 Hz in the
    # full stack (sinsei_UMIUSI_autonomy/docs/performance_tuning.md §5). Run FSM + driver on their own
    # clock and HOLD the last action in between, as the ESC/servo do.
    ctl_period = 1.0 / args.control_hz if args.control_hz > 0 else dt
    if drv is not None:
        drv.dt = ctl_period
        drv.ctl.dt = ctl_period
    # LATENCY. A detection describes the frame it was computed from, not the moment it arrives.
    lat_steps = int(round(args.perception_latency / dt))
    recall_curve = ([tuple(float(v) for v in kv.split(":")) for kv in args.recall_curve.split(",")]
                    if args.recall_curve else None)
    pending = []   # [(deliver_step, dets)]
    # PERSISTENT FALSE POSITIVES. Real ones are not uniform noise: a surface reflection or a patch of
    # floor looks the same every frame, so the tracker CONFIRMS it and the FSM chases it. Model them as
    # fixed world points (half just under the surface, half on the floor) that are never poppable.
    phantoms = []
    for i in range(args.phantoms):
        y = scn.POOL_DEPTH - 0.1 if i % 2 == 0 else scn.FLOOR_Y + 0.1
        phantoms.append(np.array([rng.uniform(0.5, 5.0), y, rng.uniform(-2.0, 2.0)]))

    pin_offset = None
    if args.pin_aware:
        cam = sim.model.camera("front_cam").pos
        tip = args.pin_tip if args.pin_tip is not None else scn.PIN_TIP
        pin_offset = (tip[0] - cam[0], tip[1] - cam[1], tip[2] - cam[2])
    fsm = BalloonBehavior(frame_h=CAM_H, frame_w=CAM_W, fovy_deg=FOVY_DEG, dt=ctl_period, pin_offset=pin_offset,
                          ki_heave=args.ki_heave, propagate_bearing=args.propagate_bearing)

    n_steps = int(round(args.minutes * 60 * rate))
    popped, score, prev_pin, held = set(), 0, None, []
    snag_prev, wire_events, t_clear = set(), 0, None
    heights, first_pop, occ = [], None, {}
    t_ctl, action, cmd = ctl_period, np.zeros(8), {"surge": 0.0, "heave": 0.0, "yaw": 0.0}
    tracer = _Tracer(args.trace_events, ep_seed=args._ep_seed) if args.trace_events else None
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
                if d is not None and recall_curve:
                    # MEASURED detector: recall by apparent size (box width / image width), e.g. the
                    # 2026-10-03 F320 model on held-out JAMSTEC frames. Drop the detection with 1 - recall.
                    frac = (d.bbox[2] - d.bbox[0]) / CAM_W
                    rc = next((r for f, r in recall_curve if frac < f), recall_curve[-1][1])
                    if rng.random() >= rc:
                        d = None
                if d is not None:
                    dets.append(d)
            for _ in range(int(args.fp_per_frame) + (rng.random() < args.fp_per_frame % 1)):
                fp = false_positive(rng)
                if fp is not None:
                    dets.append(fp)
            if args.fp_rate > 0 and rng.random() < args.fp_rate:
                fp = false_positive(rng)
                if fp is not None:
                    dets.append(fp)
            for ph in phantoms:
                if rng.random() >= args.phantom_p:
                    continue
                proj = project_balloon(R.T @ (ph - cam_pos))
                dproj = degrade_projection(proj, rng, args) if proj is not None else None
                d = make_detection(*dproj, args.phantom_colour, 0) if dproj is not None else None
                if d is not None:
                    dets.append(d)
            pending.append((k + lat_steps, dets))
        fresh = False
        while pending and pending[0][0] <= k:
            held, fresh = pending.pop(0)[1], True
        t_ctl += dt
        if t_ctl >= ctl_period - 1e-9:      # control tick: FSM + driver on the control clock
            t_ctl -= ctl_period
            heading = float(math.atan2((R @ [1.0, 0, 0])[2], (R @ [1.0, 0, 0])[0]))
            cmd, info = fsm.step(held, float(st["ang_vel"][1]), heading=heading, dt=ctl_period, fresh=fresh)
            occ[fsm.state] = occ.get(fsm.state, 0) + 1
            if tracer is not None:
                tracer.tick(k * dt, fsm, sim, balloons, popped, cam_id, pin_sid, held, fresh)
            action = (feedforward_allocation([0, 0, cmd["yaw"]], [-cmd["surge"], 0, cmd["heave"]])
                      if drv is None else drv.step(cmd))
        sim.step(action)
        heights.append(float(sim.data.xpos[sim.base_id][1]))

        pin_tip = sim.data.site_xpos[pin_sid].copy()
        axis = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0, 0])
        vel = (pin_tip - prev_pin) / dt if prev_pin is not None else np.zeros(3)
        prev_pin = pin_tip
        for b in balloons:
            if b["name"] not in popped and scn.popped(pin_tip, b["pos"], axis, vel,
                                                      min_speed=args.min_pop_speed,
                                                      angle_tol_deg=args.pop_angle_tol):
                popped.add(b["name"])
                score += b["points"]
                if first_pop is None:
                    first_pop = (k + 1) * dt
        snag = set(scn.entanglement(sim.data.xpos[sim.base_id], balloons, popped))
        wire_events += len(snag - snag_prev)
        if tracer is not None:
            tracer.step(k * dt, fsm, sim, balloons, popped, pin_sid, vel, snag - snag_prev)
        snag_prev = snag
        if positive <= popped:  # all positive cleared
            t_clear = (k + 1) * dt
            break

    if tracer is not None:
        tracer.close(n_steps * dt, popped)
    blue_popped = sum(1 for b in balloons if b["name"] in popped and b["points"] < 0)
    h = np.asarray(heights)
    return {"cleared": t_clear is not None, "t_clear": t_clear, "score": score,
            "blue_popped": blue_popped, "wire": wire_events,
            "n_positive": len(positive), "n_pos_popped": len(positive & popped),
            "first_pop": first_pop,
            "n_red": sum(1 for b in balloons if b["name"] in popped and b["colour"] == "red"),
            "surface_frac": float(np.mean(h > scn.POOL_DEPTH - SURFACE_BAND)),
            "floor_frac": float(np.mean(h < scn.FLOOR_Y + SURFACE_BAND)),
            "h_min": float(h.min()), "h_max": float(h.max()),
            "occ": occ, "n_ram": fsm.n_ram, "n_miss": fsm.n_miss, "n_recover": fsm.n_recover,
            "n_abandon": fsm.n_abandon}


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
    ap.add_argument("--driver", choices=("deploy", "ff"), default="deploy")
    ap.add_argument("--max-duty", type=float, default=0.3, help="esc cap (field: 0.3, then 0.4)")
    ap.add_argument("--yaw-sign", type=float, default=-1.0,
                    help="-1 = FSM contract (+yaw turns toward +Z); +1 = what the robot navigator does")
    ap.add_argument("--hold-yaw", action="store_true")
    ap.add_argument("--net-buoy", type=float, default=None, help="plant buoyancy - weight [N], + floats")
    ap.add_argument("--cob-fwd", type=float, default=0.0, help="plant CoB ahead of CoM [m]")
    ap.add_argument("--cob-lat", type=float, default=0.0, help="plant CoB to starboard of CoM [m]")
    ap.add_argument("--start-height", type=float, default=START[1], help="start height above floor [m]")
    ap.add_argument("--sweep", action="store_true", help="run the net-buoy x cob-fwd grid")
    ap.add_argument("--ki", type=float, default=None, help="override the bundle's attitude integrator gain")
    ap.add_argument("--ki-heave", type=float, default=0.0,
                    help="FSM heave bias learned from the camera elevation error (0 = off)")
    ap.add_argument("--ram-surge", type=float, default=None,
                    help="EXPERIMENT: override behavior.RAM_SURGE (FSM surge units; x surge_scale 0.35 = m/s)")
    ap.add_argument("--ram-max-steps", type=int, default=None,
                    help="EXPERIMENT: override behavior.RAM_MAX_STEPS (control steps before a ram counts as a miss)")
    ap.add_argument("--propagate-bearing", action="store_true",
                    help="FSM advances held detections' azimuth by the gyro yaw since the frame")
    ap.add_argument("--pop-angle-tol", type=float, default=scn.POP_ANGLE_TOL_DEG,
                    help="max pin-axis vs tip->centre angle for a pop [deg]. UNMEASURED: 20 is a guess; it is "
                         "evaluated when the tip first enters radius+margin, where 4 cm off-centre is already ~18-24 deg")
    ap.add_argument("--recall-curve", default=None,
                    help="measured recall by box width / image width, 'frac:recall,...' ascending, e.g. "
                         "'0.025:0.24,0.05:0.64,1.0:1.0' (F320 on JAMSTEC, 2026-10-03)")
    ap.add_argument("--fp-per-frame", type=float, default=0.0,
                    help="mean random false positives per detector frame (F320 on JAMSTEC: ~1.0)")
    ap.add_argument("--trace-events", default=None, help="append JSONL diagnostic events (state/ram/wire) here")
    ap.add_argument("--servo-aware", action="store_true",
                    help="EXPERIMENT: esc from the angle the servo has REACHED, not the target (deploy_driver)")
    ap.add_argument("--bundle-exp", type=float, default=None, help="thrust-curve exponent the CONTROLLER assumes")
    ap.add_argument("--plant-exp", type=float, default=None, help="thrust-curve exponent the PLANT has")
    ap.add_argument("--control-hz", type=float, default=0.0,
                    help="FSM + controller rate (0 = every 50 Hz sim step). Robot full stack: 33.8 Hz")
    ap.add_argument("--perception-latency", type=float, default=0.0, help="detection age on arrival [s]")
    ap.add_argument("--phantoms", type=int, default=0,
                    help="persistent false positives: fixed points under the surface / on the floor")
    ap.add_argument("--phantom-p", type=float, default=0.7, help="per-frame detection prob of each phantom")
    ap.add_argument("--phantom-colour", default="red", help="what the detector calls them (red = worst case)")
    ap.add_argument("--w-effort", type=float, default=None,
                    help="override the bundle allocator's w_effort (null-space circulation price)")
    ap.add_argument("--min-pop-speed", type=float, default=scn.MIN_POP_SPEED,
                    help="pin closing speed a pop needs [m/s]. UNMEASURED: 0.18 was set against the ff "
                         "path's ~0.5 m/s lunge; the deploy chain rams at ~0.09 m/s (cap 0.3)")
    args = ap.parse_args()
    # The FSM's ram timing was tuned against the ff path's ~0.5 m/s lunge; on the deploy chain the ram
    # closes at ~0.09 m/s and times out (RAM_MAX_STEPS = 1.7 s) ~0.15 m short. Module constants, so
    # patch them for the run — experiment knobs, not a tuned value.
    from umiusi_perception.autonomy import behavior as _bh
    if args.ram_surge is not None:
        _bh.RAM_SURGE = args.ram_surge
    if args.ram_max_steps is not None:
        _bh.RAM_MAX_STEPS = args.ram_max_steps
    args.pin_tip = tuple(float(v) for v in args.pin_tip.split(",")) if args.pin_tip else None
    args.pin_base = tuple(float(v) for v in args.pin_base.split(",")) if args.pin_base else None
    args.yellow_frac = 0.0  # unused here (field sampled from config), but ram_eval helpers read args

    xml_path = Path(tempfile.gettempdir()) / "umiusi_sim" / "competition_eval.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    pin = "default" if not args.pin_aware and args.pin_tip is None else \
        f"{args.pin_tip} aware={args.pin_aware}"
    print(f"competition_eval: {args.episodes} episodes  timeout={args.minutes}min  pin={pin}  "
          f"perception(hz={args.perception_hz} bearing={args.bearing_noise_deg} dropout={args.dropout})")
    print(f"  driver={args.driver} cap={args.max_duty} yaw_sign={args.yaw_sign:+.0f} hold_yaw={args.hold_yaw}"
          f"  start_h={args.start_height}  min_pop_speed={args.min_pop_speed}  w_effort={args.w_effort}"
          f"  ki={args.ki}  ki_heave={args.ki_heave}  ram_surge={args.ram_surge}  ram_max={args.ram_max_steps}")
    print(f"  perception_hz={args.perception_hz} latency={args.perception_latency}s control_hz={args.control_hz}"
          f"  dropout={args.dropout} fp_rate={args.fp_rate} phantoms={args.phantoms}x{args.phantom_colour}"
          f"@p{args.phantom_p}  servo_aware={args.servo_aware} prop={args.propagate_bearing} bundle_exp={args.bundle_exp} plant_exp={args.plant_exp}")
    if args.sweep:
        return sweep(args, xml_path)
    summarize(run_set(args, xml_path), args)
    return 0


def run_set(args, xml_path):
    rows = []
    for e in range(args.episodes):
        args._ep_seed = 1000 + args.seed + e
        r = run_episode(np.random.default_rng(args._ep_seed), args, xml_path)
        rows.append(r)
        if args.verbose:
            tc = f"{r['t_clear']:.1f}s" if r["cleared"] else "TIMEOUT"
            print(f"  ep {e:2d}: {'CLEAR' if r['cleared'] else 'fail '} {tc:>8s}  "
                  f"pos {r['n_pos_popped']}/{r['n_positive']}  score {r['score']:+d}  "
                  f"blue {r['blue_popped']}  wire {r['wire']}  red {r['n_red']}  "
                  f"h[{r['h_min']:.2f},{r['h_max']:.2f}] surf {r['surface_frac']:.0%} "
                  f"floor {r['floor_frac']:.0%}", flush=True)
    return rows


def summarize(rows, args):
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
    fp = [r["first_pop"] for r in rows if r["first_pop"] is not None]
    print(f"  >=1 pop (competition floor: 0 pops = 0 points): {len(fp)}/{n}"
          + (f"  first pop median {np.median(fp):.1f}s" if fp else ""))
    tot = {}
    for r in rows:
        for k, v in r["occ"].items():
            tot[k] = tot.get(k, 0) + v
    n_ticks = max(1, sum(tot.values()))
    print("  FSM time: " + "  ".join(f"{k} {v / n_ticks:.0%}" for k, v in sorted(tot.items(), key=lambda kv: -kv[1]))
          + f"  | per ep: ram {np.mean([r['n_ram'] for r in rows]):.1f} miss {np.mean([r['n_miss'] for r in rows]):.1f}"
          f" recover {np.mean([r['n_recover'] for r in rows]):.1f} abandon {np.mean([r['n_abandon'] for r in rows]):.1f}")
    print(f"  time at surface {np.mean([r['surface_frac'] for r in rows]):.0%}  "
          f"on floor {np.mean([r['floor_frac'] for r in rows]):.0%}  "
          f"red popped {np.mean([r['n_red'] for r in rows]):.1f}/ep")


SWEEP_NET_BUOY = (-1.0, -0.5, 0.0, 0.5, 1.0)     # N; +1.1 N was the pre-ballast hull
SWEEP_COB_FWD = (-0.01, 0.0, 0.01)               # m; stern-heavy / centred / bow-heavy


def sweep(args, xml_path):
    """Same episodes (same seeds) on every plant variant, so rows differ only by the plant."""
    out = []
    for nb in SWEEP_NET_BUOY:
        for cf in SWEEP_COB_FWD:
            args.net_buoy, args.cob_fwd = nb, cf
            rows = run_set(args, xml_path)
            n = len(rows)
            out.append((nb, cf, sum(r["first_pop"] is not None for r in rows) / n,
                        np.mean([r["score"] for r in rows]), np.mean([r["n_red"] for r in rows]),
                        np.mean([r["surface_frac"] for r in rows]), np.mean([r["floor_frac"] for r in rows])))
            print(f"  net_buoy {nb:+.1f} N  cob_fwd {cf * 1000:+.0f} mm  ->  >=1pop {out[-1][2]:.0%}  "
                  f"score {out[-1][3]:+.0f}  red {out[-1][4]:.1f}  surf {out[-1][5]:.0%}  "
                  f"floor {out[-1][6]:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
