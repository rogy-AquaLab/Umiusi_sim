"""tools/ram_eval.py — control-isolated RAM failure-mode diagnostics.

The competition-score bottleneck is the balloon RAM/pop step (~15% hit rate; see ai/backlog.md #1).
That headline number is measured end-to-end, so it mixes PERCEPTION errors with the underlying
CONTROL. This tool ISOLATES the control problem: it drives the REAL deployed behaviour FSM
(``umiusi_perception.autonomy.BalloonBehavior``) with SYNTHETIC GROUND-TRUTH detections projected
from the true balloon pose — i.e. PERFECT perception — over many domain-randomized single-balloon
trials, and classifies WHY each ram fails. Whatever misses remain under perfect perception are the
control wall; the gap to the end-to-end rate is the perception cost.

  * NO camera / detector / GL: detections are computed geometrically from the front_cam pose using
    the exact pinhole relation the real detectors use (so they pass the tracker's size/aspect gate).
    Runs headless with no MUJOCO_GL backend.
  * The FSM, tracker, feed-forward allocation and the ``scn.popped`` pop judge are the SAME code the
    robot runs — only the detections are substituted for ground truth.

Wire/tether contact is a FIRST-CLASS FAILURE here (per the design discussion): a trial in which the
hull UNDER-PASSES the target's vertical tether wire (``scn.entanglement``) is classed UNDER_TETHER —
counted as a FAILURE even if a pop later registers, because the real vehicle would be tangled. A
clean head-on ram never enters the wire keep-out, so UNDER_TETHER also flags "passed underneath"
(the classic tall-yellow vertical miss).

Failure taxonomy (per trial, priority order):
  POP           the pin popped it head-on, no wire contact  -> SUCCESS
  UNDER_TETHER  hull under-passed the target's wire          -> tangled / passed underneath
  MISS_ANGLE    pin reached the pop sphere but glancing (axis angle > POP_ANGLE_TOL_DEG)
  MISS_SLOW     pin reached the pop sphere head-on but too slow (closing speed < MIN_POP_SPEED)
  MISS_NEAR     pin grazed close (< GRAZE_M) but never entered the pop sphere (lateral/vertical miss)
  MISS_WIDE     pin never got near the balloon though a ram was committed
  NO_COMMIT     never committed a ram within the time budget (stuck approaching / unstable)

Perception-degradation sweep
----------------------------
The detections default to PERFECT, but a per-frame degradation model can be dialled in to measure how
much detector quality the ram actually needs: ``--bearing-noise-deg`` (Gaussian az/el noise sigma),
``--range-noise`` (multiplicative range sigma), ``--dropout`` (per-frame miss probability), and
``--fp-rate`` (per-frame false-positive injection). Sweeping one knob at a time turns "it's perception"
into a spec: the pop rate vs each knob shows the tolerance band and which failure mode it drives.

``--perception-hz`` is the detector RATE (distinct from ``--dropout``): the detector runs every
``round(control_rate/hz)`` steps and the FSM HOLDS the last detection between ticks (the realistic
slow-Pi model, unlike dropout which drops to nothing). Sweep it to answer "how many Hz is enough".

Usage:
    uv run python -m tools.ram_eval --trials 60 --seed 0
    uv run python -m tools.ram_eval --trials 40 --current 0.06 --csv /tmp/ram.csv --verbose
    uv run python -m tools.ram_eval --trials 40 --dropout 0.5 --bearing-noise-deg 4   # perception sweep
    uv run python -m tools.ram_eval --trials 40 --pin-tip 0.28,0.02,0 --pin-base 0.15,0.02,0 --pin-aware

Pin study
---------
``--pin-tip``/``--pin-base`` set the popping-pin mount (body frame) so a placement study can sweep it;
``--pin-aware`` turns on geometry-solving aim: the FSM reconstructs the balloon's 3-D position from the
detection bearing + range + the (pin_tip - camera) offset and drives the PIN (not the camera axis) onto
it. This decouples where the pin is mounted (choose it for camera FOV) from how it is aimed, and nearly
eliminates the tether-snag failure. Needs a calibrated pin<->camera offset and a usable range estimate.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from umiusi_perception.autonomy import BalloonBehavior
from umiusi_perception.balloon_detector import BALLOON_DIAMETER_M, Detection, _pinhole
from umiusi_perception.control import feedforward_allocation
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator

CAM_W, CAM_H, FOVY_DEG = 320, 240, 60.0
BALLOON_ASPECT = 1.25  # rendered oval height/width (see appearance.BALLOON_ASPECT); keeps boxes plausible

# Closest-approach diagnostic bands (built on the scn pop gate).
POP_PROX = scn.BALLOON_RADIUS + scn.POP_MARGIN  # 0.13 m: inside this the pin is "on" the balloon
GRAZE_M = 0.30                                   # pin came near but outside the pop sphere -> MISS_NEAR

FAIL_ORDER = ["POP", "UNDER_TETHER", "MISS_ANGLE", "MISS_SLOW", "MISS_NEAR", "MISS_WIDE", "NO_COMMIT"]


def _yaw_quat(yaw):
    """MuJoCo quat (w,x,y,z) for a rotation of ``yaw`` rad about the +Y (up) axis."""
    return (math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)


def project_balloon(delta_body):
    """(az, el, range) of a balloon at body-frame offset ``delta_body`` (x fwd, y up, z right), or
    None if it is behind the camera plane. ``+az`` toward body +Z (image-right), ``+el`` up — the FSM
    convention."""
    dx, dy, dz = float(delta_body[0]), float(delta_body[1]), float(delta_body[2])
    if dx <= 1e-3:
        return None  # behind / beside the camera plane
    return math.atan2(dz, dx), math.atan2(dy, math.hypot(dx, dz)), math.sqrt(dx * dx + dy * dy + dz * dz)


def make_detection(az, el, r, colour, points):
    """A size-consistent ``Detection`` from a bearing+range, or None if out of FOV / sub-pixel. The
    box size follows the detector's pinhole relation (range = D*fx / mean_box_px) so the box passes
    ``tracker.size_consistent`` by construction (range and box stay coupled even after range noise)."""
    if r <= 1e-3:
        return None
    fx, fy, cx, cy = _pinhole(CAM_H, CAM_W, FOVY_DEG)
    if abs(math.atan2(CAM_W / 2.0, fx)) < abs(az) or abs(math.atan2(CAM_H / 2.0, fy)) < abs(el):
        return None  # centre left the frame -> the real camera loses it
    mean_px = BALLOON_DIAMETER_M * fx / r
    w = mean_px / (0.5 * (1.0 + BALLOON_ASPECT))
    h = BALLOON_ASPECT * w
    if min(w, h) < 6:                       # sub-pixel: a real detector would not see it (far balloon)
        return None
    u = cx + fx * math.tan(az)
    v = cy - fy * math.tan(el)
    bbox = (int(u - w / 2), int(v - h / 2), int(u + w / 2), int(v + h / 2))
    return Detection(colour=colour, points=points, bbox=bbox, centroid=(float(u), float(v)),
                     area_px=int(0.785 * w * h), bearing=(az, el), range_m=r, confidence=0.9)


def synth_detection(delta_body, colour, points):
    """Perfect ground-truth ``Detection`` (no perception noise). Thin wrapper over project + build."""
    proj = project_balloon(delta_body)
    return None if proj is None else make_detection(*proj, colour, points)


def degrade_projection(proj, rng, args):
    """Apply the per-frame PERCEPTION-degradation model to a ``(az, el, range)`` projection: with
    probability ``dropout`` the detector MISSES the frame (return None); otherwise add Gaussian bearing
    noise (``bearing_noise_deg`` sigma per axis) and multiplicative range noise (``range_noise`` sigma).
    Returns the perturbed ``(az, el, range)`` or None."""
    if args.dropout > 0 and rng.random() < args.dropout:
        return None
    az, el, r = proj
    if args.bearing_noise_deg > 0:
        s = math.radians(args.bearing_noise_deg)
        az += float(rng.normal(0.0, s))
        el += float(rng.normal(0.0, s))
    if args.range_noise > 0:
        r *= max(0.1, 1.0 + float(rng.normal(0.0, args.range_noise)))
    return az, el, r


def false_positive(rng):
    """A spurious in-FOV detection (random colour/bearing/range) — the detector's false-positive noise
    that stresses target selection, wire/blue avoidance and pop confirmation. None if sub-pixel."""
    fx, fy, _, _ = _pinhole(CAM_H, CAM_W, FOVY_DEG)
    colour = str(rng.choice(("red", "yellow", "blue")))
    az = float(rng.uniform(-0.9, 0.9)) * math.atan2(CAM_W / 2.0, fx)
    el = float(rng.uniform(-0.9, 0.9)) * math.atan2(CAM_H / 2.0, fy)
    r = float(rng.uniform(0.8, 4.0))
    return make_detection(az, el, r, colour, scn.BALLOON_SPECS[colour]["points"])


def run_trial(rng, args, nominal_thrust, xml_path):
    """Run ONE domain-randomized single-balloon ram trial; return a result dict."""
    colour = "yellow" if rng.random() < args.yellow_frac else "red"
    bx = float(rng.uniform(2.2, 3.0))
    bz = float(rng.uniform(-args.balloon_z, args.balloon_z))
    layout = [(f"balloon_{colour}", colour, bx, bz)]

    # Compose + compile the single-balloon scene through the normal path (so UmiusiSimulator's hydro /
    # inertia precompute runs on the real composed model, exactly like competition_run / autonomy_run).
    pin_base, pin_tip = getattr(args, "pin_base", None), getattr(args, "pin_tip", None)
    pin_kw = {}
    if pin_base is not None:
        pin_kw["pin_base"] = pin_base
    if pin_tip is not None:
        pin_kw["pin_tip"] = pin_tip
    xml_path.write_text(scn.build_spec(layout=layout, **pin_kw).to_xml())
    sim = UmiusiSimulator(model_path=xml_path)
    model = sim.model

    # Randomized start pose: lateral (z), depth (y), heading error (yaw about +Y).
    start = (0.0,
             float(np.clip(1.0 + rng.uniform(-args.start_depth, args.start_depth), 0.4, 2.5)),
             float(rng.uniform(-args.start_z, args.start_z)))
    yaw0 = math.radians(float(rng.uniform(-args.heading_deg, args.heading_deg)))
    sim.reset(pos=start, quat=_yaw_quat(yaw0))
    # DR disturbances (applied AFTER reset, which zeroes them). Steady current + optional thrust jitter.
    if args.current > 0:
        sim.current_world = np.array([rng.uniform(-args.current, args.current), 0.0,
                                      rng.uniform(-args.current, args.current)])
    if args.thrust_jitter > 0:
        sim.thrust_per_cmd = nominal_thrust * (1.0 + rng.uniform(-args.thrust_jitter,
                                                                 args.thrust_jitter, size=4))

    balloons = scn.balloon_table(layout=layout)
    bpos = balloons[0]["pos"]
    points = balloons[0]["points"]
    cam_id = model.camera("front_cam").id
    pin_sid = model.site("pin_tip").id
    control_rate = float(sim.cfg["sim"]["control_rate_hz"])
    control_dt = 1.0 / control_rate
    pin_offset = None
    if getattr(args, "pin_aware", False):  # pin-aware aiming needs the body-frame pin_tip -> camera offset
        cam = model.camera("front_cam").pos
        tip = pin_tip if pin_tip is not None else scn.PIN_TIP
        pin_offset = (tip[0] - cam[0], tip[1] - cam[1], tip[2] - cam[2])
    behavior = BalloonBehavior(frame_h=CAM_H, frame_w=CAM_W, fovy_deg=FOVY_DEG, dt=control_dt,
                               pin_offset=pin_offset)
    # Detector rate: control runs every step at control_rate; the DETECTOR runs every ``perc_stride``
    # steps and the FSM HOLDS the last detection between ticks (fresh=False) — the realistic slow-Pi
    # model (mirrors tools/autonomy_run). 0/>=control_rate = detect every step.
    perc_stride = max(1, round(control_rate / args.perception_hz)) if args.perception_hz > 0 else 1

    n_steps = int(round(args.seconds / control_dt))
    min_dist = float("inf")
    best_angle = 180.0     # min axis angle seen while inside the pop sphere
    best_speed = -1.0      # max closing speed seen while inside the pop sphere
    el_at_min = 0.0
    committed = False
    tether = False
    popped = False
    prev_pin = None
    held = []              # last detector output, re-presented between detector ticks

    for k in range(n_steps):
        st = sim.get_state()
        # DETECTOR TICK (every perc_stride steps): re-run the detector + perception degradation. Between
        # ticks the FSM re-drives on the HELD (now stale) detection with fresh=False.
        fresh = (k % perc_stride == 0)
        R = sim.data.xmat[sim.base_id].reshape(3, 3)
        if fresh:
            cam_pos = sim.data.cam_xpos[cam_id]
            delta_body = R.T @ (bpos - cam_pos)
            proj = None if popped else project_balloon(delta_body)
            det = None
            if proj is not None:
                dproj = degrade_projection(proj, rng, args)
                if dproj is not None:
                    det = make_detection(*dproj, colour, points)
            dets = [det] if det is not None else []
            if args.fp_rate > 0 and rng.random() < args.fp_rate:  # inject a false positive this tick
                fp = false_positive(rng)
                if fp is not None:
                    dets.append(fp)
            held = dets
        else:
            dets = held

        fwd = R @ np.array([1.0, 0.0, 0.0])
        heading = float(math.atan2(fwd[2], fwd[0]))
        cmd, info = behavior.step(dets, float(st["ang_vel"][1]), heading=heading, dt=control_dt, fresh=fresh)
        if info["state"] == "RAM":
            committed = True
        action = feedforward_allocation([0.0, 0.0, cmd["yaw"]], [-cmd["surge"], 0.0, cmd["heave"]])
        sim.step(action)

        # Pop gate diagnostics (mirrors scn.popped internals) at the pin tip.
        pin_tip = sim.data.site_xpos[pin_sid].copy()
        pin_axis = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        pin_vel = (pin_tip - prev_pin) / control_dt if prev_pin is not None else np.zeros(3)
        prev_pin = pin_tip
        delta = bpos - pin_tip
        dist = float(np.linalg.norm(delta))
        if dist < min_dist:
            min_dist = dist
            el_at_min = float(math.atan2(delta_body[1], math.hypot(delta_body[0], delta_body[2])))
        if dist < POP_PROX and dist > 1e-6:
            u = delta / dist
            ang = math.degrees(math.acos(float(np.clip(np.dot(pin_axis / (np.linalg.norm(pin_axis)
                                                                          + 1e-9), u), -1, 1))))
            closing = float(np.dot(pin_vel, u))
            best_angle = min(best_angle, ang)
            best_speed = max(best_speed, closing)

        if not popped and scn.popped(pin_tip, bpos, pin_axis, pin_vel):
            popped = True
            break
        # Tether under-pass of the (still un-popped) target = tangled / passed underneath.
        if not popped and scn.entanglement(sim.data.xpos[sim.base_id], balloons):
            tether = True

    outcome = _classify(popped, tether, committed, min_dist, best_angle, best_speed)
    return {
        "colour": colour, "outcome": outcome, "min_dist": min_dist, "el_at_min_deg": math.degrees(el_at_min),
        "best_angle": best_angle if best_angle < 180 else float("nan"),
        "best_speed": best_speed if best_speed >= 0 else float("nan"),
        "tether": tether, "committed": committed,
        "n_ram": behavior.n_ram, "n_miss": behavior.n_miss, "n_pop": behavior.n_confirmed_pop,
    }


def _classify(popped, tether, committed, min_dist, best_angle, best_speed):
    if tether:
        return "UNDER_TETHER"          # first-class failure even if a pop registered (tangled)
    if popped:
        return "POP"
    if min_dist < POP_PROX:            # reached the pop sphere but a gate rejected it
        if best_angle > scn.POP_ANGLE_TOL_DEG:
            return "MISS_ANGLE"
        if best_speed < scn.MIN_POP_SPEED:
            return "MISS_SLOW"
        return "MISS_NEAR"             # in the sphere, gates passed at some point, yet never all-at-once
    if min_dist < GRAZE_M:
        return "MISS_NEAR"
    if committed:
        return "MISS_WIDE"
    return "NO_COMMIT"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=60, help="number of DR trials")
    ap.add_argument("--seed", type=int, default=0, help="master RNG seed")
    ap.add_argument("--seconds", type=float, default=15.0, help="per-trial time budget")
    ap.add_argument("--yellow-frac", type=float, default=0.5, help="fraction of trials with a tall yellow")
    ap.add_argument("--balloon-z", type=float, default=0.4, help="+/- lateral spread of the balloon [m]")
    ap.add_argument("--start-z", type=float, default=0.5, help="+/- lateral start offset [m]")
    ap.add_argument("--start-depth", type=float, default=0.3, help="+/- start depth offset [m]")
    ap.add_argument("--heading-deg", type=float, default=30.0, help="+/- initial heading error [deg]")
    ap.add_argument("--current", type=float, default=0.04, help="+/- steady water current per axis [m/s]")
    ap.add_argument("--thrust-jitter", type=float, default=0.0, help="+/- per-thruster gain jitter [frac]")
    # --- perception-degradation model (default 0 = perfect perception) ---------------------------
    ap.add_argument("--perception-hz", type=float, default=0.0,
                    help="detector rate [Hz]; FSM holds the last detection between ticks. 0 = every "
                         "control step (~50 Hz). Answers 'how many Hz is enough'.")
    ap.add_argument("--bearing-noise-deg", type=float, default=0.0,
                    help="per-frame Gaussian bearing noise sigma [deg] added to az AND el")
    ap.add_argument("--range-noise", type=float, default=0.0,
                    help="per-frame multiplicative range-noise sigma [fraction]")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="per-frame probability the detector MISSES the balloon [0..1]")
    ap.add_argument("--fp-rate", type=float, default=0.0,
                    help="per-frame probability of injecting a false-positive detection [0..1]")
    # --- pin geometry + pin-aware aiming ---------------------------------------------------------
    ap.add_argument("--pin-tip", type=str, default=None, help="pin tip 'x,y,z' body-frame (default scenario)")
    ap.add_argument("--pin-base", type=str, default=None, help="pin base 'x,y,z' body-frame (default scenario)")
    ap.add_argument("--pin-aware", action="store_true",
                    help="pin-aware aiming: FSM drives the PIN (not the camera axis) onto the balloon")
    ap.add_argument("--csv", type=str, default="", help="write per-trial rows to this CSV path")
    ap.add_argument("--verbose", action="store_true", help="print each trial")
    args = ap.parse_args()
    args.pin_tip = tuple(float(v) for v in args.pin_tip.split(",")) if args.pin_tip else None
    args.pin_base = tuple(float(v) for v in args.pin_base.split(",")) if args.pin_base else None

    rng = np.random.default_rng(args.seed)
    nominal_thrust = float(UmiusiSimulator().thrust_per_cmd) if args.thrust_jitter > 0 else None
    xml_path = Path(tempfile.gettempdir()) / "umiusi_sim" / "ram_eval_scene.xml"

    rows = []
    print(f"ram_eval: {args.trials} trials  yellow_frac={args.yellow_frac}  "
          f"start(z=+/-{args.start_z}, depth=+/-{args.start_depth}, heading=+/-{args.heading_deg}deg)  "
          f"current=+/-{args.current} m/s  thrust_jitter=+/-{args.thrust_jitter}  budget={args.seconds}s")
    if any((args.bearing_noise_deg, args.range_noise, args.dropout, args.fp_rate, args.perception_hz)):
        print(f"perception model: rate={args.perception_hz or 'control'}Hz bearing_noise={args.bearing_noise_deg}deg "
              f"range_noise={args.range_noise} dropout={args.dropout} fp_rate={args.fp_rate}")
    else:
        print("perfect perception (GT detections) -> whatever misses remain are the CONTROL wall")
    print("-" * 92)
    for i in range(args.trials):
        r = run_trial(rng, args, nominal_thrust, xml_path)
        rows.append(r)
        if args.verbose:
            print(f"  trial {i:3d}  {r['colour']:6s} {r['outcome']:12s} "
                  f"min_dist={r['min_dist']:.3f} el@min={r['el_at_min_deg']:+5.1f}deg "
                  f"angle={r['best_angle']:5.1f} speed={r['best_speed']:5.2f} "
                  f"{'TETHER ' if r['tether'] else ''}")

    _report(rows, args)
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"wrote per-trial CSV -> {args.csv}")
    return 0


def _report(rows, args):
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    n_pop = counts.get("POP", 0)
    n_tether = sum(1 for r in rows if r["tether"])
    print("-" * 92)
    print(f"OUTCOMES over {n} trials  (pop rate {n_pop / max(1, n):.0%},  "
          f"wire-touch rate {n_tether / max(1, n):.0%})")
    for k in FAIL_ORDER:
        c = counts.get(k, 0)
        if c:
            bar = "#" * round(40 * c / max(1, n))
            print(f"  {k:13s} {c:4d}  {c / n:5.0%}  {bar}")
    # Per-colour split (tall yellows are the known hard case).
    for colour in ("red", "yellow"):
        sub = [r for r in rows if r["colour"] == colour]
        if sub:
            p = sum(1 for r in sub if r["outcome"] == "POP")
            t = sum(1 for r in sub if r["tether"])
            print(f"  [{colour:6s}] {len(sub):3d} trials  pop {p / len(sub):4.0%}  wire-touch {t / len(sub):4.0%}")
    # FSM retry activity (a trial ends on the FIRST physical pop, so these count the CHURN on the way:
    # a clean trial commits one ram and stops; failures rack up re-align / re-approach misses).
    tot_ram = sum(r["n_ram"] for r in rows)
    tot_miss = sum(r["n_miss"] for r in rows)
    print(f"FSM retry activity: {tot_ram} ram commits, {tot_miss} FSM-registered misses over {n} "
          f"trials  (more misses = more re-align/re-approach churn before a hit)")
    succ = [r["min_dist"] for r in rows if r["outcome"] == "POP"]
    if succ:
        print(f"successful rams: mean closest pin-tip dist {np.mean(succ):.3f} m")


def _write_csv(rows, path):
    import csv
    keys = ["colour", "outcome", "min_dist", "el_at_min_deg", "best_angle", "best_speed",
            "tether", "committed", "n_ram", "n_miss", "n_pop"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


if __name__ == "__main__":
    raise SystemExit(main())
