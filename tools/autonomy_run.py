"""Perception-in-the-loop autonomous competition run — vision replaces the ground-truth driver.

The robot detects balloons from its OWN underwater-degraded onboard camera and drives to approach
and pop them, scoring red +30 / yellow +10 / blue -10 (must pop >=1). This is the perception-driven
counterpart to ``tools/competition_run`` (which cheated with ground-truth balloon positions): here
the ONLY balloon information the controller gets is what the learned detector reports from the live
degraded ``front_cam`` frame.

Pipeline, per control step:
  1. render the degraded onboard camera (``sim.render_camera(degrade=True)``) — the SAME appearance
     the detector was trained on (oval balloons, pin geom hidden, subtle tethers, sunlit pool; see
     ``perception.render_appearance``) plus the physically-based underwater degradation;
  2. run the learned detector -> ``[Detection]`` (colour / bearing / range / bbox);
  3. the behaviour FSM (``tools/behavior``) picks the nearest red/yellow, steers onto its bearing,
     surges in, RAMs, and avoids blue;
  4. drive via the analytical feed-forward allocation; score geometric pin-tip pops (ground-truth
     geometry — the "did it physically pop" check, exactly as in competition_run).

Usage (headless render needs an offscreen GL backend, e.g. EGL):
    MUJOCO_GL=egl uv run --extra perception python -m tools.autonomy_run --headless \
        --seed 1 --steps 900 --out /home/satoi/mujoco_ws/ai_out
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tools.behavior import BalloonBehavior
from umiusi_sim.control import feedforward_allocation
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.perception import render_appearance as ra
from umiusi_sim.perception import underwater_sim as us
from umiusi_sim.perception.learned_detector import load_learned_detector
from umiusi_sim.perception.tracker import sanitise_near_colours
from umiusi_sim.simulator import UmiusiSimulator

_DEFAULT_MODEL = "examples/balloon_detector/model.pt"  # shipped best detector (camp3 mix, colour-invariant)
_DEFAULT_OUT = Path("/home/satoi/mujoco_ws/ai_out")
START = (0.0, 1.0, 0.0)  # ~1 m off the pool floor, on the +X approach axis
CAM_W, CAM_H = 320, 240

_DRAW_RGB = {"red": (255, 60, 60), "yellow": (255, 220, 40), "blue": (80, 140, 255)}
_STATE_RGB = {"SEARCH": (180, 180, 180), "APPROACH": (120, 220, 120), "ALIGN": (90, 200, 255),
              "RAM": (255, 140, 40), "RECOVER": (255, 90, 90)}


def build_perception_model(layout):
    """Compose the REAL competition scene, then apply the shared perception appearance (oval
    balloons, hidden pin geom, subtle tethers, sunlit pool) — the detector's training look, on the
    real 3.3 m pool (NOT the enlarged training field). Returns a written, loadable MJCF path."""
    spec = scn.build_spec(layout)
    ra.apply_perception_appearance(
        spec, center_x=scn.POOL_CENTER_X, depth=scn.POOL_DEPTH,
        len_x=scn.POOL_LEN_X, len_z=scn.POOL_LEN_Z, floor_y=scn.FLOOR_Y,
    )
    return spec


def annotate_onboard(frame, detections, state, score, popped, total, t):
    """Overlay detection boxes + colour/range, the FSM state, and the running score on the cam."""
    img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        u0, v0, u1, v1 = d.bbox
        col = _DRAW_RGB.get(d.colour, (255, 255, 255))
        draw.rectangle([u0, v0, u1, v1], outline=col, width=2)
        label = f"{d.colour} {d.range_m:.2f}m {d.confidence:.2f}"
        ty = max(0, v0 - 11)
        draw.rectangle([u0, ty, u0 + 6 * len(label), ty + 11], fill=(0, 0, 0))
        draw.text((u0 + 1, ty), label, fill=col)
    # HUD: state + score banner
    banner = f"{state}  t={t:4.1f}s  score={score:+d}  popped {popped}/{total}"
    draw.rectangle([0, 0, CAM_W, 13], fill=(0, 0, 0))
    draw.text((3, 2), banner, fill=_STATE_RGB.get(state, (255, 255, 255)))
    return np.asarray(img)


def _run_live_viewer(sim, control_dt, autonomy_step, *, annotate_onboard_fn, balloons,
                     popped_set, out, score_getter):
    """Drive the autonomy LIVE inside the MuJoCo passive viewer (reuses ``UmiusiViewer`` exactly as
    ``tools/drive.py`` does: ``UmiusiViewer(model, data, ...).run(step_fn)``). Each viewer tick runs
    ONE ``autonomy_step`` (perception->FSM->drive->pop/score), so the user watches the robot
    SEARCH/APPROACH/ALIGN/RAM/pop in 3rd-person 3D. FSM state transitions + pops print to the
    console; the latest onboard-cam-with-detections frame is dumped to disk periodically (the 2D box
    overlay can't be injected into the passive 3D viewer). Degrades gracefully with no display."""
    # Pre-check a display: launch_passive -> GLFW hard-EXITS the process (not a catchable Python
    # exception) with no X11 server, so guard on $DISPLAY here instead of try/except. The passive
    # viewer uses GLFW/X11; the user runs --render with DISPLAY set (e.g. DISPLAY=:0).
    if not os.environ.get("DISPLAY"):
        print("[--render] no X11 display ($DISPLAY unset); the passive viewer needs one "
              "(e.g. DISPLAY=:0). Skipping the live viewer — run --headless for the mp4.", flush=True)
        return
    try:  # local import so the headless path never needs a display / mujoco.viewer
        from umiusi_sim.viewer import UmiusiViewer
    except Exception as e:  # pragma: no cover - import guard
        print(f"[--render] viewer import failed ({type(e).__name__}: {e}); use --headless.", flush=True)
        return
    dump_path = out / "autonomy_live_onboard.png"
    n_balloons = len(balloons)
    ctr = [0]

    def step_fn():
        rgb, detections, info = autonomy_step()
        if ctr[0] % 25 == 0:  # cheap: glimpse of what the robot sees, every ~0.5 s
            over = annotate_onboard_fn(rgb, detections, info["state"], score_getter(),
                                       len(popped_set), n_balloons, ctr[0] * control_dt)
            _imwrite_png(dump_path, over)
        ctr[0] += 1

    print(f"[--render] live autonomy in the passive viewer — watch SEARCH/APPROACH/ALIGN/RAM/pop in "
          f"3D. Onboard+detections dumped to {dump_path} (~every 25 steps). Close the window to stop.",
          flush=True)
    try:
        UmiusiViewer(sim.model, sim.data, base_id=sim.base_id, cam="track",
                     control_rate_hz=1.0 / control_dt,
                     extra_keys={"(autonomy)": "perception->FSM drives the robot; nothing to steer"}
                     ).run(step_fn)
    except Exception as e:  # no display / launch_passive unavailable -> don't crash the run
        print(f"[--render] passive viewer unavailable ({type(e).__name__}: {e}). This needs a "
              "display (e.g. DISPLAY=:0); run --headless for the mp4 instead.", flush=True)


def _imwrite_png(path, rgb):
    import imageio
    imageio.imwrite(path, np.ascontiguousarray(rgb))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=_DEFAULT_MODEL, help="learned detector checkpoint")
    ap.add_argument("--seed", type=int, default=1, help="layout + water seed; <0 = fixed layout")
    ap.add_argument("--steps", type=int, default=900, help="max control steps (short self-test default)")
    ap.add_argument("--full-run", "--record-full", dest="full_run", action="store_true",
                    help="record the ENTIRE competition to mp4: run until all POSITIVE balloons are "
                         "popped OR the competition-length timeout (--minutes); forces recording. "
                         "The primary way to watch the autonomy (the --render live viewer is heavy).")
    ap.add_argument("--minutes", type=float, default=4.0,
                    help="competition length for --full-run (default 4 min)")
    ap.add_argument("--perception-hz", type=float, default=10.0,
                    help="detector + onboard-camera rate [Hz]; FSM/control run at the sim rate and "
                         "HOLD the last detections between perception ticks (realistic Pi-4 timing, "
                         "and keeps the full-run render tractable). 0/negative = every control step.")
    ap.add_argument("--conf", type=float, default=None, help="detector confidence floor override")
    ap.add_argument("--murk", type=float, default=None, help="fix water murk [0,1] (else sampled)")
    ap.add_argument("--headless", action="store_true", help="no display; only render to mp4")
    ap.add_argument("--render", action="store_true",
                    help="watch the autonomy LIVE in the MuJoCo passive viewer (needs a display, "
                         "e.g. DISPLAY=:0); implies --no-video. The 3D scene shows the robot "
                         "SEARCH/APPROACH/ALIGN/RAM/pop; FSM state transitions print to the console "
                         "and the latest onboard+detections frame is dumped to the out dir.")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output dir for mp4 + log")
    ap.add_argument("--no-video", action="store_true", help="skip mp4 (fastest self-test)")
    ap.add_argument("--third-person", action="store_true", default=True,
                    help="also render a 3rd-person track view beside the onboard cam")
    args = ap.parse_args()
    if args.render:
        args.no_video = True  # live 3D viewer replaces the mp4 for this run
    if args.full_run:
        args.no_video = False  # full-run always records the mp4 (the deliverable)

    rng = np.random.default_rng(max(0, args.seed))
    layout = scn.BALLOON_LAYOUT if args.seed < 0 else scn.sample_layout(rng)
    args.out.mkdir(parents=True, exist_ok=True)

    spec = build_perception_model(layout)
    xml_path = args.out / "autonomy_scene.xml"
    xml_path.write_text(spec.to_xml())
    sim = UmiusiSimulator(model_path=xml_path)
    sim.model.vis.map.znear = 0.02 / max(sim.model.stat.extent, 1e-3)  # allow ram close-ups
    sim.reset(pos=START)

    # Fix the water condition for the whole run (stable cast; per-frame noise still varies).
    sim.camera_degrade = True
    sim.water_params = us.random_params(rng, murk=args.murk)
    detector = load_learned_detector(args.model, conf_thresh=args.conf)

    balloons = scn.balloon_table(layout=layout)
    pin_sid = sim.model.site("pin_tip").id
    control_rate = sim.cfg["sim"]["control_rate_hz"]
    control_dt = 1.0 / control_rate
    behavior = BalloonBehavior(frame_h=CAM_H, dt=control_dt)

    # Perception decoupling: run the detector + degraded onboard render every ``perc_stride`` control
    # steps (~perception_hz), holding the last detections/frame between — realistic Pi-4 timing and
    # far cheaper over a full run. FSM + control still run every control step.
    perc_stride = max(1, round(control_rate / args.perception_hz)) if args.perception_hz > 0 else 1
    max_steps = int(round(args.minutes * 60 * control_rate)) if args.full_run else args.steps
    positive_names = {b["name"] for b in balloons if b["points"] > 0}

    n_positive = len(positive_names)
    print(f"model={args.model}  seed={args.seed}  layout={len(balloons)} balloons "
          f"({n_positive} positive)  water murk={sim.water_params.murk:.2f}  "
          f"perception={args.perception_hz:.0f} Hz (every {perc_stride} steps)  max_steps={max_steps}")
    for b in balloons:
        print(f"  {b['name']:22s} {b['colour']:6s} {b['points']:+3d} pts @ {np.round(b['pos'], 2)}")
    print("-" * 76)

    recording = not args.no_video and not args.render
    third = None
    if args.third_person and recording:
        third = mujoco.Renderer(sim.model, height=CAM_H, width=CAM_W)
    # STREAM frames straight to the mp4 (a full 4-min run is ~12k frames — never hold them in RAM).
    writer = None
    if recording:
        import imageio
        tag = "full" if args.full_run else f"seed{args.seed}"
        out_mp4 = args.out / f"autonomy_{tag}_seed{args.seed}.mp4"
        writer = imageio.get_writer(out_mp4, fps=round(1.0 / control_dt))

    n_frames = [0]
    popped_set, score = set(), 0
    det_counts = {"red": 0, "yellow": 0, "blue": 0}  # detector output tally (quality report)
    state_counts = {"SEARCH": 0, "APPROACH": 0, "ALIGN": 0, "RAM": 0, "CONFIRM": 0, "RECOVER": 0}
    n_frames_with_det = 0
    n_perc_ticks = 0
    step_ctr = [0]          # mutable step counter (shared by the loop and the viewer step_fn)
    last_state = [None]     # for printing FSM state transitions in the live viewer
    held = {"rgb": None, "dets": []}  # last onboard frame + detections (held between perception ticks)
    prev_pin = {"xyz": None}  # last step's pin-tip world pos, for finite-difference pin-tip velocity
    # Tether-entanglement tally (the wire-avoidance headline metric): count UNDER-PASS events —
    # each time the hull enters the tether keep-out of an un-popped balloon while below it (rising
    # edge, so a single lingering under-pass counts once). ``names`` = the distinct balloons snagged.
    entangle = {"prev": set(), "events": 0, "names": set()}

    def autonomy_step():
        """ONE control step: (every perc_stride) render degraded onboard cam -> detect; (every step)
        FSM -> feed-forward drive -> pop/score. Returns (rgb, detections, info). ``rgb``/``detections``
        are the last perception output (held between ticks). Shared by the headless loop and the
        live-viewer step_fn so both drive the vehicle identically."""
        nonlocal score, n_frames_with_det, n_perc_ticks
        i = step_ctr[0]
        fresh = held["rgb"] is None or i % perc_stride == 0
        if fresh:  # perception tick: re-render + re-detect
            held["rgb"] = sim.render_camera("front_cam", width=CAM_W, height=CAM_H, degrade=True)
            # Close-range red/blue colour confirmation from the actual pixels: underwater the blue
            # cast makes near red/blue labels risky, so re-check them before the FSM can commit to a
            # pop — a near "red" that reads blue is relabelled blue and thus AVOIDED (never pop blue).
            held["dets"] = sanitise_near_colours(held["rgb"], detector(held["rgb"]))
            n_perc_ticks += 1
            for d in held["dets"]:
                det_counts[d.colour] = det_counts.get(d.colour, 0) + 1
            if held["dets"]:
                n_frames_with_det += 1
        rgb, detections = held["rgb"], held["dets"]

        st = sim.get_state()
        fwd = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])  # +X = pin axis
        heading = float(np.arctan2(fwd[2], fwd[0]))  # world heading for the search sweep
        cmd, info = behavior.step(detections, float(st["ang_vel"][1]), heading=heading,
                                  dt=control_dt, fresh=fresh)
        state_counts[info["state"]] = state_counts.get(info["state"], 0) + 1
        if info["state"] != last_state[0]:  # print each FSM transition (visible in --render too)
            tgt = "" if info["target"] is None else f"  ({info['target']} @ {info['range']:.2f} m)"
            print(f"t={i * control_dt:5.1f}s  -> {info['state']}{tgt}", flush=True)
            last_state[0] = info["state"]
        # command -> 8-D action via feed-forward allocation. Forward surge = NEGATIVE Vx; heave = +Vz.
        action = feedforward_allocation([0.0, 0.0, cmd["yaw"]], [-cmd["surge"], 0.0, cmd["heave"]])
        sim.step(action)

        # PHYSICAL pop event (the game's authoritative judge): near-frontal pin contact. On a pop we
        # HIDE the balloon geom so it vanishes from the camera — the FSM then confirms the pop from
        # the camera alone (it never reads popped_set / scn.popped to decide what to do next).
        pin_tip = sim.data.site_xpos[pin_sid]
        pin_axis = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        # Pin-tip world velocity by finite difference across this control step (captures the full
        # motion — hull surge + yaw/roll lever arm) so the speed gate sees the true drive-through
        # closing rate. First step (no previous position) -> zero velocity (never at a balloon yet).
        pin_vel = (pin_tip - prev_pin["xyz"]) / control_dt if prev_pin["xyz"] is not None \
            else np.zeros(3)
        prev_pin["xyz"] = pin_tip.copy()
        for b in balloons:
            if b["name"] in popped_set:
                continue
            # near-frontal + driving-in pop (shared head-on + speed model)
            if scn.popped(pin_tip, b["pos"], pin_axis, pin_vel):
                popped_set.add(b["name"])
                score += b["points"]
                scn.hide_balloon(sim.model, b["name"])  # deflate: vanish from the onboard camera
                print(f"t={i * control_dt:5.1f}s  POP {b['name']} ({b['colour']}) {b['points']:+d} "
                      f"-> total {score}   [state={info['state']}]", flush=True)

        # Tether entanglement: which un-popped balloons is the hull currently under-passing? Count the
        # rising edges (fresh snags) — the wire-avoidance metric surfaced in the run summary.
        snag = set(scn.entanglement(sim.data.xpos[sim.base_id], balloons, popped_set))
        fresh_snag = snag - entangle["prev"]
        if fresh_snag:
            entangle["events"] += len(fresh_snag)
            entangle["names"].update(fresh_snag)
            for nm in sorted(fresh_snag):
                print(f"t={i * control_dt:5.1f}s  TANGLE under-pass of un-popped {nm} "
                      f"(entanglement events {entangle['events']})", flush=True)
        entangle["prev"] = snag
        step_ctr[0] += 1
        return rgb, detections, info

    if args.render:
        _run_live_viewer(sim, control_dt, autonomy_step, annotate_onboard_fn=annotate_onboard,
                         balloons=balloons, popped_set=popped_set, out=args.out,
                         score_getter=lambda: score)
    else:
        import time as _time
        _t0 = _time.time()
        for _ in range(max_steps):
            rgb, detections, info = autonomy_step()
            if writer is not None:
                t = (step_ctr[0] - 1) * control_dt
                over = annotate_onboard(rgb, detections, info["state"], score, len(popped_set),
                                        len(balloons), t)
                if third is not None:
                    third.update_scene(sim.data, camera="track")
                    over = np.concatenate([over, third.render()], axis=1)  # onboard | 3rd-person
                writer.append_data(np.ascontiguousarray(over))
                n_frames[0] += 1
            if positive_names <= popped_set:  # all POSITIVE balloons popped -> episode over
                print(f"t={(step_ctr[0] - 1) * control_dt:5.1f}s  all positive balloons popped "
                      "— episode complete", flush=True)
                break
        _wall = _time.time() - _t0

    if third is not None:
        third.close()
    step = step_ctr[0] - 1  # last executed step index (for the report)

    # --- report -------------------------------------------------------------------------------
    remaining = [b for b in balloons if b["name"] not in popped_set]
    n_pos_popped = sum(1 for b in balloons if b["name"] in popped_set and b["points"] > 0)
    n_blue_popped = sum(1 for b in balloons if b["name"] in popped_set and b["points"] < 0)
    print("-" * 76)
    print(f"FINAL SCORE: {score:+d}   ({len(popped_set)}/{len(balloons)} popped; "
          f"{n_pos_popped} positive, {n_blue_popped} blue)")
    print(f"ran {step + 1} control steps ({(step + 1) * control_dt:.1f} s sim) with "
          f"{n_perc_ticks} perception ticks (~{args.perception_hz:.0f} Hz)")
    print(f"detector output over {n_perc_ticks} perception ticks: red={det_counts['red']} "
          f"yellow={det_counts['yellow']} blue={det_counts['blue']}  "
          f"(ticks with >=1 detection: {n_frames_with_det}/{max(1, n_perc_ticks)})")
    print(f"FSM time: SEARCH={state_counts['SEARCH']} APPROACH={state_counts['APPROACH']} "
          f"ALIGN={state_counts['ALIGN']} RAM={state_counts['RAM']} "
          f"CONFIRM={state_counts['CONFIRM']} RECOVER={state_counts['RECOVER']} steps")
    attempts = behavior.n_confirmed_pop + behavior.n_miss  # camera outcomes of the ram attempts
    print(f"ram attempts={attempts}: camera-confirmed pops={behavior.n_confirmed_pop} "
          f"misses={behavior.n_miss}  (pop rate "
          f"{behavior.n_confirmed_pop / max(1, attempts):.0%})  abandoned targets={behavior.n_abandon}")
    print(f"pops: GT-scored={len(popped_set)}/{n_positive} positive  "
          f"camera-confirmed(FSM belief)={behavior.n_confirmed_pop}")
    print(f"tether entanglement: {entangle['events']} under-pass events over "
          f"{len(entangle['names'])} distinct un-popped balloons "
          f"{sorted(entangle['names']) if entangle['names'] else ''}  (lower is better — the "
          f"wire-avoidance metric)")
    if remaining:
        print("remaining:")
        for b in remaining:
            print(f"  {b['name']:22s} {b['colour']:6s} {b['points']:+3d} @ {np.round(b['pos'], 2)}")

    if writer is not None:
        writer.close()
        nf = n_frames[0]
        print(f"wrote {out_mp4}  ({nf} frames = {nf * control_dt:.0f} s @ {round(1.0 / control_dt)} "
              f"fps, onboard{'+3rd-person' if third is not None else ''}; "
              f"render+encode {_wall:.1f} s wall, {_wall / max(1, nf) * 1000:.0f} ms/frame)")

    ok = score > 0 and len(popped_set) >= 1 and n_blue_popped == 0
    print(f"SELF-TEST: {'PASS' if ok else 'FAIL'} "
          f"(score>0 & popped>=1 & no blue popped)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
