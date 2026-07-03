"""Competition balloon-popping run harness — the "sim is runnable end-to-end" deliverable.

Composes the competition world (base robot + pool + tethered balloons + popping pin), then
drives the vehicle with a ground-truth greedy seek + the analytical feed-forward controller
(``umiusi_sim.control.feedforward_allocation``) — NO reinforcement learning. Each control step
it aims at the nearest un-popped positive-scoring balloon, yaws to face it, surges forward and
heaves to the right depth, and scores geometric pin-tip pops.

Perception is future work: the driver reads balloon world positions directly (ground truth),
so this exercises the world + controller + scoring loop, not a vision pipeline.

Usage (headless render needs an offscreen GL backend, e.g. EGL):
    MUJOCO_GL=egl python -m tools.competition_run --seconds 40
    python -m tools.competition_run --render                 # passive viewer (needs a display)
"""

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from umiusi_sim.control import feedforward_allocation
from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator

_SCRATCH = Path(
    "/tmp/claude-1000/-home-satoi-mujoco-ws/0cf18c2f-3f06-4906-a070-c3f6db043305/scratchpad"
)
START = (0.0, 1.0, 0.0)  # ~1 m off the pool floor, on the +X approach axis (scenario assumption)

# --- driver gains (feed-forward command convention; see umiusi_sim/control.py docstring) ------
# Command axes map to sim motion as: Vx -> body -X (surge), Vz -> body +Y (heave), Phi_z -> yaw.
SPEED_CAP = 0.35        # max surge/heave command magnitude (~"modest speed", per the brief)
KP_HEAVE = 1.5          # vertical P gain: Vz command per metre of depth error
KP_YAW = 1.2            # yaw P gain: Phi_z command per radian of heading error
KD_YAW = 0.15           # yaw D gain (damps the yaw rate; ground-truth ang_vel about +Y)
FACE_TOL = math.radians(50.0)  # only surge hard once within this heading error of the target


def _quat2mat(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _pick_target(pos, balloons, popped_set):
    """Nearest NOT-yet-popped balloon with POSITIVE points (blue is never a target)."""
    best, best_d = None, float("inf")
    for b in balloons:
        if b["name"] in popped_set or b["points"] <= 0:
            continue
        d = float(np.linalg.norm(b["pos"] - pos))
        if d < best_d:
            best, best_d = b, d
    return best


def _driver_action(state, pin_tip, target):
    """Ground-truth greedy seek -> 8-D action via the feed-forward allocation.

    Homes the PIN TIP (the popping point, offset ahead of the hull) onto the target: yaw to
    face it (Phi_z, PD on heading + measured yaw rate), surge forward along body +X once
    roughly facing it, and heave to the target depth. Sway is unusable on this vehicle's
    symmetric thrusters (it produces a yaw couple), so lateral offset is closed by yawing then
    surging rather than strafing.
    """
    quat, ang_vel = state["quat"], state["ang_vel"]
    R = _quat2mat(quat)
    fwd = R @ np.array([1.0, 0.0, 0.0])  # body +X in world
    delta = target["pos"] - pin_tip     # seek with the pin tip, not the hull centre

    # Heading (horizontal X-Z plane): rotate the nose toward the target bearing. Freeze the yaw
    # command when nearly on top of the target (bearing is ill-conditioned there) — just damp.
    horiz_dist = math.hypot(delta[0], delta[2])
    if horiz_dist < 0.12:
        phiz = float(np.clip(KD_YAW * ang_vel[1], -1.0, 1.0))
        herr = 0.0
    else:
        cur_head = math.atan2(fwd[2], fwd[0])
        des_head = math.atan2(delta[2], delta[0])
        herr = _wrap(des_head - cur_head)
        # +Phi_z increases the heading angle (empirically); damp with the measured yaw rate (ang_vel_y).
        phiz = float(np.clip(KP_YAW * herr + KD_YAW * ang_vel[1], -1.0, 1.0))

    # Surge forward (body +X). Vx command maps to body -X, so a forward push is a NEGATIVE Vx.
    # Scale down while badly mis-pointed so the vehicle turns before charging off-bearing.
    surge = min(horiz_dist, SPEED_CAP) * max(0.0, math.cos(herr))
    if abs(herr) > FACE_TOL:
        surge *= 0.3
    vx = -surge

    # Heave to target depth (world +Y). +Vz command maps to +Y.
    vz = float(np.clip(KP_HEAVE * delta[1], -SPEED_CAP, SPEED_CAP))

    return feedforward_allocation([0.0, 0.0, phiz], [vx, 0.0, vz])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=60.0, help="episode horizon (demo default 60 s)")
    ap.add_argument("--record", default=str(_SCRATCH / "competition_run.mp4"),
                    help="mp4 output path (headless; run with MUJOCO_GL=egl). '' disables.")
    ap.add_argument("--render", action="store_true", help="watch in the MuJoCo passive viewer (needs a display)")
    ap.add_argument("--seed", type=int, default=0, help="layout seed; <0 uses the fixed placeholder layout")
    args = ap.parse_args()

    # World: fixed placeholder layout by default (front yellow ahead); a seed samples XY.
    if args.seed is not None and args.seed >= 0:
        layout = scn.sample_layout(np.random.default_rng(args.seed))
    else:
        layout = scn.BALLOON_LAYOUT
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    xml_path = scn.write_xml(_SCRATCH / "competition_run.xml", layout=layout)
    sim = UmiusiSimulator(model_path=xml_path)
    sim.reset(pos=START)
    balloons = scn.balloon_table(layout=layout)
    pin_sid = sim.model.site("pin_tip").id
    control_dt = 1.0 / sim.cfg["sim"]["control_rate_hz"]
    n_steps = int(round(args.seconds / control_dt))
    n_positive = sum(1 for b in balloons if b["points"] > 0)

    print(f"composed model: nbody={sim.model.nbody} ngeom={sim.model.ngeom}  layout={len(balloons)} balloons "
          f"({n_positive} positive)  horizon={args.seconds:.0f}s / {n_steps} steps")
    for b in balloons:
        print(f"  {b['name']:22s} {b['colour']:6s} {b['points']:+3d} pts  @ {np.round(b['pos'], 2)}")
    print("-" * 72)

    recorder, frames = None, []
    if args.record:
        recorder = mujoco.Renderer(sim.model, height=480, width=640)
    viewer = None
    if args.render:
        from umiusi_sim.viewer import UmiusiViewer  # local import; only when a display is available

        viewer = UmiusiViewer(sim.model, sim.data, base_id=sim.base_id, cam="track",
                              control_rate_hz=sim.cfg["sim"]["control_rate_hz"]).launch()

    popped_set, score, timeline = set(), 0, []
    for step in range(n_steps):
        state = sim.get_state()
        pin_tip = sim.data.site_xpos[pin_sid].copy()
        target = _pick_target(pin_tip, balloons, popped_set)
        if target is None:  # all positive balloons popped
            print(f"t={step * control_dt:5.1f}s  all positive balloons popped — stopping early")
            break
        sim.step(_driver_action(state, pin_tip, target))

        pin_tip = sim.data.site_xpos[pin_sid]
        for b in balloons:
            if b["name"] in popped_set:
                continue
            if scn.popped(pin_tip, b["pos"]):
                popped_set.add(b["name"])
                score += b["points"]
                t = step * control_dt
                timeline.append((t, b, score))
                print(f"t={t:5.1f}s  popped {b['name']} ({b['colour']}) {b['points']:+d} -> total {score}")

        if recorder is not None:
            recorder.update_scene(sim.data, camera="track")
            frames.append(recorder.render())
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()
    if recorder is not None:
        import imageio

        imageio.mimsave(args.record, frames, fps=round(1.0 / control_dt))
        recorder.close()
        print(f"wrote {args.record}  ({len(frames)} frames)")

    remaining = [b for b in balloons if b["name"] not in popped_set]
    print("-" * 72)
    print(f"FINAL SCORE: {score}   ({len(popped_set)}/{len(balloons)} balloons popped)")
    if timeline:
        print("pop timeline:")
        for t, b, tot in timeline:
            print(f"  t={t:5.1f}s  {b['name']:22s} {b['colour']:6s} {b['points']:+3d}  -> {tot}")
    if remaining:
        print("remaining:")
        for b in remaining:
            print(f"  {b['name']:22s} {b['colour']:6s} {b['points']:+3d} pts  @ {np.round(b['pos'], 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
