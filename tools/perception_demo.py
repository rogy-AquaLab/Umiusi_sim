"""Balloon-detector demo + ground-truth validation (headless via EGL/OSMesa).

Composes the competition world (base robot + pool + tethered balloons), places the vehicle at
a start pose that sees several balloons, renders ``front_cam``, and runs
``perception.detect_balloons``. It then VALIDATES the detections against ground truth: every
balloon in ``competition_balloon.balloon_table`` is projected into the camera (world->camera
via ``data.cam_xpos``/``cam_xmat``, then pinhole) to get its expected pixel, TRUE bearing and
TRUE range; detections are matched to the nearest in-frame ground-truth balloon and per-balloon
errors are reported. A summary (detection rate, colour accuracy, mean bearing/range error,
false positives) is printed, and an annotated frame is written to a portable temp dir.

Usage:
    MUJOCO_GL=egl python -m tools.perception_demo [out_dir]
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import imageio
import numpy as np
from PIL import Image, ImageDraw

from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_perception import detect_balloons
from umiusi_perception.balloon_detector import _pinhole
from umiusi_sim.simulator import UmiusiSimulator

_TMP = pathlib.Path(tempfile.gettempdir()) / "umiusi_sim"
OUT_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _TMP

W, H = 480, 360
FOVY = 60.0
MATCH_PX = 40.0  # a detection matches a GT balloon if its centroid is within this many pixels

# Draw colours (RGB) for annotation, keyed by detected colour.
_DRAW_RGB = {"red": (255, 60, 60), "yellow": (255, 220, 40), "blue": (60, 120, 255)}


def project_gt(sim, cam_id):
    """Project every ground-truth balloon into the camera. Returns a list of dicts with the
    expected pixel (u, v), in-frame flag, TRUE range [m] and TRUE bearing (az, el) [rad]."""
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()  # cols = cam x,y,z axes in world
    fx, fy, cx, cy = _pinhole(H, W, FOVY)
    out = []
    for b in scn.balloon_table():
        pc = cam_mat.T @ (b["pos"] - cam_pos)  # into camera frame (looks down -Z)
        depth = -pc[2]
        rng = float(np.linalg.norm(b["pos"] - cam_pos))
        rec = {"name": b["name"], "colour": b["colour"], "range_m": rng,
               "u": None, "v": None, "inframe": False,
               "bearing": (float(np.arctan2(pc[0], depth)) if depth > 0 else np.nan,
                           float(np.arctan2(pc[1], depth)) if depth > 0 else np.nan)}
        if depth > 0:
            u = cx + fx * pc[0] / depth
            v = cy - fy * pc[1] / depth
            rec["u"], rec["v"] = float(u), float(v)
            rec["inframe"] = (0 <= u < W) and (0 <= v < H)
        out.append(rec)
    return out


def annotate(frame, detections):
    """Draw detection bboxes + colour/range/bearing labels onto a copy of the frame."""
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    for d in detections:
        u0, v0, u1, v1 = d.bbox
        col = _DRAW_RGB.get(d.colour, (255, 255, 255))
        draw.rectangle([u0, v0, u1, v1], outline=col, width=2)
        az_deg = np.degrees(d.bearing[0])
        label = f"{d.colour} {d.range_m:.2f}m az{az_deg:+.0f}"
        ty = max(0, v0 - 11)
        draw.rectangle([u0, ty, u0 + 7 * len(label), ty + 11], fill=(0, 0, 0))
        draw.text((u0 + 1, ty), label, fill=col)
    return np.asarray(img)


def evaluate_pose(sim, pos, quat, tag):
    """Render one pose, detect, match to ground truth, print per-balloon + summary; return
    (metrics dict, annotated frame)."""
    sim.reset(pos=pos, quat=quat)
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)  # settle kinematics for the render/projection

    cam_id = sim.model.camera("front_cam").id
    frame = sim.render_camera("front_cam", W, H)
    dets = detect_balloons(frame, fovy_deg=FOVY)
    gts = project_gt(sim, cam_id)
    visible = [g for g in gts if g["inframe"]]

    print(f"\n=== pose '{tag}'  pos={np.round(pos, 2)} quat={np.round(quat, 3)} ===")
    print(f"visible ground-truth balloons: {len(visible)}   raw detections: {len(dets)}")

    # Match each visible GT to the nearest unused detection within MATCH_PX.
    used = set()
    bearing_errs, range_errs, range_pcts = [], [], []
    n_detected = n_colour_ok = 0
    print(f"  {'balloon':22s} {'col':6s} {'det?':4s} {'colOK':5s} "
          f"{'bearErr':>8s} {'rngGT':>6s} {'rngEst':>7s} {'rngErr%':>8s}")
    for g in visible:
        best, best_d = None, MATCH_PX
        for j, d in enumerate(dets):
            if j in used:
                continue
            dist = np.hypot(d.centroid[0] - g["u"], d.centroid[1] - g["v"])
            if dist < best_d:
                best, best_d = j, dist
        if best is None:
            print(f"  {g['name']:22s} {g['colour']:6s} {'no':4s} {'-':5s} "
                  f"{'-':>8s} {g['range_m']:6.2f} {'-':>7s} {'-':>8s}")
            continue
        used.add(best)
        d = dets[best]
        n_detected += 1
        col_ok = d.colour == g["colour"]
        n_colour_ok += int(col_ok)
        # Bearing error = angle between estimated and true (az, el) direction vectors.
        be = np.hypot(d.bearing[0] - g["bearing"][0], d.bearing[1] - g["bearing"][1])
        bearing_errs.append(np.degrees(be))
        re = abs(d.range_m - g["range_m"])
        range_errs.append(re)
        range_pcts.append(100.0 * re / g["range_m"])
        print(f"  {g['name']:22s} {g['colour']:6s} {'yes':4s} {str(col_ok):5s} "
              f"{np.degrees(be):7.2f}d {g['range_m']:6.2f} {d.range_m:7.2f} {range_pcts[-1]:7.1f}%")

    n_false = len(dets) - len(used)
    metrics = {
        "visible": len(visible), "detected": n_detected, "colour_ok": n_colour_ok,
        "false_pos": n_false,
        "mean_bearing_deg": float(np.mean(bearing_errs)) if bearing_errs else float("nan"),
        "mean_range_m": float(np.mean(range_errs)) if range_errs else float("nan"),
        "mean_range_pct": float(np.mean(range_pcts)) if range_pcts else float("nan"),
    }
    dr = n_detected / len(visible) if visible else float("nan")
    ca = n_colour_ok / n_detected if n_detected else float("nan")
    print(f"  -> detection rate {n_detected}/{len(visible)} = {dr:.0%}   "
          f"colour acc {n_colour_ok}/{n_detected} = {ca:.0%}   false-pos {n_false}")
    print(f"  -> mean bearing err {metrics['mean_bearing_deg']:.2f} deg   "
          f"mean range err {metrics['mean_range_m']:.3f} m ({metrics['mean_range_pct']:.1f}%)")
    return metrics, annotate(frame, dets)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = scn.write_xml(OUT_DIR / "competition_balloon.xml")
    sim = UmiusiSimulator(model_path=xml_path)
    print(f"composed model: nbody={sim.model.nbody} ngeom={sim.model.ngeom} -> {xml_path}")

    def yaw_quat(deg):  # rotation about +Y (up) axis
        a = np.radians(deg) / 2
        return (np.cos(a), 0.0, np.sin(a), 0.0)

    poses = [
        ("start", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ("start_yaw+15", (0.0, 1.0, 0.0), yaw_quat(15)),
        ("closer_low", (1.0, 0.7, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ]

    all_metrics, primary_frame = [], None
    for tag, pos, quat in poses:
        m, annotated = evaluate_pose(sim, pos, quat, tag)
        all_metrics.append((tag, m))
        if primary_frame is None:
            primary_frame = annotated  # annotate the primary "start" pose

    # Aggregate across poses.
    tot_vis = sum(m["visible"] for _, m in all_metrics)
    tot_det = sum(m["detected"] for _, m in all_metrics)
    tot_col = sum(m["colour_ok"] for _, m in all_metrics)
    tot_fp = sum(m["false_pos"] for _, m in all_metrics)
    bears = [m["mean_bearing_deg"] for _, m in all_metrics if not np.isnan(m["mean_bearing_deg"])]
    rngs = [m["mean_range_pct"] for _, m in all_metrics if not np.isnan(m["mean_range_pct"])]
    print("\n===== AGGREGATE (all poses) =====")
    print(f"detection rate  {tot_det}/{tot_vis} = {tot_det / tot_vis:.0%}")
    print(f"colour accuracy {tot_col}/{tot_det} = {tot_col / tot_det:.0%}")
    print(f"false positives {tot_fp}")
    print(f"mean bearing error {np.mean(bears):.2f} deg")
    print(f"mean range error   {np.mean(rngs):.1f} %")

    out_png = OUT_DIR / "perception.png"
    imageio.imwrite(out_png, primary_frame)
    print(f"\nannotated frame -> {out_png}  shape={primary_frame.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
