"""Validate the balloon detector against a labelled REAL underwater dataset (COCO format).

Runs ``detect_balloons`` over a split, IoU-matches detections to COCO GT boxes (IoU>=0.3 = TP),
and reports precision / recall / F1 per colour and overall, the false-positive count, and how
many of those FPs are water/reflection artefacts (characterised by image band + colour). It runs
BOTH the BEFORE config (sim thresholds, no reflection reject) and the AFTER config
(real thresholds + reflection reject) so the improvement is quantified side by side. A few
annotated frames (GT boxes + colour-coded detections; rejected reflections marked) are written to
``<tmp>/umiusi_sim/perception_eval_*.png`` for visual review.

The dataset lives OUTSIDE the repo (a user-provided folder); point at it with --data-root.
Category map: 1=balloon_red, 2=balloon_blue, 3=balloon_yellow. COCO bbox = [x, y, w, h].

Usage:
    python -m tools.perception_eval --split val  --profile real --reject-reflections
    python -m tools.perception_eval --split train --profile sim
    python -m tools.perception_eval --split val  --compare        # BEFORE vs AFTER table only

Needs the `dev` extra (imageio, Pillow) and the `perception` extra (scipy).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from collections import defaultdict

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from umiusi_perception import REAL_THRESHOLDS, SIM_THRESHOLDS, detect_balloons
from umiusi_perception.hough_detector import detect_combined, detect_hough

DATA_ROOT = pathlib.Path("/home/satoi/mujoco_ws/ai/balloon")
CATMAP = {1: "red", 2: "blue", 3: "yellow"}
COLOURS = ["red", "yellow", "blue"]
IOU_TP = 0.3
FOVY = 60.0                 # dataset has no camera intrinsics; only bbox IoU is used, not bearing
MAX_AREA_FRAC = 0.02        # real profile: drop blobs > 2% of the frame (large ragged water/wall)
OUT_DIR = pathlib.Path(tempfile.gettempdir()) / "umiusi_sim"
_DRAW = {"red": (255, 60, 60), "yellow": (255, 220, 40), "blue": (60, 120, 255)}


def load_split(root: pathlib.Path, split: str):
    """Return (images list, image_id -> [ (colour, [x,y,w,h]) ] GT dict)."""
    d = json.load(open(root / "annotations" / f"{split}.json"))
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    gt = defaultdict(list)
    for a in d["annotations"]:
        gt[a["image_id"]].append((CATMAP[a["category_id"]], list(a["bbox"])))
    images = [(im["id"], id2file[im["id"]]) for im in d["images"]]
    return images, gt, split


def read_rgb(path: pathlib.Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def det_to_xywh(d):
    u0, v0, u1, v1 = d.bbox
    return [u0, v0, u1 - u0, v1 - v0]


def match(dets, gts):
    """Greedy IoU match (highest IoU first). Returns (tp_pairs, matched_det_idx, matched_gt_idx).

    Colour must agree for a match (a red detection on a blue GT is not a TP for red)."""
    cand = []
    for di, d in enumerate(dets):
        for gi, (gcol, gbox) in enumerate(gts):
            if d.colour != gcol:
                continue
            iou = iou_xywh(det_to_xywh(d), gbox)
            if iou >= IOU_TP:
                cand.append((iou, di, gi))
    cand.sort(reverse=True)
    used_d, used_g, pairs = set(), set(), []
    for iou, di, gi in cand:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        pairs.append((di, gi))
    return pairs, used_d, used_g


def make_detfn(method, thresholds, reject, max_area_frac):
    """Return a ``rgb -> [Detection]`` callable for the chosen METHOD (color/hough/combined).

    * ``color``    — the HSV colour detector (``detect_balloons``); the existing baseline.
    * ``hough``    — the shape-based Hough-circle detector (``detect_hough``); colour assigned by
                     inner-disc HSV vote so its Detections are directly comparable.
    * ``combined`` — colour detections, plus round-and-coloured circles the colour path missed
                     (``detect_combined`` mode="recover").
    """
    if method == "color":
        return lambda rgb: detect_balloons(rgb, fovy_deg=FOVY, thresholds=thresholds,
                                           reject_reflections=reject, max_area_frac=max_area_frac)
    if method == "hough":
        return lambda rgb: detect_hough(rgb, fovy_deg=FOVY, thresholds=thresholds)
    if method == "combined":
        return lambda rgb: detect_combined(rgb, fovy_deg=FOVY, thresholds=thresholds,
                                           reject_reflections=reject, max_area_frac=max_area_frac,
                                           mode="recover")
    raise ValueError(f"unknown method {method!r}")


def evaluate(images, gt, root, split, thresholds, reject, max_area_frac, detfn=None):
    """Run the detector over the split; accumulate per-colour TP/FP/FN and FP characterisation.

    ``detfn`` optionally overrides how detections are produced (see ``make_detfn``); by default the
    colour detector is used with the given thresholds/reject/area cap."""
    if detfn is None:
        detfn = make_detfn("color", thresholds, reject, max_area_frac)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    fp_lowerband = defaultdict(int)   # FPs whose centroid is in the bottom third of the frame
    per_image = []                    # (image_id, file, rgb, dets, gts, used_d) for annotation
    for image_id, fname in images:
        rgb = read_rgb(root / f"{split}2017" / fname)
        H = rgb.shape[0]
        dets = detfn(rgb)
        gts = gt[image_id]
        pairs, used_d, used_g = match(dets, gts)
        for _, gi in pairs:
            tp[gts[gi][0]] += 1
        for gi, (gcol, _) in enumerate(gts):
            if gi not in used_g:
                fn[gcol] += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                fp[d.colour] += 1
                if d.centroid[1] > 2.0 * H / 3.0:
                    fp_lowerband[d.colour] += 1
        per_image.append((image_id, fname, rgb, dets, gts, used_d))
    return tp, fp, fn, fp_lowerband, per_image


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = 2 * p * r / (p + r) if (p and r and not np.isnan(p) and not np.isnan(r) and (p + r) > 0) else \
        (0.0 if (tp + fp + fn) else float("nan"))
    return p, r, f


def print_report(tag, tp, fp, fn, fp_lowerband):
    print(f"\n===== {tag} =====")
    print(f"  {'colour':7s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s} "
          f"{'FP@lower⅓':>10s}")
    T = F = N = L = 0
    for c in COLOURS:
        p, r, f = prf(tp[c], fp[c], fn[c])
        T += tp[c]
        F += fp[c]
        N += fn[c]
        L += fp_lowerband[c]
        print(f"  {c:7s} {tp[c]:4d} {fp[c]:4d} {fn[c]:4d} {p:6.2f} {r:6.2f} {f:6.2f} "
              f"{fp_lowerband[c]:10d}")
    p, r, f = prf(T, F, N)
    print(f"  {'ALL':7s} {T:4d} {F:4d} {N:4d} {p:6.2f} {r:6.2f} {f:6.2f} {L:10d}")
    print(f"  -> {F} false positives total, {L} of them in the lower third of the frame "
          f"(likely water surface / reflections / pool floor)")
    return {"tp": T, "fp": F, "fn": N, "prec": p, "rec": r, "f1": f, "fp_lower": L}


def annotate(rgb, dets, gts, used_d, path):
    """GT boxes in white (dashed feel via thin), detections colour-coded (matched=solid,
    unmatched/FP=thin), rejected reflections in magenta. Saves a PNG."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for gcol, (x, y, w, h) in gts:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=1)
    for di, d in enumerate(dets):
        u0, v0, u1, v1 = d.bbox
        col = _DRAW.get(d.colour, (255, 255, 255))
        wdt = 3 if di in used_d else 1
        draw.rectangle([u0, v0, u1, v1], outline=col, width=wdt)
    # rejected reflections (still present on the objects with is_reflection=True are dropped from
    # `dets`, so re-run detection without rejection to draw what got cut)
    img.save(path)


def draw_rejected(rgb, path):
    """Second pass: show which blobs the reflection filter removed (magenta) vs kept."""
    kept = detect_balloons(rgb, fovy_deg=FOVY, thresholds=REAL_THRESHOLDS,
                           reject_reflections=True, max_area_frac=MAX_AREA_FRAC)
    alld = detect_balloons(rgb, fovy_deg=FOVY, thresholds=REAL_THRESHOLDS,
                          reject_reflections=False, max_area_frac=MAX_AREA_FRAC)
    kept_boxes = {d.bbox for d in kept}
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for d in alld:
        u0, v0, u1, v1 = d.bbox
        if d.bbox in kept_boxes:
            draw.rectangle([u0, v0, u1, v1], outline=_DRAW[d.colour], width=3)
        else:
            draw.rectangle([u0, v0, u1, v1], outline=(255, 0, 255), width=2)  # rejected reflection
    img.save(path)


def _detfn_for(method: str, weights: str, input_size: int | None, conf: float):
    """rgb->[Detection] callable for a classical method or the learned detector (lazy torch import)."""
    if method == "learned":
        from umiusi_perception.learned_detector import load_learned_detector  # lazy: keeps torch optional
        return load_learned_detector(weights, input_size=input_size, conf_thresh=conf)
    return make_detfn(method, REAL_THRESHOLDS, True, MAX_AREA_FRAC)


def compare(weights: str, data_root: pathlib.Path, split: str, methods: list[str],
            input_size: int | None = None, conf: float = 0.3) -> dict:
    """Run each method over the split, print per-colour reports + an overall comparison table.

    The shared IoU evaluation harness (learned TinyBalloonNet vs classical color/hough/combined on
    the SAME split). Returns ``{method: summary_dict}`` (summary as produced by ``print_report``)."""
    images, gt, split = load_split(data_root, split)
    n_gt = sum(len(v) for v in gt.values())
    print(f"dataset: {data_root}  split={split}  images={len(images)}  GT balloons={n_gt}")
    print("head-to-head: classical (real profile) vs learned (TinyBalloonNet, int8-ready)\n")

    summary = {}
    for m in methods:
        detfn = _detfn_for(m, weights, input_size, conf)
        res = evaluate(images, gt, data_root, split, REAL_THRESHOLDS, True, MAX_AREA_FRAC, detfn)
        summary[m] = print_report(f"METHOD = {m}", res[0], res[1], res[2], res[3])

    if len(methods) > 1:
        print("\n===== method comparison (overall) =====")
        print(f"  {'method':10s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'FP':>5s}")
        for m in methods:
            s = summary[m]
            print(f"  {m:10s} {s['prec']:6.2f} {s['rec']:6.2f} {s['f1']:6.2f} {s['fp']:5d}")
    return summary


def run_methods(args):
    """Compare detection METHODS (color / hough / combined) on the same split, real profile.

    Reports per-colour precision/recall/F1 + FP for each method so shape vs colour vs their
    combination are directly comparable, and (unless --no-images) saves annotated frames
    (GT=white, detections colour-coded) to <tmp>/umiusi_sim/hough_<method>_<split>_<stem>.png."""
    if not (args.data_root / "annotations" / f"{args.split}.json").exists():
        print(f"ERROR: dataset not found at {args.data_root} (need annotations/{args.split}.json)")
        return 1
    images, gt, split = load_split(args.data_root, args.split)
    n_gt = sum(len(v) for v in gt.values())
    print(f"dataset: {args.data_root}  split={split}  images={len(images)}  GT balloons={n_gt}")
    print("profile=real (thresholds + reflection reject + area cap); Hough=gray+CLAHE, "
          "GRADIENT_ALT param2=0.4, radii 10-100px")

    methods = [args.method] if args.method else ["color", "hough", "combined"]
    summary = {}
    per_image_by_method = {}
    for m in methods:
        detfn = make_detfn(m, REAL_THRESHOLDS, True, MAX_AREA_FRAC)
        res = evaluate(images, gt, args.data_root, split, REAL_THRESHOLDS, True, MAX_AREA_FRAC, detfn)
        summary[m] = print_report(f"METHOD = {m}", res[0], res[1], res[2], res[3])
        per_image_by_method[m] = res[4]

    if len(methods) > 1:
        print("\n===== method comparison (overall) =====")
        print(f"  {'method':10s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'FP':>5s}")
        for m in methods:
            s = summary[m]
            print(f"  {m:10s} {s['prec']:6.2f} {s['rec']:6.2f} {s['f1']:6.2f} {s['fp']:5d}")

    if not args.no_images:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for m in methods:
            for image_id, fname, rgb, dets, gts, used_d in per_image_by_method[m][:4]:
                p = OUT_DIR / f"hough_{m}_{split}_{pathlib.Path(fname).stem}.png"
                annotate(rgb, dets, gts, used_d, p)
                saved.append(p)
        print("\nannotated frames (white=GT, thick=matched det, thin=FP; boxes are the circle's "
              "square for hough/combined):")
        for p in saved:
            print(f"  {p}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--profile", choices=["sim", "real"], default="real")
    ap.add_argument("--reject-reflections", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="always run BEFORE(sim) vs AFTER(real+reject) both (default behaviour)")
    ap.add_argument("--data-root", type=pathlib.Path, default=DATA_ROOT)
    ap.add_argument("--no-images", action="store_true", help="skip writing annotated PNGs")
    ap.add_argument("--methods", action="store_true",
                    help="compare color vs hough vs combined (real profile) side by side and exit")
    ap.add_argument("--method", choices=["color", "hough", "combined"], default=None,
                    help="evaluate a single detection method (real profile) and exit")
    args = ap.parse_args()

    if args.methods or args.method:
        return run_methods(args)

    if not (args.data_root / "annotations" / f"{args.split}.json").exists():
        print(f"ERROR: dataset not found at {args.data_root} (need annotations/{args.split}.json)")
        return 1

    images, gt, split = load_split(args.data_root, args.split)
    n_gt = sum(len(v) for v in gt.values())
    print(f"dataset: {args.data_root}  split={split}  images={len(images)}  GT balloons={n_gt}")

    # BEFORE: sim thresholds, no reflection reject, no area cap (the original behaviour on real).
    before = evaluate(images, gt, args.data_root, split, SIM_THRESHOLDS, False, None)
    b = print_report("BEFORE  (sim thresholds, no reflection reject)",
                     before[0], before[1], before[2], before[3])

    # AFTER: real thresholds + reflection reject + area cap.
    after = evaluate(images, gt, args.data_root, split, REAL_THRESHOLDS, True, MAX_AREA_FRAC)
    a = print_report("AFTER   (real thresholds + reflection reject + area cap)",
                     after[0], after[1], after[2], after[3])

    print("\n===== BEFORE -> AFTER (overall) =====")
    print(f"  recall  {b['rec']:.2f} -> {a['rec']:.2f}")
    print(f"  precision {b['prec']:.2f} -> {a['prec']:.2f}")
    print(f"  F1      {b['f1']:.2f} -> {a['f1']:.2f}")
    print(f"  false positives {b['fp']} -> {a['fp']}   "
          f"(lower-third FPs {b['fp_lower']} -> {a['fp_lower']})")

    # If the user asked for a specific single profile run, also state it explicitly.
    if not args.compare:
        prof = REAL_THRESHOLDS if args.profile == "real" else SIM_THRESHOLDS
        cap = MAX_AREA_FRAC if args.profile == "real" else None
        single = evaluate(images, gt, args.data_root, split, prof,
                          args.reject_reflections, cap)
        print_report(f"REQUESTED  (--profile {args.profile} "
                     f"{'--reject-reflections' if args.reject_reflections else ''})",
                     single[0], single[1], single[2], single[3])

    if not args.no_images:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # Annotate the AFTER run for a few images, and a rejected-reflections view.
        _, _, _, _, per_image = after
        saved = []
        for k, (image_id, fname, rgb, dets, gts, used_d) in enumerate(per_image[:4]):
            p = OUT_DIR / f"perception_eval_{split}_{pathlib.Path(fname).stem}.png"
            annotate(rgb, dets, gts, used_d, p)
            saved.append(p)
            pr = OUT_DIR / f"perception_eval_{split}_{pathlib.Path(fname).stem}_reflections.png"
            draw_rejected(rgb, pr)
            saved.append(pr)
        print("\nannotated frames (white=GT, thick=matched det, thin=FP, magenta=rejected reflection):")
        for p in saved:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
