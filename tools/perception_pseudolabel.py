"""Auto-generate COCO pseudo-labels + preview images for raw frames with the learned detector.

Runs ``TinyBalloonNet`` over a directory of UNLABELLED frames and writes (a) a COCO-format
annotation json a human only has to CORRECT (importable into CVAT / Label Studio / Roboflow),
and (b) annotated preview JPGs for quick visual review. Pseudo-labelling is far faster than
labelling from scratch; the detector's mistakes (extra/missing/mis-coloured boxes) are edited,
not drawn.

Usage:
    uv run --extra learn --extra perception python -m tools.perception_pseudolabel \
        --img-dir ai/balloon/raw/realsense --out ai/balloon/raw/pseudo_realsense.json \
        --preview-dir ai/balloon/raw/preview_realsense
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from umiusi_sim.perception.learned_detector import load_learned_detector

# COCO category ids match the existing annotations (annotations/train.json).
CAT_ID = {"red": 1, "blue": 2, "yellow": 3}
BGR = {"red": (0, 0, 255), "blue": (255, 128, 0), "yellow": (0, 215, 255)}
EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img-dir", required=True, help="directory of raw frames to pseudo-label")
    ap.add_argument("--model", default="models/perception_learned/tiny_balloon.pt")
    ap.add_argument("--out", required=True, help="output COCO json path")
    ap.add_argument("--preview-dir", default=None, help="if set, write annotated preview JPGs here")
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold for a pseudo-box")
    args = ap.parse_args()

    detect = load_learned_detector(args.model, conf_thresh=args.conf)
    img_dir = Path(args.img_dir)
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annos: list[dict] = []
    per_colour = {c: 0 for c in CAT_ID}
    aid = 0
    for img_id, path in enumerate(paths):
        rgb = np.asarray(imageio.imread(path))[:, :, :3]
        h, w = rgb.shape[:2]
        images.append({"id": img_id, "file_name": path.name, "width": int(w), "height": int(h)})
        dets = detect(rgb)
        prev = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if preview_dir else None
        for d in dets:
            u0, v0, u1, v1 = d.bbox
            annos.append({
                "id": aid, "image_id": img_id, "category_id": CAT_ID[d.colour],
                "bbox": [int(u0), int(v0), int(u1 - u0), int(v1 - v0)],
                "area": int((u1 - u0) * (v1 - v0)), "score": round(float(d.confidence), 3),
                "iscrowd": 0,
            })
            per_colour[d.colour] += 1
            aid += 1
            if prev is not None:
                c = BGR[d.colour]
                cv2.rectangle(prev, (u0, v0), (u1, v1), c, 2)
                cv2.putText(prev, f"{d.colour} {d.confidence:.2f}", (u0, max(0, v0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if prev is not None:
            imageio.imwrite(preview_dir / f"{path.stem}_pl.jpg", cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))

    coco = {
        "images": images, "annotations": annos,
        "categories": [{"id": i, "name": n} for n, i in CAT_ID.items()],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(coco, f, indent=1)
    n_img = len(images)
    with_box = len({a["image_id"] for a in annos})
    print(f"{img_dir}: {n_img} images -> {len(annos)} pseudo-boxes "
          f"({with_box} imgs have >=1) | per-colour {per_colour}")
    print(f"  COCO json -> {args.out}" + (f" | previews -> {preview_dir}" if preview_dir else ""))


if __name__ == "__main__":
    main()
