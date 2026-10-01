"""Pseudo-label pool footage for review: candidate boxes from the current detector, rendered for checking.

Writes, per selected frame, the image (copied to <out>/images), a JSON of candidates, and a REVIEW image:
the frame at 1280 px wide with each candidate box numbered (#k colour conf) and a labelled 100 px grid, so a
reviewer can say "keep #1, #3 is blue, delete #2, missed balloon at x 420-510 y 300-390" in ORIGINAL pixels.

    uv run python -m tools.label_assist select --src DIR --out OUT --n 320 --conf 0.15
    uv run python -m tools.label_assist refine --out OUT          # GrabCut-tighten reviewer-added boxes
    uv run python -m tools.label_assist coco --out OUT --val-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import cv2
import numpy as np

COLS = {"red": (0, 0, 255), "yellow": (0, 220, 255), "blue": (255, 120, 0)}
CAT = {"red": 1, "blue": 2, "yellow": 3}          # perception_train COCO convention


def _grid(img):
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def render(img, boxes, title):
    g = img.copy()
    _grid(g)
    for k, b in enumerate(boxes):
        x0, y0, x1, y1 = map(int, b["bbox"])
        cv2.rectangle(g, (x0, y0), (x1, y1), COLS[b["colour"]], 2)
        cv2.putText(g, f"#{k} {b['colour'][0]} {b.get('conf', 1.0):.2f}", (x0, max(30, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(g, f"#{k} {b['colour'][0]} {b.get('conf', 1.0):.2f}", (x0, max(30, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLS[b["colour"]], 1)
    cv2.putText(g, title, (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return g


def select(a):
    from umiusi_perception.learned_detector import load_learned_detector
    out = pathlib.Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    srcs = sorted(p for d in a.src for p in pathlib.Path(d).glob("*.jpg"))
    pick = [srcs[i] for i in np.unique(np.linspace(0, len(srcs) - 1, min(a.n, len(srcs))).astype(int))]
    det = load_learned_detector(a.model, conf_thresh=a.conf)
    index = []
    for p in pick:
        name = f"{a.prefix}_{p.stem}.jpg"
        img = cv2.imread(str(p))
        shutil.copy(p, out / "images" / name)
        ds = det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cands = [{"bbox": [float(v) for v in d.bbox], "colour": d.colour,
                  "conf": round(float(getattr(d, "confidence", 0.0)), 3)} for d in ds]
        (out / "images" / (name + ".cand.json")).write_text(json.dumps(cands))
        cv2.imwrite(str(out / "review" / name), render(img, cands, name), [cv2.IMWRITE_JPEG_QUALITY, 85])
        index.append({"image": name, "w": img.shape[1], "h": img.shape[0], "n_cand": len(cands)})
    (out / f"index_{a.prefix}.json").write_text(json.dumps(index, indent=1))
    print(f"selected {len(pick)} of {len(srcs)} -> {out}; candidates {sum(i['n_cand'] for i in index)}")


def grabcut_box(img, box, pad=0.25):
    """Tighten a rough box around ONE balloon: GrabCut inside the (padded) box, take the largest blob."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    rx0, ry0 = max(0, int(x0 - pad * bw)), max(0, int(y0 - pad * bh))
    rx1, ry1 = min(w - 1, int(x1 + pad * bw)), min(h - 1, int(y1 + pad * bh))
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return box, False
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (int(x0), int(y0), int(bw), int(bh)), bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return box, False
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)[ry0:ry1, rx0:rx1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return box, False
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, ww, hh, area = stats[k]
    nb = [rx0 + x, ry0 + y, rx0 + x + ww, ry0 + y + hh]
    # sanity: keep only if the tightened box still overlaps the rough one substantially
    ix = max(0, min(nb[2], x1) - max(nb[0], x0)) * max(0, min(nb[3], y1) - max(nb[1], y0))
    ok = area > 0.15 * bw * bh and ix > 0.3 * bw * bh
    return (nb if ok else box), ok


def refine(a):
    """Apply reviewer decisions (<image>.review.json) and GrabCut-tighten ADDED boxes -> <image>.label.json."""
    out = pathlib.Path(a.out)
    n_img = n_add = n_tight = 0
    for rv in sorted((out / "images").glob("*.review.json")):
        name = rv.name[: -len(".review.json")]
        img = cv2.imread(str(out / "images" / name))
        cands = json.loads((out / "images" / (name + ".cand.json")).read_text())
        r = json.loads(rv.read_text())
        labels = []
        for k, b in enumerate(cands):
            # an unreviewed candidate is NOT a label (5/720 reviews skipped or misnumbered one)
            d = r.get("boxes", {}).get(str(k), "delete")
            if d == "delete":
                continue
            colour = b["colour"] if d == "keep" else d
            labels.append({"bbox": b["bbox"], "colour": colour, "src": "model"})
        for m in r.get("missed", []):
            box, ok = grabcut_box(img, m["bbox"])
            n_add += 1
            n_tight += ok
            labels.append({"bbox": [float(v) for v in box], "colour": m["colour"], "src": "added",
                           "tight": bool(ok)})
        # merge duplicates: same colour, IoU > 0.5 -> keep the larger
        labels = _nms(labels)
        (out / "images" / (name + ".label.json")).write_text(json.dumps(
            {"labels": labels, "uncertain": r.get("uncertain", False), "note": r.get("note", "")}))
        cv2.imwrite(str(out / "check" / name) if (out / "check").exists() else str(out / "review" / ("L_" + name)),
                    render(img, [{**x, "conf": 1.0} for x in labels], "LABEL " + name), [cv2.IMWRITE_JPEG_QUALITY, 85])
        n_img += 1
    print(f"refined {n_img} images; added {n_add} boxes, GrabCut-tightened {n_tight}")


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(labels):
    labels = sorted(labels, key=lambda x: -(x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    keep = []
    for x in labels:
        if all(not (x["colour"] == k["colour"] and _iou(x["bbox"], k["bbox"]) > 0.5) for k in keep):
            keep.append(x)
    return keep


def drop_upper_of_pairs(labels):
    """Remove the UPPER box of a vertically stacked same-colour pair.

    Near the surface the underside of the water mirrors the scene, and a balloon whose top touches it shows
    up as a touching, knot-mirrored "pair" (2026-09-13 footage). Two boxes of the same colour that overlap
    horizontally by >= 60 % of the narrower one and whose vertical gap is under 30 % of the box height are
    treated as balloon + reflection; the upper one (smaller y) is dropped.
    """
    drop = set()
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j or a["colour"] != b["colour"]:
                continue
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            ov = max(0.0, min(ax1, bx1) - max(ax0, bx0)) / max(1e-6, min(ax1 - ax0, bx1 - bx0))
            h = max(ay1 - ay0, by1 - by0)
            if ov >= 0.6 and ay1 <= by0 + 0.3 * h and by0 - ay1 < 0.3 * h:   # a is above b, close
                drop.add(i)
    return [x for k, x in enumerate(labels) if k not in drop]


def _block(name, size=300):
    """Time block of a frame for the train/val split: same source video + 300-frame window.

    Labelling rounds use different prefixes for the SAME video (p0913, p0913b, ...); the trailing letters
    are stripped so neighbouring frames from different rounds land in the same block (no near-duplicate
    leaking from train into val). The 10/01 hit/negative selections keep their run tag (r1/r2).
    """
    stem, frame = name.rsplit("_f", 1)
    src = stem.split("_")[0].rstrip("abcdefghijklmnopqrstuvwxyz") + "_" + "_".join(stem.split("_")[1:])
    src = src.replace("hits", "").replace("neg", "")
    return f"{src}_{int(frame[:5]) // size}"


def coco(a):
    out = pathlib.Path(a.out)
    files = sorted((out / "images").glob("*.label.json"))
    rng = np.random.default_rng(a.seed)
    # split by TIME BLOCK, not by frame: neighbouring frames are near-duplicates, a random split leaks
    names = [f.name[: -len(".label.json")] for f in files]
    blocks = sorted({_block(n, a.block) for n in names})
    val_blocks = set(rng.choice(blocks, max(1, int(round(a.val_frac * len(blocks)))), replace=False).tolist())
    sets = {"train": {"images": [], "annotations": []}, "val": {"images": [], "annotations": []}}
    judged = {}
    if a.pairs == "judged":
        # pairs.json: [[name, [[colour, bbox], ...]], ...] (upper boxes, U0, U1, ... in order);
        # pairs_verdict_*.json: {name: {"U0": "real" | "reflection" | "unsure"}}. Unsure is kept.
        uppers = {n: ups for n, ups in json.loads((out / "pairs.json").read_text())}
        for vf in sorted(out.glob("pairs_verdict_*.json")):
            for n, v in json.loads(vf.read_text()).items():
                judged[n] = [uppers[n][int(k[1:])] for k, verdict in v.items() if verdict == "reflection"]
    aid = 1
    for i, (f, n) in enumerate(zip(files, names)):
        blk = _block(n, a.block)
        s = "val" if blk in val_blocks else "train"
        d = json.loads(f.read_text())
        if d.get("uncertain") and a.drop_uncertain:
            continue
        img = cv2.imread(str(out / "images" / n))
        if a.pairs == "lower":
            d["labels"] = drop_upper_of_pairs(d["labels"])
        elif a.pairs == "judged":
            refl = judged.get(n, [])
            d["labels"] = [lb for lb in d["labels"]
                           if not any(lb["colour"] == c and [int(v) for v in lb["bbox"]] == b for c, b in refl)]
        sets[s]["images"].append({"id": i + 1, "file_name": n, "width": img.shape[1], "height": img.shape[0]})
        for lb in d["labels"]:
            x0, y0, x1, y1 = lb["bbox"]
            sets[s]["annotations"].append({"id": aid, "image_id": i + 1, "category_id": CAT[lb["colour"]],
                                           "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                           "iscrowd": 0})
            aid += 1
    cats = [{"id": v, "name": k} for k, v in CAT.items()]
    for s, d in sets.items():
        d["categories"] = cats
        (out / f"{a.name}_{s}.json").write_text(json.dumps(d))
        print(f"{s}: {len(d['images'])} images, {len(d['annotations'])} boxes "
              f"({ {k: sum(1 for x in d['annotations'] if x['category_id'] == v) for k, v in CAT.items()} })")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--src", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=320)
    s.add_argument("--conf", type=float, default=0.15)
    s.add_argument("--prefix", default="p0913")
    s.add_argument("--model", default="examples/balloon_detector/model.pt")
    r = sub.add_parser("refine")
    r.add_argument("--out", required=True)
    c = sub.add_parser("coco")
    c.add_argument("--out", required=True)
    c.add_argument("--name", default="pool")
    c.add_argument("--val-frac", type=float, default=0.15)
    c.add_argument("--block", type=int, default=300, help="frames per time block for the train/val split")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--drop-uncertain", action="store_true")
    c.add_argument("--pairs", choices=("both", "lower", "judged"), default="both",
                   help="stacked same-colour pairs: keep both / drop every upper / drop uppers judged reflections")
    a = ap.parse_args()
    {"select": select, "refine": refine, "coco": coco}[a.cmd](a)


if __name__ == "__main__":
    main()
