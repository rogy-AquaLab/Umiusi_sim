"""Train the RECOMMENDED learned balloon detector (``TinyBalloonNet``) on the COCO balloon set.

This is Part 2: a training pipeline built to STRETCH the current 40-image set with heavy
augmentation and to scale gracefully as more data arrives. It trains the CenterNet-lite tiny CNN
(the Pi-4-safe architecture ``tools/perception_bench`` recommends), saves an int8-ready checkpoint,
and reports val precision/recall/F1 per colour vs the classical color/hough detectors (via
``tools/perception_eval_learned``, which reuses the same IoU harness).

Adding MORE data "just works": drop images into ``train2017/`` and append their boxes to
``annotations/train.json`` (COCO: category 1=red, 2=blue, 3=yellow). Nothing here is hard-coded to
40 images -- the dataset path/json/img-dir are all CLI args, so ``--data-root`` (or the individual
``--train-json`` / ``--img-dir``) is the only thing to point at a growing set.

Heavy augmentation (albumentations, applied to image + boxes together):
  * geometric   : horizontal flip, shift/scale/rotate, random resized crop
  * photometric : STRONG brightness/contrast, hue/sat/value jitter, RGB shift
  * blur        : gaussian / motion / median (motion blur mimics a moving robot)
  * UNDERWATER  : an AGGRESSIVE water colour-cast that SWEEPS blue<->green-grey (+ red attenuation,
                  depth veil, desaturation) -- the key sim->real gap-closer, so the model can't key
                  on absolute colour (sim renders greener now, real is green-grey, both must work)
  * RESOLUTION  : random downscale-then-upscale + JPEG recompression -- emulates the real ~332x176
                  compressed footage so the model is robust to low-res/blocky input
Every transform is bounded so boxes stay valid; ``A.Resize`` to the network input closes the chain
(the resolution degradation runs after, at network resolution).

Usage (short baseline on the current 40 imgs):
    uv run python -m tools.perception_train --epochs 40 --batch 8 --input-size 256
    uv run python -m tools.perception_train --no-eval                 # train only, skip val report
    uv run python -m tools.perception_train --data-root /path/to/more/data   # more data, same code
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import albumentations as A
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from umiusi_perception.learned_detector import (
    COLOUR_TO_IDX,
    STRIDE,
    TinyBalloonNet,
)

def _default_data_root() -> pathlib.Path:
    """Balloon dataset root. Override with $UMIUSI_BALLOON_DATA; otherwise default to the
    user-provided ``ai/balloon`` folder that sits alongside the repo (../ai/balloon)."""
    env = os.environ.get("UMIUSI_BALLOON_DATA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[1].parent / "ai" / "balloon"


DATA_ROOT = _default_data_root()
CATMAP = {1: "red", 2: "blue", 3: "yellow"}     # COCO category_id -> colour name
DEFAULT_OUT = pathlib.Path("models/perception_learned/tiny_balloon.pt")


# ----------------------------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------------------------
def _underwater(image: np.ndarray, **kwargs) -> np.ndarray:
    """AGGRESSIVE underwater domain-randomisation colour-cast (the key sim->real gap-closer).

    Sweeps the WHOLE water-colour range the footage spans — from saturated BLUE (clean pool) to
    desaturated GREEN-GREY (murky competition pool) — via wide, independent per-channel white-balance
    gains, a depth-veil blend toward a randomly-coloured water tint, and a variable desaturation pull
    toward grey (real footage is low-saturation and low-contrast). Applied strongly and to ALL
    training frames (real AND sim), so the tiny CNN CANNOT key on an absolute colour ("blue == water");
    it must use shape/contrast instead. Red is attenuated (physical: red light dies first underwater).
    Randomised per call; pixels never move, so boxes stay exact."""
    img = image.astype(np.float32)
    # Wide per-channel white-balance / gain. Red attenuated (physical); green & blue vary widely and
    # independently so the cast sweeps blue-dominant <-> green-grey-dominant.
    r_att = np.random.uniform(0.35, 0.95)
    g_gain = np.random.uniform(0.80, 1.35)
    b_gain = np.random.uniform(0.65, 1.35)
    img[..., 0] *= r_att
    img[..., 1] *= g_gain
    img[..., 2] *= b_gain
    # Depth veil: blend toward a water colour drawn ANYWHERE from green-grey to blue (red weakest).
    g_anchor = np.random.uniform(55, 130)
    veil = np.array([np.random.uniform(15, g_anchor),                 # red weakest
                     g_anchor,
                     np.random.uniform(40, g_anchor + 35)],           # blue < or > green (both casts)
                    dtype=np.float32)
    alpha = np.random.uniform(0.06, 0.45)          # depth-haze strength (blend toward the water colour)
    img = (1 - alpha) * img + alpha * veil
    # Desaturate toward grey (real footage is low-saturation) — variable amount.
    gray = img.mean(axis=2, keepdims=True)
    desat = np.random.uniform(0.0, 0.55)
    img = (1 - desat) * img + desat * gray
    return np.clip(img, 0, 255).astype(np.uint8)


def build_aug(input_size: int, train: bool) -> A.Compose:
    """albumentations pipeline (image + COCO boxes). Heavy for train, resize-only for val."""
    bbox = A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.25, min_area=16)
    if not train:
        return A.Compose([A.Resize(input_size, input_size)], bbox_params=bbox)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-12, 12), p=0.6),
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.5, 1.0), ratio=(0.75, 1.33),
                            p=0.5),
        # STRONG photometric DR (real footage brightness/contrast/colour varies a lot).
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=45, val_shift_limit=30, p=0.6),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9)),
                 A.MedianBlur(blur_limit=5)], p=0.5),
        # THE key domain-gap aug: aggressive water colour-cast sweeping blue<->green-grey.
        A.Lambda(image=_underwater, p=0.85, name="underwater_cast"),
        A.Resize(input_size, input_size),
        # RESOLUTION / COMPRESSION degradation: emulate the real ~332x176 compressed footage
        # (downscale-then-upscale loses fine detail; JPEG adds blocking). Pixel-only -> boxes intact.
        A.Downscale(scale_range=(0.25, 0.6), p=0.5),
        A.ImageCompression(quality_range=(28, 75), p=0.5),
    ], bbox_params=bbox)


# ----------------------------------------------------------------------------------------------
# CenterNet target rendering
# ----------------------------------------------------------------------------------------------
def _gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CenterNet gaussian radius so a box shifted within the radius still overlaps GT by min_overlap."""
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / 2
    return max(0.0, min(r1, r2, r3))


def _draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int):
    """Splat an unnormalised 2D gaussian (peak 1.0) into one heatmap channel (in-place max)."""
    r = max(1, int(radius))
    sigma = r / 3.0
    xs = np.arange(-r, r + 1)
    g = np.exp(-(xs[None, :] ** 2 + xs[:, None] ** 2) / (2 * sigma * sigma))
    H, W = hm.shape
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    gx0, gy0 = x0 - (cx - r), y0 - (cy - r)
    sub = hm[y0:y1, x0:x1]
    gsub = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
    np.maximum(sub, gsub, out=sub)


def build_targets(boxes: list, labels: list, input_size: int):
    """Boxes ([x,y,w,h] coco, in input px) + colour labels -> (hm, wh, reg_mask) target tensors."""
    fh = fw = input_size // STRIDE
    hm = np.zeros((3, fh, fw), dtype=np.float32)
    wh = np.zeros((2, fh, fw), dtype=np.float32)
    mask = np.zeros((fh, fw), dtype=np.float32)
    for (x, y, w, h), col in zip(boxes, labels):
        if w <= 1 or h <= 1:
            continue
        ci = COLOUR_TO_IDX[col]
        cx = (x + w / 2) / STRIDE
        cy = (y + h / 2) / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < fw and 0 <= iy < fh):
            continue
        radius = _gaussian_radius(h / STRIDE, w / STRIDE)
        _draw_gaussian(hm[ci], ix, iy, radius)
        hm[ci, iy, ix] = 1.0
        wh[0, iy, ix] = w / input_size
        wh[1, iy, ix] = h / input_size
        mask[iy, ix] = 1.0
    return torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(mask)


# ----------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------
class BalloonDataset(Dataset):
    """COCO balloon set -> (image tensor, hm, wh, reg_mask). Parameterised path so more data drops in."""

    def __init__(self, root: pathlib.Path, json_name: str, img_subdir: str, input_size: int,
                 train: bool):
        self.root = pathlib.Path(root)
        self.img_dir = self.root / img_subdir
        self.input_size = input_size
        self.aug = build_aug(input_size, train)
        d = json.load(open(self.root / "annotations" / json_name))
        id2file = {im["id"]: im["file_name"] for im in d["images"]}
        per_img: dict[int, list] = {im["id"]: [] for im in d["images"]}
        for a in d["annotations"]:
            per_img[a["image_id"]].append((CATMAP[a["category_id"]], list(a["bbox"])))
        self.samples = [(id2file[i], per_img[i]) for i in id2file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, anns = self.samples[idx]
        img = imageio.imread(self.img_dir / fname)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        H, W = img.shape[:2]
        boxes, labels = [], []
        for col, (x, y, w, h) in anns:
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)     # clamp to image (avoid albumentations reject)
            if w > 1 and h > 1:
                boxes.append([x, y, w, h])
                labels.append(col)
        try:
            out = self.aug(image=img, bboxes=boxes, labels=labels)
            img_a, boxes_a, labels_a = out["image"], out["bboxes"], out["labels"]
        except Exception:  # noqa: BLE001 -- an aug that drops all boxes: fall back to plain resize
            out = A.Compose([A.Resize(self.input_size, self.input_size)],
                            bbox_params=A.BboxParams(format="coco", label_fields=["labels"])
                            )(image=img, bboxes=boxes, labels=labels)
            img_a, boxes_a, labels_a = out["image"], out["bboxes"], out["labels"]
        x = torch.from_numpy(np.ascontiguousarray(img_a)).float().permute(2, 0, 1) / 255.0
        hm, wh, mask = build_targets(list(boxes_a), list(labels_a), self.input_size)
        return x, hm, wh, mask


# ----------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------
def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CenterNet). ``pred_logits`` are pre-sigmoid."""
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    neg_w = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    n_pos = pos.sum()
    pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()
    return -(neg_loss if n_pos == 0 else (pos_loss + neg_loss) / n_pos)


def wh_loss(pred_wh: torch.Tensor, gt_wh: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 on box size at GT centre cells only."""
    m = mask.unsqueeze(1)
    n = m.sum().clamp(min=1.0)
    return (torch.abs(pred_wh - gt_wh) * m).sum() / n


# ----------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------
def train(args) -> pathlib.Path:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(args.threads)
    ds = BalloonDataset(args.data_root, args.train_json, args.img_dir, args.input_size, train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=len(ds) > args.batch)
    model = TinyBalloonNet(width=args.width)
    if args.init_from is not None:
        # Fine-tune: warm-start from a pretrained checkpoint (e.g. sim-pretrain -> real-finetune).
        # Same architecture (width must match), so a strict load. Pair with a lower --lr.
        ckpt_in = torch.load(args.init_from, map_location="cpu")
        state = ckpt_in["state_dict"] if isinstance(ckpt_in, dict) and "state_dict" in ckpt_in \
            else ckpt_in
        model.load_state_dict(state)
        print(f"init-from: warm-started weights from {args.init_from}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"train: {len(ds)} images, {args.epochs} epochs, batch {args.batch}, input {args.input_size}, "
          f"width {args.width}, lr {args.lr}, threads {args.threads}")
    model.train()
    for ep in range(args.epochs):
        tot = hm_t = wh_t = 0.0
        nb = 0
        for x, hm, wh, mask in dl:
            phm, pwh = model(x)
            lhm = focal_loss(phm, hm)
            lwh = wh_loss(pwh, wh, mask) * args.wh_weight
            loss = lhm + lwh
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            hm_t += float(lhm.detach())
            wh_t += float(lwh.detach())
            nb += 1
        sched.step()
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}/{args.epochs}  loss {tot / nb:7.4f}  "
                  f"(hm {hm_t / nb:6.4f}  wh {wh_t / nb:6.4f})")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"width": args.width, "input_size": args.input_size,
                        "conf_thresh": args.conf}}, out)
    print(f"saved checkpoint -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=pathlib.Path, default=DATA_ROOT)
    ap.add_argument("--train-json", default="train.json")
    ap.add_argument("--img-dir", default="train2017")
    ap.add_argument("--val-split", default="val", help="split name for the post-train val report")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--input-size", type=int, default=256, help="256 & 320 both clear Pi4 10 fps")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--init-from", type=pathlib.Path, default=None,
                    help="warm-start weights from this checkpoint before training (fine-tune; "
                         "use a lower --lr). Width must match.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--wh-weight", type=float, default=0.1)
    ap.add_argument("--conf", type=float, default=0.3, help="stored eval confidence floor")
    ap.add_argument("--threads", type=int, default=2, help="torch threads (keep low: RL runs in bg)")
    ap.add_argument("--workers", type=int, default=0, help="dataloader workers (0 = no subprocess)")
    ap.add_argument("--no-eval", action="store_true", help="skip the val comparison vs classical")
    args = ap.parse_args()

    if not (args.data_root / "annotations" / args.train_json).exists():
        print(f"ERROR: {args.data_root / 'annotations' / args.train_json} not found")
        return 1

    ckpt = train(args)

    if not args.no_eval:
        print("\n" + "=" * 60 + "\nVAL COMPARISON: learned vs color vs hough\n" + "=" * 60)
        from umiusi_perception.eval import compare
        compare(str(ckpt), args.data_root, args.val_split,
                ["learned", "color", "hough"], input_size=args.input_size, conf=args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
