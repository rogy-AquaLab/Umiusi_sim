"""Learned balloon detector ("path B") — a LATENCY-FIRST tiny CNN for the Raspberry Pi 4.

Why this exists
---------------
The classical detectors hit a physical recall wall underwater (see ``balloon_detector``'s REAL
notes and ``hough_detector``): red attenuates to a dark maroon (~7% recall) and blue is almost the
same cyan as the pool water. A small learned model can, in principle, use texture/shading/shape
cues colour thresholds cannot — but ONLY if it still runs on the robot's Pi 4. The hard constraint
is therefore latency: it must clear >=5-10 fps on a Pi 4, so the architecture here is deliberately
tiny (a CenterNet-lite head, ~0.1-0.5 GFLOPs), NOT a full-size detector at 640px.

Architecture (``TinyBalloonNet``) — a CenterNet-lite, fully-convolutional, stride-8 detector:

    RGB (3xHxW, /255) ─▶ conv stem (downsample to H/8) ─▶ two heads at stride 8:
        * heatmap head : 3 channels (red/yellow/blue centre-ness, sigmoid)  ─▶ where balloons are
        * size head    : 2 channels (w, h, normalised to input)             ─▶ how big they are

Decoding: sigmoid the heatmap, keep local maxima (3x3 maxpool) above a confidence floor, read the
box size from the size head at each peak, and emit one ``Detection`` per peak — the SAME dataclass
the classical detectors return, so it drops straight into ``tools/perception_eval``.

``PatchVerifierNet`` is the tiny 32x32 patch classifier for the Hough-proposal + CNN-verifier
HYBRID candidate benchmarked in ``tools/perception_bench`` (Hough finds circles classically; this
net classifies each patch red/yellow/blue/background). It is defined here so the benchmark and any
future training share one definition; the RECOMMENDED shipped path is ``TinyBalloonNet``.

Nothing in this module changes the classical detectors; it only imports their geometry helpers
(``_pinhole``, ``COLOUR_POINTS``, ``BALLOON_DIAMETER_M``) so ranges/bearings match exactly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from umiusi_perception.balloon_detector import (
    BALLOON_DIAMETER_M,
    COLOUR_POINTS,
    Detection,
    _pinhole,
    rgb_to_hsv,
)

# Channel order for the 3-colour heatmap. This is an INTERNAL ordering, NOT the COCO id order
# (COCO is red=1/blue=2/yellow=3). It is safe because targets and decoding both map by NAME
# (COLOUR_TO_IDX[name] / COLOURS[channel]) — never by (category_id - 1). Keep it identical to the
# eval harness' COLOURS list; do NOT align channels to category_id (that would swap blue/yellow).
COLOURS = ["red", "yellow", "blue"]
COLOUR_TO_IDX = {c: i for i, c in enumerate(COLOURS)}
STRIDE = 8  # network output stride (input H,W -> heatmap H/8, W/8)


def _conv_bn(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU block (padding keeps spatial size except where stride downsamples)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyBalloonNet(nn.Module):
    """CenterNet-lite balloon detector. ~0.1-0.5 GFLOPs depending on input size / ``width``.

    Output: ``hm`` (N,3,H/8,W/8) heatmap logits and ``wh`` (N,2,H/8,W/8) box sizes normalised to
    the input dimensions. Kept fully-convolutional so any square input size works (160/256/320).
    """

    def __init__(self, width: int = 16, n_colours: int = 3):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(
            _conv_bn(3, w, k=3, s=2),        # H/2
            _conv_bn(w, 2 * w, k=3, s=2),    # H/4
            _conv_bn(2 * w, 2 * w, k=3, s=1),
            _conv_bn(2 * w, 4 * w, k=3, s=2),  # H/8
            _conv_bn(4 * w, 4 * w, k=3, s=1),
        )
        self.hm_head = nn.Sequential(
            _conv_bn(4 * w, 2 * w, k=3, s=1),
            nn.Conv2d(2 * w, n_colours, 1),
        )
        self.wh_head = nn.Sequential(
            _conv_bn(4 * w, 2 * w, k=3, s=1),
            nn.Conv2d(2 * w, 2, 1),
        )
        # Bias the heatmap toward "no object" so early training is stable (CenterNet trick).
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.hm_head(feat), self.wh_head(feat)


class PatchVerifierNet(nn.Module):
    """Tiny 3x32x32 -> 4-class (bg/red/yellow/blue) classifier for the Hough+verifier hybrid."""

    def __init__(self, width: int = 16, n_classes: int = 4):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _conv_bn(3, w, k=3, s=2),        # 16
            _conv_bn(w, 2 * w, k=3, s=2),    # 8
            _conv_bn(2 * w, 2 * w, k=3, s=2),  # 4
        )
        self.head = nn.Linear(2 * w, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(f)


# ----------------------------------------------------------------------------------------------
# Inference / decoding: TinyBalloonNet output -> list[Detection] (the classical-detector interface)
# ----------------------------------------------------------------------------------------------
def preprocess(rgb: np.ndarray, input_size: int) -> torch.Tensor:
    """(H,W,3) uint8 RGB -> (1,3,input_size,input_size) float tensor in [0,1] (bilinear resize)."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return F.interpolate(t, size=(input_size, input_size), mode="bilinear", align_corners=False)


def _nms_peaks(hm: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local maxima of the heatmap (3x3 maxpool NMS); returns a peak-masked heatmap."""
    pad = kernel // 2
    mx = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    return hm * (mx == hm).float()


def decode(hm_logits: torch.Tensor, wh: torch.Tensor, orig_h: int, orig_w: int, input_size: int,
           conf_thresh: float = 0.3, topk: int = 40, fovy_deg: float = 60.0) -> list[Detection]:
    """Decode one image's (hm, wh) tensors into Detections in ORIGINAL image pixel coordinates."""
    hm = torch.sigmoid(hm_logits)
    hm = _nms_peaks(hm)
    n_col, fh, fw = hm.shape
    scale_x = orig_w / input_size          # network-input px -> original px
    scale_y = orig_h / input_size
    fx, _fy, cx0, cy0 = _pinhole(orig_h, orig_w, fovy_deg)

    dets: list[Detection] = []
    flat = hm.reshape(n_col, -1)
    k = min(topk, flat.shape[1])
    for ci in range(n_col):
        scores, idx = torch.topk(flat[ci], k)
        colour = COLOURS[ci]
        for s, i in zip(scores.tolist(), idx.tolist()):
            if s < conf_thresh:
                break
            gy, gx = divmod(i, fw)
            # centre in network-input pixels (cell centre at stride 8), then to original px
            cx_in = (gx + 0.5) * STRIDE
            cy_in = (gy + 0.5) * STRIDE
            w_in = float(wh[0, gy, gx].clamp(min=1e-3)) * input_size
            h_in = float(wh[1, gy, gx].clamp(min=1e-3)) * input_size
            cx = cx_in * scale_x
            cy = cy_in * scale_y
            bw = w_in * scale_x
            bh = h_in * scale_y
            u0 = int(round(max(0, cx - bw / 2)))
            v0 = int(round(max(0, cy - bh / 2)))
            u1 = int(round(min(orig_w, cx + bw / 2)))
            v1 = int(round(min(orig_h, cy + bh / 2)))
            if u1 <= u0 or v1 <= v0:
                continue
            az = float(np.arctan2(cx - cx0, fx))
            el = float(np.arctan2(cy0 - cy, fx))
            apparent_d = 0.5 * (bw + bh)
            range_m = float(BALLOON_DIAMETER_M * fx / apparent_d) if apparent_d > 0 else float("inf")
            dets.append(Detection(
                colour=colour,
                points=COLOUR_POINTS[colour],
                bbox=(u0, v0, u1, v1),
                centroid=(float(cx), float(cy)),
                area_px=int(bw * bh),
                bearing=(az, el),
                range_m=range_m,
                confidence=float(s),
            ))
    dets.sort(key=lambda d: d.area_px, reverse=True)
    return dets


@torch.no_grad()
def detect_learned(rgb: np.ndarray, model: TinyBalloonNet, input_size: int = 256,
                   conf_thresh: float = 0.3, fovy_deg: float = 60.0) -> list[Detection]:
    """Run ``TinyBalloonNet`` on one RGB frame and return classical-compatible ``Detection``s."""
    model.eval()
    H, W = np.asarray(rgb).shape[:2]
    x = preprocess(rgb, input_size)
    hm, wh = model(x)
    return decode(hm[0], wh[0], H, W, input_size, conf_thresh=conf_thresh, fovy_deg=fovy_deg)


def load_learned_detector(weights_path: str, input_size: int | None = None, width: int = 16,
                          conf_thresh: float | None = None, fovy_deg: float = 60.0):
    """Load trained weights and return a ``rgb -> [Detection]`` callable for ``perception_eval``.

    ``input_size`` and ``conf_thresh`` are caller-overridable: an explicit arg WINS, else the
    checkpoint's stored value, else the default. ``width`` comes from the checkpoint when present
    (it must match the saved weights) and the ``width=`` arg is only the fallback for bare state_dicts.
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("cfg", {})
        width = cfg.get("width", width)  # weights fix the architecture; checkpoint wins
        input_size = input_size or cfg.get("input_size", 256)
        if conf_thresh is None:  # caller wins; else the checkpoint's stored floor
            conf_thresh = cfg.get("conf_thresh", 0.3)
        state = ckpt["state_dict"]
    else:  # bare state_dict
        state = ckpt
        input_size = input_size or 256
    if conf_thresh is None:
        conf_thresh = 0.3
    model = TinyBalloonNet(width=width)
    model.load_state_dict(state)
    model.eval()

    def _fn(rgb: np.ndarray) -> list[Detection]:
        return detect_learned(rgb, model, input_size=input_size, conf_thresh=conf_thresh,
                              fovy_deg=fovy_deg)

    return _fn


# rgb_to_hsv is re-exported for callers that want the same HSV as the classical path (unused here
# directly, but keeps the learned module a one-stop import for perception tooling).
__all__ = [
    "COLOURS", "STRIDE", "TinyBalloonNet", "PatchVerifierNet",
    "preprocess", "decode", "detect_learned", "load_learned_detector", "rgb_to_hsv",
]
