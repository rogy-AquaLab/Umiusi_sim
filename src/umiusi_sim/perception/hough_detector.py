"""Shape-based (Hough-circle) balloon detector — an EXPERIMENT complementing the colour detector.

Motivation
----------
Underwater, colour thresholding hits a physical wall (see ``balloon_detector``'s REAL profile
notes): red light attenuates within a metre or two so red balloons read as a dark, desaturated
maroon that the colour path barely recovers (~7% recall), and blue balloons are almost the same
cyan as the pool water, so the colour path can only separate them with a tight/high-S window that
throws away recall. Colour is running out of signal.

Shape is a second, independent cue: balloons are round, so their circular silhouette against the
water may survive where colour fails — a faint red balloon can still have a detectable circular
edge. This module runs ``cv2.HoughCircles`` to find those silhouettes, then assigns each circle a
colour by sampling the HSV *inside* it against the same REAL_THRESHOLDS windows the colour
detector uses, so the outputs are directly comparable / mergeable with ``detect_balloons``.

It is deliberately classical and Pi4-light: one grayscale conversion, one blur, one Hough call,
and a cheap per-circle HSV vote. No learning, no GPU. (The learned path B — a small detector CNN —
is the alternative if neither colour nor shape clears the bar; this quantifies how far classical
shape gets first.)

Pipeline
--------
    RGB ─▶ single channel (gray / value / …) ─▶ (optional CLAHE) ─▶ blur
        ─▶ cv2.HoughCircles ─▶ per circle: sample inner-disc HSV, vote a colour vs REAL_THRESHOLDS
        ─▶ Detection(colour, bbox=circle's square, centroid, area=πr², bearing, range, confidence)

Combining with colour (see ``detect_combined``)
    * RECOVER: add round-and-coloured circles the colour path missed (aimed at red).
    * CONFIRM: optionally keep only colour blobs that coincide with a circle (round AND coloured),
      trading recall for precision (fewer water/tile false positives).

Tuning (all exposed as named constants below, set from the dataset GT box sizes)
    On this dataset GT balloon *radii* run ~13 px (5th pct) to ~94 px (95th pct), median ~28 px,
    with a long tail of close-up balloons up to ~300 px. MIN_RADIUS/MAX_RADIUS cover the bulk
    (13–95 px, widened a little); the giant close-ups are intentionally out of range (Hough on a
    600 px disc is slow and the colour path already sees those big blobs). Balloons image slightly
    egg-shaped (median w/h ≈ 0.8), so HOUGH_GRADIENT_ALT with a forgiving ``param2`` is used — it
    scores partial/imperfect circles better than the classic accumulator.
"""

from __future__ import annotations

import cv2
import numpy as np

from umiusi_sim.perception.balloon_detector import (
    BALLOON_DIAMETER_M,
    COLOUR_POINTS,
    REAL_THRESHOLDS,
    Detection,
    _colour_mask,
    _pinhole,
    rgb_to_hsv,
)

# --- Hough configuration (named + tunable; derived from GT box sizes, see module docstring) ----
# Which single channel to run edge/gradient detection on. Underwater, red attenuates but a faint
# brightness edge usually survives, so plain luminance ("gray") is the most reliable silhouette
# cue; "value" (HSV V) is near-identical, "sat"/"red"/"blue" are offered for experiments.
HOUGH_CHANNEL = "gray"                 # "gray" | "value" | "sat" | "red" | "green" | "blue"
USE_CLAHE = True                       # local contrast boost — lifts faint (esp. red) edges cheaply
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
BLUR_KSIZE = 5                         # median blur kernel (odd); kills speckle without eating edges

# HOUGH_GRADIENT_ALT is more robust to noise/partial circles than the classic accumulator.
HOUGH_METHOD = "gradient_alt"          # "gradient_alt" | "gradient"
HOUGH_DP = 1.5                         # inverse accumulator resolution
MIN_RADIUS = 10                        # ~GT 5th-pct radius (13 px), a touch lower
MAX_RADIUS = 100                       # ~GT 95th-pct radius (94 px); giant close-ups left to colour
MIN_DIST = 24                          # balloons cluster in vertical columns, so allow them close
PARAM1 = 120                           # Canny high threshold (both methods)
# param2: GRADIENT_ALT -> circle "perfectness" 0..1 (lower = accept rougher circles); GRADIENT ->
# accumulator vote threshold (lower = more circles). Underwater edges are soft, so run forgiving.
PARAM2_ALT = 0.4
PARAM2_STD = 30.0

# --- colour assignment (reuse the colour detector's REAL windows) ------------------------------
# Fraction of a circle's inner disc whose HSV must fall inside a colour window to accept that
# colour. Below this the circle is treated as background (uncoloured) and dropped, so a bare Hough
# circle on a tile seam or lane line is not emitted as a phantom balloon.
COLOUR_MIN_FRAC = 0.18
INNER_DISC_FRAC = 0.72                 # sample the inner 72% of the radius (skip the dark rim/edge)

# --- combine tuning ---------------------------------------------------------------------------
# A Hough detection RECOVERS a balloon only if it does not already overlap a colour detection
# (same or any colour) by more than this IoU — otherwise the colour detection is kept.
RECOVER_DEDUP_IOU = 0.3
# For RECOVER we demand a stronger colour vote than for plain hough mode: a recovered balloon is
# added on top of the colour path, so we want it to be confidently coloured, not a marginal circle.
RECOVER_MIN_FRAC = 0.25


def _single_channel(rgb: np.ndarray, channel: str) -> np.ndarray:
    """Return a uint8 single-channel image for Hough, per ``channel``."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if channel == "gray":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if channel in ("value", "sat"):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv[..., 2] if channel == "value" else hsv[..., 1]
    if channel in ("red", "green", "blue"):
        return rgb[..., {"red": 0, "green": 1, "blue": 2}[channel]]
    raise ValueError(f"unknown HOUGH_CHANNEL {channel!r}")


def _preprocess(rgb: np.ndarray, channel: str) -> np.ndarray:
    """Channel select -> optional CLAHE -> median blur; the image Hough runs its gradient on."""
    img = _single_channel(rgb, channel)
    if USE_CLAHE:
        img = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID)).apply(img)
    if BLUR_KSIZE and BLUR_KSIZE >= 3:
        img = cv2.medianBlur(img, BLUR_KSIZE)
    return img


def _run_hough(gray: np.ndarray) -> np.ndarray:
    """Call cv2.HoughCircles with the configured method/params. Returns (N,3) [cx,cy,r] or empty."""
    if HOUGH_METHOD == "gradient_alt":
        method, param2 = cv2.HOUGH_GRADIENT_ALT, PARAM2_ALT
    else:
        method, param2 = cv2.HOUGH_GRADIENT, PARAM2_STD
    circles = cv2.HoughCircles(
        gray, method, dp=HOUGH_DP, minDist=MIN_DIST,
        param1=PARAM1, param2=param2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0, :, :3].astype(np.float32)


def _vote_colour(hsv: np.ndarray, cx: float, cy: float, r: float,
                 thresholds: dict) -> tuple[str | None, float]:
    """Vote a colour for one circle: fraction of its inner disc matching each REAL colour window;
    return (best_colour, fraction) or (None, best_fraction) if nothing clears COLOUR_MIN_FRAC."""
    H, W = hsv.shape[:2]
    rr = max(1.0, r * INNER_DISC_FRAC)
    x0, x1 = int(max(0, cx - rr)), int(min(W, cx + rr + 1))
    y0, y1 = int(max(0, cy - rr)), int(min(H, cy + rr + 1))
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    sub = hsv[y0:y1, x0:x1]
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disc = (xs - cx) ** 2 + (ys - cy) ** 2 <= rr * rr
    n = int(disc.sum())
    if n == 0:
        return None, 0.0
    best_col, best_frac = None, 0.0
    for colour, spec in thresholds.items():
        frac = float((_colour_mask(sub, spec) & disc).sum()) / n
        if frac > best_frac:
            best_col, best_frac = colour, frac
    if best_frac < COLOUR_MIN_FRAC:
        return None, best_frac
    return best_col, best_frac


def _circle_to_detection(cx: float, cy: float, r: float, colour: str, frac: float,
                         fx: float, cx0: float, cy0: float, H: int, W: int) -> Detection:
    """Build a Detection from a coloured circle, matching detect_balloons' geometry/fields."""
    u0, v0 = int(round(cx - r)), int(round(cy - r))
    u1, v1 = int(round(cx + r)), int(round(cy + r))
    u0, v0 = max(0, u0), max(0, v0)
    u1, v1 = min(W, u1), min(H, v1)
    az = float(np.arctan2(cx - cx0, fx))
    el = float(np.arctan2(cy0 - cy, fx))
    apparent_d = 2.0 * r
    range_m = float(BALLOON_DIAMETER_M * fx / apparent_d) if apparent_d > 0 else float("inf")
    return Detection(
        colour=colour,
        points=COLOUR_POINTS[colour],
        bbox=(u0, v0, u1, v1),
        centroid=(float(cx), float(cy)),
        area_px=int(np.pi * r * r),
        bearing=(az, el),
        range_m=range_m,
        confidence=float(np.clip(frac / (np.pi / 4.0), 0.0, 1.0)),
        mean_s=0.0,
        mean_v=0.0,
    )


def detect_hough(rgb: np.ndarray, fovy_deg: float = 60.0,
                 thresholds: dict = REAL_THRESHOLDS,
                 min_colour_frac: float = COLOUR_MIN_FRAC) -> list[Detection]:
    """Detect balloons by CIRCULAR SILHOUETTE, then assign a colour by inner-disc HSV vote.

    Returns ``Detection`` objects compatible with ``detect_balloons`` (same fields/geometry), so
    they can be evaluated or merged interchangeably. Circles whose inner disc does not match any
    REAL colour window (fraction < ``min_colour_frac``) are dropped as background.
    """
    global COLOUR_MIN_FRAC  # allow the eval to override the accept bar without editing the module
    saved = COLOUR_MIN_FRAC
    COLOUR_MIN_FRAC = min_colour_frac
    try:
        rgb = np.asarray(rgb)
        H, W = rgb.shape[:2]
        hsv = rgb_to_hsv(rgb)
        fx, _fy, cx0, cy0 = _pinhole(H, W, fovy_deg)
        gray = _preprocess(rgb, HOUGH_CHANNEL)
        circles = _run_hough(gray)
        dets: list[Detection] = []
        for cx, cy, r in circles:
            colour, frac = _vote_colour(hsv, cx, cy, r, thresholds)
            if colour is None:
                continue
            dets.append(_circle_to_detection(cx, cy, r, colour, frac, fx, cx0, cy0, H, W))
    finally:
        COLOUR_MIN_FRAC = saved
    dets.sort(key=lambda d: d.area_px, reverse=True)
    return dets


def _iou(a: Detection, b: Detection) -> float:
    au0, av0, au1, av1 = a.bbox
    bu0, bv0, bu1, bv1 = b.bbox
    ix0, iy0 = max(au0, bu0), max(av0, bv0)
    ix1, iy1 = min(au1, bu1), min(av1, bv1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (au1 - au0) * (av1 - av0) + (bu1 - bu0) * (bv1 - bv0) - inter
    return inter / ua if ua > 0 else 0.0


def combine(colour_dets: list[Detection], hough_dets: list[Detection],
            mode: str = "recover", dedup_iou: float = RECOVER_DEDUP_IOU) -> list[Detection]:
    """Merge colour and Hough detections.

    ``mode="recover"`` (default): keep ALL colour detections, then ADD Hough detections that do not
    already overlap a colour detection (any colour, IoU > ``dedup_iou``). This lets shape recover
    balloons colour missed (aimed at attenuated red) without discarding colour's wins.

    ``mode="confirm"``: keep only colour detections that COINCIDE with a Hough circle of the same
    colour (round AND coloured), i.e. require both cues — trades recall for precision (fewer
    water/tile/lane-line false positives).
    """
    if mode == "confirm":
        out = []
        for c in colour_dets:
            if any(h.colour == c.colour and _iou(c, h) > dedup_iou for h in hough_dets):
                out.append(c)
        out.sort(key=lambda d: d.area_px, reverse=True)
        return out
    # recover
    out = list(colour_dets)
    for h in hough_dets:
        if not any(_iou(h, c) > dedup_iou for c in colour_dets):
            out.append(h)
    out.sort(key=lambda d: d.area_px, reverse=True)
    return out


def detect_combined(rgb: np.ndarray, fovy_deg: float = 60.0,
                    thresholds: dict = REAL_THRESHOLDS, reject_reflections: bool = True,
                    max_area_frac: float | None = 0.02, mode: str = "recover") -> list[Detection]:
    """Convenience: run the colour detector and the Hough detector and ``combine`` them.

    Imported lazily to avoid a hard import cycle at module load."""
    from umiusi_sim.perception.balloon_detector import detect_balloons
    colour_dets = detect_balloons(rgb, fovy_deg=fovy_deg, thresholds=thresholds,
                                  reject_reflections=reject_reflections, max_area_frac=max_area_frac)
    recover_frac = RECOVER_MIN_FRAC if mode == "recover" else COLOUR_MIN_FRAC
    hough_dets = detect_hough(rgb, fovy_deg=fovy_deg, thresholds=thresholds,
                              min_colour_frac=recover_frac)
    return combine(colour_dets, hough_dets, mode=mode)
