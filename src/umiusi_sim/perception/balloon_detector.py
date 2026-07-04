"""Classical-CV balloon detector for the onboard front camera (phase 5b).

Pipeline (deliberately lightweight and replaceable):

    RGB frame ─▶ HSV ─▶ per-colour threshold masks ─▶ connected components
              ─▶ filter tiny/oversize/sliver blobs ─▶ one ``Detection`` per surviving blob
              ─▶ (optional) reject water-surface reflections

Each Detection carries the balloon COLOUR ("red"/"yellow"/"blue"), its scoring ``points``,
the image ``bbox``/``centroid``/``area_px``, a ``bearing`` (azimuth + elevation, radians) from
the pixel offset vs. the camera optic axis, an estimated ``range_m``, a ``confidence``, and the
blob's mean saturation/value (used by the reflection filter).

Geometry — pinhole from the MuJoCo camera (square pixels, vertical field-of-view ``fovy``):

    fy = (H / 2) / tan(fovy / 2)          # focal length in pixels (vertical)
    fx = fy                               # square pixels
    cx, cy = W / 2, H / 2                 # image centre (optic axis)
    bearing_az = atan2(u - cx, fx)        # +right of the optic axis
    bearing_el = atan2(cy - v, fy)        # +above the optic axis (image v grows downward)
    range_m   = BALLOON_DIAMETER_M * fx / apparent_pixel_diameter

The MuJoCo camera looks down its local -Z with +Y as image-up; here we work purely in image
space, so those conventions only matter for the world->pixel projection used to VALIDATE the
detector against ground truth (see ``tools/perception_demo.py``), not for detection itself.

The apparent pixel diameter is the mean of the blob's bbox width and height. This assumes a
roughly circular (unoccluded, isolated) balloon silhouette; occlusion or two overlapping
same-colour balloons merge into one blob and bias the range/centroid (documented limitation).

Colour PROFILES
---------------
Colour thresholds are passed in as a profile so the detector serves two very different image
domains without one regressing the other:

* ``SIM_THRESHOLDS`` (default) — tuned to the sim's clean, saturated balloon renders. Do NOT
  change these: ``tools/perception_demo`` is the regression guard for them.
* ``REAL_THRESHOLDS`` — calibrated data-drivenly from a labelled real underwater dataset (see
  the block above the definition for exactly how the windows were derived). Underwater the
  colours shift hard: yellow balloons read *green* (H~90-160), blue reads cyan and overlaps the
  pool water almost completely, and red is heavily attenuated to a dark, desaturated maroon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# --- physical / camera constants ---------------------------------------------
# Balloon spheres are BALLOON_RADIUS = 0.10 m (see competition_balloon.py) -> 0.20 m diameter.
BALLOON_DIAMETER_M = 0.20

# --- colour thresholds (HSV; H in [0,360), S and V in [0,1]) ------------------
# SIM profile: tuned to the sim renders (measured balloon centres: red H~0, yellow H~52,
# blue H~227, all S~0.7-0.8, V~0.5-0.6). Each entry is (hue_ranges, s_min, v_min); hue wraps
# at 360. This is the DEFAULT and must stay unchanged (perception_demo is its regression guard).
SIM_THRESHOLDS = {
    # red wraps around 0deg, so two hue windows.
    "red": {"hue": [(0.0, 20.0), (340.0, 360.0)], "s_min": 0.35, "v_min": 0.20},
    "yellow": {"hue": [(38.0, 72.0)], "s_min": 0.35, "v_min": 0.20},
    # start blue window ABOVE the pool-water cyan (~190-210 deg) to avoid scenery false positives.
    "blue": {"hue": [(205.0, 260.0)], "s_min": 0.40, "v_min": 0.18},
}

# REAL profile — DATA-DRIVEN calibration. Method: for every GT box in the train split we sampled
# the HSV of the box's central 50% (excludes rim/tether/edges), dropped specular highlights
# (V>0.9 & S<0.15) and near-black rim (V<0.06), and read robust percentiles of H/S/V per colour.
# We also sampled pool water/background (pixels >8px outside every box) to set the blue floors
# high enough to reject most water. Derived windows (train, ~1.1M px; see the report):
#
#   colour  |  balloon H (5-95%)   S (med)  V (med) | how it reads underwater
#   --------+----------------------+-----------------+-------------------------
#   red     |  344-358 & 0-15      |  0.30   |  0.70 | dark, DESATURATED maroon (red attenuates)
#   yellow  |  97-160 (green!)     |  0.50   |  1.00 | bright yellow-GREEN
#   blue    |  188-220 (cyan)      |  0.90   |  0.90 | cyan — overlaps pool water almost entirely
#
# Water/background: H 163-213 (median 187), S up to 1.0 at the 95th pct. => hue cannot separate
# blue balloons from water, and even S only partly can. The final windows below were then swept on
# the train split against GT IoU to trade FP against recall (see tools/perception_eval and the
# report). The blue window is deliberately tight + HIGH s_min/v_min: that alone cut the blue
# false-positive flood from ~740 to ~120 on train (recall ~0.17) — colour cannot do better against
# same-colour water, so blob shape/size caps (``max_area_frac``, fill ratio) mop up the rest.
REAL_THRESHOLDS = {
    # attenuated maroon: hue is still red, but saturation collapses to ~0.3, so a LOW s_min. Red
    # underwater is close to invisible; this window is permissive and still only recovers ~7%.
    "red": {"hue": [(0.0, 15.0), (335.0, 360.0)], "s_min": 0.15, "v_min": 0.15},
    # yellow reads GREEN underwater; window sits in the greens, clear of the cyan water at 180+.
    "yellow": {"hue": [(90.0, 160.0)], "s_min": 0.35, "v_min": 0.55},
    # cyan, sitting right on top of the water distribution -> a TIGHT hue + high S/V does the
    # separating (loses recall, but this is what cuts the water flood).
    "blue": {"hue": [(188.0, 212.0)], "s_min": 0.90, "v_min": 0.65},
}

# Back-compat alias (was the module-level name before profiles were introduced).
COLOUR_THRESHOLDS = SIM_THRESHOLDS

# Colour -> scoring points (competition rule; mirrors BALLOON_SPECS in competition_balloon.py).
COLOUR_POINTS = {"red": 30, "yellow": 10, "blue": -10}

# --- blob filtering -----------------------------------------------------------
MIN_AREA_PX = 40          # drop specks / anti-aliasing fringe; ~7px-diameter blob
MIN_FILL_RATIO = 0.35     # blob_area / bbox_area; a solid disc is ~pi/4 = 0.785, reject slivers


@dataclass
class Detection:
    """One detected balloon in a single frame."""

    colour: str                       # "red" | "yellow" | "blue"
    points: int                       # scoring value of that colour
    bbox: tuple[int, int, int, int]   # (u0, v0, u1, v1) inclusive-exclusive pixel box
    centroid: tuple[float, float]     # (u, v) blob centroid in pixels
    area_px: int                      # number of masked pixels in the blob
    bearing: tuple[float, float]      # (azimuth, elevation) in RADIANS, vs. optic axis
    range_m: float                    # estimated distance to the balloon centre [m]
    confidence: float                 # heuristic 0..1 (blob circularity/fill)
    mean_s: float = 0.0               # mean HSV saturation of the blob (for reflection filter)
    mean_v: float = 0.0               # mean HSV value/brightness of the blob (for reflection filter)
    is_reflection: bool = False       # flagged True by the reflection filter (kept for inspection)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB(uint8/float) -> HSV. Returns float array, H in [0,360), S,V in [0,1].

    Pure-numpy (no OpenCV dependency) so the detector needs only scipy for labelling.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    safe = np.where(d == 0, 1.0, d)  # avoid div-by-zero; hue is 0 where d==0 anyway

    h = np.zeros_like(mx)
    r_is_max = (mx == r) & (d > 0)
    g_is_max = (mx == g) & (d > 0) & ~r_is_max
    b_is_max = (mx == b) & (d > 0) & ~r_is_max & ~g_is_max
    h = np.where(r_is_max, ((g - b) / safe) % 6.0, h)
    h = np.where(g_is_max, (b - r) / safe + 2.0, h)
    h = np.where(b_is_max, (r - g) / safe + 4.0, h)
    h = (h * 60.0) % 360.0

    s = np.where(mx == 0, 0.0, d / np.where(mx == 0, 1.0, mx))
    v = mx
    return np.stack([h, s, v], axis=-1)


def _colour_mask(hsv: np.ndarray, spec: dict) -> np.ndarray:
    """Boolean mask of pixels matching one colour spec (hue windows + s/v floors)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, dtype=bool)
    for lo, hi in spec["hue"]:
        hue_ok |= (h >= lo) & (h <= hi)
    return hue_ok & (s >= spec["s_min"]) & (v >= spec["v_min"])


def _pinhole(height: int, width: int, fovy_deg: float):
    """Return (fx, fy, cx, cy) for a square-pixel pinhole with vertical FOV ``fovy_deg``."""
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    fx = fy
    return fx, fy, width / 2.0, height / 2.0


def _reject_reflections(detections: list[Detection], height: int) -> list[Detection]:
    """Flag/drop water-surface reflections by GEOMETRY (colour can't — a reflection is the same
    colour as its balloon).

    Rule (per colour, since a reflection mirrors its own balloon):
      For a pair of same-colour blobs that are strongly horizontally aligned (u-centroids within
      0.4x the wider blob's width) and vertically separated, the LOWER blob is dropped as a
      reflection only if ALL of: (i) it sits in the bottom ~55% of the frame (where the water
      surface / floor reflections live), (ii) it is dimmer by a margin (lower mean V), AND (iii) it
      is smaller than its source. Requiring dimmer-AND-smaller-AND-low keeps it conservative so it
      rarely deletes a genuine balloon.

    Failure modes (documented, accepted): these balloons are tethered to the floor and float in
    near-vertical COLUMNS, so a real lower balloon that happens to be dimmer+smaller than the one
    above it gets suppressed (measured: ~a fifth of true yellow blobs on this set) — the single
    biggest cost of the filter. Conversely a reflection brighter/larger than its source (specular),
    or one whose source is out of frame, survives. Colour genuinely cannot help (a reflection is
    the same colour), so this geometric rule is the only lever; it is off by default (the sim has
    no free water surface)."""
    H = height
    keep = [True] * len(detections)
    for i, a in enumerate(detections):
        for j, b in enumerate(detections):
            if i == j or a.colour != b.colour:
                continue
            # order a above b
            if a.centroid[1] >= b.centroid[1]:
                continue
            # candidate reflection must sit in the lower band (near/below the water line)
            if b.centroid[1] < 0.45 * H:
                continue
            wa = a.bbox[2] - a.bbox[0]
            wb = b.bbox[2] - b.bbox[0]
            # strong horizontal alignment: centroids within 0.4x the wider blob's width
            if abs(a.centroid[0] - b.centroid[0]) > 0.4 * max(wa, wb):
                continue
            # vertical separation (avoid splitting one blob's own noise)
            if (b.centroid[1] - a.centroid[1]) < 0.6 * (a.bbox[3] - a.bbox[1]):
                continue
            # conservative: dimmer AND smaller (a fainter, broken-up mirror image)
            if (b.mean_v < a.mean_v - 0.04) and (b.area_px < 0.8 * a.area_px):
                keep[j] = False
                detections[j].is_reflection = True
    return [d for d, k in zip(detections, keep) if k]


def detect_balloons(rgb: np.ndarray, fovy_deg: float = 60.0,
                    thresholds: dict = SIM_THRESHOLDS, reject_reflections: bool = False,
                    min_area_px: int = MIN_AREA_PX, max_area_frac: float | None = None) -> list[Detection]:
    """Detect balloons in an (H, W, 3) uint8 RGB frame; return a list of ``Detection``.

    ``fovy_deg`` is the camera's vertical field of view (MuJoCo front_cam = 60 deg).
    ``thresholds`` is a colour PROFILE (``SIM_THRESHOLDS`` default, or ``REAL_THRESHOLDS`` for real
    underwater imagery). ``reject_reflections`` enables the water-surface reflection post-filter
    (leave off for the sim). ``max_area_frac`` optionally drops blobs larger than that fraction of
    the frame area (large ragged water regions in real images); ``None`` disables the cap.

    Detections are returned sorted by descending pixel area (nearest/largest first).
    """
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    hsv = rgb_to_hsv(rgb)
    fx, fy, cx, cy = _pinhole(H, W, fovy_deg)
    max_area_px = max_area_frac * H * W if max_area_frac is not None else float("inf")

    detections: list[Detection] = []
    for colour, spec in thresholds.items():
        mask = _colour_mask(hsv, spec)
        labels, n = ndimage.label(mask)
        if n == 0:
            continue
        # Per-label pixel counts and bounding slices in one pass.
        areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        slices = ndimage.find_objects(labels)
        centroids = ndimage.center_of_mass(mask, labels, index=np.arange(1, n + 1))
        for i in range(n):
            area = int(areas[i])
            if area < min_area_px or area > max_area_px:
                continue
            vsl, usl = slices[i]
            u0, u1 = usl.start, usl.stop
            v0, v1 = vsl.start, vsl.stop
            w_px, h_px = (u1 - u0), (v1 - v0)
            bbox_area = max(1, w_px * h_px)
            fill = area / bbox_area
            if fill < MIN_FILL_RATIO:
                continue  # sliver / non-blob (e.g. thin colour fringe or lane line), not a balloon
            cv, cu = centroids[i]  # center_of_mass returns (row=v, col=u)

            # Mean saturation/brightness over the blob (for the reflection filter).
            blob = (labels[vsl, usl] == (i + 1))
            sub = hsv[vsl, usl]
            mean_s = float(sub[..., 1][blob].mean())
            mean_v = float(sub[..., 2][blob].mean())

            # Bearing from the centroid offset vs. the optic axis.
            az = float(np.arctan2(cu - cx, fx))
            el = float(np.arctan2(cy - cv, fy))

            # Range from apparent size: mean of bbox width/height as the pixel diameter.
            apparent_d = 0.5 * (w_px + h_px)
            range_m = float(BALLOON_DIAMETER_M * fx / apparent_d) if apparent_d > 0 else float("inf")

            # Confidence: how disc-like the blob is (ideal solid circle fill ~= pi/4).
            confidence = float(np.clip(fill / (np.pi / 4.0), 0.0, 1.0))

            detections.append(Detection(
                colour=colour,
                points=COLOUR_POINTS[colour],
                bbox=(int(u0), int(v0), int(u1), int(v1)),
                centroid=(float(cu), float(cv)),
                area_px=area,
                bearing=(az, el),
                range_m=range_m,
                confidence=confidence,
                mean_s=mean_s,
                mean_v=mean_v,
            ))

    if reject_reflections:
        detections = _reject_reflections(detections, H)

    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections
