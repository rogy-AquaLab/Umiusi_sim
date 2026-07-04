"""Classical-CV balloon detector for the onboard front camera (phase 5b).

Pipeline (deliberately lightweight and replaceable):

    RGB frame ─▶ HSV ─▶ per-colour threshold masks ─▶ connected components
              ─▶ filter tiny blobs ─▶ one ``Detection`` per surviving blob

Each Detection carries the balloon COLOUR ("red"/"yellow"/"blue"), its scoring ``points``,
the image ``bbox``/``centroid``/``area_px``, a ``bearing`` (azimuth + elevation, radians) from
the pixel offset vs. the camera optic axis, an estimated ``range_m``, and a ``confidence``.

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

Colour thresholds are HSV constants tuned to the sim's clean, saturated balloon renders. They
would need widening / white-balancing / illumination-robustifying for real underwater images
(colour cast, turbidity, specular highlights, lower saturation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# --- physical / camera constants ---------------------------------------------
# Balloon spheres are BALLOON_RADIUS = 0.10 m (see competition_balloon.py) -> 0.20 m diameter.
BALLOON_DIAMETER_M = 0.20

# --- colour thresholds (HSV; H in [0,360), S and V in [0,1]) ------------------
# Tuned to the sim renders (measured balloon centres: red H~0, yellow H~52, blue H~227,
# all S~0.7-0.8, V~0.5-0.6). Each entry is (hue_ranges, s_min, v_min); hue wraps at 360.
# NOTE: widen these (and add white-balance / saturation floors per site) for real imagery.
COLOUR_THRESHOLDS = {
    # red wraps around 0deg, so two hue windows.
    "red": {"hue": [(0.0, 20.0), (340.0, 360.0)], "s_min": 0.35, "v_min": 0.20},
    "yellow": {"hue": [(38.0, 72.0)], "s_min": 0.35, "v_min": 0.20},
    # start blue window ABOVE the pool-water cyan (~190-210 deg) to avoid scenery false positives.
    "blue": {"hue": [(205.0, 260.0)], "s_min": 0.40, "v_min": 0.18},
}

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


def detect_balloons(rgb: np.ndarray, fovy_deg: float = 60.0,
                    min_area_px: int = MIN_AREA_PX) -> list[Detection]:
    """Detect balloons in an (H, W, 3) uint8 RGB frame; return a list of ``Detection``.

    ``fovy_deg`` is the camera's vertical field of view (MuJoCo front_cam = 60 deg). Detections
    are returned sorted by descending pixel area (nearest/largest first).
    """
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    hsv = rgb_to_hsv(rgb)
    fx, fy, cx, cy = _pinhole(H, W, fovy_deg)

    detections: list[Detection] = []
    for colour, spec in COLOUR_THRESHOLDS.items():
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
            if area < min_area_px:
                continue
            vsl, usl = slices[i]
            u0, u1 = usl.start, usl.stop
            v0, v1 = vsl.start, vsl.stop
            w_px, h_px = (u1 - u0), (v1 - v0)
            bbox_area = max(1, w_px * h_px)
            fill = area / bbox_area
            if fill < MIN_FILL_RATIO:
                continue  # sliver / non-blob (e.g. thin colour fringe), not a balloon
            cv, cu = centroids[i]  # center_of_mass returns (row=v, col=u)

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
            ))

    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections
