"""Physically-grounded underwater image-formation degradation (SYNTHETIC data generator).

This is the *forward* model — the opposite of ``underwater.py`` (which *restores* colour). Given
a clean rendered RGB frame and its metric depth buffer, it produces a degraded frame that looks
like real murky underwater footage, so we can auto-generate free labelled training data (the GT
boxes come from the segmentation buffer — pixels don't move, so labels transfer through degrade()).

Model (Jaffe-McGlamery / Sea-thru, simplified, per-pixel using depth):

    I_c = J_c * t_c + B_c * (1 - t_c),   t_c = exp(-beta_c * z)

  * ``J_c``   clean (in-air) radiance of the scene (the render).
  * ``z``     per-pixel distance from the camera [m] (the depth buffer).
  * ``t_c``   per-channel transmission. Red water-absorption coefficient is the LARGEST
              (beta_red > beta_green > beta_blue), so distant red darkens toward the water colour
              first — exactly why red balloons (+30) read as dark/blue in real footage.
  * ``B_c``   veiling / backscatter colour (blue-green): what a distant pixel (t->0) fades to.

On top of that base image, optional, physically-motivated nuisances (each toggled + scaled by
``params``): turbidity blur (depth-scaled), backscatter shot noise, caustics (low-freq light
ripple), exposure/gain + white-balance jitter, and a water-surface reflection distractor near the
top of the frame (a known false-detection source for red/orange balloons).

Cheap: numpy + cv2 (optional — a numpy gaussian fallback is used if cv2 is missing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # cv2 is in the `perception` extra; degrade() works without it (numpy fallback for blur).
    import cv2
except Exception:  # pragma: no cover - exercised only on installs without opencv
    cv2 = None


# --- parameters --------------------------------------------------------------
@dataclass
class WaterParams:
    """One water/imaging condition. Defaults = a moderately murky pool (mid difficulty).

    beta_*  : per-channel attenuation [1/m]. beta_red > beta_green > beta_blue (red dies first).
              Bigger => murkier / shorter visibility. Clear water ~ (0.25, 0.06, 0.03);
              murky pool ~ (1.2, 0.5, 0.35).
    B       : backscatter/veiling colour in [0,1] RGB — the blue-green a far pixel fades to.
    turbidity      : depth-scaled gaussian blur strength (0 = off). Distant pixels blur more.
    backscatter_noise : std (in 0..255 units) of veiling shot noise, scaled by (1-t).
    caustics       : amplitude in [0,1] of the low-freq sinusoidal light ripple (0 = off).
    reflection     : strength in [0,1] of the mirrored water-surface distractor (0 = off).
    exposure       : global multiplicative gain on the final image.
    wb_gain        : per-channel white-balance multiplier (camera colour cast).
    """

    beta: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.40, 0.28]))
    B: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.30, 0.34]))
    turbidity: float = 0.6
    backscatter_noise: float = 6.0
    caustics: float = 0.10
    caustics_freq: float = 4.0
    reflection: float = 0.25
    exposure: float = 1.0
    wb_gain: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))

    def __post_init__(self):
        self.beta = np.asarray(self.beta, dtype=np.float64).reshape(3)
        self.B = np.asarray(self.B, dtype=np.float64).reshape(3)
        self.wb_gain = np.asarray(self.wb_gain, dtype=np.float64).reshape(3)


# Domain-randomization ranges (documented; sampled uniformly unless noted). Chosen so the sampled
# conditions span "clear-ish pool" -> "very murky", bracketing real competition-pool footage.
DR_RANGES = {
    # attenuation [1/m], sampled per channel within these bands (order enforces red>green>blue)
    "beta_red": (0.45, 1.40),
    "beta_green": (0.18, 0.60),
    "beta_blue": (0.10, 0.42),
    # veiling colour: green + blue dominant, red faint (typical pool water)
    "B_red": (0.02, 0.18),
    "B_green": (0.18, 0.42),
    "B_blue": (0.20, 0.48),
    "turbidity": (0.0, 1.4),          # blur strength; 0 => sharp
    "backscatter_noise": (2.0, 12.0),  # veiling shot-noise std (0..255)
    "caustics": (0.0, 0.18),          # ripple amplitude; 0 => off
    "caustics_freq": (2.5, 6.0),      # ripples across the frame
    "exposure": (0.75, 1.20),         # global gain
    "wb_jitter": 0.12,                 # +/- fraction per channel around 1.0
    "reflection": (0.10, 0.40),       # mirrored-surface distractor strength (when enabled)
    "reflection_prob": 0.6,            # probability the reflection distractor is present
}


def random_params(rng: np.random.Generator) -> WaterParams:
    """Sample a domain-randomization ``WaterParams`` from ``DR_RANGES`` (see that dict)."""
    def u(key):
        lo, hi = DR_RANGES[key]
        return float(rng.uniform(lo, hi))

    beta = np.array([u("beta_red"), u("beta_green"), u("beta_blue")])
    # keep the physical ordering red > green > blue even after independent sampling
    beta = np.sort(beta)[::-1]
    B = np.array([u("B_red"), u("B_green"), u("B_blue")])
    j = DR_RANGES["wb_jitter"]
    wb = 1.0 + rng.uniform(-j, j, size=3)
    reflection = u("reflection") if rng.random() < DR_RANGES["reflection_prob"] else 0.0
    return WaterParams(
        beta=beta,
        B=B,
        turbidity=u("turbidity"),
        backscatter_noise=u("backscatter_noise"),
        caustics=u("caustics"),
        caustics_freq=u("caustics_freq"),
        reflection=reflection,
        exposure=u("exposure"),
        wb_gain=wb,
    )


# --- helpers -----------------------------------------------------------------
def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur an (H,W,3) float image; cv2 if available, separable-numpy fallback otherwise."""
    if sigma <= 0:
        return img
    if cv2 is not None:
        k = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigma)
    # separable numpy fallback (avoids a hard cv2 dependency)
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    ker = np.exp(-(x**2) / (2 * sigma**2))
    ker /= ker.sum()
    out = img.copy()
    for ax in (0, 1):
        out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), ax, out)
    return out


def _depth_scaled_blur(img: np.ndarray, depth_norm: np.ndarray, strength: float) -> np.ndarray:
    """Turbidity: blur more where the water column is longer (distant pixels)."""
    if strength <= 0:
        return img
    # A few blur levels, linearly interpolated per-pixel by (normalized) depth — a cheap
    # approximation of a spatially-varying kernel (nearer = sharper, farther = blurrier).
    levels = [0.0, 1.0, 2.5, 4.5]
    blurred = [img if s == 0 else _gaussian_blur(img, s * strength) for s in levels]
    dn = np.clip(depth_norm, 0.0, 1.0)
    pos = dn * (len(levels) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(levels) - 2)
    frac = (pos - i0)[..., None]
    out = np.empty_like(img)
    for i in range(len(levels) - 1):
        m = i0 == i
        if m.any():
            out[m] = blurred[i][m] * (1 - frac[m]) + blurred[i + 1][m] * frac[m]
    return out


def _caustics(shape, amp: float, freq: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency crossing sinusoids -> a slowly-varying multiplicative light ripple in [1-a, 1+a]."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx /= w
    yy /= h
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    ang = rng.uniform(0, np.pi)
    pattern = (
        np.sin(2 * np.pi * freq * (xx * np.cos(ang) + yy * np.sin(ang)) + ph1)
        + 0.6 * np.sin(2 * np.pi * (freq * 0.7) * (xx - yy) + ph2)
    )
    pattern = pattern / np.abs(pattern).max()
    return 1.0 + amp * pattern


# --- main --------------------------------------------------------------------
def degrade(
    rgb_uint8: np.ndarray,
    depth_m: np.ndarray,
    params: WaterParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the underwater image-formation model to a clean render.

    Args:
        rgb_uint8: (H, W, 3) uint8 clean RGB (the MuJoCo render).
        depth_m:   (H, W) float metric depth from the camera [m] (Renderer depth mode).
        params:    a ``WaterParams`` (defaults = moderate murk). Use ``random_params`` for DR.
        rng:       optional Generator for the stochastic terms (noise/caustics/reflection phase).

    Returns:
        (H, W, 3) uint8 degraded RGB. Pixel positions are UNCHANGED, so segmentation-derived
        bounding boxes remain exact.
    """
    if params is None:
        params = WaterParams()
    if rng is None:
        rng = np.random.default_rng()

    J = np.asarray(rgb_uint8, dtype=np.float64)[..., :3] / 255.0
    z = np.asarray(depth_m, dtype=np.float64)
    # Guard the render's far-plane / background (huge z) so it just saturates to the veil colour.
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=1e3), 0.0, 1e3)

    # 1) per-channel transmission and the veiling composite  I = J t + B (1 - t)
    beta = params.beta.reshape(1, 1, 3)
    B = params.B.reshape(1, 1, 3)
    t = np.exp(-beta * z[..., None])            # (H,W,3) in (0,1]
    img = J * t + B * (1.0 - t)

    # A normalized depth (for depth-scaled effects) — 95th pct of finite scene depth as the scale.
    finite = z[z < 50.0]
    z_scale = np.percentile(finite, 95) if finite.size else 1.0
    z_scale = max(z_scale, 1e-3)
    depth_norm = np.clip(z / z_scale, 0.0, 1.0)

    # 2) water-surface reflection distractor (add BEFORE blur/noise so it also gets veiled/blurred).
    if params.reflection > 0:
        img = _add_surface_reflection(img, t, params.reflection)

    # 3) turbidity blur (depth-scaled: farther => blurrier)
    if params.turbidity > 0:
        img = _depth_scaled_blur(img, depth_norm, params.turbidity)

    # 4) caustics — multiplicative light ripple, strongest on near/lit surfaces (weight by t).
    if params.caustics > 0:
        caust = _caustics(img.shape[:2], params.caustics, params.caustics_freq, rng)[..., None]
        img = img * (1.0 + (caust - 1.0) * t.mean(axis=2, keepdims=True))

    # 5) backscatter shot noise — grows with the veiling fraction (1 - t)
    if params.backscatter_noise > 0:
        veil = (1.0 - t).mean(axis=2, keepdims=True)
        noise = rng.normal(0.0, params.backscatter_noise / 255.0, img.shape) * veil
        img = img + noise

    # 6) exposure / white-balance (camera gain + colour cast)
    img = img * params.exposure * params.wb_gain.reshape(1, 1, 3)

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _add_surface_reflection(img: np.ndarray, t: np.ndarray, strength: float) -> np.ndarray:
    """Overlay a faint vertically-mirrored, attenuated copy of the scene in the top band of the frame.

    Mimics the bright water surface acting as a mirror — a real false-detection source (mirrored
    red/orange balloons). The mirrored copy is attenuated by the local transmission so it reads as
    a hazy ghost, and fades out with depth from the top edge.
    """
    h, w, _ = img.shape
    band = max(1, int(round(h * 0.28)))          # top ~28% of the frame
    mirror = img[:band][::-1]                     # flip the top band vertically
    # vertical fade: strongest at the very top, gone by the band bottom
    fade = np.linspace(1.0, 0.0, band).reshape(band, 1, 1) ** 1.5
    ghost = mirror * fade * strength
    out = img.copy()
    out[:band] = np.clip(out[:band] + ghost, 0.0, 1.0)
    return out


# Difficulty presets (handy for the eval set / quick dials) --------------------
def preset(name: str) -> WaterParams:
    """Named difficulty dials: 'clear' | 'moderate' | 'murky'."""
    if name == "clear":
        return WaterParams(beta=np.array([0.30, 0.10, 0.06]), B=np.array([0.06, 0.20, 0.26]),
                           turbidity=0.15, backscatter_noise=2.0, caustics=0.06, reflection=0.10)
    if name == "murky":
        return WaterParams(beta=np.array([1.30, 0.58, 0.42]), B=np.array([0.14, 0.34, 0.40]),
                           turbidity=1.3, backscatter_noise=11.0, caustics=0.15, reflection=0.35)
    return WaterParams()  # moderate
