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
  * ``B_c``   veiling / backscatter colour (green-grey, matching the turbid competition pool): what a
              distant pixel (t->0) fades to.

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
    B       : backscatter/veiling colour in [0,1] RGB — the green-grey a far pixel fades to.
    turbidity      : depth-scaled gaussian blur strength (0 = off). Distant pixels blur more.
    backscatter_noise : std (in 0..255 units) of veiling shot noise, scaled by (1-t).
    caustics       : amplitude in [0,1] of the low-freq sinusoidal light ripple (0 = off).
    reflection     : strength in [0,1] of the mirrored water-surface distractor (0 = off).
    exposure       : global multiplicative gain on the final image.
    wb_gain        : per-channel white-balance multiplier (camera colour cast).
    murk           : the [0,1] difficulty level this condition was sampled at (0 = clear, 1 = murky);
                     recorded for per-condition stratification/reporting.
    cast           : the [0,1] water-COLOUR hue this frame was sampled at (0 = blue-green,
                     ~0.5 = green-grey like the real pool, 1 = murky yellow-green). Randomized per
                     frame so the training set spans casts (domain randomization). Recorded for eval.
    cast_sat       : the [0,1] saturation of that cast (0 = near-neutral grey, 1 = strongly cast).
    """

    beta: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.30, 0.36]))
    B: np.ndarray = field(default_factory=lambda: np.array([0.14, 0.32, 0.30]))
    turbidity: float = 0.6
    backscatter_noise: float = 6.0
    caustics: float = 0.10
    caustics_freq: float = 4.0
    particles: float = 0.0
    reflection: float = 0.25
    exposure: float = 1.0
    wb_gain: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    motion_blur: float = 0.0        # directional motion-blur kernel LENGTH [px] (0 = off); the moving vehicle
    motion_angle: float = 0.0       # motion-blur direction [rad] (0 = horizontal)
    vignette: float = 0.0           # lens corner-darkening strength in [0,1] (0 = off)
    murk: float = 0.5
    cast: float = 0.5
    cast_sat: float = 0.6
    # --- opt-in "pool0913" terms (defaults = OFF, so existing callers are unchanged) -------------
    cam_gain: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))  # NoIR WB error
    nir_gain: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # NIR -> RGB leak
    nir_beta: float = 2.5           # water NIR absorption [1/m] (NIR dies within ~1 m)
    caustic_web: float = 0.0        # amplitude of the cellular caustic network (0 = off)
    caustic_cell: float = 40.0      # caustic cell size [px]
    soft_focus: float = 0.0         # global defocus gaussian sigma [px] (0 = off)

    def __post_init__(self):
        self.beta = np.asarray(self.beta, dtype=np.float64).reshape(3)
        self.B = np.asarray(self.B, dtype=np.float64).reshape(3)
        self.wb_gain = np.asarray(self.wb_gain, dtype=np.float64).reshape(3)
        self.cam_gain = np.asarray(self.cam_gain, dtype=np.float64).reshape(3)
        self.nir_gain = np.asarray(self.nir_gain, dtype=np.float64).reshape(3)


# TWO independent random axes per frame:
#  (1) MURK  in [0,1] — difficulty / magnitude: sets attenuation strength, turbidity blur, veiling
#      noise and veil brightness (0 = clear, 1 = murky). Endpoints bracket real pool footage.
#  (2) CAST  in [0,1] — water COLOUR hue, sampled INDEPENDENTLY so the veil colour is randomized per
#      frame (domain randomization). This is the real fix for the Round-1 failure where the detector
#      keyed on an absolute "blue == water" cast: the training set now spans casts, so the model must
#      become colour-INVARIANT. cast=0 -> blue-green (ocean-like), ~0.5 -> green-grey (the real v1b
#      pool), 1 -> murky yellow-green. The attenuation `beta` green/blue balance is made CONSISTENT
#      with the cast (bluer water => blue penetrates => beta_blue smallest; greener/turbid water =>
#      green penetrates => beta_green smallest), but beta_RED is ALWAYS the largest (red dies first:
#      physical). A per-frame SATURATION (some frames strongly cast, some near-neutral grey) breaks
#      any residual colour key.
CLEAR = {"beta_red": 0.30, "turbidity": 0.05, "backscatter_noise": 2.0, "veil_val": 0.20}  # murk 0
MURKY = {"beta_red": 1.35, "turbidity": 1.40, "backscatter_noise": 12.0, "veil_val": 0.46}  # murk 1

# Unit-peak veil chromaticity anchors along the cast arc (blue-green <-> green-grey <-> yellow-green).
_VEIL_BLUE = np.array([0.34, 0.72, 1.00])   # cast 0.0  bluer-green / cyan (ocean-like)
_VEIL_GREY = np.array([0.60, 1.00, 0.86])   # cast 0.5  green-grey (real competition pool)
_VEIL_YGRN = np.array([0.82, 1.00, 0.52])   # cast 1.0  murky yellow-green


def _veil_chroma(cast: float) -> np.ndarray:
    """Unit-peak veil chromaticity along the blue<->green-grey<->yellow-green arc (cast in [0,1])."""
    if cast < 0.5:
        c = _VEIL_BLUE * (1 - 2 * cast) + _VEIL_GREY * (2 * cast)
    else:
        c = _VEIL_GREY * (2 - 2 * cast) + _VEIL_YGRN * (2 * cast - 1)
    return c / c.max()


def cast_bucket(cast: float | None) -> str:
    """Bucket a cast hue for eval stratification: blue / green_grey / yellow_green (None -> clean)."""
    if cast is None:
        return "clean"
    if cast < 0.34:
        return "blue"
    if cast < 0.67:
        return "green_grey"
    return "yellow_green"


DR_RANGES = {
    "caustics": (0.0, 0.28),          # ripple amplitude; 0 => off (widened for more variety)
    "caustics_freq": (2.5, 7.0),      # ripples across the frame (widened)
    "particles": (0.0, 0.6),          # suspended-particle / marine-snow speck strength (0 => off)
    # Exposure/gain + white-balance model the Raspberry-Pi camera's AUTO-EXPOSURE + AUTO-WHITE-BALANCE,
    # which swing hard in the operating range as the vehicle approaches a bright/dark balloon. Widened
    # (was 0.75-1.20 / 0.12) so the detector cannot key on a fixed brightness/colour temperature.
    "exposure": (0.55, 1.45),         # global gain (raspi auto-exposure swing)
    "wb_jitter": 0.18,                 # +/- fraction per channel around 1.0 (auto-white-balance)
    # Motion blur (the vehicle MOVES while the rolling shutter integrates) — a directional streak whose
    # length ~ speed. Present on ~half the frames; boxes stay exact (linear kernel is centred).
    "motion_blur": (0.0, 9.0),        # kernel length [px]
    "motion_blur_prob": 0.5,
    # Lens vignetting: gentle radial corner-darkening (cheap wide-angle optics). Photometric only.
    "vignette": (0.10, 0.50),
    "vignette_prob": 0.6,
    "cast_sat": (0.15, 1.0),           # veil colour saturation (0.15 = near-grey, 1.0 = strong cast)
    "veil_val_jitter": 0.15,           # +/- fraction on veil brightness around the murk-set value
    # --- water-surface reflection distractor (the #1 false-detection source: mirrored balloons on
    # the underside of the surface). STRONG + almost always present so the detector learns them as
    # HARD NEGATIVES. Reflections are geometric (a mirrored-camera render, see gen_sim_dataset) and
    # UNLABELLED — only real balloon_*_geom get boxes. These params control the ripple/veil ON TOP.
    "reflection": (0.35, 0.80),        # composite strength (when present) — clearly visible
    "reflection_prob": 0.90,           # present ~90% of frames (also gated by seeing the surface)
    "reflection_ripple_px": (4.0, 14.0),  # peak horizontal wave displacement [px] (vertical ~0.35x)
    "reflection_waves": (2, 3),           # number of superposed sinusoidal wave components
    "reflection_bluemix": (0.30, 0.55),   # how far the reflection is pulled toward the veil colour B
}


# Murk is sampled around a realistic CENTRE (not uniform 0-1): a normal centred at MURK_CENTRE with
# MURK_STD spread, so most frames sit at the "just right" difficulty (calibrated to real footage:
# ~between sim_sample frames 0000=0.68 and 0011=0.47) while the tails still give some clearer and
# some murkier frames for robustness.
MURK_CENTRE = 0.57
MURK_STD = 0.18
# Cast (water COLOUR hue) is sampled around green-grey (the real v1b look) but with a WIDE spread so
# every frame gets a different cast and the tails reach blue-green and yellow-green — the training
# distribution spans casts (domain randomization) so the detector cannot key on an absolute colour.
CAST_CENTRE = 0.50
CAST_STD = 0.30


def random_params(
    rng: np.random.Generator, murk: float | None = None, cast: float | None = None
) -> WaterParams:
    """Sample a ``WaterParams``; ``murk`` in [0,1] (difficulty) and ``cast`` in [0,1] (colour hue).

    ``murk`` sets magnitude (attenuation strength / turbidity / noise / veil brightness), drawn from a
    normal centred at MURK_CENTRE. ``cast`` sets the water COLOUR — the veil hue AND the beta green/blue
    balance — drawn INDEPENDENTLY from a wide normal centred at CAST_CENTRE (green-grey), so casts span
    blue-green<->green-grey<->yellow-green per frame. beta_red is always the largest (red dies first).
    A per-frame saturation randomizes how strongly cast vs near-grey each frame is.
    caustics/exposure/white-balance/reflection are drawn independently.
    """
    def u(key):
        lo, hi = DR_RANGES[key]
        return float(rng.uniform(lo, hi))

    m = (float(np.clip(rng.normal(MURK_CENTRE, MURK_STD), 0.03, 0.97))
         if murk is None else float(np.clip(murk, 0.0, 1.0)))
    c = (float(np.clip(rng.normal(CAST_CENTRE, CAST_STD), 0.0, 1.0))
         if cast is None else float(np.clip(cast, 0.0, 1.0)))

    # Attenuation: red magnitude from murk; green/blue balance from the cast. bluer water (c->0) =>
    # blue penetrates (beta_blue smallest); greener/turbid water (c->1) => green penetrates
    # (beta_green smallest). beta_red is ALWAYS the largest.
    red = CLEAR["beta_red"] * (1 - m) + MURKY["beta_red"] * m
    frac_g = 0.50 - 0.20 * c    # 0.50 (blue water) -> 0.30 (green water)
    frac_b = 0.30 + 0.28 * c    # 0.30 (blue water) -> 0.58 (green water)
    beta = np.array([red, red * frac_g, red * frac_b])

    # Veil colour: chromaticity from the cast hue, desaturated toward grey by a per-frame saturation,
    # brightness from murk (murkier => brighter veil).
    sat = u("cast_sat")
    chroma = _veil_chroma(c)
    chroma = (1 - sat) * chroma.mean() + sat * chroma       # desaturate toward grey
    vj = DR_RANGES["veil_val_jitter"]
    veil_val = (CLEAR["veil_val"] * (1 - m) + MURKY["veil_val"] * m) * (1.0 + rng.uniform(-vj, vj))
    B = chroma * veil_val

    turbidity = CLEAR["turbidity"] * (1 - m) + MURKY["turbidity"] * m
    noise = CLEAR["backscatter_noise"] * (1 - m) + MURKY["backscatter_noise"] * m
    j = DR_RANGES["wb_jitter"]
    wb = 1.0 + rng.uniform(-j, j, size=3)
    reflection = u("reflection") if rng.random() < DR_RANGES["reflection_prob"] else 0.0
    # suspended particles: present on ~half the frames, denser in murkier water.
    particles = u("particles") * (0.5 + 0.5 * m) if rng.random() < 0.5 else 0.0
    # motion blur (moving vehicle) + lens vignetting — near-range robustness DR.
    motion_blur = u("motion_blur") if rng.random() < DR_RANGES["motion_blur_prob"] else 0.0
    motion_angle = float(rng.uniform(0.0, np.pi))
    vignette = u("vignette") if rng.random() < DR_RANGES["vignette_prob"] else 0.0
    return WaterParams(
        beta=beta, B=B, turbidity=turbidity, backscatter_noise=noise,
        caustics=u("caustics"), caustics_freq=u("caustics_freq"), particles=particles,
        reflection=reflection, exposure=u("exposure"), wb_gain=wb,
        motion_blur=motion_blur, motion_angle=motion_angle, vignette=vignette, murk=m,
        cast=c, cast_sat=sat,
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


def _add_particles(img: np.ndarray, t: np.ndarray, strength: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sprinkle suspended-particle specks (marine snow) onto a float image in-place-ish.

    Mostly small BRIGHT motes (lit backscatter) with a few darker flecks, denser where the water is
    more veiled (weight by the veiling fraction 1 - t). ``strength`` in [0,1] sets density/contrast.
    Cheap: a sparse random mask, softened once. Boxes are untouched (pixels don't move).
    """
    h, w = img.shape[:2]
    veil = (1.0 - t).mean(axis=2)                     # 0 near, ->1 far
    density = 0.0008 + 0.010 * strength               # fraction of pixels seeded
    seed = rng.random((h, w))
    thresh = density * (0.3 + 0.7 * veil)             # more specks in the far/veiled region
    bright = (seed < thresh).astype(np.float64)
    dark = (seed > 1.0 - 0.25 * thresh).astype(np.float64)
    speck = _gaussian_blur((bright - 0.6 * dark)[..., None] * np.ones((1, 1, 3)), 0.6)
    return img + speck * (0.35 + 0.5 * strength)


def _motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
    """Directional (linear) motion-blur streak — the vehicle moves while the frame integrates.

    Kernel is a centred line of length ``length`` px at ``angle`` rad, so the blur is symmetric about
    each pixel and box centres/extents are unchanged (labels stay exact). cv2 if present, else a cheap
    separable-ish numpy fallback along the nearest axis.
    """
    k = int(round(length))
    if k < 2:
        return img
    ker = np.zeros((k, k), dtype=np.float32)
    c = (k - 1) / 2.0
    dx, dy = np.cos(angle), np.sin(angle)
    for t in np.linspace(-c, c, k * 2):
        xi = int(round(c + t * dx))
        yi = int(round(c + t * dy))
        if 0 <= xi < k and 0 <= yi < k:
            ker[yi, xi] = 1.0
    s = ker.sum()
    if s <= 0:
        return img
    ker /= s
    if cv2 is not None:
        return cv2.filter2D(img.astype(np.float32), -1, ker).astype(img.dtype)
    # numpy fallback: blur along the dominant axis only (approximation)
    ax = 1 if abs(dx) >= abs(dy) else 0
    lin = np.ones(k, dtype=np.float64) / k
    return np.apply_along_axis(lambda m: np.convolve(m, lin, mode="same"), ax, img)


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Gentle radial corner-darkening (cheap wide-angle optics). Multiplicative, boxes untouched."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2
    mask = 1.0 - strength * np.clip(r2 / 2.0, 0.0, 1.0)   # centre=1, corners=1-strength
    return img * mask[..., None]


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
    nir: np.ndarray | None = None,
    caustic_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the underwater image-formation model to a clean render.

    Args:
        rgb_uint8: (H, W, 3) uint8 clean RGB (the MuJoCo render).
        depth_m:   (H, W) float metric depth from the camera [m] (Renderer depth mode).
        params:    a ``WaterParams`` (defaults = moderate murk). Use ``random_params`` for DR.
        rng:       optional Generator for the stochastic terms (noise/caustics/reflection phase).
        nir:       optional (H, W) near-infrared reflectance map in [0,1] (per material, from the
                   segmentation). Used only when ``params.nir_gain`` is non-zero (pool0913 profile):
                   the NoIR camera adds ``nir_gain * nir * exp(-nir_beta * z)`` to R/G/B.
        caustic_weight: optional (H, W) weight in [0,1] where the cellular caustic network lands
                   (lit floor/walls/surface; 0 on balloons). Used only when ``params.caustic_web`` > 0.

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

    # NB: the water-surface reflection is GEOMETRIC (a mirrored-camera render) and is composited
    # into the CLEAN RGB by the generator BEFORE degrade() — see apply_surface_reflection. degrade()
    # then veils it along with the rest of the scene ("veil on top of the correct reflection").

    # 1b) Pi NoIR near-infrared leak (opt-in): the sensor has no IR-cut filter, so NIR reflected by
    # each material leaks mostly into R (some B). Water absorbs NIR within ~1 m, so the leak is
    # strongest for close objects / near the surface and falls off with range.
    if nir is not None and np.any(params.nir_gain > 0):
        leak = np.asarray(nir, dtype=np.float64)[..., None] * np.exp(-params.nir_beta * z[..., None])
        img = img + leak * params.nir_gain.reshape(1, 1, 3)

    # 2b) cellular caustic network (opt-in): bright web-like lines of sunlight focused by the waves.
    if params.caustic_web > 0:
        web = caustic_web(img.shape[:2], params.caustic_cell, rng)
        wgt = t.mean(axis=2) if caustic_weight is None else caustic_weight * t.mean(axis=2)
        img = img * (1.0 + params.caustic_web * (web * wgt)[..., None])

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

    # 5b) suspended particles / marine snow — sparse bright specks (lit backscatter motes) plus a few
    # darker flecks, denser in the veiled (far) region. A common real-footage distractor; adds
    # texture the detector must ignore. Pixel positions unchanged => boxes stay exact.
    if params.particles > 0:
        img = _add_particles(img, t, params.particles, rng)

    # 6) exposure / white-balance (camera gain + colour cast; raspi auto-exposure/AWB swing). The
    # opt-in ``cam_gain`` is the NoIR camera's pink/magenta white-balance error (R up, G down).
    img = img * params.exposure * params.wb_gain.reshape(1, 1, 3) * params.cam_gain.reshape(1, 1, 3)
    if params.soft_focus > 0:
        img = _gaussian_blur(img, params.soft_focus)

    # 7) motion blur (moving vehicle) then lens vignetting — near-range robustness DR. Both are
    # applied last (on the ~final image) and keep pixel positions symmetric, so boxes stay exact.
    if params.motion_blur > 0:
        img = _motion_blur(img, params.motion_blur, params.motion_angle)
    if params.vignette > 0:
        img = _vignette(img, params.vignette)

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def caustic_web(shape, cell_px: float, rng: np.random.Generator) -> np.ndarray:
    """Cellular (Worley F2-F1) caustic network in [0,1]: thin bright lines along cell borders,
    wobbled by a smooth warp — the web-like light pattern sunlight makes through a rippled surface."""
    from scipy.spatial import cKDTree

    h, w = shape
    n = max(8, int(h * w / (cell_px * cell_px)))
    pts = rng.uniform([0, 0], [w, h], size=(n, 2))
    small = 4                                     # evaluate at 1/4 res, then upsample (cheap)
    hs, ws = (h + small - 1) // small, (w + small - 1) // small
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float64) * small
    warp = _gaussian_blur(rng.normal(0, 1, (hs, ws, 2)), 3.0)
    warp = warp / max(np.abs(warp).max(), 1e-9) * cell_px * 0.35
    q = np.stack([xx + warp[..., 0], yy + warp[..., 1]], -1).reshape(-1, 2)
    d, _ = cKDTree(pts).query(q, k=2)
    edge = (d[:, 1] - d[:, 0]).reshape(hs, ws)
    web = np.exp(-edge / (0.04 * cell_px))
    if cv2 is not None:
        web = cv2.resize(web.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:  # pragma: no cover
        web = np.kron(web, np.ones((small, small)))[:h, :w]
    return np.clip(web, 0.0, 1.0)


# pool0913 profile (opt-in): the real pool footage of 2026-09-13 / 10-01 (Pi NoIR camera just below
# the surface, clear bright water). Ranges are measured/eyeballed from ai/balloon/pool0913 frames.
POOL0913_RANGES = {
    "beta_red": (0.25, 0.55),        # clear water: red attenuates, but distances are short
    "veil": ([0.30, 0.62, 0.62], [0.42, 0.76, 0.74]),  # cyan veil (before the camera gain)
    "cam_gain_r": (1.08, 1.35),      # NoIR white-balance error: R gain (per episode/day)
    "cam_gain_g": (0.85, 0.98),
    "cam_gain_b": (0.80, 0.95),
    "nir_r": (0.50, 1.00),           # NIR leak into R ...
    "nir_b_frac": (0.40, 0.70),      # ... and a fraction of that into B
    "nir_beta": (0.8, 2.0),          # effective water absorption over the NoIR band (~700-800 nm)
    "exposure": (0.78, 1.10),        # often over-exposed (washes the pale balloons out)
    "caustic_web": (0.20, 0.70),
    "caustic_cell": (18.0, 60.0),
    "soft_focus": (0.6, 2.2),
    "motion_blur": (0.0, 7.0),
}


def random_params_pool0913(rng: np.random.Generator) -> WaterParams:
    """Sample a ``WaterParams`` for the opt-in pool0913 look (clear cyan water + NoIR camera)."""
    def u(key):
        lo, hi = POOL0913_RANGES[key]
        return float(rng.uniform(lo, hi))

    red = u("beta_red")
    beta = np.array([red, red * rng.uniform(0.18, 0.30), red * rng.uniform(0.22, 0.35)])
    lo, hi = (np.asarray(v) for v in POOL0913_RANGES["veil"])
    B = rng.uniform(lo, hi)
    cam_gain = np.array([u("cam_gain_r"), u("cam_gain_g"), u("cam_gain_b")])
    nir_r = u("nir_r")
    # silicon responds to NIR in every Bayer channel, most in R: the leak is pink-WHITE, not pure red
    nir_gain = np.array([nir_r, nir_r * float(rng.uniform(0.25, 0.45)), nir_r * u("nir_b_frac")])
    return WaterParams(
        beta=beta, B=B, turbidity=float(rng.uniform(0.1, 0.5)), backscatter_noise=float(rng.uniform(1, 4)),
        caustics=0.0, particles=0.0, reflection=float(rng.uniform(0.75, 0.95)),
        exposure=u("exposure"), wb_gain=1.0 + rng.uniform(-0.05, 0.05, size=3),
        motion_blur=u("motion_blur") if rng.random() < 0.5 else 0.0,
        motion_angle=float(rng.uniform(0.0, np.pi)),
        vignette=float(rng.uniform(0.0, 0.25)), murk=0.2, cast=0.2, cast_sat=0.4,
        cam_gain=cam_gain, nir_gain=nir_gain, nir_beta=u("nir_beta"),
        caustic_web=u("caustic_web"), caustic_cell=u("caustic_cell"), soft_focus=u("soft_focus"),
    )


def _remap(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at (map_y, map_x) — cv2.remap if available, nearest-neighbour numpy fallback."""
    h, w = img.shape[:2]
    if cv2 is not None:
        return cv2.remap(
            img.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
    xi = np.clip(np.round(map_x).astype(int), 0, w - 1)
    yi = np.clip(np.round(map_y).astype(int), 0, h - 1)
    return img[yi, xi]


def apply_surface_reflection(clean_rgb, reflection_rgb, reflect_mask, B, strength, rng):
    """Composite a GEOMETRICALLY-CORRECT water-surface reflection onto the clean render.

    Unlike a naive image flip, ``reflection_rgb`` is a render of the scene with the balloons mirrored
    across the true water-surface plane (y≈3.3 m), seen by the SAME camera — so it obeys perspective
    and lands exactly where the surface projects. ``reflect_mask`` (bool HxW) is the intersection of
    "the water surface is actually visible here" (primary segmentation) and "a mirrored balloon
    projects here" (reflection segmentation) — so reflections are anchored to the surface line and are
    ONLY the mirrored balloons, not the whole band. On TOP of that correct reflection we add the ripple
    (a wavy displacement) and pull the colour toward the veil ``B`` (attenuate + blue-tint). Reflections
    carry NO GT box — they are hard negatives.

    Returns uint8 HxWx3. If nothing reflects (empty mask / strength 0) the clean image is unchanged.
    """
    base = np.asarray(clean_rgb, dtype=np.float64)[..., :3] / 255.0
    if strength <= 0 or not np.any(reflect_mask):
        return np.clip(base * 255.0, 0, 255).astype(np.uint8)
    refl = np.asarray(reflection_rgb, dtype=np.float64)[..., :3] / 255.0
    B = np.asarray(B, dtype=np.float64).reshape(1, 1, 3)
    h, w = base.shape[:2]
    alpha = reflect_mask.astype(np.float64)

    # Ripple: a wavy sinusoidal displacement applied to BOTH the reflection colour and its mask, so
    # the mirrored balloons stretch/break up like a real rippled surface (on top of correct geometry).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    amp = rng.uniform(*DR_RANGES["reflection_ripple_px"])
    n_waves = int(rng.integers(DR_RANGES["reflection_waves"][0], DR_RANGES["reflection_waves"][1] + 1))
    disp_x = np.zeros((h, w))
    disp_y = np.zeros((h, w))
    for _ in range(n_waves):
        a = amp * rng.uniform(0.4, 1.0) / max(n_waves, 1)
        fr = rng.uniform(0.8, 2.6)
        fc = rng.uniform(0.6, 2.4)
        ph = rng.uniform(0, 2 * np.pi)
        phase = 2 * np.pi * (fr * yy / h + fc * xx / w) + ph
        disp_x += a * np.sin(phase)
        disp_y += 0.35 * a * np.cos(phase)
    refl = _remap(refl, xx + disp_x, yy + disp_y)
    alpha = _remap(alpha, xx + disp_x, yy + disp_y)

    # Attenuate + blue-tint: pull the reflection toward the veil colour B (recognizable but watery).
    bluemix = float(rng.uniform(*DR_RANGES["reflection_bluemix"]))
    refl = refl * (1.0 - bluemix) + B * bluemix
    refl = _gaussian_blur(refl, 1.0)  # a touch soft, like a real reflection

    alpha = alpha * strength
    out = base * (1.0 - alpha[..., None]) + refl * alpha[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


SNELL_COS = float(np.cos(np.arcsin(1.0 / 1.333)))  # cos of the critical angle (~48.6 deg)


def composite_surface_mirror(clean_rgb, mirror_rgb, surface_mask, cos_inc, rng, *, window_rgb,
                             strength=0.9, extra_maps=()):
    """pool0913 profile: the underside of the water surface as seen from just below it.

    Outside Snell's window (incidence beyond the critical angle) the surface is a near-perfect mirror
    (total internal reflection) of the underwater scene — ``mirror_rgb`` is a render of the WHOLE
    scene mirrored across the surface plane from the same camera, so each balloon's upside-down copy
    sits directly above it, touching it at the surface line. Inside the window the bright sky shows
    through (``window_rgb``) with a weak Fresnel reflection. A wavy displacement ripples both.
    ``extra_maps`` (HxW float arrays, e.g. a NIR map of the mirrored scene) are displaced with the
    same ripple. Returns (uint8 image, tir_alpha HxW, [displaced extra maps]). Reflections carry NO
    label — only the primary segmentation produces boxes.
    """
    base = np.asarray(clean_rgb, dtype=np.float64)[..., :3] / 255.0
    refl = np.asarray(mirror_rgb, dtype=np.float64)[..., :3] / 255.0
    h, w = base.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    amp = rng.uniform(3.0, 16.0)
    disp_x = np.zeros((h, w))
    disp_y = np.zeros((h, w))
    for _ in range(int(rng.integers(2, 5))):
        a = amp * rng.uniform(0.3, 1.0) / 2.0
        fr = rng.uniform(3.0, 14.0)                 # mostly horizontal stripes (perspective-squashed)
        fc = rng.uniform(0.3, 2.5)
        phase = 2 * np.pi * (fr * yy / h + fc * xx / w) + rng.uniform(0, 2 * np.pi)
        disp_x += a * np.sin(phase)
        disp_y += 0.3 * a * np.cos(phase)
    refl = _remap(refl, xx + disp_x, yy + disp_y)
    maps = [_remap(np.asarray(m, dtype=np.float32), xx + disp_x, yy + disp_y) for m in extra_maps]
    tir = np.clip((SNELL_COS + 0.05 - np.asarray(cos_inc)) / 0.10, 0.0, 1.0)
    alpha = strength * (0.10 + 0.90 * tir) * surface_mask
    win = np.asarray(window_rgb, dtype=np.float64).reshape(1, 1, 3) * (1.0 + 0.08 * np.sin(
        2 * np.pi * (yy / h * rng.uniform(4, 12)) + disp_x / 3.0))[..., None]
    surf = refl * alpha[..., None] + win * (1.0 - alpha[..., None])
    out = np.where(surface_mask[..., None], surf, base)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8), alpha, maps


# Difficulty presets (handy for the eval set / quick dials) --------------------
def preset(name: str) -> WaterParams:
    """Named difficulty dials: 'clear' | 'moderate' | 'murky'."""
    if name == "clear":
        return WaterParams(beta=np.array([0.30, 0.09, 0.11]), B=np.array([0.10, 0.24, 0.22]),
                           turbidity=0.15, backscatter_noise=2.0, caustics=0.06, reflection=0.10)
    if name == "murky":
        return WaterParams(beta=np.array([1.30, 0.50, 0.62]), B=np.array([0.26, 0.44, 0.38]),
                           turbidity=1.3, backscatter_noise=11.0, caustics=0.15, reflection=0.35)
    return WaterParams()  # moderate
