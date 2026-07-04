"""Underwater colour restoration — recover the attenuated red channel + remove the blue-green cast.

Underwater, red light is absorbed fastest, so red objects look dark/blue and telling red from blue is
hard — a real problem for the competition (red +30 vs blue -10; confusing them is costly). This restores
colour so red / blue / yellow separate again. Use it two ways:

* to prep frames for **labelling** (a human can then tell red from blue), and
* as a fixed **preprocessing** step for the detector, so training and inference see the same de-cast
  image. Bounding boxes are unchanged (colour restoration does not move pixels), so labels transfer.

Cheap (per-image channel gains + one CLAHE pass), Pi-4-friendly.
"""
from __future__ import annotations

import cv2
import numpy as np


def _red_compensate(rgb: np.ndarray) -> np.ndarray:
    """Ancuti-style: refill the attenuated red channel from green (where red signal is lost)."""
    r, g = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0
    r2 = r + (g.mean() - r.mean()) * (1.0 - r) * g
    out = rgb.copy()
    out[..., 0] = np.clip(r2 * 255.0, 0.0, 255.0)
    return out


def _gray_world(rgb: np.ndarray, gain_lo: float = 0.5, gain_hi: float = 2.5) -> np.ndarray:
    """Per-image white balance: scale each channel so the means match (removes the colour cast).

    Gains are clamped so a bright, low-signal scene isn't blown out / pushed to a false cast.
    """
    m = rgb.reshape(-1, 3).mean(0)
    gain = np.clip(m.mean() / np.clip(m, 1e-3, None), gain_lo, gain_hi)
    return np.clip(rgb * gain, 0.0, 255.0)


def _clahe_l(rgb: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Local contrast on the L channel (Lab) — lifts faint underwater detail without colour shift."""
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def underwater_correct(rgb: np.ndarray, *, red_compensate: bool = True, clahe: bool = True) -> np.ndarray:
    """Colour-restore an underwater RGB frame (uint8 HxWx3 -> uint8 HxWx3).

    Recovers red and removes the blue-green cast so red/blue/yellow separate. Deterministic, no fit.
    """
    x = np.asarray(rgb)[..., :3].astype(np.float32)
    if red_compensate:
        x = _red_compensate(x)
    x = _gray_world(x)
    if clahe:
        x = _clahe_l(x)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)
