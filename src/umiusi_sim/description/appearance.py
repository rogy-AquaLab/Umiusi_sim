"""Shared "perception appearance" spec-prep for the balloon detector's rendered view.

The learned detector (``learned_detector``) is trained by ``tools/gen_sim_dataset`` on frames
whose *appearance* is deliberately shaped so the synthetic balloons look like the real teardrop
balloons under bright pool light: oval (ellipsoid) balloons, the foreground pin geom hidden,
near-invisible fishing-line tethers, and a sunlit pool (bright near-uniform fill + an overhead
sun + a water-surface ceiling + far fog). The underwater degradation (``underwater_sim``) is then
applied on top.

Any tool that feeds the detector a LIVE render must reproduce this same appearance, or the
detector sees an out-of-distribution image (dark spheres with a foreground needle) and fails. So
the appearance edits live here, shared by BOTH ``gen_sim_dataset.prep_render_spec`` (which ALSO
enlarges the pool for the training field) and ``tools/autonomy_run`` (which keeps the REAL 3.3 m
competition pool geometry). Only the balloon/pin/tether/lighting *appearance* is shared; the pool
size is the caller's choice.

All edits mutate a composed ``mujoco.MjSpec`` IN PLACE, between ``build_spec`` and ``compile`` —
``competition_balloon.py`` and the physics (pin geometry, pop detection) are untouched: the pin is
only made invisible (alpha=0), not removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Real competition balloons are egg/teardrop-shaped (taller than wide). We approximate them as
# ELLIPSOIDS with a vertical (+Y) major axis at this aspect (height/width). Seg->bbox stays exact.
BALLOON_ASPECT = 1.25
# Subtle underwater tether look (near-invisible fishing line): thin + low-contrast, near the
# water/background colour with low alpha. Turbidity blur fades it further with distance.
TETHER_RGBA = (0.28, 0.42, 0.52, 0.32)
TETHER_RADIUS = 0.0015


def style_balloons_pin_tethers(spec: mujoco.MjSpec, hide_tethers: bool = False) -> None:
    """Balloon spheres -> vertical ellipsoids; hide the foreground pin geom; make tethers subtle.

      * balloon ``*_geom`` spheres -> ellipsoids (vertical major axis, aspect BALLOON_ASPECT);
      * the ``pin`` geom -> alpha 0 (invisible in renders; geometry/physics/pop-detection unchanged);
      * ``*_tether`` cylinders -> thin + low-contrast + low alpha, or hidden if ``hide_tethers``.
    """
    for g in spec.geoms:
        name = g.name or ""
        if name.startswith("balloon_") and name.endswith("_geom"):
            r = float(g.size[0])
            g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            g.size[:] = [r, r * BALLOON_ASPECT, r]  # +Y up => vertical major axis
        elif name == "pin":
            g.rgba[3] = 0.0  # invisible in renders; geometry/physics unchanged
        elif name.endswith("_tether"):
            if hide_tethers:
                g.rgba[3] = 0.0
            else:
                g.rgba[:] = TETHER_RGBA
                g.size[0] = TETHER_RADIUS


@dataclass
class LightingParams:
    """One lighting condition. Defaults REPRODUCE the original fixed sunlit-pool look EXACTLY, so
    callers that pass nothing (e.g. tools/autonomy_run) are byte-for-byte unchanged. The dataset
    generator instead draws a randomized ``LightingParams`` per frame (see ``sample_lighting``) so
    the detector generalises across lighting — the biggest remaining sim gap after the colour-cast fix.

    ambient/diffuse : headlight fill (orientation-independent water-scatter fill).
    sun_dir/sun_diffuse : overhead directional light (sunlight through the surface). Randomising the
                     DIRECTION adds shading cues (currently dead-flat straight-down); intensity varies
                     scene brightness.
    surface_rgb    : colour/brightness of the opaque water-surface "ceiling" seen when looking UP —
                     randomising it varies the surface glare, directly diversifying the hard look-up
                     stratum.
    fog_rgb/fogstart/fogend : far background water fog.
    """

    ambient: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.57, 0.60]))
    diffuse: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.55, 0.55]))
    sun_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    sun_diffuse: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.55, 0.58]))
    surface_rgb: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.72, 0.80]))
    fog_rgb: np.ndarray = field(default_factory=lambda: np.array([0.18, 0.46, 0.55]))
    fogstart: float = 6.0
    fogend: float = 16.0
    # summary tags recorded for eval stratification (0 => the fixed default look).
    light_level: float = 0.55
    sun_tilt_deg: float = 0.0


def sample_lighting(rng: np.random.Generator) -> LightingParams:
    """Randomize a plausible sunlit-pool lighting condition (domain randomisation).

    Varies: overall brightness (dimmer/brighter than default), a warm<->cool colour temperature on
    the fill + sun, the overhead sun DIRECTION (tilt off-nadir + azimuth, for shading variety), the
    sun intensity, the surface-ceiling brightness/tint (look-up glare), and the fog depth. All bounded
    and clipped so it stays a physically-plausible underwater-pool look.
    """
    # overall brightness and a warm(-)/cool(+) colour temperature tint (normalised around 1).
    level = float(rng.uniform(0.35, 0.75))
    temp = float(rng.uniform(-1.0, 1.0))
    tint = np.array([1.0 - 0.12 * temp, 1.0, 1.0 + 0.12 * temp])   # warm=more red, cool=more blue
    jit = lambda s=0.04: rng.uniform(-s, s, size=3)  # noqa: E731
    ambient = np.clip(level * tint + jit(), 0.05, 0.90)
    diffuse = np.clip(level * rng.uniform(0.8, 1.2) * tint + jit(), 0.05, 0.90)
    # sun direction: tilt off straight-down by up to ~32 deg at a random azimuth (shading cues).
    tilt = float(rng.uniform(0.0, np.radians(32.0)))
    azi = float(rng.uniform(0.0, 2.0 * np.pi))
    sun_dir = np.array([np.sin(tilt) * np.cos(azi), -np.cos(tilt), np.sin(tilt) * np.sin(azi)])
    sun_diffuse = np.clip(float(rng.uniform(0.30, 0.80)) * tint, 0.0, 0.90)
    surf_b = float(rng.uniform(0.42, 1.00))
    surface_rgb = np.clip(surf_b * np.array([0.83, 1.0, 1.11]) * tint, 0.05, 1.0)
    fog_rgb = np.clip(np.array([0.18, 0.46, 0.55]) * tint + jit(0.03), 0.03, 0.9)
    return LightingParams(
        ambient=ambient, diffuse=diffuse, sun_dir=sun_dir, sun_diffuse=sun_diffuse,
        surface_rgb=surface_rgb, fog_rgb=fog_rgb,
        fogstart=float(rng.uniform(4.0, 9.0)), fogend=float(rng.uniform(13.0, 20.0)),
        light_level=round(level, 3), sun_tilt_deg=round(float(np.degrees(tilt)), 1),
    )


def brighten_like_pool(spec: mujoco.MjSpec, center_x: float, depth: float, len_x: float,
                       len_z: float, floor_y: float = 0.0, add_surface: bool = True,
                       lighting: LightingParams | None = None) -> None:
    """Light the scene like a real sunlit pool: bright + near-uniform, lit from ABOVE.

    Reproduces the dataset render's lighting so the detector sees its training distribution:

      * headlight ambient/diffuse raised -> strong orientation-independent fill (the water-scattering
        look, near-uniform brightness) instead of the dark, lit-from-below raw scenario;
      * a broad DIRECTIONAL overhead light points down from above the surface (sunlight through the
        surface), no shadow so it stays even;
      * ``add_surface`` adds an opaque water-surface "ceiling" at the top of the pool so a camera
        tilted up sees a bright surface (as underwater), not the black void above the walls;
      * far underwater fog fades the background (beyond the walls) to a water colour so the frame
        reads as "in water"; kept far so it barely touches near balloons (the murk veil comes from
        the depth-based degradation).

    ``center_x``/``depth``/``len_x``/``len_z``/``floor_y`` are the pool geometry (real competition
    pool or the enlarged training field) so the sun/surface/fog track whatever pool is in use.
    ``lighting`` (a ``LightingParams``) selects the exact levels/colours/direction; ``None`` uses the
    fixed default look (identical to the original behaviour).
    """
    lp = lighting or LightingParams()
    hl = spec.visual.headlight
    hl.ambient[:] = lp.ambient           # near-uniform bright fill (raw scenario ~0.10)
    hl.diffuse[:] = lp.diffuse
    hl.specular[:] = [0.10, 0.10, 0.10]
    hl.active = 1
    sun = spec.worldbody.add_light()
    sun.name = "perception_sun"
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = [center_x, floor_y + depth + 2.0, 0.0]  # above the water surface
    sun.dir[:] = lp.sun_dir
    sun.diffuse[:] = lp.sun_diffuse
    sun.specular = [0.0, 0.0, 0.0]
    sun.castshadow = False
    if add_surface:
        surf = spec.worldbody.add_geom()
        surf.name = "perception_surface"
        surf.type = mujoco.mjtGeom.mjGEOM_BOX
        surf.pos = [center_x, floor_y + depth, 0.0]
        surf.size = [len_x / 2, 0.02, len_z / 2]
        surf.rgba = [*lp.surface_rgb, 1.0]  # bright surface seen from below
        surf.contype = 0
        surf.conaffinity = 0
    # Far background fog -> a bright water colour so the region beyond the walls reads as water.
    spec.visual.rgba.fog[:] = [*lp.fog_rgb, 1.0]
    spec.visual.map.fogstart = lp.fogstart
    spec.visual.map.fogend = lp.fogend


def apply_perception_appearance(spec: mujoco.MjSpec, *, center_x: float, depth: float,
                                len_x: float, len_z: float, floor_y: float = 0.0,
                                hide_tethers: bool = False, add_surface: bool = True) -> None:
    """Apply the FULL shared perception appearance (balloon/pin/tether styling + pool lighting).

    Convenience wrapper used by ``tools/autonomy_run`` on the real competition pool. The dataset
    generator composes the two halves itself (it interleaves its own pool-enlargement resize).
    """
    style_balloons_pin_tethers(spec, hide_tethers=hide_tethers)
    brighten_like_pool(spec, center_x, depth, len_x, len_z, floor_y, add_surface=add_surface)


__all__ = [
    "BALLOON_ASPECT", "TETHER_RGBA", "TETHER_RADIUS", "LightingParams", "sample_lighting",
    "style_balloons_pin_tethers", "brighten_like_pool", "apply_perception_appearance",
]
