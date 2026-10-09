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


# ------------------------------------------------------------------------------------------------
# OPT-IN "pool0913" appearance profile (real pool footage 2026-09-13 / 10-01). Nothing above uses it;
# only ``tools/gen_sim_dataset --profile pool0913`` calls these. See docs/sim_pool_appearance.md.
#   * balloons: near-round (aspect ~1.0-1.15), glossy, bright latex; visible colours are the real
#     dyes (blue / crimson / orange-yellow). The pale-lavender look of blue in the footage is NOT a
#     paint colour: it comes from the Pi NoIR camera's near-infrared leak (see underwater_sim
#     ``add_nir_leak`` + ``POOL0913_NIR``), applied per material after rendering;
#   * a small knot under each balloon, sometimes a red plastic clip;
#   * tethers: thin DARK RED-BROWN wavy ribbons (box-segment chain) from the knot down to the floor.
# ------------------------------------------------------------------------------------------------
POOL0913_BALLOON_RGB = {   # base latex colour (before water + camera gain), jittered per balloon
    "red": (0.92, 0.04, 0.22),
    "blue": (0.42, 0.56, 0.78),   # a real BLUE dye; the NoIR camera's NIR leak makes it lavender
    "yellow": (1.00, 0.66, 0.04),
}
POOL0913_ASPECT = (1.0, 1.15)          # height/width of the ellipsoid, sampled per balloon
POOL0913_RIBBON_RGB = (0.36, 0.10, 0.08)  # dark red-brown ribbon
POOL0913_CLIP_PROB = 0.35
# Near-infrared reflectance per material class (0..1) for the NoIR leak model. Blue dyes are nearly
# transparent in NIR, so the latex reflects NIR strongly (-> pale lavender); red/yellow somewhat;
# white plaster walls/floor high; the dark ribbon low.
POOL0913_NIR = {"blue": 0.95, "red": 0.55, "yellow": 0.60, "ribbon": 0.15, "clip": 0.45,
                "pool": 0.55, "robot": 0.30}


def _quat_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) of the rotation whose columns are the given orthonormal axes."""
    mat = np.stack([x, y, z], axis=1).reshape(9)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def _add_ribbon(world, name: str, top: np.ndarray, bottom: np.ndarray, rng: np.random.Generator,
                rgb=POOL0913_RIBBON_RGB) -> None:
    """A wavy flat ribbon from ``top`` to ``bottom`` as a chain of thin twisted box segments."""
    length = float(np.linalg.norm(top - bottom))
    if length < 1e-3:
        return
    axis = (bottom - top) / length
    side = np.cross(axis, [0.0, 1.0, 0.0])
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
    side2 = np.cross(axis, side)
    amp = rng.uniform(0.008, 0.05)
    lam = rng.uniform(0.12, 0.45)
    ph = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    n = int(np.clip(length / 0.035, 4, 48))
    s = np.linspace(0.0, length, n + 1)
    # the wave dies out toward the balloon end (the knot holds it) and the anchor
    env = np.clip(np.sin(np.pi * s / length), 0.0, 1.0) ** 0.5
    pts = (top[None] + axis[None] * s[:, None]
           + side[None] * (amp * env * np.sin(2 * np.pi * s / lam + ph))[:, None]
           + side2[None] * (0.5 * amp * env * np.sin(2 * np.pi * s / (1.7 * lam) + ph2))[:, None])
    half_w = rng.uniform(0.0012, 0.0028)
    twist0 = rng.uniform(0, np.pi)
    rgba = [*np.clip(np.asarray(rgb) * rng.uniform(0.8, 1.2), 0, 1), 1.0]
    for k in range(n):
        a, b = pts[k], pts[k + 1]
        d = b - a
        seg = float(np.linalg.norm(d))
        if seg < 1e-6:
            continue
        d = d / seg
        tw = twist0 + 2.5 * s[k]
        ref = np.cos(tw) * side + np.sin(tw) * side2
        ref = ref - d * float(ref @ d)
        ref /= max(np.linalg.norm(ref), 1e-9)
        g = world.add_geom()
        g.name = f"{name}_ribbon{k}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = list((a + b) / 2)
        g.quat = list(_quat_from_axes(ref, d, np.cross(ref, d)))
        g.size = [half_w, seg / 2 + 0.002, 0.0006]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0


def style_pool0913(spec: mujoco.MjSpec, rng: np.random.Generator, heights: dict, floor_y: float,
                   radius: float) -> dict:
    """Apply the pool0913 balloon/knot/clip/ribbon look IN PLACE (opt-in profile).

    ``heights`` maps balloon body name -> centre height y [m] (overrides the scenario's per-colour
    heights, so balloons can float just under the surface). Returns ``{body_name: aspect}``.
    The original ``*_tether`` / ``*_weight`` geoms and the pin are hidden (alpha 0, physics intact).
    Only ``balloon_*_geom`` stays a balloon in segmentation: knots/clips/ribbons get other names.
    """
    aspects = {}
    world = spec.worldbody
    for b in spec.bodies:
        bname = b.name or ""
        if not bname.startswith("balloon_") or bname not in heights:
            continue
        colour = bname.split("_")[1]
        b.pos[1] = heights[bname]
        asp = float(rng.uniform(*POOL0913_ASPECT))
        aspects[bname] = asp
        rgb = np.clip(np.asarray(POOL0913_BALLOON_RGB[colour]) + rng.uniform(-0.05, 0.05, 3), 0, 1)
        mat = spec.add_material()
        mat.name = f"{bname}_latex"
        mat.rgba = [*rgb, 1.0]
        mat.specular = float(rng.uniform(0.3, 0.7))
        mat.shininess = float(rng.uniform(0.4, 0.8))
        mat.emission = float(rng.uniform(0.15, 0.40))   # translucent latex glows in sunlight
        for g in b.geoms:
            if (g.name or "").endswith("_geom"):
                g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
                g.size[:] = [radius, radius * asp, radius]
                g.material = mat.name
                g.rgba[:] = [*rgb, 1.0]
        # knot (same latex) + optional red plastic clip, hanging under the balloon (body frame)
        knot_y = -radius * asp - 0.008
        k = b.add_geom()
        k.name = f"{bname}_knot"
        k.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        k.size = [0.010, 0.014, 0.010]
        k.pos = [0.0, knot_y, 0.0]
        k.rgba = [*rgb * 0.85, 1.0]
        k.contype = k.conaffinity = 0
        bottom_y = knot_y - 0.014
        if rng.random() < POOL0913_CLIP_PROB:
            c = b.add_geom()
            c.name = f"{bname}_clip"
            c.type = mujoco.mjtGeom.mjGEOM_BOX
            c.size = [0.018, 0.008, 0.004]
            c.pos = [0.0, bottom_y - 0.008, 0.0]
            c.rgba = [0.85, 0.08, 0.10, 1.0]
            c.contype = c.conaffinity = 0
            bottom_y -= 0.016
        top = np.array([b.pos[0], b.pos[1] + bottom_y, b.pos[2]])
        anchor = np.array([b.pos[0] + rng.uniform(-0.25, 0.25), floor_y,
                           b.pos[2] + rng.uniform(-0.25, 0.25)])
        _add_ribbon(world, bname, top, anchor, rng)
    for g in spec.geoms:
        name = g.name or ""
        if name == "pin" or (name.startswith("balloon_") and name.endswith(("_tether", "_weight"))):
            g.rgba[3] = 0.0
    return aspects


def mirror_world_across(spec: mujoco.MjSpec, surface_y: float, keep_bodies=("base_link",)) -> None:
    """Reflect the whole visual world (world geoms, balloon bodies, lights) across the plane
    y = ``surface_y`` IN PLACE, so a render from the unchanged camera shows the scene's MIRROR image
    in the water surface (total internal reflection seen from below). The robot (``keep_bodies``)
    is not moved. Rotations map R -> M R M with M = diag(1,-1,1) (a proper rotation)."""
    M = np.diag([1.0, -1.0, 1.0])

    def _mirror_quat(q):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=float))
        r = M @ mat.reshape(3, 3) @ M
        out = np.zeros(4)
        mujoco.mju_mat2Quat(out, r.reshape(9))
        return out

    for g in spec.worldbody.geoms:
        if np.all(np.isfinite(g.fromto)):          # fromto-defined geom (e.g. the scenario tether)
            g.fromto[1] = 2 * surface_y - g.fromto[1]
            g.fromto[4] = 2 * surface_y - g.fromto[4]
            continue
        g.pos[1] = 2 * surface_y - g.pos[1]
        g.quat[:] = _mirror_quat(g.quat)
    for b in spec.worldbody.bodies:
        if (b.name or "") in keep_bodies:
            continue
        b.pos[1] = 2 * surface_y - b.pos[1]
        b.quat[:] = _mirror_quat(b.quat)
        for g in b.geoms:
            g.pos[1] = -g.pos[1]
            g.quat[:] = _mirror_quat(g.quat)
    for lt in spec.worldbody.lights:
        lt.pos[1] = 2 * surface_y - lt.pos[1]
        lt.dir[1] = -lt.dir[1]
        lt.castshadow = False


__all__ = [
    "BALLOON_ASPECT", "TETHER_RGBA", "TETHER_RADIUS", "LightingParams", "sample_lighting",
    "style_balloons_pin_tethers", "brighten_like_pool", "apply_perception_appearance",
    "POOL0913_BALLOON_RGB", "POOL0913_ASPECT", "POOL0913_RIBBON_RGB", "POOL0913_NIR", "style_pool0913",
    "mirror_world_across",
]
