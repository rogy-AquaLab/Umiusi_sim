"""Generate a labelled SYNTHETIC underwater balloon dataset (free COCO-labelled training data).

Pipeline, per frame:
  1. Randomize the world: ``sample_layout()`` scatters balloons + we randomize the robot base
     pose (x/z in the pool, height, yaw) so balloons appear at varied distances/angles/counts.
  2. Render three co-registered buffers from one camera: RGB, metric DEPTH, and SEGMENTATION
     (per-geom ids). (MuJoCo Renderer: enable_depth_rendering / enable_segmentation_rendering.)
  3. Degrade the clean RGB with the physically-based underwater model (``underwater_sim``) using
     the depth buffer — unless ``--clean`` (reference render, no degradation).
  4. Extract GROUND-TRUTH boxes from the segmentation: each balloon sphere geom -> pixel mask ->
     bbox; drop by size / occlusion / off-screen (see ``_boxes_from_seg``). Pixels never move, so
     the boxes are exact on the degraded image too.

Output: ``<out>/images/frame_XXXX.jpg`` (degraded RGB) + ``<out>/annotations.json`` (COCO, with
the same 3 categories as the real ai/balloon set: balloon_red=1, balloon_blue=2, balloon_yellow=3)
+ ``<out>/preview/frame_XXXX.jpg`` (first few frames, GT boxes drawn) for a human eyeball check.

Usage (headless):
    MUJOCO_GL=egl uv run python -m tools.gen_sim_dataset \
        --n 12 --out /tmp/umiusi_sim/simds --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.description import appearance as ra
from umiusi_sim.rendering import underwater_sim as us

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# colour -> COCO category id / name (matches ai/balloon: red=1, blue=2, yellow=3)
CATEGORIES = [
    {"id": 1, "name": "balloon_red", "supercategory": "balloon"},
    {"id": 2, "name": "balloon_blue", "supercategory": "balloon"},
    {"id": 3, "name": "balloon_yellow", "supercategory": "balloon"},
]
COLOUR_TO_CAT = {"red": 1, "blue": 2, "yellow": 3}
# BGR draw colours for preview boxes (cv2 draws in BGR)
DRAW_BGR = {"red": (40, 40, 220), "blue": (220, 120, 40), "yellow": (40, 210, 230)}

OCCLUSION_MAX = 0.80  # drop a box if >80% of its expected projected area is missing (occluded/clipped)
CLOSEUP_KEEP_FRAC = 0.02  # a frame-clipped balloon covering >=2% of the frame is a close-up -> KEEP it

# --- DISTANCE-LIMITED LABELS ---------------------------------------------------------------------
# Only balloons whose camera distance <= MAX_LABEL_RANGE are emitted as POSITIVE labels; balloons
# farther than this stay UNLABELED, so the detector trains them as background/negatives and learns
# NOT to fire on far/tiny balloons — baking the deployment range gate into the model and cutting the
# far false-positive source at the SOURCE. Set to ~7 m (NOT tighter): the robot still needs far
# balloons as coarse navigation cues to approach them, and the downstream tracker + range gate +
# size/range filter mop up residual far FPs — going blind past ~4-5 m is worse than a few far FPs.
# Configurable via --max-label-range (0 or negative disables the limit, e.g. for a full-range eval).
MAX_LABEL_RANGE = 7.0  # metres

# Balloon shape (ellipsoid aspect) is shared with the live perception render — see
# perception.render_appearance. Seg->bbox stays exact; _boxes_from_seg uses this aspect.
BALLOON_ASPECT = ra.BALLOON_ASPECT
# Bright, opaque pool-wall colour (dataset render): the raw scenery walls are low + translucent,
# leaving a black void above them; a real pool is an enclosed bright box.
WALL_RGBA = (0.50, 0.60, 0.66, 1.0)

# --- bigger, real-pool-scale GENERATED world (dataset-only; competition_run keeps its own layout) ---
# The scenario's default box is tight (8 x 5 m). For synthetic data we enlarge it a lot so the
# balloon field spreads over a real-pool-scale area -> more distance range -> distant balloons +
# a surface full of their reflections. Walls/floor/ceiling are moved out to CONTAIN the whole field
# (no balloon outside the walls). Water depth is kept ~3.3 m (scn.POOL_DEPTH).
POOL_CENTER_X = 7.0     # was 2.0
POOL_LEN_X = 18.0       # forward extent (+X), was 8.0
POOL_LEN_Z = 12.0       # lateral extent (±Z), was 5.0
FIELD_X = (0.8, 14.0)   # balloon x-range (inside the walls, ahead of the start pose)
FIELD_Z = (-5.0, 5.0)   # balloon z-range (inside the ±6 m walls)
BALLOON_COUNT = (12, 30)                       # randomized number of balloons per frame
COLOUR_WEIGHTS = {"red": 0.4, "blue": 0.3, "yellow": 0.3}  # realistic competition mix


def sample_big_layout(rng: np.random.Generator):
    """Randomized real-pool-scale layout: 12-30 balloons at varied x/z across the enlarged field,
    with a realistic red/blue/yellow mix. Returns a BALLOON_LAYOUT-shaped list of (name, colour, x, z)
    with UNIQUE names (index-suffixed) so seg->box auto-labelling stays 1:1. Heights come from
    BALLOON_SPECS per colour (via build_spec). competition_run's own sample_layout is untouched.

    Arrangement DR: each frame is either SCATTERED (uniform over the field) or CLUSTERED (a few
    cluster centres with balloons tightly grouped around them, plus a sparse uniform sprinkle) — so
    the detector sees both spread-out and bunched-up spacing/counts (overlap, occlusion variety).
    """
    n = int(rng.integers(BALLOON_COUNT[0], BALLOON_COUNT[1] + 1))
    colours = list(COLOUR_WEIGHTS)
    probs = np.array([COLOUR_WEIGHTS[c] for c in colours], dtype=float)
    probs /= probs.sum()

    clustered = rng.random() < 0.45
    if clustered:
        n_clusters = int(rng.integers(2, 5))
        centres = [(float(rng.uniform(*FIELD_X)), float(rng.uniform(*FIELD_Z)))
                   for _ in range(n_clusters)]
        spread = float(rng.uniform(0.5, 1.8))   # cluster tightness [m]

    layout = []
    for i in range(n):
        c = colours[int(rng.choice(len(colours), p=probs))]
        if clustered and rng.random() < 0.8:    # 80% of balloons snap to a cluster, 20% sprinkle
            cxx, czz = centres[int(rng.integers(len(centres)))]
            x = float(np.clip(cxx + rng.normal(0, spread), *FIELD_X))
            z = float(np.clip(czz + rng.normal(0, spread), *FIELD_Z))
        else:
            x = float(rng.uniform(*FIELD_X))
            z = float(rng.uniform(*FIELD_Z))
        layout.append((f"balloon_{c}_{i}", c, x, z))
    return layout


SURFACE_Y = scn.POOL_DEPTH  # water-surface plane height [m] (=3.3); reflections mirror across this


def prep_render_spec(spec: mujoco.MjSpec, hide_tethers: bool = False, mirror: bool = False,
                     lighting=None, wall_rgba=None, floor_rgb=None, headlamp: float | None = None) -> None:
    """Mutate a composed MjSpec IN PLACE for the rendered dataset (competition_balloon.py stays
    untouched — these edits happen only here, between build_spec and compile):

      * balloon spheres -> vertical ellipsoids (egg-shaped, aspect BALLOON_ASPECT);
      * hide the base_link pin (foreground needle, not in a real onboard view) via alpha=0 —
        the geom stays in the model so competition_run pop-detection is unaffected;
      * make tethers subtle (thin, low-contrast, low alpha) or hide them entirely.

    ``mirror=True`` builds the REFLECTION model: every balloon body is reflected across the water
    surface plane (y=SURFACE_Y) and the opaque surface-ceiling is omitted, so a render from the SAME
    camera shows the geometrically-correct mirror image of the balloons (the reflection). Tethers are
    hidden in the mirror model (their mirror is meaningless).
    """
    hide_tethers = hide_tethers or mirror
    # Shared appearance (balloons -> ellipsoids, hide pin, subtle/hidden tethers). Same styling the
    # live competition tool applies — see perception.render_appearance.
    ra.style_balloons_pin_tethers(spec, hide_tethers=hide_tethers)
    # Dataset-only: enlarge the pool (floor/water/walls) to the training-field scale.
    wall_rgba = WALL_RGBA if wall_rgba is None else wall_rgba
    for g in spec.geoms:
        name = g.name or ""
        if name == "pool_floor":
            g.pos[:] = [POOL_CENTER_X, scn.FLOOR_Y - 0.02, 0.0]
            g.size[:] = [POOL_LEN_X / 2, 0.02, POOL_LEN_Z / 2]
            if floor_rgb is not None:
                g.rgba[:3] = floor_rgb
        elif name == "pool_water":
            g.pos[:] = [POOL_CENTER_X, scn.FLOOR_Y + scn.POOL_DEPTH / 2, 0.0]
            g.size[:] = [POOL_LEN_X / 2, scn.POOL_DEPTH / 2, POOL_LEN_Z / 2]
        elif name.startswith("pool_wall_"):
            _resize_wall(g, name, wall_rgba)
    if mirror:
        # Reflect each balloon body across the surface plane: y -> 2*SURFACE_Y - y (lands above the
        # surface, i.e. where its reflection appears when viewed from below).
        for b in spec.bodies:
            if (b.name or "").startswith("balloon_"):
                b.pos[1] = 2 * SURFACE_Y - b.pos[1]
    # Shared sunlit-pool lighting (randomized per frame via ``lighting``), over the ENLARGED training
    # field. The surface geom is named 'perception_surface' (used below for the reflection mask).
    ra.brighten_like_pool(spec, POOL_CENTER_X, scn.POOL_DEPTH, POOL_LEN_X, POOL_LEN_Z,
                          scn.FLOOR_Y, add_surface=not mirror, lighting=lighting)
    # Robot's own forward light cone (near-range shading cue). Not on the mirror/reflection scene.
    if headlamp is not None and not mirror:
        add_headlamp(spec, diffuse=headlamp)


# --- robot headlamp (near-range light cone) ------------------------------------------------------
# The vehicle carries its own forward light. Attached to base_link at the camera offset, pointing +X
# (the camera look axis), with distance ATTENUATION so it lights NEAR balloons strongly and fades to
# nothing on the far field — exactly the operating-range shading cue the detector should rely on.
# Dataset-only (added here, not in the shared render_appearance), so live-inference appearance and
# competition physics are untouched.
HEADLAMP_PROB = 0.6                 # fraction of frames the headlamp is on
HEADLAMP_DIFFUSE = (0.35, 1.05)     # per-frame lamp brightness


def add_headlamp(spec: mujoco.MjSpec, diffuse: float, cutoff_deg: float = 38.0) -> None:
    """Attach a forward SPOT light to base_link at the front_cam offset (robot's own light cone)."""
    base = spec.body("base_link")
    lt = base.add_light()
    lt.name = "perception_headlamp"
    lt.type = mujoco.mjtLightType.mjLIGHT_SPOT
    lt.pos = [float(CAM_OFFSET[0]), float(CAM_OFFSET[1]), float(CAM_OFFSET[2])]
    lt.dir = [1.0, 0.0, 0.0]                       # +X = camera look axis (base frame)
    lt.diffuse = [diffuse, diffuse, diffuse * 0.97]  # near-white, a hair warm
    lt.specular = [0.0, 0.0, 0.0]
    lt.cutoff = float(cutoff_deg)
    lt.exponent = 12.0
    lt.attenuation = [0.5, 0.7, 1.1]               # 1/(c0+c1 d+c2 d^2): near lit, far dark
    lt.castshadow = False


# --- mild lens distortion (near-range DR) --------------------------------------------------------
# A cheap wide-angle lens barrel/pincushion-distorts the image. We warp the RGB, DEPTH and SEG buffers
# with the SAME map (nearest for depth/seg) BEFORE box extraction, so boxes recomputed from the warped
# seg stay EXACT. Mild (|k1| small) and applied to a fraction of frames.
LENS_DISTORT_PROB = 0.45
LENS_K1 = 0.14  # max |k1| radial coefficient (barrel<0 / pincushion>0), normalised-radius model


def _lens_distort_maps(h: int, w: int, k1: float):
    """cv2.remap sample maps for a radial distortion with coefficient ``k1`` (normalised radius)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1.0 + k1 * r2
    map_x = (cx + nx * f * cx).astype(np.float32)
    map_y = (cy + ny * f * cy).astype(np.float32)
    return map_x, map_y


def _apply_distort(rgb, depth, seg, map_x, map_y):
    """Warp co-registered RGB(bilinear)/DEPTH(nearest)/SEG(nearest) by a shared distortion map."""
    rgb_d = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    depth_d = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    seg_d = cv2.remap(seg, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    if seg_d.ndim == 2:                    # remap may drop a size-1 channel axis
        seg_d = seg_d[..., None]
    return rgb_d, depth_d, seg_d


def _resize_wall(g, name: str, wall_rgba=WALL_RGBA) -> None:
    """Move + resize a scenery wall to the enlarged pool extent, full water depth, bright + opaque,
    so the whole (bigger) balloon field is contained and the frame is filled with pool."""
    hy = scn.POOL_DEPTH                       # full-height walls
    y = scn.FLOOR_Y + hy / 2
    if name.endswith(("xpos", "xneg")):       # walls spanning ±Z at the ends of +X
        sign = 1.0 if name.endswith("xpos") else -1.0
        g.pos[:] = [POOL_CENTER_X + sign * POOL_LEN_X / 2, y, 0.0]
        g.size[:] = [0.02, hy / 2, POOL_LEN_Z / 2]
    else:                                     # walls spanning ±X at the ends of Z
        sign = 1.0 if name.endswith("zpos") else -1.0
        g.pos[:] = [POOL_CENTER_X, y, sign * POOL_LEN_Z / 2]
        g.size[:] = [POOL_LEN_X / 2, hy / 2, 0.02]
    g.rgba[:] = wall_rgba


# Camera pitch is sampled across three buckets so conditions span looking UP (toward the surface —
# sees reflections), LEVEL, and DOWN (toward the floor). Positive pitch (about lateral +Z) tips the
# +X-looking front_cam UP toward the surface.
PITCH_BUCKETS = {
    "up": (12.0, 45.0),     # look toward the surface -> surface + reflections prominent
    "level": (-8.0, 8.0),   # roughly horizontal
    "down": (-45.0, -14.0),  # look toward the floor
}

# Extreme close-up ("about to ram") frames: place the camera right in front of one balloon so it
# fills much of the frame. The real robot approaches until a balloon is this close, so the detector
# (and eval) must cover it — including the frame-clipped case. Raised from 0.30 to bias the training
# mix toward the OPERATING range (near detection is what the mission needs).
CLOSEUP_PROB = 0.42
# camera-to-balloon-CENTRE distance [m]. Balloon radius is 0.10 m: stay well outside it (and outside
# the renderer near-plane) while filling much of the frame — 0.25 m ~= 70% frame height, 0.6 m ~= 30%.
CLOSEUP_DIST = (0.25, 0.6)
# NEAR-APPROACH frames: a softer approach than a ram close-up — the camera is 0.8-2.8 m behind a target
# balloon (still looking at it) so it sits at operating range with OTHER balloons also in view. Fills
# the near-distance band between ram close-ups and the wide field, where the pop happens. Applied to a
# fraction of the front_cam frames that aren't ram close-ups.
NEAR_APPROACH_PROB = 0.30
NEAR_APPROACH_DIST = (0.8, 2.8)
CAM_OFFSET = np.array([0.10, 0.12, 0.0])  # front_cam pos in the base frame (from umiusi.xml)


def _closeup_pose(data: mujoco.MjData, rng: np.random.Generator, layout) -> dict:
    """Place the base so the CAMERA is ~0.25-0.6 m in front of one balloon (ram-approach view).

    Base orientation is identity (base +X = world +X = camera look axis, +Y = up), and the base is
    positioned so the front_cam (offset CAM_OFFSET, looking +X) sits ``cc`` m in world -X of the target
    balloon -> the balloon is dead-centre and fills much of the frame. A small yaw/pitch jitter adds
    variety while keeping it in view.
    """
    _, colour, bx, bz = layout[int(rng.integers(len(layout)))]
    by = float(scn.BALLOON_SPECS[colour]["height"])
    cc = float(rng.uniform(*CLOSEUP_DIST))
    # camera world pose: at (bx-cc, by, bz) looking +X; base = camera - CAM_OFFSET (identity base).
    data.qpos[0:3] = [bx - cc - CAM_OFFSET[0], by - CAM_OFFSET[1], bz - CAM_OFFSET[2]]
    yaw = float(rng.uniform(-0.10, 0.10))    # +-6 deg heading; balloon stays near centre at this range
    pitch = float(rng.uniform(-0.08, 0.08))
    q = Rot.from_euler("yz", [yaw, pitch]).as_quat()  # about +Y then +Z; [x,y,z,w]
    data.qpos[3:7] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    return {"pitch_bucket": "level", "pitch_deg": round(float(np.degrees(pitch)), 1),
            "roll_deg": 0.0, "closeup": True, "dist_bucket": "closeup"}


def _near_approach_pose(data: mujoco.MjData, rng: np.random.Generator, layout) -> dict:
    """Place the CAMERA 0.8-2.8 m behind a target balloon (operating-range approach, other balloons
    still in view). Like ``_closeup_pose`` but farther and with more yaw/pitch/height jitter so the
    target isn't always dead-centre — fills the near-distance band where the pop actually happens.
    """
    _, colour, bx, bz = layout[int(rng.integers(len(layout)))]
    by = float(scn.BALLOON_SPECS[colour]["height"])
    cc = float(rng.uniform(*NEAR_APPROACH_DIST))
    yaw = float(rng.uniform(-0.22, 0.22))     # +-13 deg heading (target near, not exactly, centre)
    pitch = float(rng.uniform(-0.20, 0.20))
    dy = float(rng.uniform(-0.25, 0.25))      # small height offset so it isn't perfectly level
    data.qpos[0:3] = [bx - cc - CAM_OFFSET[0], by - CAM_OFFSET[1] + dy, bz - CAM_OFFSET[2]]
    q = Rot.from_euler("yz", [yaw, pitch]).as_quat()  # about +Y then +Z; [x,y,z,w]
    data.qpos[3:7] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    return {"pitch_bucket": "level", "pitch_deg": round(float(np.degrees(pitch)), 1),
            "roll_deg": 0.0, "closeup": False, "dist_bucket": "near"}


def randomize_base_pose(data: mujoco.MjData, rng: np.random.Generator, camera: str, layout=None) -> dict:
    """Randomize the free base pose so the balloon field is seen from varied viewpoints.

    Frame: +Y up, forward = +X. ``front_cam`` looks +X (down the balloon run); ``down_cam`` is
    nadir. Randomizes position (x/z/height), yaw, plus PITCH (up/level/down bucket) and a little roll
    so conditions span near/far and looking-up/level/down; ~30% of front_cam frames are extreme
    close-ups (ram approach). Returns a condition dict for tagging.
    """
    if camera == "front_cam" and layout:
        if rng.random() < CLOSEUP_PROB:
            return _closeup_pose(data, rng, layout)
        if rng.random() < NEAR_APPROACH_PROB:
            return _near_approach_pose(data, rng, layout)
    x = float(rng.uniform(-1.5, 3.0))              # along the (long) run, behind/among the near balloons
    z = float(rng.uniform(-3.0, 3.0))              # lateral, within the (wider) pool
    if camera == "down_cam":
        y = float(rng.uniform(1.6, 3.0))
        yaw = float(rng.uniform(-np.pi, np.pi))
        pitch_bucket, pitch, roll = "down", -90.0, 0.0  # nadir camera is inherently looking down
        data.qpos[0:3] = [x, y, z]
        data.qpos[3:7] = [np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0]
        return {"pitch_bucket": pitch_bucket, "pitch_deg": pitch, "roll_deg": roll,
                "closeup": False, "dist_bucket": "field"}

    y = float(rng.uniform(0.6, 2.8))               # mid-water height
    yaw = float(rng.uniform(-0.6, 0.6))            # +/-34 deg heading; keep balloons in view
    pitch_bucket = rng.choice(list(PITCH_BUCKETS))  # up / level / down, equally likely
    pitch = float(rng.uniform(*PITCH_BUCKETS[pitch_bucket]))
    roll = float(rng.uniform(-8.0, 8.0))           # a little roll
    data.qpos[0:3] = [x, y, z]
    # Compose base orientation (intrinsic, body frame): yaw about +Y (up), then pitch about the new
    # lateral +Z, then roll about the new forward +X — natural "turn, tilt, roll" camera aiming.
    quat_xyzw = Rot.from_euler("yzx", [yaw, np.radians(pitch), np.radians(roll)]).as_quat()
    data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # -> [w, x, y, z]
    return {"pitch_bucket": pitch_bucket, "pitch_deg": round(pitch, 1), "roll_deg": round(roll, 1),
            "closeup": False, "dist_bucket": "field"}


def _geom_colour(model: mujoco.MjModel, gid: int) -> str | None:
    """Return the balloon colour for a segmentation geom id, or None if it's not a balloon sphere."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    if not name or not name.endswith("_geom") or not name.startswith("balloon_"):
        return None
    # name like 'balloon_red_1_geom' / 'balloon_yellow_start_geom' -> colour is the 2nd token
    colour = name.split("_")[1]
    return colour if colour in COLOUR_TO_CAT else None


def _boxes_from_seg(model, seg, depth, fpx, min_area_px, max_label_range=MAX_LABEL_RANGE):
    """Extract GT balloon boxes from the segmentation buffer.

    For each balloon sphere geom present:
      * mask = pixels whose seg objid == geom id (already the VISIBLE mask — occluders overwrite it)
      * bbox = [x, y, w, h] from the mask extent, clipped to the frame
      * expected full projected area from the sphere: pi * (fpx * R / d)^2, d = median mask depth
      * occlusion/clip fraction = 1 - visible_px / full_area
    Drop rules -> a box is kept only if: colour is a balloon, visible bbox area >= min_area_px,
    occlusion fraction <= OCCLUSION_MAX (this also catches boxes mostly off-screen, since a clipped
    balloon has visible_px << full_area), AND distance <= ``max_label_range`` (far balloons stay
    UNLABELED background; set <=0 to disable the range limit).

    Returns (boxes, drops) where boxes = list of dicts and drops = counts by reason.
    """
    h, w = seg.shape[:2]
    objid = seg[..., 0]
    boxes = []
    drops = {"size": 0, "occluded": 0, "offscreen": 0, "far": 0}
    for gid in np.unique(objid):
        if gid < 0:
            continue
        colour = _geom_colour(model, gid)
        if colour is None:
            continue
        mask = objid == gid
        n_vis = int(mask.sum())
        if n_vis == 0:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        # expected full projected area of the (ellipsoid) balloon at its distance (occlusion/clip
        # test). Vertical major axis => taller than wide: area ~ pi * a * b.
        d = float(np.median(depth[mask]))
        a_px = fpx * scn.BALLOON_RADIUS / max(d, 1e-3)
        b_px = a_px * BALLOON_ASPECT
        full_area = np.pi * a_px * b_px
        occ = 1.0 - n_vis / max(full_area, 1.0)
        # off-screen: the balloon touches a frame border AND most of it is missing
        touches_border = x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1
        # A big balloon clipped by the FRAME edge (a ram-approach close-up) must be KEPT: frame
        # clipping inflates `occ` but is not real occlusion. Keep border boxes that cover a large
        # area; still drop small border slivers (far, off-screen) and interior high-occ boxes
        # (genuinely occluded by another balloon).
        closeup_keep = touches_border and n_vis >= CLOSEUP_KEEP_FRAC * w * h
        if bw * bh < min_area_px:
            drops["size"] += 1
            continue
        if occ > OCCLUSION_MAX and not closeup_keep:
            drops["offscreen" if touches_border else "occluded"] += 1
            continue
        # DISTANCE-LIMITED LABEL: a genuinely far balloon is left UNLABELED (trains as background) so
        # the detector learns not to fire on far/tiny balloons. Close-up frame-clipped balloons are
        # near by definition (kept). max_label_range <= 0 disables the limit (full-range labels).
        if max_label_range > 0 and d > max_label_range and not closeup_keep:
            drops["far"] += 1
            continue
        boxes.append({
            "colour": colour,
            "category_id": COLOUR_TO_CAT[colour],
            "bbox": [x0, y0, bw, bh],
            "area": n_vis,          # visible pixel (segmentation) area
            "occlusion": round(occ, 3),
            "distance_m": round(d, 3),
        })
    return boxes, drops


def _geom_only_option() -> mujoco.MjvOption:
    """A visualisation option that renders ONLY real geoms — no decorations.

    Sites (the red ``pin_tip`` marker), camera/light glyphs, contact points, inertia/COM boxes,
    joints, actuators, etc. are all disabled so they never appear in the RGB, depth, OR
    segmentation buffers (a decoration in seg would otherwise inject a bogus object id).
    """
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0  # sites have no vis-flag; they are gated per group (all off => no sites)
    for f in (
        mujoco.mjtVisFlag.mjVIS_CAMERA, mujoco.mjtVisFlag.mjVIS_LIGHT,
        mujoco.mjtVisFlag.mjVIS_JOINT, mujoco.mjtVisFlag.mjVIS_ACTUATOR,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_INERTIA, mujoco.mjtVisFlag.mjVIS_COM,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT, mujoco.mjtVisFlag.mjVIS_PERTFORCE,
    ):
        opt.flags[f] = 0
    return opt


def render_buffers(renderer, model, data, camera, opt):
    """Render RGB, metric depth, and per-geom segmentation from one camera (shared scene).

    ``opt`` (see ``_geom_only_option``) strips all non-geom decorations from every pass.
    """
    renderer.update_scene(data, camera=camera, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1  # fog only on the RGB pass
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=opt)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera, scene_option=opt)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, depth, seg


def render_reflection_mask(layout, base_qpos, camera, opt, width, height,
                           lighting=None, wall_rgba=None, floor_rgb=None):
    """Build + render the MIRRORED-balloon scene from the same camera pose; return (rgb, mask).

    The scene is the same layout with every balloon reflected across the water surface (y=SURFACE_Y)
    and no opaque ceiling — so this render, seen by the SAME camera, is the geometrically-correct
    reflection of the balloons (correct perspective, anchored to the surface line). ``mask`` marks the
    pixels covered by a (mirrored) balloon, from segmentation.
    """
    spec_ref = scn.build_spec(layout)
    prep_render_spec(spec_ref, mirror=True, lighting=lighting, wall_rgba=wall_rgba,
                     floor_rgb=floor_rgb)
    model_ref = spec_ref.compile()
    data_ref = mujoco.MjData(model_ref)
    data_ref.qpos[0:7] = base_qpos[0:7]  # identical base/camera pose
    mujoco.mj_forward(model_ref, data_ref)
    r = mujoco.Renderer(model_ref, height=height, width=width)
    try:
        r.update_scene(data_ref, camera=camera, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(data_ref, camera=camera, scene_option=opt)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
    finally:
        r.close()
    mask = np.zeros(seg.shape[:2], dtype=bool)
    for gid in np.unique(seg[..., 0]):
        if gid >= 0 and _geom_colour(model_ref, gid):
            mask |= seg[..., 0] == gid
    return rgb, mask


def draw_preview(rgb, boxes):
    """Draw GT boxes + colour labels onto a copy of the (degraded) RGB for eyeballing."""
    img = np.ascontiguousarray(rgb[..., ::-1])  # RGB -> BGR for cv2
    for b in boxes:
        x, y, bw, bh = b["bbox"]
        c = DRAW_BGR[b["colour"]]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), c, 2)
        label = f"{b['colour']} {b['distance_m']:.1f}m"
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return img[..., ::-1]  # back to RGB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="number of frames")
    ap.add_argument("--out", type=Path, required=True, help="output dataset dir")
    ap.add_argument("--camera", default="front_cam", choices=["front_cam", "down_cam"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area-px", type=int, default=64, help="drop boxes smaller than this (w*h)")
    ap.add_argument("--max-label-range", type=float, default=MAX_LABEL_RANGE,
                    help="only balloons within this many metres are labelled POSITIVE; farther ones "
                         "stay unlabelled background (bakes in the range gate). <=0 disables the limit "
                         "(full-range labels — use for a fair eval set).")
    ap.add_argument("--clean", action="store_true", help="skip degradation (reference render)")
    ap.add_argument("--hide-tethers", action="store_true", help="hide balloon tethers entirely")
    ap.add_argument("--force-reflection", type=float, default=None, metavar="STRENGTH",
                    help="force the water-surface reflection ON at this strength every frame "
                         "(e.g. 0.75) — for the reflection demo")
    ap.add_argument("--preview-n", type=int, default=8, help="how many frames to also save as previews")
    args = ap.parse_args()

    if cv2 is None:
        print("WARNING: cv2 not available; previews will be skipped (install the 'perception' extra).")

    # Two independent streams so scene geometry is reproducible regardless of degradation: the
    # scene stream drives layout+pose (=> --clean and degraded produce IDENTICAL scenes/boxes),
    # the water stream drives the degradation params/noise only.
    ss = np.random.SeedSequence(args.seed)
    scene_rng = np.random.default_rng(ss.spawn(1)[0])
    water_rng = np.random.default_rng(ss.spawn(1)[0])
    rng = scene_rng
    img_dir = args.out / "images"
    prev_dir = args.out / "preview"
    img_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    total_boxes = {"red": 0, "blue": 0, "yellow": 0}
    total_drops = {"size": 0, "occluded": 0, "offscreen": 0, "far": 0}
    depth_lo, depth_hi = np.inf, -np.inf
    seen_balloon_ids = set()
    scene_opt = _geom_only_option()  # geoms-only (no sites/decorations) for every render pass

    total_pitch = {"up": 0, "level": 0, "down": 0}
    total_refl_frames = 0
    total_closeups = 0
    total_near = 0
    total_headlamp = 0
    total_distort = 0
    for i in range(args.n):
        layout = sample_big_layout(rng)
        # Per-frame LIGHTING + scene-tone domain randomisation (biggest gap after the colour fix):
        # randomised overhead-light direction/intensity/colour-temp + ambient fill + surface glare,
        # and randomised wall/floor tone. Drawn from the SCENE stream so --clean matches degraded.
        lighting = ra.sample_lighting(rng)
        wtone = float(rng.uniform(0.35, 0.72))
        wtint = np.array([rng.uniform(0.9, 1.1), 1.0, rng.uniform(0.9, 1.1)])
        wall_rgba = (*np.clip(wtone * wtint, 0.10, 0.90), 1.0)
        floor_rgb = tuple(np.clip(float(rng.uniform(0.30, 0.62)) * wtint, 0.10, 0.85))
        # Near-range DR drawn from the SCENE stream (so --clean matches degraded): robot headlamp
        # (forward light cone) present ~HEADLAMP_PROB of frames, and a mild lens distortion k1.
        headlamp = float(rng.uniform(*HEADLAMP_DIFFUSE)) if rng.random() < HEADLAMP_PROB else None
        distort_k1 = (float(rng.uniform(-LENS_K1, LENS_K1))
                      if cv2 is not None and rng.random() < LENS_DISTORT_PROB else None)
        spec = scn.build_spec(layout)
        prep_render_spec(spec, hide_tethers=args.hide_tethers, lighting=lighting,
                         wall_rgba=wall_rgba, floor_rgb=floor_rgb, headlamp=headlamp)
        model = spec.compile()
        # Small near-clip plane so extreme close-up balloons (~0.2 m) aren't clipped. The default
        # near plane scales with the (now large) scene extent -> ~0.9 m, which would hide ram close-ups.
        model.vis.map.znear = min(0.01, 0.02 / max(model.stat.extent, 1e-3))
        data = mujoco.MjData(model)
        pose_cond = randomize_base_pose(data, rng, args.camera, layout)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        try:
            rgb, depth, seg = render_buffers(renderer, model, data, args.camera, scene_opt)
        finally:
            renderer.close()

        # Mild lens distortion: warp RGB/DEPTH/SEG by a shared radial map BEFORE box extraction so the
        # seg-derived boxes stay exact on the distorted image. Maps reused for the reflection composite.
        dmap = None
        if distort_k1 is not None:
            dmap = _lens_distort_maps(args.height, args.width, distort_k1)
            rgb, depth, seg = _apply_distort(rgb, depth, seg, *dmap)

        # focal length in pixels from the vertical FOV (for the occlusion/projection test)
        cam_id = model.camera(args.camera).id
        fovy = np.radians(model.cam_fovy[cam_id])
        fpx = (args.height / 2.0) / np.tan(fovy / 2.0)

        # depth stats over the finite scene (background far-plane excluded)
        finite = depth[depth < 50.0]
        if finite.size:
            depth_lo = min(depth_lo, float(finite.min()))
            depth_hi = max(depth_hi, float(finite.max()))
        for gid in np.unique(seg[..., 0]):
            if gid >= 0 and _geom_colour(model, gid):
                seen_balloon_ids.add(int(gid))

        boxes, drops = _boxes_from_seg(model, seg, depth, fpx, args.min_area_px,
                                       max_label_range=args.max_label_range)
        for k in total_drops:
            total_drops[k] += drops[k]
        total_pitch[pose_cond["pitch_bucket"]] += 1
        total_closeups += int(pose_cond.get("closeup", False))
        total_near += int(pose_cond.get("dist_bucket") == "near")
        total_headlamp += int(headlamp is not None)
        total_distort += int(distort_k1 is not None)

        # water condition (sampled at a uniform 'murk' difficulty) + optional forced reflection
        params = us.random_params(water_rng)
        if args.force_reflection is not None:
            params.reflection = args.force_reflection

        # GEOMETRIC water-surface reflection: composite the mirrored-camera render onto the clean RGB,
        # anchored to where the surface is actually visible (correct perspective + boundary). Only
        # when the surface is in view (looking up/level) and reflection is on. Unlabelled (no boxes).
        surf_id = model.geom("perception_surface").id
        surface_mask = seg[..., 0] == surf_id
        reflected = False
        if params.reflection > 0 and surface_mask.any():
            refl_rgb, refl_balloon = render_reflection_mask(
                layout, data.qpos, args.camera, scene_opt, args.width, args.height,
                lighting=lighting, wall_rgba=wall_rgba, floor_rgb=floor_rgb)
            if dmap is not None:  # warp the reflection buffers by the same map so they stay aligned
                refl_rgb = cv2.remap(refl_rgb, dmap[0], dmap[1], cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
                refl_balloon = cv2.remap(refl_balloon.astype(np.uint8), dmap[0], dmap[1],
                                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(bool)
            reflect_mask = surface_mask & refl_balloon
            if reflect_mask.any():
                rgb = us.apply_surface_reflection(
                    rgb, refl_rgb, reflect_mask, params.B, params.reflection, water_rng)
                reflected = True
                total_refl_frames += 1

        out_rgb = rgb if args.clean else us.degrade(rgb, depth, params, water_rng)

        fname = f"frame_{i:04d}.jpg"
        _imwrite(img_dir / fname, out_rgb)

        dists = [b["distance_m"] for b in boxes]
        condition = {
            "pitch_bucket": pose_cond["pitch_bucket"],
            "pitch_deg": pose_cond["pitch_deg"],
            "roll_deg": pose_cond["roll_deg"],
            "closeup": pose_cond.get("closeup", False),
            "dist_bucket": pose_cond.get("dist_bucket", "field"),
            "n_balloons": len(boxes),
            "median_dist_m": round(float(np.median(dists)), 2) if dists else None,
            "murk": None if args.clean else round(params.murk, 3),
            "turbidity": None if args.clean else round(params.turbidity, 3),
            "cast": None if args.clean else round(params.cast, 3),
            "cast_sat": None if args.clean else round(params.cast_sat, 3),
            "cast_bucket": None if args.clean else us.cast_bucket(params.cast),
            "light_level": lighting.light_level,
            "sun_tilt_deg": lighting.sun_tilt_deg,
            "reflection": round(params.reflection, 3) if reflected else 0.0,
            "headlamp": round(headlamp, 3) if headlamp is not None else 0.0,
            "distort_k1": round(distort_k1, 3) if distort_k1 is not None else 0.0,
            "motion_blur": None if args.clean else round(params.motion_blur, 2),
            "vignette": None if args.clean else round(params.vignette, 3),
        }
        coco["images"].append({
            "id": i + 1, "file_name": fname, "width": args.width, "height": args.height,
            "condition": condition,
        })
        for b in boxes:
            total_boxes[b["colour"]] += 1
            coco["annotations"].append({
                "id": ann_id, "image_id": i + 1, "category_id": b["category_id"],
                "bbox": [float(v) for v in b["bbox"]], "area": float(b["area"]),
                "iscrowd": 0, "occlusion": b["occlusion"], "distance_m": b["distance_m"],
            })
            ann_id += 1

        if cv2 is not None and i < args.preview_n:
            prev = draw_preview(out_rgb, boxes)
            _imwrite(prev_dir / fname, prev)

        print(f"frame {i:04d}: kept={len(boxes):2d} "
              f"drops(sz={drops['size']},occ={drops['occluded']},off={drops['offscreen']},"
              f"far={drops['far']}) "
              f"look={pose_cond['pitch_bucket']:5s}({pose_cond['pitch_deg']:+.0f}deg) "
              f"murk={condition['murk']} cast={condition['cast']}({condition['cast_bucket']}) "
              f"refl={reflected}")

    with open(args.out / "annotations.json", "w") as f:
        json.dump(coco, f, indent=1)

    print("\n=== dataset summary ===")
    print(f"frames: {args.n}  camera: {args.camera}  {args.width}x{args.height}  "
          f"mode: {'CLEAN' if args.clean else 'DEGRADED'}")
    print(f"boxes by colour: red={total_boxes['red']} blue={total_boxes['blue']} "
          f"yellow={total_boxes['yellow']}  total={sum(total_boxes.values())}")
    print(f"boxes DROPPED: size(<{args.min_area_px}px)={total_drops['size']} "
          f"occluded={total_drops['occluded']} offscreen={total_drops['offscreen']} "
          f"far(>{args.max_label_range:g}m)={total_drops['far']}  "
          f"total={sum(total_drops.values())}")
    if np.isfinite(depth_lo):
        print(f"depth buffer (scene, m): min={depth_lo:.3f} max={depth_hi:.3f}  "
              f"[far-plane background >50 m excluded]")
    print(f"pitch buckets: up={total_pitch['up']} level={total_pitch['level']} down={total_pitch['down']}"
          f"   close-ups: {total_closeups}/{args.n}   near-approach: {total_near}/{args.n}"
          f"   surface reflection: {total_refl_frames}/{args.n}")
    print(f"near-range DR: headlamp={total_headlamp}/{args.n}  lens-distort={total_distort}/{args.n}"
          f"   label range: <= {args.max_label_range:g} m")
    print(f"unique balloon seg geom ids seen: {sorted(seen_balloon_ids)}")
    print(f"wrote: {img_dir}/  {args.out/'annotations.json'}  previews: {prev_dir}/")
    return 0


def _imwrite(path: Path, rgb: np.ndarray) -> None:
    """Write an (H,W,3) uint8 RGB array as JPEG (cv2 if present, else imageio)."""
    if cv2 is not None:
        cv2.imwrite(str(path), rgb[..., ::-1])  # RGB -> BGR
    else:  # pragma: no cover
        import imageio
        imageio.imwrite(path, rgb)


if __name__ == "__main__":
    raise SystemExit(main())
