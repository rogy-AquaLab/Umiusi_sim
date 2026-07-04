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
    MUJOCO_GL=egl uv run --extra perception python -m tools.gen_sim_dataset \
        --n 12 --out /tmp/umiusi_sim/simds --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.perception import underwater_sim as us

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

# Real competition balloons are egg/teardrop-shaped (taller than wide) — matched to
# ai/balloon/train2017/11.jpg (clear near-field teardrops). We approximate them as ELLIPSOIDS
# with a vertical (+Y) major axis at this aspect (height/width). ~1.25x. Seg->bbox stays exact.
BALLOON_ASPECT = 1.25
# Subtle underwater tether look (near-invisible fishing line): thin + low-contrast, near the
# water/background colour with low alpha. Turbidity blur fades it further with distance.
TETHER_RGBA = (0.28, 0.42, 0.52, 0.32)
TETHER_RADIUS = 0.0015  # was 0.003 in the scenario
# Bright, opaque pool-wall colour (dataset render): the raw scenery walls are low + translucent,
# leaving a black void above them; a real pool is an enclosed bright box.
WALL_RGBA = (0.50, 0.60, 0.66, 1.0)


def prep_render_spec(spec: mujoco.MjSpec, hide_tethers: bool = False) -> None:
    """Mutate a composed MjSpec IN PLACE for the rendered dataset (competition_balloon.py stays
    untouched — these edits happen only here, between build_spec and compile):

      * balloon spheres -> vertical ellipsoids (egg-shaped, aspect BALLOON_ASPECT);
      * hide the base_link pin (foreground needle, not in a real onboard view) via alpha=0 —
        the geom stays in the model so competition_run pop-detection is unaffected;
      * make tethers subtle (thin, low-contrast, low alpha) or hide them entirely.
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
        elif name.startswith("pool_wall_"):
            # Raise the low (1.2 m) scenery walls to the full water depth and make them bright +
            # opaque, so the frame is FILLED with pool (no black "over-the-wall" void) — a bright,
            # enclosed pool look. Height is size[1]/pos[1]; the other dims are left untouched.
            g.pos[1] = scn.POOL_DEPTH / 2
            g.size[1] = scn.POOL_DEPTH / 2
            g.rgba[:] = WALL_RGBA
    _brighten_like_pool(spec)


def _brighten_like_pool(spec: mujoco.MjSpec) -> None:
    """Light the scene like a real sunlit pool: bright + near-uniform, lit from ABOVE (not the
    dark, lit-from-below look of the raw scenario). Dataset-render only (this spec is compiled and
    thrown away per frame — the scenario's own lights / competition_run are untouched).

      * Headlight ambient is raised a lot -> strong orientation-independent fill (uniform brightness,
        the water-scattering look); diffuse raised for frontal modelling.
      * A broad DIRECTIONAL overhead light points straight down from above the surface (+Y up),
        standing in for sunlight through the surface; no shadow so it stays even.
    """
    hl = spec.visual.headlight
    hl.ambient[:] = [0.55, 0.57, 0.60]   # was 0.10 — near-uniform bright fill
    hl.diffuse[:] = [0.55, 0.55, 0.55]   # was 0.40
    hl.specular[:] = [0.10, 0.10, 0.10]
    hl.active = 1
    sun = spec.worldbody.add_light()
    sun.name = "dataset_sun"
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.pos = [scn.POOL_CENTER_X, scn.POOL_DEPTH + 2.0, 0.0]  # above the water surface
    sun.dir = [0.0, -1.0, 0.0]                                # straight down
    sun.diffuse = [0.55, 0.55, 0.58]
    sun.specular = [0.0, 0.0, 0.0]
    sun.castshadow = False
    # Water-surface "ceiling" at the top of the pool: caps the open top so a camera tilted/high
    # up sees a bright surface (as underwater) instead of the black void above the walls.
    surf = spec.worldbody.add_geom()
    surf.name = "dataset_surface"
    surf.type = mujoco.mjtGeom.mjGEOM_BOX
    surf.pos = [scn.POOL_CENTER_X, scn.POOL_DEPTH, 0.0]
    surf.size = [scn.POOL_LEN_X / 2, 0.02, scn.POOL_LEN_Z / 2]
    surf.rgba = [0.60, 0.72, 0.80, 1.0]  # bright surface seen from below
    surf.contype = 0
    surf.conaffinity = 0
    # Underwater fog: fade the far BACKGROUND (the black region above/beyond the pool walls, where
    # there is no geometry) to a bright water colour so the frame reads as "in water" instead of a
    # dark room. Kept far (fogstart 6 m) so it barely touches balloons within ~5 m — the real murk
    # still comes from the depth-based veil in degrade(). Enabled per-pass via mjRND_FOG (RGB only).
    spec.visual.rgba.fog[:] = [0.18, 0.46, 0.55, 1.0]
    spec.visual.map.fogstart = 6.0
    spec.visual.map.fogend = 16.0


def randomize_base_pose(data: mujoco.MjData, rng: np.random.Generator, camera: str) -> None:
    """Randomize the free base pose so the balloon field is seen from varied viewpoints.

    Frame: +Y up, forward = +X. ``front_cam`` looks +X (down the balloon run); ``down_cam`` is
    nadir. We keep the robot toward the -X / near end and yaw modestly so balloons stay framed.
    """
    x = float(rng.uniform(-1.2, 1.8))              # along the run, behind/among the near balloons
    z = float(rng.uniform(-1.6, 1.6))              # lateral, within the pool
    if camera == "down_cam":
        y = float(rng.uniform(1.6, 3.0))           # high, looking down at the field
        yaw = float(rng.uniform(-np.pi, np.pi))    # any heading for nadir
    else:
        y = float(rng.uniform(0.6, 2.4))           # mid-water height
        yaw = float(rng.uniform(-0.6, 0.6))        # +/-34 deg heading; keep balloons in view
    data.qpos[0:3] = [x, y, z]
    # yaw about the +Y (up) axis
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0]


def _geom_colour(model: mujoco.MjModel, gid: int) -> str | None:
    """Return the balloon colour for a segmentation geom id, or None if it's not a balloon sphere."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    if not name or not name.endswith("_geom") or not name.startswith("balloon_"):
        return None
    # name like 'balloon_red_1_geom' / 'balloon_yellow_start_geom' -> colour is the 2nd token
    colour = name.split("_")[1]
    return colour if colour in COLOUR_TO_CAT else None


def _boxes_from_seg(model, seg, depth, fpx, min_area_px):
    """Extract GT balloon boxes from the segmentation buffer.

    For each balloon sphere geom present:
      * mask = pixels whose seg objid == geom id (already the VISIBLE mask — occluders overwrite it)
      * bbox = [x, y, w, h] from the mask extent, clipped to the frame
      * expected full projected area from the sphere: pi * (fpx * R / d)^2, d = median mask depth
      * occlusion/clip fraction = 1 - visible_px / full_area
    Drop rules -> a box is kept only if: colour is a balloon, visible bbox area >= min_area_px,
    and occlusion fraction <= OCCLUSION_MAX (this also catches boxes mostly off-screen, since a
    clipped balloon has visible_px << full_area).

    Returns (boxes, drops) where boxes = list of dicts and drops = counts by reason.
    """
    h, w = seg.shape[:2]
    objid = seg[..., 0]
    boxes = []
    drops = {"size": 0, "occluded": 0, "offscreen": 0}
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
        if bw * bh < min_area_px:
            drops["size"] += 1
            continue
        if occ > OCCLUSION_MAX:
            drops["offscreen" if touches_border else "occluded"] += 1
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
    total_drops = {"size": 0, "occluded": 0, "offscreen": 0}
    depth_lo, depth_hi = np.inf, -np.inf
    seen_balloon_ids = set()
    scene_opt = _geom_only_option()  # geoms-only (no sites/decorations) for every render pass

    for i in range(args.n):
        layout = scn.sample_layout(rng, n_random=int(rng.integers(3, 6)))
        spec = scn.build_spec(layout)
        prep_render_spec(spec, hide_tethers=args.hide_tethers)
        model = spec.compile()
        data = mujoco.MjData(model)
        randomize_base_pose(data, rng, args.camera)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        try:
            rgb, depth, seg = render_buffers(renderer, model, data, args.camera, scene_opt)
        finally:
            renderer.close()

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

        boxes, drops = _boxes_from_seg(model, seg, depth, fpx, args.min_area_px)
        for k in total_drops:
            total_drops[k] += drops[k]

        # degrade (or keep clean)
        if args.clean:
            out_rgb = rgb
        else:
            params = us.random_params(water_rng)
            if args.force_reflection is not None:
                params.reflection = args.force_reflection  # force the distractor ON for the demo
            out_rgb = us.degrade(rgb, depth, params, water_rng)

        fname = f"frame_{i:04d}.jpg"
        _imwrite(img_dir / fname, out_rgb)

        coco["images"].append({
            "id": i + 1, "file_name": fname, "width": args.width, "height": args.height,
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

        print(f"frame {i:04d}: balloons_kept={len(boxes):2d} "
              f"drops(size={drops['size']},occ={drops['occluded']},off={drops['offscreen']}) "
              f"pose={np.round(data.qpos[0:3], 2)}")

    with open(args.out / "annotations.json", "w") as f:
        json.dump(coco, f, indent=1)

    print("\n=== dataset summary ===")
    print(f"frames: {args.n}  camera: {args.camera}  {args.width}x{args.height}  "
          f"mode: {'CLEAN' if args.clean else 'DEGRADED'}")
    print(f"boxes by colour: red={total_boxes['red']} blue={total_boxes['blue']} "
          f"yellow={total_boxes['yellow']}  total={sum(total_boxes.values())}")
    print(f"boxes DROPPED: size(<{args.min_area_px}px)={total_drops['size']} "
          f"occluded={total_drops['occluded']} offscreen={total_drops['offscreen']}  "
          f"total={sum(total_drops.values())}")
    if np.isfinite(depth_lo):
        print(f"depth buffer (scene, m): min={depth_lo:.3f} max={depth_hi:.3f}  "
              f"[far-plane background >50 m excluded]")
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
