"""Competition "balloon-popping" world — composed on top of the base UMIUSI robot.

Builds the fully-autonomous underwater balloon-popping WORLD around the untouched base
robot (``src/umiusi_sim/description/umiusi.xml``):

  * a pool (floor + semi-transparent water box + low walls) — VISUAL only; the analytical
    hydrodynamics in ``simulator.py`` already provide buoyancy/drag, so the water is scenery.
  * colour-coded, tethered balloons floating at fixed heights above the floor.
  * a small rigid "pin" popping tool protruding forward (+X) from ``base_link``.

Composition uses ``mujoco.MjSpec`` (programmatic) rather than an MJCF ``<include>``: the pin
must be a rigid CHILD geom of ``base_link``, and ``<include>`` can only append *sibling*
elements to ``<worldbody>`` — it cannot inject a geom into an already-defined body. MjSpec
lets us load the base spec, reach into ``base_link``, and attach the pin there while leaving
``umiusi.xml`` (and therefore ``validate_sim``) completely unchanged.

Frame (inherited from the base): CAD frame, +Y up, gravity (0, -9.81, 0), forward = +X.
The base robot starts (per the sim/env) around y = 1.0 m, i.e. ~1 m off the pool floor.

Everything added here has ``contype = conaffinity = 0`` (no collision): the vehicle floats
freely under analytical hydro, exactly like the base model. Balloons are therefore STATIC
scenery and "popping" is detected geometrically (pin-tip vs. balloon-centre distance) — see
``balloon_table`` / ``popped`` below and the scenario-abstraction proposal.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

_BASE_MODEL = Path(__file__).resolve().parents[1] / "umiusi.xml"

# --- pool (VISUAL only) ------------------------------------------------------
# ASSUMPTION: the competition uses the "3.3 m-deep area" but the real pool length/width are
# not published. We model an 8 m (X, forward) x 5 m (Z, lateral) footprint, floor top at y=0,
# water up to y = 3.3 m. The footprint is shifted +X so the whole run toward the balloons
# stays inside it. None of this affects physics (collisions are off) — it is framing/scenery.
POOL_DEPTH = 3.3
POOL_LEN_X = 8.0
POOL_LEN_Z = 5.0
POOL_CENTER_X = 2.0
FLOOR_Y = 0.0

# --- balloons ----------------------------------------------------------------
BALLOON_RADIUS = 0.10  # ~20 cm diameter spheres

# Scoring rule -> colour. RED @0.5 m (+30), BLUE @0.7 m (-10, decoy), YELLOW @1.5 m (+10).
BALLOON_SPECS = {
    "red": {"height": 0.5, "points": 30, "rgba": (0.85, 0.12, 0.12, 1.0)},
    "blue": {"height": 0.7, "points": -10, "rgba": (0.12, 0.28, 0.85, 1.0)},
    "yellow": {"height": 1.5, "points": 10, "rgba": (0.92, 0.82, 0.12, 1.0)},
}

# Fixed placeholder layout: (name, colour, x, z). Height comes from BALLOON_SPECS.
# The FIRST balloon is YELLOW, fixed ~1.5 m in front of the start pose (start at x=0, +X).
# The others are a small fixed set standing in for PER-EPISODE RANDOM XY — see
# ``sample_layout`` for the documented randomization placeholder.
BALLOON_LAYOUT = [
    ("balloon_yellow_start", "yellow", 1.5, 0.0),
    ("balloon_red_1", "red", 2.6, 0.8),
    ("balloon_blue_1", "blue", 2.2, -0.9),
    ("balloon_yellow_2", "yellow", 3.4, -0.4),
    ("balloon_red_2", "red", 3.9, 0.6),
]

# --- pin (popping tool, rigid child of base_link) ----------------------------
# Protrudes forward (+X) from the hull front (~x=0.09) at hull mid-height (y~0.10, level with
# front_cam). Thin + low-mass so it barely perturbs the measured inertial. ``pin_tip`` site is
# the geometric point used for pop detection.
PIN_BASE = (0.15, 0.10, 0.0)
PIN_TIP = (0.40, 0.10, 0.0)
PIN_RADIUS = 0.006
PIN_MASS = 0.02

# Pop detection: a balloon is "popped" when the pin tip comes within this of its centre AND the
# hit is near-FRONTAL. Real balloons only pop to a straight jab of the needle, not a glancing/
# sideways brush, so ``popped`` additionally requires the pin axis (robot +X forward) to point at
# the balloon within POP_ANGLE_TOL_DEG. This makes the RAM actually aim straight at the target.
POP_MARGIN = 0.03  # m; effective radius = BALLOON_RADIUS + POP_MARGIN
POP_ANGLE_TOL_DEG = 25.0  # max angle between the pin axis and the pin-tip->balloon direction


def _add_geom(body, **kw):
    """Add a non-colliding VISUAL geom to ``body`` from keyword attributes."""
    g = body.add_geom()
    g.contype = 0
    g.conaffinity = 0
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def build_spec(layout=BALLOON_LAYOUT):
    """Load the base robot and compose the competition world; return a ``mujoco.MjSpec``.

    ``layout`` is a list of ``(name, colour, x, z)`` tuples (colour keys BALLOON_SPECS).
    """
    spec = mujoco.MjSpec.from_file(str(_BASE_MODEL))
    world = spec.worldbody

    # Extra overhead fill light over the pool (the base model's single light leaves the far
    # scene dark). Purely for the demo/camera framing.
    light = world.add_light()
    light.name = "pool_fill"
    light.pos = [POOL_CENTER_X, POOL_DEPTH, 0.0]
    light.dir = [0.0, -1.0, 0.0]
    light.diffuse = [0.5, 0.5, 0.5]

    # Pool floor (thin box) + semi-transparent water box + low translucent walls. Visual only.
    _add_geom(
        world, name="pool_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[POOL_CENTER_X, FLOOR_Y - 0.02, 0.0],
        size=[POOL_LEN_X / 2, 0.02, POOL_LEN_Z / 2], rgba=[0.80, 0.78, 0.72, 1.0],
    )
    _add_geom(
        world, name="pool_water", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[POOL_CENTER_X, FLOOR_Y + POOL_DEPTH / 2, 0.0],
        size=[POOL_LEN_X / 2, POOL_DEPTH / 2, POOL_LEN_Z / 2], rgba=[0.12, 0.42, 0.62, 0.12],
        group=2,
    )
    wall_h = 1.2  # only draw the lower part of the walls (scenery; keeps cameras from clipping)
    for name, (px, pz, sx, sz) in {
        "pool_wall_xpos": (POOL_CENTER_X + POOL_LEN_X / 2, 0.0, 0.02, POOL_LEN_Z / 2),
        "pool_wall_xneg": (POOL_CENTER_X - POOL_LEN_X / 2, 0.0, 0.02, POOL_LEN_Z / 2),
        "pool_wall_zpos": (POOL_CENTER_X, POOL_LEN_Z / 2, POOL_LEN_X / 2, 0.02),
        "pool_wall_zneg": (POOL_CENTER_X, -POOL_LEN_Z / 2, POOL_LEN_X / 2, 0.02),
    }.items():
        _add_geom(
            world, name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[px, FLOOR_Y + wall_h / 2, pz], size=[sx, wall_h / 2, sz],
            rgba=[0.30, 0.45, 0.55, 0.18], group=2,
        )

    # Balloons: each a STATIC body (no joint => welded to world) holding one sphere geom, plus a
    # thin tether + floor weight drawn as world geoms. Static + geometric pop detection is the
    # simplest reliable approach (see module docstring / proposal).
    for name, colour, x, z in layout:
        spec_c = BALLOON_SPECS[colour]
        y = FLOOR_Y + spec_c["height"]
        b = world.add_body(name=name, pos=[x, y, z])
        _add_geom(
            b, name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[BALLOON_RADIUS, 0, 0], rgba=list(spec_c["rgba"]), mass=1e-4,
        )
        # Tether (floor -> balloon bottom) + a small weight box on the floor. World coords.
        _add_geom(
            world, name=f"{name}_tether", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[x, FLOOR_Y, z, x, y - BALLOON_RADIUS, z], size=[0.003, 0, 0],
            rgba=[0.15, 0.15, 0.15, 1.0],
        )
        _add_geom(
            world, name=f"{name}_weight", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, FLOOR_Y + 0.02, z], size=[0.03, 0.02, 0.03], rgba=[0.2, 0.2, 0.2, 1.0],
        )

    # Pin: rigid child geom of base_link (+ a tip site for pop detection).
    base = spec.body("base_link")
    _add_geom(
        base, name="pin", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[*PIN_BASE, *PIN_TIP], size=[PIN_RADIUS, 0, 0], rgba=[0.85, 0.85, 0.9, 1.0],
        mass=PIN_MASS,
    )
    s = base.add_site()
    s.name = "pin_tip"
    s.pos = list(PIN_TIP)
    s.size = [0.008, 0, 0]
    s.rgba = [1.0, 0.4, 0.0, 1.0]
    return spec


def build_model(layout=BALLOON_LAYOUT):
    """Compose and compile the competition scene; return a ``mujoco.MjModel``."""
    return build_spec(layout).compile()


def write_xml(path, layout=BALLOON_LAYOUT):
    """Compose the scene and write the flattened MJCF to ``path`` (self-contained, loadable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_spec(layout).to_xml())
    return path


def balloon_table(layout=BALLOON_LAYOUT):
    """Scoring/pop-detection metadata: list of dicts (name, colour, points, height, pos)."""
    table = []
    for name, colour, x, z in layout:
        spec_c = BALLOON_SPECS[colour]
        table.append({
            "name": name, "colour": colour, "points": spec_c["points"],
            "height": spec_c["height"],
            "pos": np.array([x, FLOOR_Y + spec_c["height"], z], dtype=float),
        })
    return table


def popped(pin_tip_world, balloon_pos, pin_axis=None, margin=POP_MARGIN,
           angle_tol_deg=POP_ANGLE_TOL_DEG):
    """True if the pin pops the balloon: the tip is within (radius + margin) of the centre AND,
    when a ``pin_axis`` (robot +X forward, world frame) is given, the hit is near-FRONTAL — the pin
    axis points at the balloon within ``angle_tol_deg`` of the pin-tip->centre direction.

    ``pin_axis=None`` keeps the legacy proximity-only test (any caller that does not model aiming).
    Both ``tools/competition_run`` and ``tools/autonomy_run`` pass the axis, so a glancing/sideways
    contact no longer scores — the vehicle must approach the balloon head-on.
    """
    delta = np.asarray(balloon_pos, dtype=float) - np.asarray(pin_tip_world, dtype=float)
    dist = float(np.linalg.norm(delta))
    if dist >= BALLOON_RADIUS + margin:
        return False
    if pin_axis is None:
        return True
    a = np.asarray(pin_axis, dtype=float)
    na = float(np.linalg.norm(a))
    if na < 1e-9 or dist < 1e-6:  # degenerate axis or tip essentially at the centre -> count it
        return True
    cos_ang = float(np.dot(a / na, delta / dist))
    return cos_ang >= math.cos(math.radians(angle_tol_deg))


def hide_balloon(model, name):
    """Make a popped balloon visually VANISH: set its geom alpha to 0 on the COMPILED ``model``.

    Only the render changes — the geom stays in the model and collisions were already off
    (``contype = conaffinity = 0``), so physics, ``scn.popped`` scoring and every other balloon are
    unaffected. Used by ``tools/autonomy_run`` so that when a balloon pops it disappears from the
    onboard camera (deflated), letting the vision FSM confirm the pop from the camera alone.
    """
    gid = model.geom(f"{name}_geom").id
    model.geom_rgba[gid, 3] = 0.0


def sample_layout(rng, n_random=4):
    """PLACEHOLDER per-episode randomization: keep the YELLOW start balloon fixed, sample the
    rest at random XY within the pool footprint. Returns a BALLOON_LAYOUT-shaped list.

    NOTE: heights are fixed by colour (rule); only XY randomizes. Colours here follow the same
    small mix as the fixed layout — tune counts/keep-out zones when the real rules are settled.
    """
    layout = [BALLOON_LAYOUT[0]]  # yellow, ~1.5 m in front of start (fixed by rule)
    colours = ["red", "blue", "yellow", "red"]
    x_lo, x_hi = POOL_CENTER_X - POOL_LEN_X / 2 + 0.5, POOL_CENTER_X + POOL_LEN_X / 2 - 0.5
    z_lo, z_hi = -POOL_LEN_Z / 2 + 0.5, POOL_LEN_Z / 2 - 0.5
    for i in range(n_random):
        c = colours[i % len(colours)]
        x = float(rng.uniform(max(x_lo, 1.0), x_hi))  # keep clear of the start pose
        z = float(rng.uniform(z_lo, z_hi))
        layout.append((f"balloon_{c}_{i}", c, x, z))
    return layout
