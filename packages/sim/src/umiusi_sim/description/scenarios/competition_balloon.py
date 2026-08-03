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
import yaml

_SCN_DIR = Path(__file__).resolve()
_BASE_MODEL = _SCN_DIR.parents[1] / "umiusi.xml"
# Repo root: .../src/umiusi_sim/description/scenarios/competition_balloon.py -> parents[4].
_DEFAULT_CONFIG = _SCN_DIR.parents[4] / "configs" / "umiusi.yaml"

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
# The FIRST balloon is YELLOW, fixed ~1.5 m in front of the start pose (start at x=0, +X). This
# entry is also the deterministic "near target" that ``sample_layout`` always keeps as balloon 0.
# The others are a small fixed set kept only for the legacy fixed-layout path (seed < 0); the real
# per-episode field is drawn by ``sample_layout`` from the YAML-configured COUNTS below.
BALLOON_LAYOUT = [
    ("balloon_yellow_start", "yellow", 1.5, 0.0),
    ("balloon_red_1", "red", 2.6, 0.8),
    ("balloon_blue_1", "blue", 2.2, -0.9),
    ("balloon_yellow_2", "yellow", 3.4, -0.4),
    ("balloon_red_2", "red", 3.9, 0.6),
]

# --- per-episode field defaults (overridable from configs/umiusi.yaml -> competition.balloons) --
# PLACEHOLDER per-colour counts for the sampled field. The exact rulebook red/yellow/blue ratio is
# TBD / user-tunable: the competition videos suggest ~6-8 reds with yellows and blues a comparable
# spread, so we default to red 7 / yellow 7 / blue 5. Edit configs/umiusi.yaml (NOT this file) to
# retune. The deterministic tall yellow start balloon COUNTS AS ONE of the yellows.
DEFAULT_BALLOON_COUNTS = {"red": 7, "yellow": 7, "blue": 5}
# Minimum centre-to-centre XY separation between balloons [m] enforced by the Poisson-disk /
# rejection sampler, so the field stays evenly spread (not clustered). Overridable via YAML.
DEFAULT_MIN_SEPARATION = 0.6

# --- tether entanglement -----------------------------------------------------
# Each balloon is tethered by a vertical wire from a floor anchor at its XY up to the balloon (see
# the ``*_tether`` geoms in build_spec). The robot ENTANGLES a wire if it under-passes the balloon:
# its horizontal position comes within TETHER_RADIUS of the balloon's XY while it is BELOW the
# balloon. Small radius ~ the wire "capture" zone; see ``entanglement`` below.
TETHER_RADIUS = 0.20  # m; horizontal capture radius around a balloon's vertical tether wire

# --- pin (popping tool, rigid child of base_link) ----------------------------
# Protrudes forward (+X) from the hull front (~x=0.09) at hull mid-height (y~0.10, level with
# front_cam). Thin + low-mass so it barely perturbs the measured inertial. ``pin_tip`` site is
# the geometric point used for pop detection.
PIN_BASE = (0.15, 0.10, 0.0)
PIN_TIP = (0.40, 0.10, 0.0)
PIN_RADIUS = 0.006
PIN_MASS = 0.02

# Pop detection: a balloon is "popped" when the pin tip comes within this of its centre AND the
# hit is near-FRONTAL AND the tip is actually driving INTO the balloon fast enough. Real balloons
# only pop to a straight, committed jab of the needle — not a glancing/sideways brush, and not a
# slow drift-by that merely nudges the skin. So ``popped`` requires three things together:
#   1. proximity: tip within (BALLOON_RADIUS + POP_MARGIN) of the centre;
#   2. head-on:   the pin axis (robot +X forward) points at the balloon within POP_ANGLE_TOL_DEG of
#                 the pin-tip->centre direction (rejects glancing contacts, forces a straight aim);
#   3. speed:     the pin-tip's closing speed toward the balloon (velocity projected onto the
#                 pin-tip->centre direction) is at least MIN_POP_SPEED (the needle must be moving
#                 into the skin, not resting against / drifting past it).
# POP_ANGLE_TOL_DEG was tightened 25 -> 20 deg. 20 deg is the calibrated sweet spot: measuring the
# pin-tip vs. balloon geometry during real drive-through lunges, a well-aimed head-on ram enters the
# pop sphere (dist ~0.10-0.12 m) at an axis error of ~19-24 deg — NOT ~0 deg — because the pin sits
# forward AND laterally offset from the camera/COM, so the +X axis is a few deg off the tip->centre
# line even on a dead-centred camera approach. 15 deg rejected those good rams (they only re-qualified
# on a later, messier pass, or not at all -> reds stopped popping and the run degenerated); 25 deg was
# the old value that also let wide glancing brushes score. 20 deg keeps the clean fast ram while
# still rejecting the 20-25 deg "somewhat off" band and all the >30 deg sideways brushes. MIN_POP_SPEED
# = 0.18 m/s sits well below the drive-through lunge closing speed (measured ~0.5 m/s at contact) but
# above the slow drift a mis-timed pass produces, so the lunge clears the gate and a drift-by does not.
POP_MARGIN = 0.03  # m; effective radius = BALLOON_RADIUS + POP_MARGIN
POP_ANGLE_TOL_DEG = 20.0  # max angle between the pin axis and the pin-tip->balloon direction
MIN_POP_SPEED = 0.18      # m/s; min pin-tip closing speed toward the balloon for a pop to register


def _add_geom(body, **kw):
    """Add a non-colliding VISUAL geom to ``body`` from keyword attributes."""
    g = body.add_geom()
    g.contype = 0
    g.conaffinity = 0
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def build_spec(layout=BALLOON_LAYOUT, pin_base=PIN_BASE, pin_tip=PIN_TIP):
    """Load the base robot and compose the competition world; return a ``mujoco.MjSpec``.

    ``layout`` is a list of ``(name, colour, x, z)`` tuples (colour keys BALLOON_SPECS).
    ``pin_base``/``pin_tip`` set the popping-pin mount (body frame; default = module constants); they
    are exposed so a pin-placement study can sweep the mount without editing this module.
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
        fromto=[*pin_base, *pin_tip], size=[PIN_RADIUS, 0, 0], rgba=[0.85, 0.85, 0.9, 1.0],
        mass=PIN_MASS,
    )
    s = base.add_site()
    s.name = "pin_tip"
    s.pos = list(pin_tip)
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


def popped(pin_tip_world, balloon_pos, pin_axis=None, pin_vel=None, margin=POP_MARGIN,
           angle_tol_deg=POP_ANGLE_TOL_DEG, min_speed=MIN_POP_SPEED):
    """True if the pin pops the balloon: the tip is within (radius + margin) of the centre AND,
    when a ``pin_axis`` (robot +X forward, world frame) is given, the hit is near-FRONTAL — the pin
    axis points at the balloon within ``angle_tol_deg`` of the pin-tip->centre direction — AND, when
    a ``pin_vel`` (pin-tip world linear velocity [m/s]) is given, the tip is actually driving INTO
    the balloon: its closing speed (velocity projected onto the pin-tip->centre direction) is at
    least ``min_speed``.

    ``pin_axis=None`` / ``pin_vel=None`` each independently drop the corresponding gate, so the
    legacy proximity-only call (e.g. ``tools/scenario_demo``) still works. Both
    ``tools/competition_run`` and ``tools/autonomy_run`` pass the axis AND the velocity, so a
    glancing/sideways contact or a slow drift-by no longer scores — the vehicle must approach the
    balloon head-on and be driving into it.
    """
    delta = np.asarray(balloon_pos, dtype=float) - np.asarray(pin_tip_world, dtype=float)
    dist = float(np.linalg.norm(delta))
    if dist >= BALLOON_RADIUS + margin:
        return False
    # Speed gate: require a minimum closing speed toward the balloon centre (the needle must be
    # jabbing INTO the skin, not resting against it or drifting past). When the tip is essentially
    # AT the centre the direction is ill-defined -> fall back to the raw speed magnitude.
    if pin_vel is not None:
        v = np.asarray(pin_vel, dtype=float)
        closing = float(np.dot(v, delta / dist)) if dist > 1e-6 else float(np.linalg.norm(v))
        if closing < min_speed:
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


def load_field_config(config_path=_DEFAULT_CONFIG):
    """Read the competition balloon-field config (per-colour COUNTS + min separation) from
    ``configs/umiusi.yaml`` (``competition.balloons``). Missing file/keys fall back to the module
    defaults, so callers work without any config present. Returns ``(counts, min_separation)``."""
    counts = dict(DEFAULT_BALLOON_COUNTS)
    min_sep = DEFAULT_MIN_SEPARATION
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return counts, min_sep
    bcfg = ((cfg.get("competition") or {}).get("balloons")) or {}
    for colour, n in (bcfg.get("counts") or {}).items():
        if colour in counts:
            counts[colour] = int(n)
    if bcfg.get("min_separation") is not None:
        min_sep = float(bcfg["min_separation"])
    return counts, min_sep


def _poisson_sample(rng, placed, min_sep, bounds, max_tries=256):
    """Rejection (Poisson-disk) sample of ONE XY at least ``min_sep`` from every point in
    ``placed`` (list of (x, z)), within ``bounds`` = (x_lo, x_hi, z_lo, z_hi). If no candidate
    clears the separation within ``max_tries``, return the best (farthest-from-neighbours)
    candidate so the exact count is still honoured (spacing best-effort). Seeded via ``rng``."""
    x_lo, x_hi, z_lo, z_hi = bounds
    best, best_d = None, -1.0
    for _ in range(max_tries):
        x = float(rng.uniform(x_lo, x_hi))
        z = float(rng.uniform(z_lo, z_hi))
        if not placed:
            return x, z
        d = min(math.hypot(x - px, z - pz) for px, pz in placed)
        if d >= min_sep:
            return x, z
        if d > best_d:
            best, best_d = (x, z), d
    return best


def sample_layout(rng, counts=None, min_separation=None, config_path=_DEFAULT_CONFIG):
    """Per-episode field sampler: scatter the configured per-colour balloon COUNTS at random XY
    within the pool footprint with roughly EVEN spacing. Returns a BALLOON_LAYOUT-shaped list of
    ``(name, colour, x, z)`` tuples (heights come from BALLOON_SPECS per colour).

    Spacing: rejection / Poisson-disk sampling enforces a minimum centre-to-centre separation
    (``min_separation``) between every pair of balloons, so the field is spread out rather than
    clustered (matching the fairly even spacing seen in the competition videos).

    Balloon 0 is ALWAYS the deterministic tall YELLOW at the start position
    (``balloon_yellow_start``, ~x=1.5) so the fixed near target is always present; it counts as one
    of the yellows. Heights are fixed by colour (rule); only XY randomizes. Seeded via ``rng`` ->
    reproducible.

    ``counts`` / ``min_separation`` default to the ``competition.balloons`` block in
    ``configs/umiusi.yaml`` (or the module defaults if absent) and may be overridden by callers.
    """
    if counts is None or min_separation is None:
        cfg_counts, cfg_sep = load_field_config(config_path)
        counts = cfg_counts if counts is None else dict(counts)
        min_separation = cfg_sep if min_separation is None else float(min_separation)

    # Balloon 0: the deterministic tall YELLOW at the start position (fixed near target, by rule).
    start = BALLOON_LAYOUT[0]                          # (name, "yellow", x, z)
    layout = [start]
    placed = [(start[2], start[3])]                    # XY of everything placed so far

    # The RANDOM balloons to scatter = the configured counts, with the fixed start yellow counting
    # as one of the yellows (so ``counts`` is the TOTAL per-colour field size).
    to_place = (["red"] * counts.get("red", 0)
                + ["yellow"] * max(0, counts.get("yellow", 0) - 1)
                + ["blue"] * counts.get("blue", 0))

    # Sampling window: inside the pool footprint (0.5 m margin) and clear of the start pose (x>=1.0).
    bounds = (
        max(POOL_CENTER_X - POOL_LEN_X / 2 + 0.5, 1.0),
        POOL_CENTER_X + POOL_LEN_X / 2 - 0.5,
        -POOL_LEN_Z / 2 + 0.5,
        POOL_LEN_Z / 2 - 0.5,
    )

    per_colour = {"red": 0, "yellow": 0, "blue": 0}
    for colour in to_place:
        x, z = _poisson_sample(rng, placed, min_separation, bounds)
        placed.append((x, z))
        per_colour[colour] += 1
        layout.append((f"balloon_{colour}_{per_colour[colour]}", colour, x, z))
    return layout


def entanglement(robot_pos, balloons, popped=None):
    """Names of the UN-POPPED balloons whose tether the robot is currently ENTANGLED in.

    Model: each balloon is tethered by a vertical wire from a floor anchor at its XY up to the
    balloon. The robot ENTANGLES a balloon when it UNDER-PASSES it — the robot's HORIZONTAL
    position is within ``TETHER_RADIUS`` of the balloon's horizontal position (i.e. it is up
    against the wire) AND the robot is BELOW the balloon (passing under it along the wire). Popped
    balloons (wire removed) never entangle.

    Frame note: CAD frame with +Y UP, so the HORIZONTAL plane is (x, z) and the VERTICAL axis is y.

    ``robot_pos`` : (x, y, z) robot body position, world frame.
    ``balloons``  : a ``balloon_table()`` list; each item has 'name' and 'pos' = [x, y(height), z].
    ``popped``    : iterable of popped balloon names to exclude (their wire is gone), or None.

    Returns the list of offending balloon names; ``len(...)`` is how many tethers are snagged.
    """
    popped = set(popped or ())
    rx, ry, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
    snagged = []
    for b in balloons:
        if b["name"] in popped:
            continue
        bx, by, bz = float(b["pos"][0]), float(b["pos"][1]), float(b["pos"][2])
        if math.hypot(rx - bx, rz - bz) <= TETHER_RADIUS and ry < by:
            snagged.append(b["name"])
    return snagged
