"""Default legible world — a grounded, well-lit scene around the untouched base robot.

The bare robot floats in a black void, which makes motion and orientation impossible to read
("the view is confusing"). This scenario composes a minimal, lightweight reference scene on
top of the base robot (``src/umiusi_sim/description/umiusi.xml``) so that in the GUI you can
always tell which way is up, which way is forward, and that the vehicle is actually moving:

  * a **checkerboard floor plane** below the start pose (Y-up: a plane with +Y normal at
    ``FLOOR_Y``) giving a fixed motion + orientation reference grid.
  * soft **fill lighting** and a faint, semi-transparent **water tint** volume for depth cues.
  * a small **world-origin axis triad** (R=+X forward, G=+Y up, B=+Z) so orientation is
    unambiguous.

Everything added is VISUAL only (``contype = conaffinity = 0``): the analytical hydrodynamics
in ``simulator.py`` already provide buoyancy/drag, so this is pure scenery and does not touch
the physics or ``umiusi.xml`` (``validate_sim`` stays unchanged). Same MjSpec ``build_spec`` /
``build_model`` pattern as ``competition_balloon``.

Frame (inherited from the base): CAD frame, +Y up, gravity (0, -9.81, 0), forward = +X. The
vehicle starts ~0.5-1.0 m above the floor.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

_BASE_MODEL = Path(__file__).resolve().parents[1] / "umiusi.xml"

# Floor a little below the start pose (start is around y = 0.5..1.0). A grid ~6 m square.
FLOOR_Y = -0.6
GRID_HALF = 3.0        # half-extent of the drawn floor plane [m]
WATER_HEIGHT = 3.0     # faint tint volume rises this far above the floor
WATER_HALF = GRID_HALF  # water tint footprint (matches the floor)
AXIS_LEN = 0.4         # world-origin triad arm length [m]
AXIS_R = 0.012         # triad arm radius [m]


def _add_geom(body, **kw):
    """Add a non-colliding VISUAL geom to ``body`` from keyword attributes."""
    g = body.add_geom()
    g.contype = 0
    g.conaffinity = 0
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def build_spec():
    """Load the base robot and compose the default legible world; return a ``mujoco.MjSpec``."""
    spec = mujoco.MjSpec.from_file(str(_BASE_MODEL))
    world = spec.worldbody

    # -- checkerboard material for the floor ----------------------------------
    tex = spec.add_texture()
    tex.name = "grid_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.width = 300
    tex.height = 300
    tex.rgb1 = [0.26, 0.36, 0.44]
    tex.rgb2 = [0.42, 0.54, 0.62]
    mat = spec.add_material()
    mat.name = "grid_mat"
    # texrepeat tiles the checker across the plane; reflectance gives a subtle floor sheen.
    mat.texrepeat = [12, 12]
    mat.texuniform = True
    mat.reflectance = 0.10
    # material.textures is a fixed-size list indexed by texture role; 2D goes in the RGB slot.
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid_tex"

    # -- soft fill lighting (base model has a single top light) ---------------
    fill = world.add_light()
    fill.name = "world_fill"
    fill.type = mujoco.mjtLightType.mjLIGHT_SPOT
    fill.pos = [2.0, 3.0, 2.0]
    fill.dir = [-0.4, -1.0, -0.4]
    fill.diffuse = [0.35, 0.35, 0.35]
    fill.cutoff = 60.0

    # -- checker floor plane (Y-up: normal +Y). A MuJoCo plane's local +Z is its normal (the lit,
    #    visible face), so rotate -90 deg about world +X to map local +Z -> world +Y (normal up). -
    _add_geom(
        world, name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, FLOOR_Y, 0.0],
        # quat = rotate -90 deg about X (w, x, y, z), maps local +Z -> world +Y.
        quat=[0.70710678, -0.70710678, 0.0, 0.0],
        size=[GRID_HALF, GRID_HALF, 0.1], material="grid_mat",
    )

    # -- faint water tint volume for depth cues (semi-transparent box) ---------
    _add_geom(
        world, name="water_tint", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, FLOOR_Y + WATER_HEIGHT / 2, 0.0],
        size=[WATER_HALF, WATER_HEIGHT / 2, WATER_HALF],
        rgba=[0.12, 0.42, 0.62, 0.06], group=2,
    )

    # -- world-origin axis triad (R=+X forward, G=+Y up, B=+Z) -----------------
    triad = world.add_body(name="world_axes", pos=[0.0, 0.0, 0.0])
    _add_geom(
        triad, name="axis_x", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, AXIS_LEN, 0, 0], size=[AXIS_R, 0, 0], rgba=[0.90, 0.20, 0.20, 1.0],
    )
    _add_geom(
        triad, name="axis_y", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, AXIS_LEN, 0], size=[AXIS_R, 0, 0], rgba=[0.20, 0.80, 0.20, 1.0],
    )
    _add_geom(
        triad, name="axis_z", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, AXIS_LEN], size=[AXIS_R, 0, 0], rgba=[0.20, 0.40, 0.90, 1.0],
    )
    return spec


def build_model():
    """Compose and compile the default world; return a ``mujoco.MjModel``."""
    return build_spec().compile()


def write_xml(path):
    """Compose the scene and write the flattened MJCF to ``path`` (self-contained, loadable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_spec().to_xml())
    return path
