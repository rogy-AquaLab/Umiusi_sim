"""Render placement snapshots of the UMIUSI model to media/*.png (headless via EGL/OSMesa).

Usage:
    MUJOCO_GL=egl python -m tools.snapshot
"""

import pathlib

import imageio
import mujoco

from umiusi_sim.simulator import _DEFAULT_MODEL

# Fixed cameras defined in the MJCF (upright for the +Y-up frame), plus free-camera angles.
FIXED_CAMS = ["iso", "top"]
FREE_VIEWS = {
    "front": dict(azimuth=90, elevation=-8, distance=2.2),
    "corner": dict(azimuth=135, elevation=-20, distance=2.2),
}


def main():
    model = mujoco.MjModel.from_xml_path(str(_DEFAULT_MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = pathlib.Path("media")
    out.mkdir(exist_ok=True)
    base = model.body("base_link").id

    with mujoco.Renderer(model, 720, 960) as r:
        for name in FIXED_CAMS:
            r.update_scene(data, camera=name)
            imageio.imwrite(out / f"umiusi_{name}.png", r.render())
            print("wrote", out / f"umiusi_{name}.png")
        for name, kw in FREE_VIEWS.items():
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = data.subtree_com[base]
            cam.azimuth, cam.elevation, cam.distance = kw["azimuth"], kw["elevation"], kw["distance"]
            r.update_scene(data, cam)
            imageio.imwrite(out / f"umiusi_{name}.png", r.render())
            print("wrote", out / f"umiusi_{name}.png")


if __name__ == "__main__":
    main()
