"""Capture an onboard camera frame from the sim and save it (headless via EGL/OSMesa).

Steps the vehicle a few control-steps with a simple forward thrust command, then grabs a
front_cam RGB frame through UmiusiSimulator.render_camera().

Usage:
    MUJOCO_GL=egl python -m tools.camera_demo [output.png]   # default: ./front_cam.png
"""

import pathlib
import sys

import imageio
import numpy as np

from umiusi_sim.simulator import UmiusiSimulator

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "front_cam.png")


def main():
    sim = UmiusiSimulator()
    sim.reset(pos=(0.0, 1.0, 0.0))
    action = np.zeros(8)
    action[4:8] = 0.5  # all thrusters forward at half command
    for _ in range(20):
        sim.step(action)

    frame = sim.render_camera(camera="front_cam", width=320, height=240)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(OUT, frame)
    print(f"wrote {OUT}  shape={frame.shape} dtype={frame.dtype}")


if __name__ == "__main__":
    main()
