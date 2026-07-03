"""Interactive MuJoCo GUI viewer for the UMIUSI simulator (real-time, WSLg/GLFW).

Runs the analytical simulation live so you can watch buoyancy, drag, servo steering and
thrust. Requires a display (WSLg provides one). Not for headless/EGL-only sessions.

Camera / mouse:
    The mouse (left-drag = orbit, right-drag = pan, scroll = zoom) only drives the FREE
    camera. This model is +Y-up but the free camera assumes +Z-up, so it looks tilted —
    hence the default is the upright, well-framed fixed "iso" camera. Press [ or ] in the
    window to cycle cameras (free <-> iso <-> top). Use --free to start on the free camera
    (framed on the vehicle) when you want to fly around with the mouse.

Usage:
    python -m tools.view              # float freely and self-level (zero action)
    python -m tools.view --demo       # sweep the servos and pulse the thrusters
    python -m tools.view --free       # start on the mouse-controllable free camera
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from umiusi_sim.simulator import UmiusiSimulator


def demo_action(t):
    """A gentle scripted action: sweep azimuth servos, pulse thrust."""
    servo = 0.6 * np.sin(2.0 * np.pi * 0.2 * t) * np.ones(4)
    esc = 0.4 * np.ones(4)
    return np.concatenate([servo, esc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="apply a scripted servo/thrust action")
    ap.add_argument("--free", action="store_true",
                    help="start on the mouse-controllable free camera (framed on the vehicle)")
    args = ap.parse_args()

    sim = UmiusiSimulator()
    sim.reset(pos=(0.0, 0.5, 0.0))
    control_dt = 1.0 / sim.cfg["sim"]["control_rate_hz"]

    print("controls: mouse orbits/pans/zooms the FREE camera; press [ or ] to cycle cameras.")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        if args.free:
            # Free camera: mouse-controllable. Frame it on the whole-vehicle CoM.
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = sim.data.subtree_com[sim.base_id]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth, viewer.cam.elevation = 135.0, -20.0
        else:
            # Upright, well-framed "iso" fixed camera (free camera assumes Z-up).
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = sim.model.camera("iso").id
        t = 0.0
        while viewer.is_running():
            tic = time.time()
            action = demo_action(t) if args.demo else np.zeros(8)
            sim.step(action)
            viewer.sync()
            t += control_dt
            time.sleep(max(0.0, control_dt - (time.time() - tic)))


if __name__ == "__main__":
    main()
