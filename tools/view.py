"""Passive GUI viewer for the UMIUSI simulator — just launch and watch (real-time, WSLg/GLFW).

**Display only.** It runs the analytical simulation and shows it (idle, or a scripted ``--demo``
motion) so you can watch buoyancy, drag, servo steering and thrust — it does NOT take control input.
To DRIVE the robot with a trained policy from the keyboard, use ``tools.drive`` instead.

By default it loads the legible **default world** (checker floor grid + lighting + world-origin axis
triad) so motion and orientation are obvious — not a robot in a black void.
Requires a display (WSLg provides one). Not for headless/EGL-only sessions.

The camera / +Y-up handling and on-screen controls are shared with every other tool via
``umiusi_sim.viewer`` — the default is the fixed ``track`` camera (upright + follows the
vehicle). Press ``[`` / ``]`` in the window to cycle cameras; ``--free`` starts on the
mouse-controllable free camera (framed on the vehicle).

Usage:
    python -m tools.view                       # grounded default world, self-level (zero action)
    python -m tools.view --demo                # sweep the servos and pulse the thrusters
    python -m tools.view --free                # start on the mouse-controllable free camera
    python -m tools.view --bare                # plain robot, no world (black void)
    python -m tools.view --scenario competition_balloon   # the balloon-popping world
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from umiusi_sim.simulator import UmiusiSimulator
from umiusi_sim.viewer import UmiusiViewer

# Composed-world MJCF is written to a portable temp dir (scenario builders flatten to a file).
_TMP = Path(tempfile.gettempdir()) / "umiusi_sim"


def demo_action(t):
    """A gentle scripted action: sweep azimuth servos, pulse thrust."""
    servo = 0.6 * np.sin(2.0 * np.pi * 0.2 * t) * np.ones(4)
    esc = 0.4 * np.ones(4)
    return np.concatenate([servo, esc])


def _build_sim(scenario, bare):
    """Return an UmiusiSimulator loaded with the chosen scene (bare robot or a composed world)."""
    if bare:
        return UmiusiSimulator()
    if scenario == "default_world":
        from umiusi_sim.description.scenarios import default_world as scn
        xml = scn.write_xml(_TMP / "view_default_world.xml")
    elif scenario == "competition_balloon":
        from umiusi_sim.description.scenarios import competition_balloon as scn
        xml = scn.write_xml(_TMP / "view_competition.xml")
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return UmiusiSimulator(model_path=xml)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="apply a scripted servo/thrust action")
    ap.add_argument("--free", action="store_true",
                    help="start on the mouse-controllable free camera (framed on the vehicle)")
    ap.add_argument("--bare", action="store_true", help="plain robot, no world (black void)")
    ap.add_argument("--scenario", choices=["default_world", "competition_balloon"],
                    default="default_world", help="composed world to load (default: default_world)")
    args = ap.parse_args()

    _TMP.mkdir(parents=True, exist_ok=True)
    sim = _build_sim(args.scenario, args.bare)
    sim.reset(pos=(0.0, 0.5, 0.0))

    t = [0.0]
    control_dt = 1.0 / sim.cfg["sim"]["control_rate_hz"]

    def step():
        action = demo_action(t[0]) if args.demo else np.zeros(8)
        sim.step(action)
        t[0] += control_dt

    with UmiusiViewer(sim.model, sim.data, base_id=sim.base_id,
                      cam="free" if args.free else "track",
                      control_rate_hz=sim.cfg["sim"]["control_rate_hz"]) as viewer:
        viewer.run(step)


if __name__ == "__main__":
    main()
