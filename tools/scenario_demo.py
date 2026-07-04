"""Render demo for the competition balloon-popping WORLD (headless via EGL/OSMesa).

Composes the scene (base robot + pool + tethered balloons + popping pin) with MjSpec, loads
it into UmiusiSimulator (base robot/physics unchanged), steps a few control-steps with a
gentle forward thrust, then saves a front_cam and a down_cam frame. Also reports the balloon
table and whether the pin tip has "popped" any balloon (geometric check).

Usage:
    MUJOCO_GL=egl python -m tools.scenario_demo [out_dir]
"""

import pathlib
import sys
import tempfile

import imageio
import numpy as np

from umiusi_sim.description.scenarios import competition_balloon as scn
from umiusi_sim.simulator import UmiusiSimulator

_TMP = pathlib.Path(tempfile.gettempdir()) / "umiusi_sim"  # portable default output dir
OUT_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _TMP
START = (0.0, 1.0, 0.0)  # ~1 m off the pool floor, on the +X approach axis


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compose -> flatten to a self-contained MJCF -> load through the normal simulator path.
    xml_path = scn.write_xml(OUT_DIR / "competition_balloon.xml")
    sim = UmiusiSimulator(model_path=xml_path)
    print(f"composed model: nbody={sim.model.nbody} ngeom={sim.model.ngeom} -> {xml_path}")

    sim.reset(pos=START)
    action = np.zeros(8)
    action[4:8] = 0.35  # gentle forward thrust
    for _ in range(30):
        sim.step(action)
    state = sim.get_state()
    print(f"after 30 control-steps: pos={np.round(state['pos'], 3)} "
          f"vel={np.round(state['lin_vel'], 3)}")

    # Pop check: pin tip vs. each balloon centre (balloons are static, so world pos == layout).
    pin_tip = sim.data.site_xpos[sim.model.site("pin_tip").id]
    print(f"pin tip @ {np.round(pin_tip, 3)}")
    for b in scn.balloon_table():
        hit = scn.popped(pin_tip, b["pos"])
        print(f"  {b['name']:22s} {b['colour']:6s} {b['points']:+3d} pts  "
              f"@ {np.round(b['pos'], 2)}  popped={hit}")

    frames = {}
    for cam in ("front_cam", "down_cam"):
        frame = sim.render_camera(camera=cam, width=480, height=360)
        out = OUT_DIR / f"scenario_{cam}.png"
        imageio.imwrite(out, frame)
        frames[cam] = frame
        # Balloon colours are saturated; a crude "are non-water colours present?" visibility hint.
        colourful = int(np.count_nonzero(frame.max(axis=2).astype(int) - frame.min(axis=2).astype(int) > 60))
        print(f"wrote {out}  shape={frame.shape}  colourful_px={colourful}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
