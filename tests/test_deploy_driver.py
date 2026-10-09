"""tools/deploy_driver.py — the competition harness's stand-in for the robot's control chain.

Pins the two facts every competition number rests on: the driver loads the deploy BUNDLE (gains and
allocator knobs, not library defaults — without prefer_deg/dead_hold a heave command averages to zero,
measured 2026-10-01), and +yaw turns the nose toward body +Z (the FSM's contract).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools.deploy_driver import DEFAULT_BUNDLE, DeployDriver  # noqa: E402
from umiusi_sim.simulator import UmiusiSimulator  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_BUNDLE.exists(), reason="deploy bundle not exported")


def _run(cmd, steps=150, **kw):
    sim = UmiusiSimulator()
    sim.set_net_buoyancy(0.0)
    sim.reset(pos=(0.0, 1.5, 0.0))
    drv = DeployDriver(sim, **kw)
    for _ in range(steps):
        st = sim.step(drv.step(cmd))
    return sim, st


def test_heave_command_moves_the_hull():
    _sim, up = _run({"heave": 1.0})
    _sim, down = _run({"heave": -1.0})
    assert up["lin_vel"][1] > 0.15 and down["lin_vel"][1] < -0.15


def test_positive_yaw_turns_toward_plus_z():
    sim, _ = _run({"yaw": 0.5})
    fwd = sim.data.xmat[sim.base_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
    assert fwd[2] > 0.3, fwd


def test_servo_aware_holds_level_on_a_flatter_thrust_curve():
    """exp 1.6 plant vs a 2.0 bundle: the target-angle duty excites a limit cycle, servo-aware damps it."""
    def wobble(servo_aware):
        sim = UmiusiSimulator()
        sim.set_net_buoyancy(0.0)
        sim.thrust_curve_exp = 1.6
        # The 10/03-calibrated plant damps roll/pitch hard enough that no limit cycle forms at all,
        # so this feature test runs on the pre-calibration roll/pitch plant, where it does.
        sim.thrust_vertical_eff = 1.0
        for i in (3, 5):
            sim.drag_lin[i] /= 8.0
            sim.drag_quad[i] /= 8.0
            sim.added_mass_diag[i] /= 3.0
        sim.reset(pos=(0.0, 1.5, 0.0))
        drv = DeployDriver(sim, servo_aware=servo_aware)
        roll = []
        for k in range(750):
            if k == 50:
                sim.data.qvel[3:6] += np.array([0.5, 0.0, 0.3])
            sim.step(drv.step({}))
            R = sim.data.xmat[sim.base_id].reshape(3, 3)
            roll.append(math.degrees(math.asin(float(np.clip((R @ [0.0, 0.0, 1.0])[1], -1, 1)))))
        return float(np.std(roll[250:]))
    assert wobble(True) < 0.5 * wobble(False)
