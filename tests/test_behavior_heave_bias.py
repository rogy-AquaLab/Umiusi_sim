"""The FSM's camera-learned heave bias (`BalloonBehavior.ki_heave`, 2026-10-01).

No depth sensor and no software buoyancy trim (D-3): a hull that is only roughly neutral meets every
balloon with a standing vertical error, because the heave loop is P-only on the target's elevation
and the vertical axis is an open-loop force. Measured in tools/competition_eval (+1 N hull): the pin
parked 13-29 cm above the yellow and never rammed. The bias integrates the elevation error the camera
reports and adds it to every heave command.

Two things must hold: OFF changes nothing (the deployed default), and ON learns the right SIGN — a
target persistently above the optic axis means the hull sinks, so the bias must push up.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "perception" / "src"))
sys.path.insert(0, str(_ROOT))

from tools.ram_eval import make_detection  # noqa: E402
from umiusi_perception.autonomy.behavior import BalloonBehavior  # noqa: E402


def _drive(fsm, el, steps=150):
    """A red balloon dead ahead at 1.5 m, held `el` rad above the optic axis, every frame fresh."""
    cmds = []
    for _ in range(steps):
        d = make_detection(0.0, el, 1.5, "red", 30)
        cmd, _info = fsm.step([d], yaw_rate=0.0, heading=0.0, dt=fsm.dt, fresh=True)
        cmds.append(cmd)
    return cmds


def test_off_is_the_previous_behaviour():
    a = _drive(BalloonBehavior(dt=0.02), el=0.15)
    b = _drive(BalloonBehavior(dt=0.02, ki_heave=0.0), el=0.15)
    assert a == b


def test_on_learns_the_sign_of_a_standing_error():
    up = BalloonBehavior(dt=0.02, ki_heave=0.3)
    _drive(up, el=0.15)                       # target stays ABOVE -> hull is sinking -> push up
    down = BalloonBehavior(dt=0.02, ki_heave=0.3)
    _drive(down, el=-0.25)                    # target stays BELOW -> hull is floating -> push down
    assert up._heave_bias > 0.0, up._heave_bias
    assert down._heave_bias < 0.0, down._heave_bias
    assert abs(down._heave_bias) <= down.heave_bias_max


def test_bias_persists_into_search():
    """The residual buoyancy belongs to the hull, not to one balloon: keep holding depth while searching."""
    fsm = BalloonBehavior(dt=0.02, ki_heave=0.3)
    _drive(fsm, el=0.15)
    learned = fsm._heave_bias
    cmd, _ = fsm.step([], yaw_rate=0.0, heading=0.0, dt=fsm.dt, fresh=True)
    assert fsm._heave_bias == learned
    assert learned != 0.0


def test_bearing_propagation_off_changes_nothing_and_on_tracks_the_gyro():
    """Held frames: with propagation the azimuth the FSM steers on moves by the gyro's yaw (+wy -> +az)."""
    d = make_detection(0.10, 0.0, 1.5, "red", 30)
    a, b = BalloonBehavior(dt=0.02), BalloonBehavior(dt=0.02, propagate_bearing=False)
    for fresh in (True, False, False, False):
        assert a.step([d], yaw_rate=0.5, dt=0.02, fresh=fresh) == b.step([d], yaw_rate=0.5, dt=0.02, fresh=fresh)
    on = BalloonBehavior(dt=0.02, propagate_bearing=True)
    on.step([d], yaw_rate=0.5, dt=0.02, fresh=True)
    for _ in range(5):
        on.step([d], yaw_rate=0.5, dt=0.02, fresh=False)
    assert abs(on._yaw_since_frame - 0.05) < 1e-9
    assert d.bearing[0] == 0.10, "the caller's detection must not be mutated"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
