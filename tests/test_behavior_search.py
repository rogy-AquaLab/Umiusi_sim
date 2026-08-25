"""Unit tests for the SEARCH sweep's IMU-outage fallback (autonomy#19-3).

The sweep judges "one full turn" by integrating |measured yaw_rate|·dt; a dead IMU (real run
2026-08-25: 15.4 s + 11.1 s outages) freezes that integral and the FSM spins in place forever.
SWEEP_TIMEOUT_S must bound it: a sweep that outlives the timeout is treated as complete.

Runnable two ways:
    python -m pytest tests/test_behavior_search.py          # if pytest is installed
    python tests/test_behavior_search.py                    # standalone (plain asserts)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "perception" / "src"))

from umiusi_perception.autonomy.behavior import SWEEP_TIMEOUT_S, BalloonBehavior  # noqa: E402


def _run_search(fsm, yaw_rate, seconds):
    """Step the FSM with no detections for `seconds`; return the number of TRANSLATING entries."""
    translations = 0
    steps = int(round(seconds / fsm.dt))
    was_translating = False
    for _ in range(steps):
        fsm.step([], yaw_rate=yaw_rate, heading=0.0, dt=fsm.dt, fresh=False)
        translating = fsm._translating > 0
        if translating and not was_translating:
            translations += 1
        was_translating = translating
    return translations


def test_dead_yaw_feedback_still_translates():
    """yaw_rate stuck at 0 (IMU outage) -> the sweep times out and the FSM still moves on."""
    fsm = BalloonBehavior(dt=0.02)
    assert fsm.state == "SEARCH"
    n = _run_search(fsm, yaw_rate=0.0, seconds=2.5 * SWEEP_TIMEOUT_S)
    assert n >= 2, f"expected >=2 timeout-driven translations, got {n}"
    assert fsm.state == "SEARCH"


def test_live_yaw_feedback_completes_by_integration():
    """A healthy yaw rate finishes the 2*pi sweep well BEFORE the timeout (fallback stays idle)."""
    fsm = BalloonBehavior(dt=0.02)
    yaw_rate = 0.56                       # measured real-robot sweep rate [rad/s]
    expect_s = 2.0 * math.pi / yaw_rate   # ~11.2 s per sweep
    n = _run_search(fsm, yaw_rate=yaw_rate, seconds=1.5 * expect_s)
    assert n == 1, f"expected exactly 1 integration-driven translation, got {n}"
    # The sweep completed by integration, not by the timeout.
    assert expect_s < SWEEP_TIMEOUT_S


def test_sweep_timer_resets_on_start_sweep():
    """_start_sweep() clears the fallback timer, so a fresh sweep gets the full budget."""
    fsm = BalloonBehavior(dt=0.02)
    _run_search(fsm, yaw_rate=0.0, seconds=0.5 * SWEEP_TIMEOUT_S)
    assert fsm._sweep_time > 0.0
    fsm._start_sweep()
    assert fsm._sweep_time == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
