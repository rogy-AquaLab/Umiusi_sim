"""Unit tests for the competition_balloon scenario: YAML-configured counts, the even-spacing
field sampler, and the tether-entanglement check.

Runnable two ways:
    python -m pytest tests/test_competition_balloon.py      # if pytest is installed
    python tests/test_competition_balloon.py                # standalone (plain asserts)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "sim" / "src"))

from umiusi_sim.description.scenarios import competition_balloon as scn  # noqa: E402


def _min_pairwise_xz(layout):
    """Smallest centre-to-centre XY (x, z) distance between any two balloons in a layout."""
    pts = [(x, z) for _, _, x, z in layout]
    return min(math.hypot(ax - bx, az - bz)
               for i, (ax, az) in enumerate(pts) for (bx, bz) in pts[i + 1:])


def test_counts_match_config():
    """The sampler emits exactly the configured per-colour counts (start yellow counts as one)."""
    counts = {"red": 7, "yellow": 7, "blue": 5}
    layout = scn.sample_layout(np.random.default_rng(0), counts=counts, min_separation=0.6)
    got = {"red": 0, "yellow": 0, "blue": 0}
    for _, colour, _, _ in layout:
        got[colour] += 1
    assert got == counts, got
    assert len(layout) == sum(counts.values())


def test_start_yellow_is_balloon_zero():
    """Balloon 0 is always the deterministic tall yellow start target."""
    layout = scn.sample_layout(np.random.default_rng(3))
    assert layout[0] == scn.BALLOON_LAYOUT[0]
    assert layout[0][0] == "balloon_yellow_start"


def test_even_spacing_no_overlap():
    """Every pair of balloons is at least min_separation apart (Poisson-disk / rejection)."""
    sep = 0.6
    layout = scn.sample_layout(np.random.default_rng(1),
                               counts={"red": 7, "yellow": 7, "blue": 5}, min_separation=sep)
    # Small numerical slack for the best-effort fallback; must be no worse than ~1 mm under sep.
    assert _min_pairwise_xz(layout) >= sep - 1e-3


def test_reproducible():
    """Same seed -> identical layout."""
    a = scn.sample_layout(np.random.default_rng(42))
    b = scn.sample_layout(np.random.default_rng(42))
    assert a == b


def test_config_defaults_loaded():
    """load_field_config returns the YAML-configured counts + separation (or module defaults)."""
    counts, sep = scn.load_field_config()
    assert set(counts) == {"red", "yellow", "blue"}
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())
    assert sep > 0.0


def test_entanglement_under_unpopped_flagged():
    """Robot horizontally under an un-popped balloon and below its height -> flagged."""
    balloons = [{"name": "b_red", "pos": np.array([2.0, 0.5, 0.5])}]
    # Robot right under the balloon's XY, below its 0.5 m height.
    tangled = scn.entanglement((2.0, 0.2, 0.5), balloons)
    assert tangled == ["b_red"]


def test_entanglement_clear_not_flagged():
    """Robot far from any wire (horizontally) -> not flagged."""
    balloons = [{"name": "b_red", "pos": np.array([2.0, 0.5, 0.5])}]
    far = scn.entanglement((2.0 + scn.TETHER_RADIUS + 0.1, 0.2, 0.5), balloons)
    assert far == []


def test_entanglement_above_balloon_not_flagged():
    """Robot near the wire XY but ABOVE the balloon (not under-passing) -> not flagged."""
    balloons = [{"name": "b_red", "pos": np.array([2.0, 0.5, 0.5])}]
    above = scn.entanglement((2.0, 0.9, 0.5), balloons)
    assert above == []


def test_entanglement_popped_excluded():
    """A popped balloon's wire is gone -> even under-passing it is not flagged."""
    balloons = [{"name": "b_red", "pos": np.array([2.0, 0.5, 0.5])}]
    res = scn.entanglement((2.0, 0.2, 0.5), balloons, popped={"b_red"})
    assert res == []


def test_entanglement_counts_multiple():
    """Two overlapping un-popped tethers under the robot -> both flagged (count = 2)."""
    balloons = [
        {"name": "b1", "pos": np.array([2.0, 0.5, 0.5])},
        {"name": "b2", "pos": np.array([2.05, 1.5, 0.5])},
    ]
    res = scn.entanglement((2.0, 0.2, 0.5), balloons)
    assert set(res) == {"b1", "b2"}
    assert len(res) == 2


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
