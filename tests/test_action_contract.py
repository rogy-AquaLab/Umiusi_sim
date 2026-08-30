"""The exported deploy contract must describe the plant the sim actually ran.

`meta.json`'s `action_contract` is what the robot reproduces (integrate -> mix -> fold). Nothing
downstream can tell that it has drifted from the sim: the bundle loads, the policy runs, and the
vehicle simply gets a different plant than the trained one. Until now the only check was an
equivalence run someone remembered to do by hand — this pins it in pytest instead.
"""

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from export_policy import action_contract  # noqa: E402
from umiusi_rl.envs.mode_mixer import DEADBAND_FRAC, _MODE_SIGNS  # noqa: E402
from umiusi_sim.simulator import UmiusiSimulator  # noqa: E402

_TRAIN_CFG = "configs/train_ppo_mode_ft.yaml"


@pytest.fixture(scope="module")
def contract():
    train_cfg = yaml.safe_load((_ROOT / _TRAIN_CFG).read_text())
    sim_cfg = yaml.safe_load((_ROOT / train_cfg["sim_config"]).read_text())
    c = action_contract({"action_mode": "modes"}, train_cfg["env"], sim_cfg)
    assert c is not None
    return c


def test_no_contract_for_esc_policies(contract):
    """Only a modes policy carries one; an esc bundle must not grow a bogus contract."""
    assert action_contract({"action_mode": "esc"}, {}, {}) is None
    assert action_contract({}, {}, {}) is None


def test_plant_constants_match_the_simulator(contract):
    """The constants must equal what the sim ran, not literals that happen to agree today.

    thrust_per_cmd is UNMEASURED pending bench calibration, so this is the check that catches a
    retune shipping a stale contract.
    """
    sim = UmiusiSimulator(config_path=_ROOT / "configs" / "umiusi.yaml")
    assert contract["thrust_per_cmd"] == pytest.approx(sim.thrust_per_cmd)
    assert contract["thrust_curve_exp"] == pytest.approx(sim.thrust_curve_exp)
    assert contract["servo_range_deg"] == pytest.approx(sim.servo_range_rad * 180.0 / 3.141592653589793)
    assert contract["control_rate_hz"] == pytest.approx(sim.cfg["sim"]["control_rate_hz"])
    assert contract["deadband_frac"] == pytest.approx(DEADBAND_FRAC)


def test_slew_matches_the_training_config(contract):
    """The policy rides the limiter, so a deploy running a different slew runs a different plant."""
    env = yaml.safe_load((_ROOT / _TRAIN_CFG).read_text())["env"]
    assert contract["mode_slew_per_s"] == pytest.approx(env["mode_slew_per_s"])
    assert contract["mode_slew_per_s"] > 0.0


def test_mode_signs_are_the_mixer_table_in_the_declared_column_order(contract):
    """The deploy side rebuilds Sh/Sv from these, so table and column order must both travel."""
    assert contract["mode_sign_columns"] == ["fx", "fy", "tz", "fz", "tx", "ty"]
    assert contract["mode_signs"] == {n: list(s) for n, s in _MODE_SIGNS.items()}
    assert contract["mode_names"] == ["fx", "fy", "fz", "tx", "ty", "tz"]


def test_all_three_stages_are_described(contract):
    """A deploy node that skips a stage gets a different plant (the A-11 failure)."""
    keys = [k for stage in contract["stages"] for k in stage]
    assert keys == ["1_integrate", "2_mix", "3_fold"]
