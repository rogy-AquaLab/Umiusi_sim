"""The sim and the robot must run the SAME controller, and this is what enforces it.

The controller lives in `umiusi_perception.classical` (the only package installed on the Pi) and
the sim drives it through a `PlantContract`. Three things can quietly break that arrangement, and
each has already happened once in some form on this project:

1. THE ROBOT PACKAGE GROWS A SIM DEPENDENCY. `umiusi_perception` must import nothing from
   `umiusi_sim` / `umiusi_rl` / mujoco, or the Pi cannot install it at all (aarch64 MuJoCo is a
   separate lift). A subprocess with only this package on the path is the only honest check.
2. THE CONTRACT STOPS MATCHING THE SIM. The whole point is that the robot reproduces the plant
   the controller was tuned against; a contract built from the sim and one loaded from JSON must
   produce the same wrench, bit for bit.
3. THE DUPLICATED `reachable_speed` DRIFTS. It is deliberately implemented twice — the sim wheel
   imports nothing from the perception wheel, so a training box that installs `packages/sim[rl]`
   alone still needs the command ceiling. Duplicating four lines of algebra is fine; letting the
   two copies disagree is not.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from umiusi_perception.classical import ClassicalController, PlantContract
from umiusi_perception.classical import reachable_speed as deploy_reachable
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config
from umiusi_rl.envs.umiusi_pose_env import reachable_speed as sim_reachable

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from classical_control import contract_from_sim  # noqa: E402


def _env():
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="modes", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg.setdefault("disturbance", {})["enabled"] = False
    return UmiusiPoseEnv(cfg)


def test_reachable_speed_agrees_between_the_two_wheels():
    """Both copies solve thrust = drag. If you change one, change the other."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.uniform(5.0, 60.0)
        exp = rng.uniform(1.0, 2.8)
        lin, quad = rng.uniform(0.0, 30.0), rng.uniform(0.0, 200.0)
        cap = rng.uniform(0.05, 1.0)
        assert np.isclose(sim_reachable(k, exp, lin, quad, cap),
                          deploy_reachable(k, exp, lin, quad, cap), rtol=1e-12), \
            f"the two reachable_speed copies disagree at k={k} exp={exp} lin={lin} quad={quad} cap={cap}"


def test_contract_round_trips_through_json_and_controls_identically(tmp_path):
    """A contract written for the robot must drive the controller exactly as the live sim one does."""
    env = _env()
    live = contract_from_sim(env.sim)
    env.close()

    path = tmp_path / "classical_bundle.json"
    live.save(path)
    loaded = PlantContract.load(path)

    for field in ("thrust_axes", "pivots_from_com", "drag_lin", "drag_quad", "added_mass_diag"):
        assert np.allclose(getattr(live, field), getattr(loaded, field)), f"{field} did not round-trip"

    rng = np.random.default_rng(1)
    a, b = ClassicalController(live, cap_tau=0.0), ClassicalController(loaded, cap_tau=0.0)
    for _ in range(20):
        args = (rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.05,
                rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.05, rng.uniform(0.2, 0.4))
        assert np.allclose(a.wrench(*args), b.wrench(*args), rtol=0, atol=0), \
            "the JSON contract and the live sim contract commanded different wrenches"


def test_contract_version_must_match():
    """A stale bundle on the robot is the A-11 failure mode; it has to fail loudly, not silently."""
    env = _env()
    d = contract_from_sim(env.sim).to_dict()
    env.close()
    d["version"] = 999
    try:
        PlantContract.from_dict(d)
    except ValueError as e:
        assert "version" in str(e)
    else:
        raise AssertionError("a mismatched contract version was accepted")


def test_the_robot_module_imports_with_neither_the_sim_wheel_nor_mujoco():
    """`umiusi_perception` is the ONLY package installed on the Pi. Keep it that way."""
    src = _ROOT / "packages" / "perception" / "src"
    code = (
        "import sys\n"
        "sys.modules['mujoco'] = None\n"           # any real import of it now raises
        "sys.modules['umiusi_sim'] = None\n"
        "sys.modules['umiusi_rl'] = None\n"
        "import umiusi_perception.classical as c\n"
        "assert c.PlantContract and c.ClassicalController and c.GeneralAllocator\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, f"the deployable module needs the sim wheel:\n{r.stderr}"


def test_mode_to_cad_wrench_matches_the_expression_it_replaced():
    """`cad_wrench_from_modes` was open-coded in eight tools; it must agree with all of them."""
    from umiusi_perception.classical import cad_wrench_from_modes

    rng = np.random.default_rng(2)
    for _ in range(20):
        m, f = rng.normal(size=6), rng.uniform(1.0, 50.0)
        old = np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f
        assert np.allclose(cad_wrench_from_modes(m, f), old, rtol=0, atol=0)


def test_export_writes_a_loadable_bundle(tmp_path):
    out = tmp_path / "classical_bundle.json"
    r = subprocess.run([sys.executable, str(_ROOT / "tools" / "export_classical.py"),
                        "--out", str(out)], capture_output=True, text=True, cwd=_ROOT)
    assert r.returncode == 0, r.stderr
    bundle = json.loads(out.read_text())
    plant = PlantContract.from_dict(bundle["contract"])
    ctl = ClassicalController(plant, **bundle["gains"])
    # the exported gains must actually be usable, and the trim must push DOWN on a buoyant hull
    m = ctl.wrench(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.25)
    assert m[2] < 0.0, f"buoyancy trim should push down, got {m[2]}"
    assert set(bundle) >= {"contract", "gains", "allocator", "frames", "uncalibrated"}
