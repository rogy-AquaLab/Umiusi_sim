"""Write the classical controller's deploy bundle: plant contract + tuned gains, as JSON.

The robot reproduces the plant from THIS COPY, never from a config it happens to have — the same
discipline `tools/export_policy.py` applies to a learned policy's `action_contract`, and for the
same reason: `thrust_per_cmd` / `thrust_curve_exp` are explicitly uncalibrated, so the day they
are re-fitted, anything that hardcoded them ships a different plant than the one the controller
was tuned against. That is the A-11 failure mode.

What the deploy node does with this file:

    from umiusi_perception.classical import ClassicalController, GeneralAllocator, PlantContract
    b = json.load(open("classical_bundle.json"))
    plant = PlantContract.from_dict(b["contract"])
    ctl = ClassicalController(plant, **b["gains"])
    alloc = GeneralAllocator(plant, **b["allocator"])
    ...
    m = ctl.wrench(ori_err, gyro, v_cmd, v_hat, max_duty)     # REP-103 body, modes in [-1, 1]
    act = alloc.allocate(cad_wrench(m, ctl.f_max_total(ctl.cap)), max_duty)   # [servo x4, esc x4]

`max_duty` is an INPUT, not a constant: the operator moves the cap by hand (0.25 -> 0.4) and
every cap-dependent quantity is solved from it inside the controller. Do not bake a cap into the
node, and do not let the node's clamp and the controller's cap disagree — pass the same number.

    python tools/export_classical.py                      # -> models/classical/classical_bundle.json
    python tools/export_classical.py --out /tmp/b.json --cruise
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from classical_control import contract_from_sim  # noqa: E402
from umiusi_perception.classical import ClassicalController, GeneralAllocator  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402

# The tuned operating point (tools/classical_tune.py --stage confirm, DR + disturbance).
# `prefer_deg` spends the allocator's null space on staying off the azimuth singularity, which is
# worth ori 0.332 -> 0.092 holding station — and is skipped automatically while cruising, where
# the required force already points away from the fold.
#
# kd: 0.35 is the optimum WITHOUT the BLDC startup dead time. With it (measured on the vehicle
# 2026-09-13: 2.5-3.2 s from rest, and a reversal while spinning is 0.23-0.89 s at the median but
# p90 ~3 s) the optimum moves UP to ~0.5 — the dead time makes overshoot far worse and kd is what
# suppresses it, so if you are unsure, go UP, never down. Two independent rigs agree on that
# direction (autonomy's tools/navigator_sim.py, and MuJoCo here). LEFT AT 0.35 ANYWAY because the
# difference is small next to the damage (ori 0.503 vs 0.482 rad at cap 0.30 — 4 %, against an 8x
# degradation from the dead time itself) and 0.5 has never been tried on the vehicle. The deploy
# side carries 0.5 as a documented recommendation (autonomy docs/field_card.md), not as a default,
# and `kp`/`kd` are ROS params there, so trying it in the field is `ros2 param set` — it does not
# need a re-export. Everything else in this bundle does.
HOLD_GAINS = {"kp": 1.0, "kd": 0.35, "k_v": 1.2, "ki": 0.0}
ALLOCATOR = {"prefer_deg": 60.0, "w_move": 3.0, "dead_hold": True}

# Runtime state, not configuration: which units are alive is decided on the robot (a dead BLDC), so
# it must not be frozen into a bundle exported weeks earlier.
_RUNTIME_ONLY = ("plant", "live")


def _effective(cls, chosen):
    """Every constructor parameter with the value the robot will actually use.

    The bundle used to carry only the tuned subset and let the rest fall through to the library
    defaults — so `cap_norm`, `cap_tau`, `k_ff`, `i_max`, `cap_margin`, `w_cap`, `w_flip` and
    `k_v_vert` reached the robot as whatever `umiusi_perception.classical` happened to default to
    that day. That is exactly the A-11 failure this bundle exists to prevent, just one level down:
    edit a default in the library and every already-exported bundle silently means something new,
    with nothing in the file to show it. Writing the full effective set makes such a change appear
    as a diff in the bundle instead.

    Derived from the signature rather than hand-listed, so a NEW knob cannot be forgotten —
    tests/test_classical_contract.py pins that the bundle names them all.
    """
    import inspect

    out = {}
    for name, prm in inspect.signature(cls.__init__).parameters.items():
        if name in ("self", *_RUNTIME_ONLY) or prm.kind is prm.VAR_KEYWORD:
            continue
        if name in chosen:
            out[name] = chosen[name]
        elif prm.default is inspect.Parameter.empty:
            raise SystemExit(f"{cls.__name__}.{name} has no default and no value chosen here")
        else:
            out[name] = prm.default
    unknown = set(chosen) - set(out)
    if unknown:
        raise SystemExit(f"{cls.__name__} has no parameter(s) {sorted(unknown)} — stale tuning?")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/train_ppo_mode_ft.yaml")
    ap.add_argument("--out", default="models/classical/classical_bundle.json")
    ap.add_argument("--cap-ref", type=float, default=None,
                    help="the esc cap the attitude gains are quoted at (default: the sim's max_duty). "
                         "This is NOT the deploy cap — the deploy cap is an input at run time.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["env"].update(task="attitude_velocity", action_mode="modes", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = False     # the contract is the NOMINAL plant
    env = UmiusiPoseEnv(cfg)
    contract = contract_from_sim(env.sim, cap_ref=args.cap_ref)
    env.close()

    bundle = {
        "contract": contract.to_dict(),
        "gains": _effective(ClassicalController, HOLD_GAINS),
        "allocator": _effective(GeneralAllocator, ALLOCATOR),
        "source_config": args.config,
        "frames": {
            "controller_io": "REP-103 body (x fwd, y left, z up)",
            "contract_geometry": "sim/CAD (+X fwd, +Y up, +Z starboard)",
            "convert": "umiusi_perception.classical.rep103_from_cad / cad_from_rep103",
        },
        "uncalibrated": ["thrust_per_cmd", "thrust_curve_exp", "drag_lin", "drag_quad",
                         "added_mass_diag"],
        "note": "絶対値は未較正。相対比較には使えるが、実機の速度・力の絶対値には使えない "
                "(docs/physics.md)。較正したら再 export すること — config だけ直しても "
                "このファイルは古いまま実機に残る。",
    }
    # Fingerprint of everything above, so a measurement can be tied back to the configuration it
    # was taken with. The 2026-09-13 pool run is the reason this exists: the bundle of the day left
    # 9 knobs to fall through to library defaults, and NOTHING in the bundle or the bag records what
    # those were — so the reversal frequency measured that day is not reproducible. The deploy node
    # should log this string at startup; then any bag says which plant and which gains produced it.
    bundle["fingerprint"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    out = Path(args.out)
    out = out if out.is_absolute() else _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    print("書き出し先:", out)
    print("  fingerprint:", bundle["fingerprint"], " (実機の起動ログに残すこと)")
    print(f"  cap_ref {contract.cap_ref}  control_rate {contract.control_rate_hz} Hz")
    for cap in (0.2, 0.25, 0.3, 0.4, 0.5):
        from umiusi_perception.classical import reachable_speed
        v = reachable_speed(contract.thrust_per_cmd, contract.thrust_curve_exp,
                            contract.drag_lin[0], contract.drag_quad[0], cap)
        print(f"  cap {cap:.2f} -> 到達速度 {v:.3f} m/s ({v / 0.5144:.2f} kt)")


if __name__ == "__main__":
    main()
