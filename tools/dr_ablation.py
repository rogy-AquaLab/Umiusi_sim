"""Which part of domain randomization actually costs the classical controller its attitude?

"DR is on" is thirteen different perturbations at once, and they are not the same KIND of problem:

    a wrong CONSTANT (thrust gain, drag, CoB height, servo neutral) — a controller can integrate
    it away, and the ki sweep showed an integrator buys ~4 %, so this is not where the loss is;
    a wrong RATE (servo slew 100-500 deg/s, ESC ramp 1-10 esc/s, action latency) — no gain fixes
    a plant that cannot move faster than it moves;
    a wrong AUTHORITY (max_duty 0.2-0.4) — observed, so it should cost nothing.

Attribution matters because it decides what is worth doing next. If the loss is rate limits, then
neither better gains nor a learned policy can recover it in the controller — it is a hardware and
deploy-contract fact, and the honest move is to bound it and stop. If it is constants, tune. This
runs the classical controller with ONE knob live at a time, everything else nominal.

    python tools/dr_ablation.py --episodes 6
    python tools/dr_ablation.py --episodes 6 --cruise
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from fault_compare import run_classical  # noqa: E402

# Everything off. _apply_domain_rand skips a range when it is falsy and a frac when it is 0.0,
# so this is DR "enabled" but perturbing nothing — it must reproduce the DR-off numbers.
NEUTRAL = {
    "buoyancy_frac": 0.0, "thrust_frac": 0.0, "drag_frac": 0.0, "added_mass_frac": 0.0,
    "servo_slew_range_deg_s": None, "thrust_slew_range": None, "servo_tau_frac": 0.0,
    "servo_offset_deg": 0.0, "thrust_exp_range": None, "buoyancy_offset_frac": 0.0,
    "max_duty_range": None, "thrust_unit_frac": 0.0, "action_latency_steps": 0,
}

# label -> the single knob restored to its train-config value, and what kind of error it is
KNOBS = [
    ("none (対照)",        {},                                          "—"),
    ("推力ゲイン ±15%",     {"thrust_frac": 0.15},                       "定数"),
    ("抗力 ±30%",          {"drag_frac": 0.30},                         "定数"),
    ("付加質量 ±40%",       {"added_mass_frac": 0.40},                   "定数"),
    ("浮力 ±5%",           {"buoyancy_frac": 0.05},                     "定数"),
    ("CoB 高さ ±60%",      {"buoyancy_offset_frac": 0.6},               "定数"),
    ("サーボ中立 ±3deg",    {"servo_offset_deg": 3.0},                   "定数"),
    ("ユニット推力 ±10%",   {"thrust_unit_frac": 0.10},                  "定数"),
    ("推力指数 1.5-2.8",   {"thrust_exp_range": [1.5, 2.8]},            "定数"),
    ("サーボ速度 100-500",  {"servo_slew_range_deg_s": [100.0, 500.0]},  "レート"),
    ("ESC ランプ 1-10/s",  {"thrust_slew_range": [1.0, 10.0]},          "レート"),
    ("サーボ tau ±50%",    {"servo_tau_frac": 0.5},                     "レート"),
    ("作動遅れ 1 step",     {"action_latency_steps": 1},                 "レート"),
    ("cap 0.2-0.4",       {"max_duty_range": [0.2, 0.4]},              "権限"),
    ("全部 (=DR on)",      None,                                        "—"),
]


def _one(job):
    knob, episodes, cruise, gains, loo = job
    if knob is None:
        dr_cfg = None                                   # everything live
    elif loo:
        dr_cfg = {k: NEUTRAL[k] for k in knob}          # everything live EXCEPT this one
    else:
        dr_cfg = {**NEUTRAL, **knob}                    # only this one live
    return run_classical(episodes, None, 0.0, True, True, False,
                         gains=gains, cruise=cruise, dr_cfg=dr_cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cruise", action="store_true")
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-out: DR fully on with ONE knob pinned — what removing that "
                         "uncertainty (by bench calibration, or by estimating it online) buys")
    ap.add_argument("--kp", type=float, default=2.2)
    ap.add_argument("--kd", type=float, default=0.45)
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--k-v", type=float, default=1.2)
    args = ap.parse_args()
    gains = {"kp": args.kp, "kd": args.kd, "ki": args.ki, "k_v": args.k_v}

    # LOO drops the "perturb nothing" control (meaningless there) but keeps full DR as the reference
    knobs = KNOBS if not args.loo else [k for k in KNOBS if k[1] is None or k[1]]
    jobs = [(k, args.episodes, args.cruise, gains, args.loo) for _, k, _ in knobs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_one, jobs))

    # single-knob mode compares against "DR on but perturbing nothing"; LOO against full DR
    ref = res[-1]["ori"] if args.loo else res[0]["ori"]
    head = "外した要素 (残りは全部 DR)" if args.loo else "有効にした要素"
    print(f"DR {'除外' if args.loo else '個別'}寄与  {'cruise' if args.cruise else 'hold'}  "
          f"episodes={args.episodes}  gains={gains}")
    print(f"{head:<22}{'種別':<7}{'ori':>8}{'基準比':>9}{'横流れ':>9}{'v̂誤差':>8}")
    for (label, _knob, kind), r in zip(knobs, res):
        print(f"{label:<22}{kind:<7}{r['ori']:8.3f}{r['ori'] / max(ref, 1e-9):8.2f}x"
              f"{r['drift']:9.4f}{r['verr']:8.3f}")


if __name__ == "__main__":
    main()
