"""How much performance is the SERVO RATE LIMIT costing, and how much of it is still on the table?

An azimuth unit steers: its thrust direction is a servo angle, which is a plant STATE with a rate
limit (250 deg/s nominal, 100-500 under DR, plus a 0.05 s lag). So the wrench reachable at the
next control step is a neighbourhood of the current one — allocation is a sequential decision
problem, not the per-step algebra `GeneralAllocator` solves. The `w_move` term is only a one-step
greedy approximation of that; a receding-horizon allocator, or a learned policy (its amortised
form), could in principle plan the servo trajectory properly.

Before building either, bound the prize. Running the same controller against a servo with an
ARTIFICIALLY FAST slew gives the performance no allocator can exceed — the residual gap between
the real-servo number and that oracle is everything a smarter allocator could ever recover:

    oracle - real  ==  0   -> the actuator dynamics cost nothing; stop here.
    oracle - real  >>  0   -> that gap is the budget for MPC / RL over the servo state.

Run it for both allocators, because the answer differs: the minimum-norm solve wastes most of the
servo's rate budget on azimuth-singularity chatter, so its gap is dominated by a defect that is
already fixed rather than by anything fundamental.

    python tools/actuator_prize.py --episodes 8
    python tools/actuator_prize.py --episodes 8 --cruise
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from fault_compare import run_classical  # noqa: E402

# label -> plant actuator limits. The oracle keeps the SAME thrust hardware and only removes the
# steering constraint, so any difference is attributable to the servo, not to extra authority.
# env._base keys (radians for the servo slew). Domain randomization normally re-draws all three,
# so DR_FIX below switches those specific draws off — otherwise the rows would not be comparable.
_S = np.radians
PLANTS = [
    ("実機相当 250deg/s", {"servo_slew": _S(250.0), "servo_tau": 0.05, "thrust_slew": 4.0}),
    ("サーボ 1000deg/s", {"servo_slew": _S(1000.0), "servo_tau": 0.05, "thrust_slew": 4.0}),
    ("サーボ 無限+遅れ0", {"servo_slew": _S(1.0e6), "servo_tau": 0.0, "thrust_slew": 4.0}),
    ("上に ESC も無限", {"servo_slew": _S(1.0e6), "servo_tau": 0.0, "thrust_slew": 1.0e6}),
]

# keep every OTHER domain randomization live; only pin the actuator-rate draws
DR_FIX = {"servo_slew_range_deg_s": None, "thrust_slew_range": None, "servo_tau_frac": 0.0}

ALLOCS = [("minnorm", None), ("特異点回避60", {"prefer_deg": 60.0, "w_move": 3.0})]


def _one(job):
    alloc, plant, episodes, dr, disturb, cruise, gains = job
    return run_classical(episodes, None, 0.0, True, dr, disturb, gains=gains, cruise=cruise,
                         alloc=alloc, plant=plant, dr_cfg=dict(DR_FIX) if dr else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--cruise", action="store_true")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--no-disturb", action="store_true")
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    args = ap.parse_args()
    gains = {"kp": args.kp, "kd": args.kd}
    dr, disturb = not args.no_dr, not args.no_disturb

    jobs = [(a, t, args.episodes, dr, disturb, args.cruise, gains)
            for _, a in ALLOCS for _, t in PLANTS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_one, jobs))

    print(f"アクチュエータ拘束の代価  DR={dr} disturb={disturb} "
          f"{'cruise' if args.cruise else 'hold'}  episodes={args.episodes}  gains={gains}")
    print(f"{'プラント':<20}" + "".join(f"{n:>26}" for n, _ in ALLOCS))
    print(f"{'':<20}" + "".join(f"{'ori    横流れ     esc':>26}" for _ in ALLOCS))
    n = len(PLANTS)
    for i, (plabel, _t) in enumerate(PLANTS):
        line = f"{plabel:<20}"
        for k in range(len(ALLOCS)):
            m = res[k * n + i]
            line += f"{m['ori']:>10.3f}{m['drift']:>9.4f}{m['esc']:>7.3f}"
        print(line)
    for k, (alabel, _a) in enumerate(ALLOCS):
        real, oracle = res[k * n]["ori"], res[k * n + 2]["ori"]
        print(f"\n  {alabel}: 実機 {real:.3f} -> サーボ無限 {oracle:.3f}   "
              f"残る賞金 {real - oracle:+.3f} rad ({(real - oracle) / max(real, 1e-9) * 100:.0f}%)"
              f" = より賢い配分器 (MPC/RL) が取りうる上限")


if __name__ == "__main__":
    main()
