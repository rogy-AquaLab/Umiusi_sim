"""Head-to-head evaluation of the LEARNED detector vs the classical color / hough detectors.

Part 2's deliverable is a fair, per-colour comparison of the learned ``TinyBalloonNet`` against the
classical baselines on the SAME val split, via the SAME IoU harness. To honour the "create only NEW
files, do not edit existing source" constraint, this reuses ``tools/perception_eval``'s machinery
(``load_split``, ``evaluate``, ``make_detfn``, ``print_report``, greedy IoU match) rather than
editing it -- it simply adds a ``learned`` method whose detection function comes from
``umiusi_sim.perception.learned_detector.load_learned_detector``. Folding ``learned`` into
``perception_eval``'s own ``--method`` later is a one-line change (add it to ``make_detfn``).

Usage:
    python -m tools.perception_eval_learned --weights models/perception_learned/tiny_balloon.pt
    python -m tools.perception_eval_learned --weights ... --method learned   # learned only
    python -m tools.perception_eval_learned --weights ... --split val        # (default)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from umiusi_sim.perception import REAL_THRESHOLDS
from umiusi_sim.perception.learned_detector import load_learned_detector

from tools.perception_eval import (
    DATA_ROOT,
    MAX_AREA_FRAC,
    evaluate,
    load_split,
    make_detfn,
    print_report,
)

CLASSICAL = ["color", "hough", "combined"]


def _detfn_for(method: str, weights: str, input_size: int | None, conf: float):
    """Return a rgb->[Detection] callable for a classical method or the learned detector."""
    if method == "learned":
        return load_learned_detector(weights, input_size=input_size, conf_thresh=conf)
    return make_detfn(method, REAL_THRESHOLDS, True, MAX_AREA_FRAC)


def compare(weights: str, data_root: pathlib.Path, split: str, methods: list[str],
            input_size: int | None = None, conf: float = 0.3) -> dict:
    """Run each method over the split, print per-colour reports + an overall comparison table.

    Returns ``{method: summary_dict}`` (summary as produced by ``print_report``)."""
    images, gt, split = load_split(data_root, split)
    n_gt = sum(len(v) for v in gt.values())
    print(f"dataset: {data_root}  split={split}  images={len(images)}  GT balloons={n_gt}")
    print("head-to-head: classical (real profile) vs learned (TinyBalloonNet, int8-ready)\n")

    summary = {}
    for m in methods:
        detfn = _detfn_for(m, weights, input_size, conf)
        res = evaluate(images, gt, data_root, split, REAL_THRESHOLDS, True, MAX_AREA_FRAC, detfn)
        summary[m] = print_report(f"METHOD = {m}", res[0], res[1], res[2], res[3])

    if len(methods) > 1:
        print("\n===== method comparison (overall) =====")
        print(f"  {'method':10s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'FP':>5s}")
        for m in methods:
            s = summary[m]
            print(f"  {m:10s} {s['prec']:6.2f} {s['rec']:6.2f} {s['f1']:6.2f} {s['fp']:5d}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, help="trained TinyBalloonNet checkpoint (.pt)")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--data-root", type=pathlib.Path, default=DATA_ROOT)
    ap.add_argument("--method", choices=["learned", *CLASSICAL, "all"], default="all")
    ap.add_argument("--input-size", type=int, default=None, help="override checkpoint input size")
    ap.add_argument("--conf", type=float, default=0.3, help="learned heatmap confidence floor")
    args = ap.parse_args()

    if not (args.data_root / "annotations" / f"{args.split}.json").exists():
        print(f"ERROR: dataset not found at {args.data_root} (need annotations/{args.split}.json)")
        return 1
    if not pathlib.Path(args.weights).exists():
        print(f"ERROR: weights not found: {args.weights} (train first with tools.perception_train)")
        return 1

    methods = ["learned", *CLASSICAL] if args.method == "all" else [args.method]
    compare(args.weights, args.data_root, args.split, methods,
            input_size=args.input_size, conf=args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
