"""CLI: head-to-head eval of the LEARNED detector vs the classical baselines on the same split.

The harness itself (``compare`` + the shared IoU machinery) now lives in ``umiusi_perception.eval``;
this file is just the ``python -m tools.perception_eval_learned`` command over it.

    python -m tools.perception_eval_learned --weights models/perception_learned/tiny_balloon.pt
    python -m tools.perception_eval_learned --weights ... --method learned   # learned only
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from umiusi_perception.eval import DATA_ROOT, compare

CLASSICAL = ["color", "hough", "combined"]


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
