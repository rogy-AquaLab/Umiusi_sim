"""Batch underwater colour-correction of a folder of frames (for labelling / detector preprocessing).

    uv run --extra dev --extra perception python -m tools.underwater_correct \
        --in ../ai/balloon/raw/youtube --out ../ai/balloon/raw/youtube_cc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from umiusi_sim.perception.underwater import underwater_correct

EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--no-clahe", action="store_true")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        rgb = np.asarray(imageio.imread(p))[:, :, :3]
        out = underwater_correct(rgb, clahe=not args.no_clahe)
        imageio.imwrite(dst / p.name, out, quality=92)
        n += 1
    print(f"corrected {n} frames: {src} -> {dst}")


if __name__ == "__main__":
    main()
