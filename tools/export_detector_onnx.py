#!/usr/bin/env python3
"""学習検出器 (.pt) を onnxruntime 用の .onnx に書き出し、torch と出力が一致することを確かめる。

    uv run python -m tools.export_detector_onnx W.pt [W2.pt ...] [--input-size N] [--images DIR]

- 出力は重みの隣の <重み名>_<input_size>.onnx (load_learned_detector(backend="onnx") が探す名前)
- input_size の既定はチェックポイントの cfg
- 確かめること: 乱数入力 8 枚で hm / wh の最大差 <= ONNX_ATOL、合成画像 (+ --images の jpg/png) で Detection が一致
- 一致しなければ書き出さず exit 1。一時ファイル経由なので既存の .onnx は壊れない
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
from umiusi_perception.learned_detector import (
    ONNX_ATOL,
    OnnxBalloonNet,
    bundled_onnx_path,
    detect_learned,
    export_onnx,
    load_learned_detector,
    onnx_max_abs_diff,
)


def synthetic_frames(h: int = 720, w: int = 1280) -> list[np.ndarray]:
    """水色の背景に赤 / 黄 / 青の円を 1 個ずつ置いた画像 3 枚 (前カメラと同じ 1280x720)。"""
    rng = np.random.default_rng(0)
    frames = []
    yy, xx = np.mgrid[:h, :w]
    for colour in ((220, 30, 30), (230, 210, 40), (40, 60, 220)):
        img = np.full((h, w, 3), (20, 80, 110), np.uint8)
        cy, cx, r = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4), rng.integers(h // 16, h // 6)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = colour
        frames.append(img)
    return frames


def image_frames(directory: str | None, n: int = 20) -> list[np.ndarray]:
    if not directory:
        return []
    from PIL import Image

    paths = sorted(p for p in pathlib.Path(directory).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths[:n]]


def same_detections(a, b, atol: float = 1e-4) -> bool:
    return len(a) == len(b) and all(
        da.colour == db.colour
        and da.bbox == db.bbox
        and np.allclose(da.bearing, db.bearing, atol=atol)
        and abs(da.range_m - db.range_m) <= atol
        for da, db in zip(a, b)
    )


def export_and_check(weights: str, input_size: int | None, frames: list[np.ndarray]) -> tuple[str, float, int, bool]:
    """書き出して確かめる。戻り値は (出力先, hm/wh の最大差, torch の検出数, Detection が全部一致したか)。"""
    det = load_learned_detector(weights, input_size=input_size, backend="torch")
    model, size = det.model, det.input_size
    out = bundled_onnx_path(weights, size)
    tmp = f"{out}.{os.getpid()}.tmp"
    try:
        export_onnx(model, size, tmp)
        runner = OnnxBalloonNet(tmp)
        diff = onnx_max_abs_diff(model, runner, size, n=8)
        n_dets, same = 0, True
        for img in frames:
            dt = detect_learned(img, model, input_size=size, conf_thresh=det.conf_thresh)
            do = detect_learned(img, runner, input_size=size, conf_thresh=det.conf_thresh)
            n_dets += len(dt)
            same = same and same_detections(dt, do)
        if diff <= ONNX_ATOL and same:
            os.replace(tmp, out)
        return out, diff, n_dets, same
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--input-size", type=int, default=None, help="既定はチェックポイントの値")
    ap.add_argument("--images", default=None, help="Detection の一致を確かめる実画像のディレクトリ (先頭 20 枚)")
    args = ap.parse_args(argv)

    frames = synthetic_frames() + image_frames(args.images)
    ok = True
    for w in args.weights:
        out, diff, n_dets, same = export_and_check(w, args.input_size, frames)
        good = diff <= ONNX_ATOL and same
        ok = ok and good
        print(f"{'OK ' if good else 'NG '} {out}: max|torch-onnx| {diff:.2e} (<= {ONNX_ATOL:g}), "
              f"Detection {'一致' if same else '不一致'} ({len(frames)} 枚 / torch の検出 {n_dets} 件)"
              f"{'' if good else ' -> 書き出していない'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
