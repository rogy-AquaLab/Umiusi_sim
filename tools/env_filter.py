#!/usr/bin/env python3
"""実映像から測った「会場フィルタ」— 濁りのベール・色むら・ぼけを別の画像に載せる。

参照フレーム 1 枚ごとに 3 つを測って bank (.npz) に貯める:
  veil      32x32 に縮めた低周波画像。濁りの色と、上下の色むら・周辺減光をそのまま持つ
  e_coarse  帯域 2〜8 px のコントラスト（256x256 灰色）。濁りが強いほど小さい
  e_fine    1 px 以下の帯域のコントラスト。ぼけが強いほど小さい
掛けるときは、対象画像の同じ 2 つの量が参照に揃うように透過率 t とぼけ σ を選ぶ:
  out = t * (img * tint) + (1 - t) * veil  →  ガウスぼけ σ
ラベルは使わない（会場で数分撮った映像だけで作れる）。画素は動かないので箱はそのまま。

  python -m tools.env_filter build --out bank.npz --refs p1003=DIR_OR_LIST:2 p0913=...:1
  python -m tools.env_filter demo --bank bank.npz --images a.jpg b.jpg --out sheet.jpg
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

SIZE = 256           # 測る解像度 = 検出器の入力
VEIL = 32
SIGMAS = (0.0, 0.6, 1.0, 1.5, 2.0, 2.8, 3.6)
T_MIN = 0.15         # これより濁らせると風船が消えて、箱だけ残る


def _bands(img: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(g, (0, 0), 2.0)
    g8 = cv2.GaussianBlur(g, (0, 0), 8.0)
    return float((g2 - g8).std()), float((g - g1).std())


def measure(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """RGB uint8 -> (veil 32x32x3 uint8, e_coarse, e_fine)."""
    im = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    veil = cv2.resize(cv2.GaussianBlur(im, (0, 0), 12.0), (VEIL, VEIL), interpolation=cv2.INTER_AREA)
    ec, ef = _bands(im)
    return veil, ec, ef


class EnvFilter:
    def __init__(self, bank_path):
        d = np.load(bank_path, allow_pickle=False)
        self.veil, self.ec, self.ef = d["veil"], d["e_coarse"], d["e_fine"]
        self.p = d["weight"] / d["weight"].sum()
        self.env = d["env"]

    def __call__(self, image: np.ndarray, ref: int | None = None, **kwargs) -> np.ndarray:
        i = int(np.random.choice(len(self.p), p=self.p)) if ref is None else ref
        h, w = image.shape[:2]
        veil = self.veil[i][:, ::-1] if np.random.rand() < 0.5 else self.veil[i]
        veil = cv2.resize(veil, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = image.astype(np.float32)
        ec, _ = _bands(image)
        t = float(np.clip(self.ec[i] / max(ec, 1e-3) * np.random.uniform(0.7, 1.4), T_MIN, 1.0))
        tint = (veil.mean(axis=(0, 1)) / max(float(veil.mean()), 1.0)) ** 0.5
        out = np.clip(t * img * tint + (1.0 - t) * veil, 0, 255).astype(np.uint8)
        target = self.ef[i] * np.random.uniform(0.8, 1.3)
        scale = max(h, w) / SIZE
        for s in SIGMAS:
            cand = out if s == 0.0 else cv2.GaussianBlur(out, (0, 0), s * scale)
            if _bands(cand)[1] <= target:
                break
        return cand


def _list_images(spec: str) -> list[pathlib.Path]:
    p = pathlib.Path(spec)
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in (".jpg", ".png"))
    return [pathlib.Path(line) for line in p.read_text().split() if line]


def build(args) -> int:
    veil, ec, ef, wt, env = [], [], [], [], []
    for spec in args.refs:
        name, rest = spec.split("=", 1)
        src, weight = rest.rsplit(":", 1)
        files = _list_images(src)
        for f in files:
            v, c, fi = measure(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
            veil.append(v)
            ec.append(c)
            ef.append(fi)
            wt.append(float(weight) / len(files))
            env.append(name)
        print(f"{name}: {len(files)} 枚  e_coarse 中央値 {np.median(ec[-len(files):]):.2f}  "
              f"e_fine 中央値 {np.median(ef[-len(files):]):.2f}")
    np.savez_compressed(args.out, veil=np.stack(veil), e_coarse=np.array(ec, np.float32),
                        e_fine=np.array(ef, np.float32), weight=np.array(wt, np.float32),
                        env=np.array(env))
    print(f"saved {len(env)} refs -> {args.out}")
    return 0


def demo(args) -> int:
    filt = EnvFilter(args.bank)
    np.random.seed(args.seed)
    envs = sorted(set(filt.env.tolist()))
    rows = []
    for f in args.images:
        im = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        tiles = [im]
        for e in envs:
            for _ in range(args.per_env):
                tiles.append(filt(im, ref=int(np.random.choice(np.flatnonzero(filt.env == e)))))
        rows.append(np.hstack(tiles))
    cv2.imwrite(args.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("列: 元 | " + " | ".join(f"{e} x{args.per_env}" for e in envs))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--refs", nargs="+", required=True, help="名前=ディレクトリか一覧ファイル:重み")
    d = sub.add_parser("demo")
    d.add_argument("--bank", required=True)
    d.add_argument("--images", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--per-env", type=int, default=2)
    d.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return build(args) if args.cmd == "build" else demo(args)


if __name__ == "__main__":
    sys.exit(main())
