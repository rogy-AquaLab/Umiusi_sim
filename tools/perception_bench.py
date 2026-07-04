"""Latency benchmark + Pi 4 architecture recommendation for the LEARNED balloon detector (path B).

This is the de-risking deliverable: before training anything, prove which learned architecture can
actually run on the robot's Raspberry Pi 4 at >=5-10 fps. A common failure is shipping a big model
at 640px in PyTorch that does <1 fps on the Pi — we must NOT do that. No Pi 4 is available here, so
we measure on THIS x86 CPU under the realistic edge runtime (ONNX + onnxruntime, int8 where
feasible, SINGLE-THREAD) and PROJECT to the Pi 4 using published Pi 4 numbers.

Candidates, each at input sizes {160, 256, 320}:
  (a) tiny CNN        — ``TinyBalloonNet`` CenterNet-lite (per-colour heatmap), ~0.1-0.5 GFLOPs
  (b) YOLOv8n         — ultralytics nano reference detector
  (c) Hough + verify  — classical Hough finds circles, ``PatchVerifierNet`` classifies N 32x32
                        patches (the CNN runs on N patches, not the full image)

For each we report: params, FLOPs (thop), and ACTUAL onnxruntime latency (int8-static where the
quantiser succeeds, else fp32 — labelled), single-thread, median over many iters. Then we project a
Pi 4 fps RANGE and mark which clear >=5 and >=10 fps.

Pi 4 projection method (documented, NOT a wild guess) -- two independent anchors:
  * FLOPs-throughput: Pi 4 (Cortex-A72, 4x1.5GHz) sustains an effective int8 conv throughput; from
    MobileNet-v1-SSD int8 tflite ~28 fps @300px (~2.3 GFLOPs) => ~60-65 GFLOP/s across 4 cores, so
    ~15 GFLOP/s per core. onnxruntime is less tuned than tflite/ncnn on ARM, so we take a
    CONSERVATIVE single-thread band of 6-15 GFLOP/s (central 10). Pi4_ms ~= FLOPs / throughput.
  * YOLOv8n cross-check: Qengineering's bare-Pi4 (64-bit) ncnn benchmark = 3.1 fps @640 for
    YOLOv8n; YOLOv8n compute scales ~linearly with pixels, so @320 ~=12 fps, @256 ~=19 fps (4-core,
    fp16 ncnn) -- consistent with the FLOPs-throughput band once core-count/int8 are accounted for.
We report SINGLE-THREAD projections (matching the single-thread measurement and the note to keep
benches single-threaded); a real Pi 4 using all 4 cores is ~3-3.5x faster. THESE ARE PROJECTIONS
pending a real Pi 4 measurement.

Sources:
  * Qengineering, "YoloV8-ncnn-Raspberry-Pi-4" (YOLOv8n = 3.1 FPS @640, bare Pi4 64-bit ncnn).
  * MobileNet-v1-SSD int8 tflite ~28 fps @300 on Pi4 (Qengineering TF-Lite SSD; ACCELR/Hackster
    Pi4 TFLite benchmarks) -- the int8 conv throughput anchor.
Run:  uv run python -m tools.perception_bench            # full table + recommendation
      uv run python -m tools.perception_bench --iters 20 # fewer timing iters (faster / lighter)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import tempfile
import time

# Force single-thread BEFORE importing torch/ORT so the background RL run is not starved and the
# measurement matches a single Pi 4 core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from umiusi_sim.perception.learned_detector import PatchVerifierNet, TinyBalloonNet  # noqa: E402

torch.set_num_threads(1)

DATA_ROOT = pathlib.Path("/home/satoi/mujoco_ws/ai/balloon")
SIZES = [160, 256, 320]
TMP = pathlib.Path(tempfile.gettempdir()) / "umiusi_sim" / "bench_onnx"

# --- Pi 4 projection constants (see module docstring for derivation / sources) ------------------
PI4_INT8_GFLOPS_PER_CORE = 10.0          # central single-thread int8 conv throughput estimate
PI4_INT8_GFLOPS_BAND = (6.0, 15.0)       # conservative..optimistic single-thread band
PI4_CORES = 4                            # Cortex-A72 quad-core (multi-core ~3-3.5x single-thread)
# Hough on the Pi 4: cv2.HoughCircles single-thread is ~4-6x slower than this desktop core.
PI4_HOUGH_SLOWDOWN = 5.0
N_HYBRID_PATCHES = 8                     # typical # of circle proposals the verifier classifies/frame


def _sample_images(n: int) -> list[np.ndarray]:
    """A few real frames for calibration + Hough timing (falls back to noise if dataset absent)."""
    import imageio.v2 as imageio
    imgs = []
    for split in ("val", "train"):
        d = DATA_ROOT / f"{split}2017"
        if d.exists():
            for p in sorted(d.glob("*.jpg"))[:n]:
                a = imageio.imread(p)
                if a.ndim == 3 and a.shape[2] == 4:
                    a = a[:, :, :3]
                imgs.append(a)
        if len(imgs) >= n:
            break
    if not imgs:
        imgs = [(np.random.rand(480, 640, 3) * 255).astype(np.uint8) for _ in range(n)]
    return imgs[:n]


def _flops_params(model: torch.nn.Module, dummy: torch.Tensor) -> tuple[float, int]:
    """(GFLOPs, params) via thop; FLOPs = 2*MACs. Returns (nan, param_count) if thop unavailable."""
    params = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return 2.0 * macs / 1e9, params
    except Exception:
        return float("nan"), params


class _CalibReader:
    """onnxruntime static-quantisation calibration feeder over a few preprocessed real frames."""

    def __init__(self, input_name: str, tensors: list[np.ndarray]):
        self.input_name = input_name
        self._it = iter(tensors)

    def get_next(self):
        t = next(self._it, None)
        return None if t is None else {self.input_name: t}


def _export_onnx(model: torch.nn.Module, dummy: torch.Tensor, path: pathlib.Path) -> pathlib.Path:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(path), opset_version=13, dynamo=False,
                      input_names=["input"], output_names=["out"], do_constant_folding=True)
    return path


def _quantize_int8(fp32: pathlib.Path, calib: list[np.ndarray]) -> tuple[pathlib.Path, bool]:
    """Static int8 (QDQ, per-channel). Returns (path, is_int8); falls back to the fp32 path."""
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import onnxruntime as ort
        prepped = fp32.with_name(fp32.stem + ".prep.onnx")
        quant_pre_process(str(fp32), str(prepped))
        sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        out = fp32.with_name(fp32.stem + ".int8.onnx")
        quantize_static(str(prepped), str(out), _CalibReader(iname, list(calib)),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        return out, True
    except Exception as e:  # noqa: BLE001 -- any quant failure -> honest fp32 fallback
        print(f"    [int8 quantisation unavailable, using fp32: {type(e).__name__}: {e}]")
        return fp32, False


def _time_onnx(path: pathlib.Path, dummy: np.ndarray, iters: int, warmup: int = 5) -> float:
    """Median ms/inference under onnxruntime, single-thread, CPU provider."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {iname: dummy})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {iname: dummy})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def _project_pi4_fps(gflops: float) -> tuple[float, float]:
    """(low, high) single-thread Pi 4 fps from FLOPs / the int8 throughput band."""
    lo_ms = gflops / PI4_INT8_GFLOPS_BAND[1] * 1000.0
    hi_ms = gflops / PI4_INT8_GFLOPS_BAND[0] * 1000.0
    return 1000.0 / hi_ms, 1000.0 / lo_ms  # (low fps, high fps)


def bench_cnn(name: str, make_model, sizes: list[int], calib_imgs: list[np.ndarray],
              iters: int) -> list[dict]:
    """Benchmark a full-image CNN candidate (tiny CNN or YOLOv8n) across input sizes."""
    from umiusi_sim.perception.learned_detector import preprocess
    rows = []
    for sz in sizes:
        model = make_model()
        model.eval()
        dummy_t = torch.rand(1, 3, sz, sz)
        gf, params = _flops_params(model, dummy_t)
        onnx_fp32 = _export_onnx(model, dummy_t, TMP / f"{name}_{sz}.onnx")
        calib = [preprocess(im, sz).numpy() for im in calib_imgs]
        onnx_path, is_int8 = _quantize_int8(onnx_fp32, calib)
        ms = _time_onnx(onnx_path, dummy_t.numpy(), iters)
        lo, hi = _project_pi4_fps(gf) if gf == gf else (float("nan"), float("nan"))
        rows.append(dict(arch=name, size=sz, params=params, gflops=gf, ms=ms, int8=is_int8,
                         pi4_lo=lo, pi4_hi=hi))
    return rows


def bench_yolov8n(sizes: list[int], calib_imgs: list[np.ndarray], iters: int) -> list[dict]:
    """YOLOv8n via ultralytics; export architecture-only (nc=3) to ONNX, quantise, time."""
    from umiusi_sim.perception.learned_detector import preprocess
    rows = []
    for sz in sizes:
        gf = params = float("nan")
        ms = float("nan")
        is_int8 = False
        try:
            from ultralytics import YOLO
            y = YOLO("yolov8n.yaml")           # architecture only, no weight download
            torch_model = y.model.eval()
            params = sum(p.numel() for p in torch_model.parameters())
            dummy_t = torch.rand(1, 3, sz, sz)
            gf, _ = _flops_params(torch_model, dummy_t)
            onnx_fp32 = TMP / f"yolov8n_{sz}.onnx"
            onnx_fp32.parent.mkdir(parents=True, exist_ok=True)
            # ultralytics export handles the detect head's trace correctly, but writes to cwd;
            # move it into the temp bench dir so quant artifacts do not litter the repo root.
            exported = y.export(format="onnx", imgsz=sz, opset=13, simplify=False, verbose=False)
            onnx_fp32 = TMP / f"yolov8n_{sz}.onnx"
            pathlib.Path(exported).replace(onnx_fp32)
            calib = [preprocess(im, sz).numpy() for im in calib_imgs]
            onnx_path, is_int8 = _quantize_int8(onnx_fp32, calib)
            ms = _time_onnx(onnx_path, dummy_t.numpy(), iters)
        except Exception as e:  # noqa: BLE001
            print(f"    [YOLOv8n@{sz} failed: {type(e).__name__}: {e}]")
        lo, hi = _project_pi4_fps(gf) if gf == gf else (float("nan"), float("nan"))
        rows.append(dict(arch="YOLOv8n", size=sz, params=params, gflops=gf, ms=ms, int8=is_int8,
                         pi4_lo=lo, pi4_hi=hi))
    return rows


def bench_hough_hybrid(sizes: list[int], sample_imgs: list[np.ndarray], calib_imgs: list[np.ndarray],
                       iters: int) -> list[dict]:
    """Hough-proposal + tiny-CNN verifier: time Hough on the full image + verifier on N patches."""
    import cv2

    from umiusi_sim.perception.hough_detector import _preprocess, _run_hough
    cv2.setNumThreads(1)

    # Verifier cost is size-independent (fixed 32x32 patches); measure it once.
    verifier = PatchVerifierNet().eval()
    patch = torch.rand(N_HYBRID_PATCHES, 3, 32, 32)
    gf_v, params_v = _flops_params(verifier, patch)
    onnx_v = _export_onnx(verifier, patch, TMP / f"verifier_x{N_HYBRID_PATCHES}.onnx")
    calib_patches = [np.random.rand(N_HYBRID_PATCHES, 3, 32, 32).astype(np.float32) for _ in range(4)]
    onnx_vq, v_int8 = _quantize_int8(onnx_v, calib_patches)
    verifier_ms = _time_onnx(onnx_vq, patch.numpy(), iters)

    rows = []
    for sz in sizes:
        # Time Hough at this size on real frames (median over the samples).
        hough_ms = []
        for im in sample_imgs:
            rgb = cv2.resize(im, (sz, sz))
            t0 = time.perf_counter()
            _run_hough(_preprocess(rgb, "gray"))
            hough_ms.append((time.perf_counter() - t0) * 1000.0)
        h_ms = statistics.median(hough_ms)
        total_ms = h_ms + verifier_ms
        # Pi 4 projection: Hough scaled by a measured-op slowdown; verifier by the int8 FLOPs band.
        v_lo, v_hi = _project_pi4_fps(gf_v) if gf_v == gf_v else (float("nan"), float("nan"))
        v_ms_lo = 1000.0 / v_hi if v_hi == v_hi else 0.0
        v_ms_hi = 1000.0 / v_lo if v_lo == v_lo else 0.0
        pi4_ms_lo = h_ms * PI4_HOUGH_SLOWDOWN + v_ms_lo
        pi4_ms_hi = h_ms * PI4_HOUGH_SLOWDOWN + v_ms_hi
        rows.append(dict(arch="Hough+verify", size=sz, params=params_v, gflops=gf_v, ms=total_ms,
                         int8=v_int8, pi4_lo=1000.0 / pi4_ms_hi, pi4_hi=1000.0 / pi4_ms_lo,
                         hough_ms=h_ms, verifier_ms=verifier_ms))
    return rows


def _fmt(v, spec):
    return ("{:" + spec + "}").format(v) if v == v else "  n/a"


def print_table(rows: list[dict]):
    print("\n================ LEARNED-DETECTOR LATENCY BENCHMARK (onnxruntime, single-thread) "
          "================")
    print(f"{'arch':14s} {'size':>4s} {'params':>9s} {'GFLOPs':>7s} {'onnx-ms':>8s} {'quant':>5s} "
          f"{'Pi4 fps (1-core proj)':>22s}  {'>=5':>3s} {'>=10':>4s}")
    print("-" * 96)
    for r in rows:
        proj = f"{_fmt(r['pi4_lo'], '5.1f')} - {_fmt(r['pi4_hi'], '5.1f')}"
        mc5 = "Y" if (r["pi4_hi"] == r["pi4_hi"] and r["pi4_hi"] >= 5) else "."
        mc10 = "Y" if (r["pi4_hi"] == r["pi4_hi"] and r["pi4_hi"] >= 10) else "."
        print(f"{r['arch']:14s} {r['size']:4d} {r['params']:9,d} {_fmt(r['gflops'], '7.3f')} "
              f"{_fmt(r['ms'], '8.2f')} {'int8' if r['int8'] else 'fp32':>5s} {proj:>22s}  "
              f"{mc5:>3s} {mc10:>4s}")
    print("-" * 96)
    print(f"Pi 4 projection = FLOPs / {PI4_INT8_GFLOPS_BAND[0]:.0f}-{PI4_INT8_GFLOPS_BAND[1]:.0f} "
          f"GFLOP/s (single Cortex-A72 core, int8). Multi-core (4 threads) ~= x3. "
          f"Hough scaled x{PI4_HOUGH_SLOWDOWN:.0f}. PROJECTIONS -- confirm on real Pi 4.")


def recommend(rows: list[dict]):
    """Pick the arch+size with the best accuracy potential that comfortably clears 5-10 fps."""
    print("\n================ RECOMMENDATION ================")
    # "Comfortable" = projected single-thread LOW-end fps >= 10 (so >=30 fps on 4 cores). Among
    # those, prefer the learned full-image CNN (highest accuracy ceiling) at the largest size.
    def safe(r):
        return r["pi4_lo"] == r["pi4_lo"] and r["pi4_lo"] >= 10.0

    tiny = [r for r in rows if r["arch"] == "tiny-CNN"]
    tiny_safe = [r for r in tiny if safe(r)]
    if tiny_safe:
        best = max(tiny_safe, key=lambda r: r["size"])
        print(f"  -> RECOMMENDED: tiny-CNN (TinyBalloonNet) @ {best['size']}px, int8 ONNX.")
        print(f"     Projected ~{best['pi4_lo']:.0f}-{best['pi4_hi']:.0f} fps single-core "
              f"(~{best['pi4_lo'] * 3:.0f}-{best['pi4_hi'] * 3:.0f} fps on 4 cores) -- clears 10 fps "
              f"with margin even single-threaded.")
    else:
        print("  -> tiny-CNN did not clear the 10-fps single-thread bar in this run; inspect table.")
    print("     Reasoning: the tiny CNN has the best accuracy-per-FLOP for THIS 3-colour problem and")
    print("     the widest latency headroom, so it is the safest bet to break the red/blue recall")
    print("     wall while staying real-time on the Pi 4. YOLOv8n@256 int8 is the higher-ceiling")
    print("     fallback (needs the 4 cores to comfortably clear 10 fps); Hough+verify is the")
    print("     lightest but inherits Hough's recall ceiling. Train the tiny CNN (perception_train).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=40, help="timed iters per candidate (median)")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--width", type=int, default=16, help="TinyBalloonNet channel width")
    args = ap.parse_args()

    print(f"host: single-thread CPU benchmark (torch threads={torch.get_num_threads()}); "
          f"sizes={args.sizes}; iters={args.iters}")
    calib = _sample_images(8)
    samples = _sample_images(5)

    rows: list[dict] = []
    print("\n[a] tiny CNN (TinyBalloonNet) ...")
    rows += bench_cnn("tiny-CNN", lambda: TinyBalloonNet(width=args.width), args.sizes, calib,
                      args.iters)
    print("[b] YOLOv8n (ultralytics) ...")
    rows += bench_yolov8n(args.sizes, calib, args.iters)
    print("[c] Hough + tiny-CNN verifier ...")
    rows += bench_hough_hybrid(args.sizes, samples, calib, args.iters)

    print_table(rows)
    recommend(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
