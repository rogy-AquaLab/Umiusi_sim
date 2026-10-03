"""learned_detector: backend="onnx" が torch と同じ Detection を返すこと。"""
import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

import torch.nn.functional as F
from umiusi_perception import learned_detector as ld
from umiusi_perception.learned_detector import load_learned_detector

REPO = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "examples" / "balloon_detector" / "model.pt"
# 実画像は repo の外 (mujoco_ws/ai/balloon/train_real) にある。無ければその分だけ skip
REAL_DIR = pathlib.Path(os.environ.get(
    "UMIUSI_REAL_IMAGES", REPO.parent / "ai" / "balloon" / "train_real"))


def _synthetic_frames():
    rng = np.random.default_rng(0)
    frames = []
    for colour in ((220, 30, 30), (230, 210, 40), (40, 60, 220)):
        img = np.full((240, 320, 3), (20, 80, 110), np.uint8)
        yy, xx = np.mgrid[:240, :320]
        cy, cx, r = rng.integers(60, 180), rng.integers(80, 240), rng.integers(15, 40)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = colour
        frames.append(img)
    return frames


def _real_frames(n=6):
    from PIL import Image

    paths = sorted(REAL_DIR.glob("*.jpg"))[:n]
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def _assert_same(a, b):
    assert len(a) == len(b)
    for da, db in zip(a, b):
        assert da.colour == db.colour
        assert da.bbox == db.bbox
        assert da.bearing == pytest.approx(db.bearing, abs=1e-4)
        assert da.range_m == pytest.approx(db.range_m, abs=1e-4)


@pytest.fixture(scope="module")
def detectors(tmp_path_factory):
    os.environ["UMIUSI_ONNX_CACHE"] = str(tmp_path_factory.mktemp("onnx"))
    return (load_learned_detector(str(WEIGHTS), backend="torch"),
            load_learned_detector(str(WEIGHTS), backend="onnx"))


def test_onnx_keeps_checkpoint_config(detectors):
    t, o = detectors
    assert o.backend == "onnx" and t.backend == "torch"
    assert (o.input_size, o.width, o.conf_thresh) == (t.input_size, t.width, t.conf_thresh)


def test_synthetic_frames_match(detectors):
    t, o = detectors
    for img in _synthetic_frames():
        _assert_same(t(img), o(img))


def test_real_frames_match(detectors):
    frames = _real_frames()
    if not frames:
        pytest.skip(f"no real images in {REAL_DIR}")
    t, o = detectors
    n_dets = 0
    for img in frames:
        dt = t(img)
        _assert_same(dt, o(img))
        n_dets += len(dt)
    assert n_dets > 0  # 一致が「両方 0 件」で通っていないこと


def test_input_size_override_exports_separately(detectors, tmp_path):
    o256 = load_learned_detector(str(WEIGHTS), input_size=256, backend="onnx")
    t256 = load_learned_detector(str(WEIGHTS), input_size=256, backend="torch")
    assert o256.input_size == 256
    img = _synthetic_frames()[0]
    _assert_same(t256(img), o256(img))


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        load_learned_detector(str(WEIGHTS), backend="tensorrt")


def _old_preprocess(rgb, input_size):
    """速くする前の preprocess (2026-10-03 まで)。出力がビット単位で同じことの基準。"""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return F.interpolate(t, size=(input_size, input_size), mode="bilinear", align_corners=False)


@pytest.mark.parametrize("shape", [(720, 1280, 3), (977, 708, 3), (240, 320, 3), (480, 640, 4), (200, 300)])
@pytest.mark.parametrize("size", [256, 320])
def test_preprocess_is_bit_identical_to_the_old_one(shape, size):
    img = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    new = ld.preprocess(img, size)
    old = _old_preprocess(img, size)
    assert torch.equal(new, old)
    assert new.stride() == old.stride()  # torch の conv はメモリ配置で結果が変わりうる


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_detections_unchanged_by_the_faster_preprocess(detectors, monkeypatch, backend):
    det = detectors[0] if backend == "torch" else detectors[1]
    frames = _synthetic_frames() + _real_frames()
    new = [det(img) for img in frames]
    monkeypatch.setattr(ld, "preprocess", _old_preprocess)
    old = [det(img) for img in frames]
    assert new == old


def _copy_weights(dst):
    import shutil

    shutil.copy(WEIGHTS, dst)
    return str(dst)


def test_bundled_onnx_is_used_when_it_matches(tmp_path, monkeypatch):
    from tools.export_detector_onnx import export_and_check

    monkeypatch.setenv("UMIUSI_ONNX_CACHE", str(tmp_path / "cache"))
    w = _copy_weights(tmp_path / "m.pt")
    out, diff, _n, same = export_and_check(w, None, _synthetic_frames())
    size = load_learned_detector(w).input_size
    assert out == str(tmp_path / f"m_{size}.onnx") and diff <= ld.ONNX_ATOL and same
    o = load_learned_detector(w, backend="onnx")
    assert o.onnx_path == out
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())


def test_stale_bundled_onnx_is_ignored(tmp_path, monkeypatch):
    """重みを差し替えて .onnx を書き出し直し忘れても、古い .onnx の結果を出さない。"""
    monkeypatch.setenv("UMIUSI_ONNX_CACHE", str(tmp_path / "cache"))
    w = _copy_weights(tmp_path / "m.pt")
    t = load_learned_detector(w)
    ld.export_onnx(ld.TinyBalloonNet(width=t.width), t.input_size, str(tmp_path / f"m_{t.input_size}.onnx"))
    o = load_learned_detector(w, backend="onnx")
    assert o.onnx_path.startswith(str(tmp_path / "cache"))
    img = _synthetic_frames()[0]
    _assert_same(t(img), o(img))


# 配備する検出器 (autonomy の models/detector)。repo の外なので無ければ skip
DEPLOY_DIR = pathlib.Path(os.environ.get(
    "UMIUSI_DETECTOR_DIR",
    REPO.parent / "ros2_ws" / "src" / "sinsei_UMIUSI_autonomy" / "umiusi_autonomy" / "models" / "detector"))


@pytest.mark.parametrize("name", ["balloon_F320_20261003.pt", "balloon_F256_20261003.pt"])
def test_export_tool_matches_torch_on_deployed_detectors(name, tmp_path):
    from tools.export_detector_onnx import export_and_check

    src = DEPLOY_DIR / name
    if not src.exists():
        pytest.skip(f"{src} が無い")
    import shutil

    w = str(shutil.copy(src, tmp_path / name))
    frames = _synthetic_frames() + _real_frames()
    out, diff, n_dets, same = export_and_check(w, None, frames)
    assert diff <= ld.ONNX_ATOL and same and n_dets > 0
    assert pathlib.Path(out).exists()
