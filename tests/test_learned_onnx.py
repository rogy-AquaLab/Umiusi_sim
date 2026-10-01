"""learned_detector: backend="onnx" が torch と同じ Detection を返すこと。"""
import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from umiusi_perception.learned_detector import load_learned_detector  # noqa: E402

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
