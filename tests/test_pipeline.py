import numpy as np
import pytest

from depthforge import classical as C, data, degrade, metrics, pipeline


@pytest.fixture(scope="module")
def scene():
    rgb, gt = data.synth_scene(120, 160, seed=1)
    return rgb, gt


@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("method", ["bicubic", "classical", "hybrid"])
def test_shapes_and_validity(scene, scale, method):
    rgb, gt = degrade.crop_to_scale(*scene, scale)
    lr = degrade.make_lr(gt, scale, *degrade.CONDITIONS["hard"], seed=0)
    out = pipeline.run(method, lr, rgb)["depth"]
    assert out.shape == gt.shape
    assert np.isfinite(out).all() and (out > 0).all()


def test_classical_beats_bicubic(scene):
    rgb, gt = degrade.crop_to_scale(*scene, 4)
    lr = degrade.make_lr(gt, 4, seed=0)
    b = metrics.rmse(pipeline.run("bicubic", lr, rgb)["depth"], gt)
    c = metrics.rmse(pipeline.run("classical", lr, rgb)["depth"], gt)
    assert c < b


def test_hole_fill_only_touches_holes():
    d = np.ones((20, 20), np.float32) * 2
    d[5:8, 5:8] = 0
    f = C.fill_holes(d)
    assert (f > 0).all() and np.allclose(f[d > 0], 2)


def test_guided_filter_identity_on_constant():
    rgb = np.random.default_rng(0).integers(0, 255, (32, 32, 3)).astype(np.uint8)
    p = np.full((32, 32), 3.0, np.float32)
    assert np.allclose(C.guided_filter(rgb, p), 3.0, atol=1e-3)
