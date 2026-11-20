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


def test_router_fit_and_choose():
    from depthforge import router as R
    rng = np.random.default_rng(0)
    X = np.c_[np.ones(200), rng.normal(size=(200, len(R.FEATURES) - 1))]
    Y = rng.normal(size=(200, len(R.CANDIDATES)))
    mu, sd, W = R.fit(X, Y, 1.0)
    rt = R.Router(dict(candidates=R.CANDIDATES, mean=mu.tolist(), std=sd.tolist(), weights=W.tolist(), margin=0.0))
    p = dict(scale=4, holes=0.3, noise=0.01, texture_risk=0.5, edge_density=0.05, lr_grad=0.02, dual_path=True, refine=True)
    assert rt.choose(p) in R.CANDIDATES
    assert R.needs("bc_ref") == (True, True, True) and R.needs("c") == (False, True, False)
    assert R.rule_candidate(p) == "bc_ref"


def test_hybrid_with_router(scene, tmp_path, monkeypatch):
    import json
    from depthforge import router as R
    n = len(R.FEATURES)
    W = np.zeros((n, len(R.CANDIDATES))); W[0, R.CANDIDATES.index("c")] = -1.0  # always prefers 'c'
    f = tmp_path / "r.json"
    R.save(f, np.zeros(n), np.ones(n), W, 0.0)
    monkeypatch.setattr(R, "_ROUTER", R.Router(json.loads(f.read_text()))); monkeypatch.setattr(R, "_LOADED", True)
    rgb, gt = degrade.crop_to_scale(*scene, 4)
    lr = degrade.make_lr(gt, 4, *degrade.CONDITIONS["clean"], seed=0)
    r = pipeline.run("hybrid", lr, rgb)
    if pipeline.load_model() is not None:
        assert r["params"]["router_choice"] == "c" and r["depth"].shape == gt.shape
        off = pipeline.run("hybrid", lr, rgb, {"router": False})
        assert "router_choice" not in off["params"]


def test_viewer_html(scene):
    from depthforge import viewer3d
    rgb, gt = scene
    page = viewer3d.viewer_html(rgb, {"a": gt, "b": gt * 1.1}, max_side=64)
    assert "WebGL" in page.upper() or "webgl2" in page
    assert "__" not in page.split("<script>")[1].split("const vs")[0]  # all placeholders substituted
    assert viewer3d.iframe(page).startswith("<iframe")
