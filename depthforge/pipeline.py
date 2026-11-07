"""One entry point for every method so the UI, CLI, evaluator and Switch agents
all run exactly the same code path."""
from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from . import classical as C
from .adaptive import choose_params

METHODS = ["nearest", "bilinear", "bicubic", "classical", "dl", "hybrid"]
HYBRID_STAGES = ["denoise", "hole_fill", "jbu", "guided", "cnn", "fusion", "refine", "snap"]

_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS = Path(os.environ.get("DEPTHFORGE_WEIGHTS", _ROOT / "weights" / "depthforge_net.pt"))
_ONNX = Path(os.environ.get("DEPTHFORGE_ONNX", _ROOT / "weights" / "depthforge_net.onnx"))
# auto: PyTorch if installed (training / dev), else ONNX Runtime (lightweight deploy image)
BACKEND = os.environ.get("DEPTHFORGE_BACKEND", "auto")
_MODEL = None
_MODEL_ERR = None
_BACKEND_USED = None


def load_model():
    """Lazy-load the CNN (PyTorch or ONNX Runtime). Returns None (and remembers why)
    if neither is available, so the app degrades gracefully to classical."""
    global _MODEL, _MODEL_ERR, _BACKEND_USED, TILE
    if _MODEL is not None or _MODEL_ERR is not None:
        return _MODEL
    errs = []
    if BACKEND in ("auto", "torch"):
        try:
            import torch
            from .model import DepthForgeNet
            if not _WEIGHTS.exists():
                raise FileNotFoundError(f"no checkpoint at {_WEIGHTS} (run scripts/train.py)")
            ck = torch.load(_WEIGHTS, map_location="cpu", weights_only=False)
            m = DepthForgeNet(ck.get("width", 32))
            m.load_state_dict(ck["state_dict"])
            m.eval()
            TILE = int(ck.get("crop", TILE))  # infer on tiles the size the net was trained on
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            _MODEL, _BACKEND_USED = m, "torch"
            return _MODEL
        except Exception as e:  # pragma: no cover - environment dependent
            errs.append(f"torch: {e}")
    if BACKEND in ("auto", "onnx"):
        try:
            import onnxruntime as ort
            if not _ONNX.exists():
                raise FileNotFoundError(f"no ONNX model at {_ONNX} (run scripts/export_onnx.py)")
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, os.cpu_count() or 1)
            # keep RAM low for free hosting tiers: no pre-allocated arena
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
            _MODEL = ort.InferenceSession(str(_ONNX), so, providers=["CPUExecutionProvider"])
            _BACKEND_USED = "onnx"
            return _MODEL
        except Exception as e:  # pragma: no cover
            errs.append(f"onnx: {e}")
    _MODEL_ERR = "; ".join(errs) or f"unknown backend {BACKEND}"
    return None


def _infer(m, d: np.ndarray, r: np.ndarray) -> np.ndarray:
    """d: Nx1xTxT, r: Nx3xTxT float32 -> NxTxT."""
    if hasattr(m, "run") and not hasattr(m, "parameters"):  # onnxruntime session
        # small mini-batches bound peak memory (~4 tiles of activations at a time)
        return np.concatenate([m.run(None, {"depth": d[i:i + 4], "rgb": r[i:i + 4]})[0][:, 0]
                               for i in range(0, len(d), 4)])
    import torch
    bf16 = os.environ.get("DEPTHFORGE_BF16", "0") == "1"
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        return m(torch.from_numpy(d), torch.from_numpy(r))[:, 0].float().numpy()


def model_status() -> str:
    load_model()
    return f"loaded ({_BACKEND_USED})" if _MODEL is not None else f"unavailable: {_MODEL_ERR}"


# flip TTA is available but OFF by default: on NYU val it slightly *hurt* (4x clean
# RMSE 0.0302 -> 0.0337), so it is not used for reported numbers
TTA = os.environ.get("DEPTHFORGE_TTA", "0") == "1"
TILE, OVERLAP = 128, 32   # inference tile = training crop size (overridden from checkpoint)


def _tiles(n, T, step):
    if n <= T:
        return [0]
    return sorted(set(list(range(0, n - T, step)) + [n - T]))


def run_cnn(coarse: np.ndarray, rgb: np.ndarray, tta: bool | None = None) -> np.ndarray:
    """CNN refinement, run on overlapping 128x128 tiles blended with a Hann window.

    Why tiles: the net was trained on 128x128 crops and GroupNorm statistics depend
    on the spatial extent seen. Whole-image inference shifted those statistics and
    made the net *worse* than its input on NYU val (4x clean RMSE 0.0445 -> 0.0467);
    tiled inference fixed it (-> 0.0307). With test-time augmentation (default)
    every tile is also run horizontally flipped and the predictions averaged
    (off by default, see TTA above)."""
    m = load_model()
    if m is None:
        raise RuntimeError(f"CNN unavailable ({_MODEL_ERR})")
    H0, W0 = coarse.shape
    # images smaller than a tile are edge-padded up to one tile, then cropped back
    ph, pw = max(0, TILE - H0), max(0, TILE - W0)
    if ph or pw:
        coarse = np.pad(coarse, ((0, ph), (0, pw)), mode="edge")
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = coarse.shape
    s = float(np.percentile(coarse[coarse > 0], 99)) if (coarse > 0).any() else 1.0
    s = max(s, 1e-3)
    T = TILE
    win = np.outer(np.hanning(T + 2)[1:-1], np.hanning(T + 2)[1:-1]).astype(np.float32) + 1e-3
    pos = [(y, x) for y in _tiles(H, T, T - OVERLAP) for x in _tiles(W, T, T - OVERLAP)]
    d = np.stack([coarse[y:y + T, x:x + T] / s for y, x in pos]).astype(np.float32)[:, None]
    r = np.ascontiguousarray(np.stack([rgb[y:y + T, x:x + T] for y, x in pos])
                             .astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    o = _infer(m, d, r)
    if TTA if tta is None else tta:
        of = _infer(m, np.ascontiguousarray(d[..., ::-1]), np.ascontiguousarray(r[..., ::-1]))
        o = 0.5 * (o + of[..., ::-1])
    out = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for (y, x), t in zip(pos, o):
        out[y:y + T, x:x + T] += t * win
        wsum[y:y + T, x:x + T] += win
    return np.clip(out / wsum * s, 0, None).astype(np.float32)[:H0, :W0]


def run(method: str, lr: np.ndarray, rgb: np.ndarray, stages: dict | None = None,
        want_viz: bool = False) -> dict:
    """Run one method. Returns dict(depth, params, viz, latency_ms, peak_mem_mb, notes)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")
    H, W = rgb.shape[:2]
    # snapping is opt-in: it helped on piecewise-planar synthetic scenes but hurt
    # on real NYU depth (curved / slanted surfaces), see docs/FAILURE_ANALYSIS.md
    st = {k: k != "snap" for k in HYBRID_STAGES}
    st.update(stages or {})
    viz: dict = {} if want_viz else None
    notes = []
    tracemalloc.start()
    t0 = time.perf_counter()
    params = choose_params(lr, rgb)

    if method in ("nearest", "bilinear", "bicubic"):
        out = C.interpolate(C.fill_holes(lr), (H, W), method)
    elif method == "classical":
        out = C.classical_pipeline(lr, rgb, params, st, viz)
    elif method == "dl":
        coarse = C.interpolate(C.fill_holes(lr), (H, W), "bicubic")
        out = run_cnn(coarse, rgb)
    else:  # hybrid: adaptive selection / combination of classical and learned outputs
        bic = C.interpolate(C.fill_holes(lr), (H, W), "bicubic")
        if not st["cnn"] or load_model() is None:
            if st["cnn"]:
                notes.append("CNN unavailable -> hybrid fell back to classical")
            out = C.classical_pipeline(lr, rgb, params, {k: st[k] for k in ("denoise", "hole_fill", "jbu", "guided")}, viz)
        else:
            d_b = run_cnn(bic, rgb)
            out = d_b
            if viz is not None:
                viz["cnn"] = d_b
            if st["fusion"] and params["dual_path"]:
                coarse = C.classical_pipeline(lr, rgb, params, {k: st[k] for k in ("denoise", "hole_fill", "jbu", "guided")}, viz)
                d_c = run_cnn(coarse, rgb)
                out = 0.5 * (d_b + d_c)
                notes.append("dual-path fusion: CNN(bicubic) + CNN(classical) averaged (many missing samples)")
            if st["refine"] and params["refine"]:
                out = C.guided_filter(rgb, out, 2, 1e-4)
                notes.append("edge-aware guided refinement applied")
        if st["snap"]:
            disc = C.depth_discontinuities(out, params["disc_thresh"])
            out = C.snap_discontinuities(out, C.band(disc, params["band"]), params["snap_k"])
            if viz is not None:
                viz["discontinuities"] = disc

    latency = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if viz is not None:
        viz["rgb_edges"] = C.rgb_edges(rgb)
    return dict(depth=out.astype(np.float32), params=params, viz=viz,
                latency_ms=latency, peak_mem_mb=peak / 2 ** 20, notes=notes)
