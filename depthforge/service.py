"""Web-service layer: everything the website needs, with no UI framework dependency.

`run_request` / `compare_request` take decoded images and return JSON-serialisable dicts
(base64 JPEG/PNG images, metrics, the adaptive decisions, 3D mesh data)."""
from __future__ import annotations

import base64
import threading
from pathlib import Path

import cv2
import numpy as np

from . import classical as C
from . import data, degrade, metrics, pipeline, viewer3d, viz

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
MAX_SIDE = 960   # keeps every request well under the 10 s budget on CPU

STAGE_LABELS = {
    "denoise": "Adaptive denoise",
    "hole_fill": "Hole filling",
    "jbu": "Joint bilateral upsampling",
    "guided": "Guided filter",
    "cnn": "DepthForge-Net (CNN)",
    "router": "Learned router",
    "fusion": "Dual-path fusion",
    "refine": "Edge-aware refinement",
    "snap": "Boundary snapping",
    "cascade": "Cascade 2×2×2 (4×/8×)",
}
CONDITION_INFO = {
    "clean": "No degradation, only decimation",
    "noisy": "Depth-proportional sensor noise",
    "holes": "Blob-shaped dropouts",
    "sparse": "Randomly missing samples",
    "hard": "Noise + holes + sparsity combined",
}
_LOCK = threading.Lock()   # the pipeline uses tracemalloc and a global model: one request at a time


class RequestError(ValueError):
    """A problem with the user's input (shown to the user as-is)."""


# ------------------------------------------------------------------ helpers
def jpg(img: np.ndarray, q: int = 90) -> str:
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


def png16(depth: np.ndarray) -> str:
    ok, b = cv2.imencode(".png", np.clip(depth * 1000, 0, 65535).astype(np.uint16))
    return "data:image/png;base64," + base64.b64encode(b.tobytes()).decode()


def list_samples() -> list[str]:
    return sorted(p.name[:-8] for p in SAMPLES.glob("*_rgb.png"))


def load_sample(name: str):
    if name not in list_samples():
        raise RequestError(f"Unknown sample '{name}'.")
    rgb = cv2.cvtColor(cv2.imread(str(SAMPLES / f"{name}_rgb.png")), cv2.COLOR_BGR2RGB)
    return rgb, data.read_depth(SAMPLES / f"{name}_depth.png")


def thumbnail(name: str, size: int = 220) -> bytes:
    rgb, _ = load_sample(name)
    f = size / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return b.tobytes()


def config() -> dict:
    from . import router as R
    return dict(
        samples=list_samples(),
        methods=pipeline.METHODS,
        conditions=[dict(key=k, info=CONDITION_INFO.get(k, "")) for k in degrade.CONDITIONS],
        stages=[dict(key=k, label=STAGE_LABELS[k], default=k not in pipeline.OPT_IN) for k in pipeline.HYBRID_STAGES],
        cnn=pipeline.model_status(),
        router=R.load() is not None,
    )


def results_summary() -> dict:
    """NYU test-split numbers written by scripts/evaluate.py, reshaped for the results chart."""
    import json
    f = ROOT / "results" / "results.json"
    if not f.exists():
        return {}
    summ = json.loads(f.read_text()).get("summary", {})
    out: dict = {}
    for key, m in summ.items():
        cond, scale, method = key.split("|")
        out.setdefault(cond, {}).setdefault(method, {})[scale] = {k: m[k] for k in ("rmse", "edge_rmse", "boundary_f1") if k in m}
    return out


def prepare(rgb, depth, gt_mode: bool, scale: int, condition: str):
    if rgb is None or depth is None:
        raise RequestError("Choose a sample, or upload both an RGB image and a depth map.")
    if condition not in degrade.CONDITIONS:
        raise RequestError(f"Unknown degradation '{condition}'.")
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = rgb[..., :3].astype(np.uint8)
    depth = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)
    if (depth > 0).mean() < 0.01:
        raise RequestError("The depth map is (almost) empty: fewer than 1% valid pixels.")
    gt = None
    if gt_mode:
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        f = min(1.0, MAX_SIDE / max(rgb.shape[:2]))
        if f < 1:
            rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb, gt = degrade.crop_to_scale(rgb, depth, scale)
        lr = degrade.make_lr(gt, scale, *degrade.CONDITIONS[condition], seed=0)
    else:
        lr = depth
        H, W = lr.shape[0] * scale, lr.shape[1] * scale
        if max(H, W) > MAX_SIDE * 1.5:
            raise RequestError(f"The output would be {W}×{H}. Choose a smaller factor or a smaller depth map.")
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb, lr, gt


def _stages(off: list[str], on: list[str]) -> dict:
    bad = [k for k in off + on if k not in pipeline.HYBRID_STAGES]
    if bad:
        raise RequestError(f"Unknown stage(s): {', '.join(bad)}")
    st = {k: k not in pipeline.OPT_IN for k in pipeline.HYBRID_STAGES}
    st.update({k: False for k in off})
    st.update({k: True for k in on})
    return st


def _decisions(params: dict, notes: list[str]) -> list[str]:
    lines = [f"Measured {params.get('scale')}× upsampling, noise {params.get('noise')}, "
             f"{float(params.get('holes', 0)) * 100:.0f}% missing samples, texture-copy risk {params.get('texture_risk')}."]
    if params.get("router_choice"):
        lines.append(f"The learned router chose path '{params['router_choice']}'.")
    lines += [n[0].upper() + n[1:] + ("" if n.endswith(".") else ".") for n in notes]
    return lines


# ------------------------------------------------------------------ requests
def run_request(rgb, depth, gt_mode: bool, scale: int, condition: str, method: str,
                off: list[str] | None = None, on: list[str] | None = None) -> dict:
    if scale not in (2, 4, 8):
        raise RequestError("The upsampling factor must be 2, 4 or 8.")
    if method not in pipeline.METHODS:
        raise RequestError(f"Unknown method '{method}'.")
    if method == "dl" and pipeline.load_model() is None:
        raise RequestError("The CNN checkpoint is not available on this server; choose classical or hybrid.")
    st = _stages(off or [], on or [])
    rgb, lr, gt = prepare(rgb, depth, gt_mode, scale, condition)
    with _LOCK:
        r = pipeline.run(method, lr, rgb, st, want_viz=True)
    d = r["depth"]
    if gt is not None:
        vmin, vmax = float(np.percentile(gt[gt > 0], 1)), float(np.percentile(gt[gt > 0], 99))
    else:
        vmin = vmax = None
    bic = C.interpolate(C.fill_holes(lr), d.shape, "bicubic")
    col = lambda x: jpg(viz.colorize(x, vmin, vmax))
    images = dict(rgb=jpg(rgb), lr=jpg(viz.nn_upscale(viz.colorize(lr, vmin, vmax), d.shape)),
                  bicubic=col(bic), result=col(d))
    stages = [dict(label="RGB guide", img=images["rgb"]),
              dict(label=f"Low-res input ({lr.shape[1]}×{lr.shape[0]})", img=images["lr"])]
    v = r["viz"] or {}
    if "upsampled" in v:
        stages.append(dict(label="After JBU / interpolation", img=col(v["upsampled"])))
    stages.append(dict(label="RGB edges (Canny)", img=jpg(viz.mask_overlay(rgb, v["rgb_edges"], (255, 255, 0)))))
    if "discontinuities" in v:
        stages.append(dict(label="Detected depth discontinuities", img=jpg(viz.mask_overlay(rgb, v["discontinuities"]))))
    if "cnn" in v:
        stages.append(dict(label="CNN output", img=col(v["cnn"])))
    stages.append(dict(label=f"Result ({method})", img=images["result"]))
    srcs = {f"Result ({method})": d, "Bicubic": bic}
    m = {}
    if gt is not None:
        images["gt"] = col(gt)
        images["error"] = jpg(viz.error_map(d, gt))
        images["bic_error"] = jpg(viz.error_map(bic, gt, vmax=float(np.percentile(np.abs(d - gt)[gt > 0], 99))))
        stages += [dict(label="Ground truth", img=images["gt"]), dict(label="|error| (bright = worse)", img=images["error"])]
        srcs["Ground truth"] = gt
        m = {k: float(x) for k, x in metrics.all_metrics(d, gt).items()}
        mb = metrics.all_metrics(bic, gt)
        m["rmse_bicubic"] = float(mb["rmse"])
    params = dict(r["params"]); params["cnn_status"] = pipeline.model_status()
    return dict(images=images, stages=stages, metrics=m, latency_ms=round(r["latency_ms"], 1),
                size=[int(d.shape[1]), int(d.shape[0])], has_gt=gt is not None,
                params={k: (round(x, 4) if isinstance(x, float) else x) for k, x in params.items() if not isinstance(x, (dict, list))},
                decisions=_decisions(params, r["notes"]), download=png16(d),
                view3d=viewer3d.payload(rgb, srcs))


def compare_request(rgb, depth, scale: int, condition: str) -> dict:
    rgb, lr, gt = prepare(rgb, depth, True, scale, condition)
    vmin, vmax = float(np.percentile(gt[gt > 0], 1)), float(np.percentile(gt[gt > 0], 99))
    rows = []
    for meth in pipeline.METHODS:
        if meth == "dl" and pipeline.load_model() is None:
            continue
        with _LOCK:
            r = pipeline.run(meth, lr, rgb)
        mm = metrics.all_metrics(r["depth"], gt)
        rows.append(dict(method=meth, rmse=mm["rmse"], mae=mm["mae"], psnr=mm["psnr"], ssim=mm["ssim"],
                         edge_rmse=mm["edge_rmse"], boundary_f1=mm["boundary_f1"], ms=round(r["latency_ms"], 1),
                         img=jpg(viz.colorize(r["depth"], vmin, vmax), 82)))
    rows.append(dict(method="ground truth", img=jpg(viz.colorize(gt, vmin, vmax), 82)))
    return dict(rows=rows)
