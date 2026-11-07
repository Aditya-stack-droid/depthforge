"""DepthForge — Gradio "switchboard" app.

Flip pipeline stages on/off, pick 2x/4x/8x and a degradation, and see every
intermediate result plus live metrics. Runs on CPU (Docker / HF Spaces).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from depthforge import classical as C
from depthforge import data, degrade, metrics, pipeline, viz

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MAX_SIDE = 960          # keeps every request well under the 10 s budget on CPU
STAGE_LABELS = {
    "denoise": "Adaptive denoise",
    "hole_fill": "Hole filling",
    "jbu": "Joint bilateral upsampling",
    "guided": "Guided filter",
    "cnn": "DepthForge-Net (CNN)",
    "fusion": "Dual-path fusion (when many holes)",
    "refine": "Edge-aware refinement",
    "snap": "Boundary snapping",
}
LABEL2KEY = {v: k for k, v in STAGE_LABELS.items()}


def list_samples():
    return sorted(p.name[:-8] for p in SAMPLES.glob("*_rgb.png"))


def load_sample(name):
    if not name:
        return None, None
    rgb = cv2.cvtColor(cv2.imread(str(SAMPLES / f"{name}_rgb.png")), cv2.COLOR_BGR2RGB)
    return rgb, str(SAMPLES / f"{name}_depth.png")


def _prepare(rgb, depth_file, input_kind, scale, condition):
    if rgb is None:
        raise gr.Error("Please upload an RGB guide image (or pick a sample).")
    if depth_file is None:
        raise gr.Error("Please upload a depth map (.png 16-bit mm, 8-bit relative, or .npy metres).")
    path = depth_file if isinstance(depth_file, str) else depth_file.name
    try:
        depth = data.read_depth(path)
    except Exception as e:
        raise gr.Error(f"Could not read depth file: {e}")
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = rgb[..., :3].astype(np.uint8)
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
    if (depth > 0).mean() < 0.01:
        raise gr.Error("Depth map is (almost) empty — fewer than 1% valid pixels.")
    scale = int(scale)
    gt = None
    if input_kind.startswith("Ground truth"):
        # Simulate a low-res sensor from a dense depth map -> metrics available
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        f = min(1.0, MAX_SIDE / max(rgb.shape[:2]))
        if f < 1:
            rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb, gt = degrade.crop_to_scale(rgb, depth, scale)
        lr = degrade.make_lr(gt, scale, *degrade.CONDITIONS[condition], seed=0)
    else:
        # Real low-res depth: output size = depth size x scale; RGB resized to match
        lr = depth
        H, W = lr.shape[0] * scale, lr.shape[1] * scale
        if max(H, W) > MAX_SIDE * 1.5:
            raise gr.Error(f"Output would be {W}x{H}; choose a smaller factor or a smaller depth map.")
        if rgb.shape[0] / rgb.shape[1] and abs(rgb.shape[0] / rgb.shape[1] - H / W) > 0.05:
            gr.Warning("RGB and depth aspect ratios differ — RGB was resized to match.")
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb, lr, gt


def run_one(rgb, depth_file, input_kind, scale, condition, method, stages):
    rgb, lr, gt = _prepare(rgb, depth_file, input_kind, scale, condition)
    st = {k: (STAGE_LABELS[k] in (stages or [])) for k in pipeline.HYBRID_STAGES}
    if method == "dl" and pipeline.load_model() is None:
        raise gr.Error("CNN checkpoint not found — run scripts/train.py or choose classical/hybrid.")
    r = pipeline.run(method, lr, rgb, st, want_viz=True)
    d = r["depth"]
    vmin, vmax = (np.percentile(gt[gt > 0], 1), np.percentile(gt[gt > 0], 99)) if gt is not None else (None, None)
    gal = [(rgb, "RGB guide"), (viz.nn_upscale(viz.colorize(lr, vmin, vmax), d.shape), f"Low-res input ({lr.shape[1]}x{lr.shape[0]})")]
    v = r["viz"] or {}
    if "upsampled" in v:
        gal.append((viz.colorize(v["upsampled"], vmin, vmax), "After JBU / interpolation"))
    gal.append((viz.mask_overlay(rgb, v["rgb_edges"], (255, 255, 0)), "RGB edges (Canny)"))
    if "discontinuities" in v:
        gal.append((viz.mask_overlay(rgb, v["discontinuities"]), "Detected depth discontinuities"))
    if "cnn" in v:
        gal.append((viz.colorize(v["cnn"], vmin, vmax), "CNN output (bicubic prior)"))
    gal.append((viz.colorize(d, vmin, vmax), f"Result — {method}"))
    if gt is not None:
        gal.append((viz.colorize(gt, vmin, vmax), "Ground truth"))
        gal.append((viz.error_map(d, gt), "|error| (bright = worse)"))
    m = metrics.all_metrics(d, gt) if gt is not None else {}
    m.update(latency_ms=round(r["latency_ms"], 1), peak_mem_mb=round(r["peak_mem_mb"], 1),
             output_size=f"{d.shape[1]}x{d.shape[0]}")
    m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    params = {k: v for k, v in r["params"].items()}
    params["cnn_status"] = pipeline.model_status()
    if r["notes"]:
        params["notes"] = r["notes"]
    out = tempfile.NamedTemporaryFile(suffix="_depth_mm.png", delete=False).name
    data.write_depth_png(out, d)
    return viz.colorize(d, vmin, vmax), gal, m, params, out


def compare_all(rgb, depth_file, input_kind, scale, condition):
    if not input_kind.startswith("Ground truth"):
        raise gr.Error("Comparison needs ground truth — choose 'Ground truth depth' as the input type.")
    rgb, lr, gt = _prepare(rgb, depth_file, input_kind, scale, condition)
    rows, imgs = [], []
    vmin, vmax = np.percentile(gt[gt > 0], 1), np.percentile(gt[gt > 0], 99)
    for m in pipeline.METHODS:
        if m == "dl" and pipeline.load_model() is None:
            continue
        r = pipeline.run(m, lr, rgb)
        mm = metrics.all_metrics(r["depth"], gt)
        rows.append([m, *(round(mm[k], 4) for k in ("rmse", "mae", "psnr", "ssim", "edge_rmse", "boundary_f1")),
                     round(r["latency_ms"], 1)])
        imgs.append((viz.colorize(r["depth"], vmin, vmax), f"{m}  RMSE {mm['rmse']:.3f}"))
    imgs.append((viz.colorize(gt, vmin, vmax), "ground truth"))
    return rows, imgs


CSS = """
#title {text-align:center} .switch-panel {border:1px solid #2a2f3a; border-radius:12px; padding:8px}
"""

with gr.Blocks(title="DepthForge — depth upsampling switchboard") as demo:
    gr.Markdown("# DepthForge  ·  edge-preserving depth upsampling\n"
                "Classical CV + a gated RGB-guided CNN + an adaptive hybrid that picks how to combine them. "
                "Flip the switches to see what each stage contributes.", elem_id="title")
    with gr.Row():
        with gr.Column(scale=1):
            sample = gr.Dropdown(list_samples(), label="Load a sample", value=(list_samples() or [None])[0])
            rgb_in = gr.Image(label="RGB guide image", type="numpy", height=220)
            depth_in = gr.File(label="Depth map (.png 16-bit mm / 8-bit / .npy m)", file_types=[".png", ".npy", ".tif", ".tiff"])
            kind = gr.Radio(["Ground truth depth (simulate sensor, get metrics)", "Low-res depth (real input)"],
                            value="Ground truth depth (simulate sensor, get metrics)", label="Depth input type")
            with gr.Row():
                scale = gr.Radio(["2", "4", "8"], value="4", label="Upsampling factor")
                cond = gr.Dropdown(list(degrade.CONDITIONS), value="hard", label="Sensor degradation")
            method = gr.Radio(pipeline.METHODS, value="hybrid", label="Method")
            stages = gr.CheckboxGroup(list(STAGE_LABELS.values()),
                                      value=[v for k, v in STAGE_LABELS.items() if k != "snap"],
                                      label="Pipeline switches (classical / hybrid)", elem_classes="switch-panel")
            with gr.Row():
                go = gr.Button("Run", variant="primary")
                cmp_btn = gr.Button("Compare all methods")
            gr.Markdown(f"CNN: `{pipeline.model_status()}`")
        with gr.Column(scale=2):
            result = gr.Image(label="Result depth", height=320)
            gallery = gr.Gallery(label="Intermediate visualisations", columns=4, height=360)
            with gr.Row():
                met = gr.JSON(label="Metrics")
                par = gr.JSON(label="Adaptive parameters chosen for this input")
            dl = gr.File(label="Download depth (16-bit PNG, millimetres)")
    with gr.Accordion("Method comparison", open=False) as acc:
        table = gr.Dataframe(headers=["method", "RMSE", "MAE", "PSNR", "SSIM", "Edge-RMSE", "Boundary-F1", "ms"])
        cmp_gal = gr.Gallery(columns=7, height=220)

    sample.change(load_sample, sample, [rgb_in, depth_in])
    demo.load(load_sample, sample, [rgb_in, depth_in])  # judges can press Run immediately
    go.click(run_one, [rgb_in, depth_in, kind, scale, cond, method, stages], [result, gallery, met, par, dl])
    cmp_btn.click(compare_all, [rgb_in, depth_in, kind, scale, cond], [table, cmp_gal]).then(
        lambda: gr.Accordion(open=True), None, acc)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), css=CSS)
