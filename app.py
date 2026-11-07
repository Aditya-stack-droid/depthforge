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


import base64 as _b64

_HERO = ROOT / "assets" / "hero.jpg"
_HERO_URI = ("data:image/jpeg;base64," + _b64.b64encode(_HERO.read_bytes()).decode()) if _HERO.exists() else ""

CSS = """
:root, .dark { --df-bg:#0b0f17; --df-card:#121826; --df-line:#232b3b; --df-acc:#ff7a1a; --df-acc2:#22d3ee; }
body, gradio-app, .gradio-container { background:
    radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(255,122,26,.10), transparent 60%),
    var(--df-bg) !important; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
#df-hero { position:relative; border-radius:22px; overflow:hidden; min-height:300px; margin:8px 0 18px;
    background: linear-gradient(90deg, rgba(8,11,18,.92) 0%, rgba(8,11,18,.62) 48%, rgba(8,11,18,.0) 100%),
                url('HERO_URI') center/cover no-repeat;
    border:1px solid var(--df-line); box-shadow: 0 20px 60px rgba(0,0,0,.45); }
#df-hero .inner { padding:38px 40px 32px; max-width:760px; }
#df-hero .kicker { display:inline-block; font:600 12px/1 ui-monospace,monospace; letter-spacing:.14em;
    color:#0b0f17; background:var(--df-acc2); padding:6px 10px; border-radius:999px; }
#df-hero h1 { font-size:44px; line-height:1.05; margin:14px 0 10px; color:#fff; font-weight:800; letter-spacing:-.02em; }
#df-hero h1 span { background:linear-gradient(90deg,#22d3ee,#a3e635,#facc15,#fb923c,#ef4444);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
#df-hero p { color:#cbd5e1; font-size:16px; margin:0 0 18px; }
#df-hero .chips { display:flex; flex-wrap:wrap; gap:8px; }
#df-hero .chip { background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.25); color:#e2e8f0;
    border-radius:12px; padding:8px 12px; font-size:13px; backdrop-filter: blur(6px); }
#df-hero .chip b { color:#fff; font-size:15px; }
#df-hero .team { margin-top:16px; color:#94a3b8; font-size:13px; }
#df-steps { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:0 0 16px; }
#df-steps .s { background:var(--df-card); border:1px solid var(--df-line); border-radius:14px; padding:12px 14px; color:#cbd5e1; font-size:13px; }
#df-steps .s b { display:block; color:#fff; font-size:14px; margin-bottom:2px; }
#df-steps .n { color:var(--df-acc); font-weight:800; margin-right:6px; }
.df-card { background:var(--df-card) !important; border:1px solid var(--df-line) !important; border-radius:16px !important; padding:10px !important; }
.df-run button, button.df-run { background:linear-gradient(90deg,#ff7a1a,#ef4444) !important; color:#fff !important; border:none !important;
    font-weight:700 !important; font-size:16px !important; box-shadow:0 8px 24px rgba(239,68,68,.35) !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .image-container { border-color: var(--df-line) !important; }
#df-foot { text-align:center; color:#64748b; font-size:12px; padding:18px 0 4px; }
#df-foot a { color:#94a3b8; }
@media (max-width: 760px) {
  #df-hero .inner { padding:24px 20px; } #df-hero h1 { font-size:30px; }
  #df-steps { grid-template-columns:repeat(2,1fr); }
}
""".replace("HERO_URI", _HERO_URI)

HERO_HTML = """
<div id="df-hero"><div class="inner">
  <span class="kicker">CV HACKATHON 2026 · P4</span>
  <h1>Depth<span>Forge</span></h1>
  <p>Turns a low-resolution, noisy, holey depth map into sharp high-resolution depth at <b>2× / 4× / 8×</b>,
     using the colour photo as a guide without copying its texture into the geometry.</p>
  <div class="chips">
    <div class="chip"><b>−34%</b> RMSE vs bicubic (4×, NYU)</div>
    <div class="chip"><b>99.1%</b> pixels within 5% of truth (4×)</div>
    <div class="chip"><b>0.5 M</b> param CNN · CPU</div>
    <div class="chip"><b>3</b> pipelines: classical · DL · hybrid</div>
  </div>
  <div class="team">Team Tiger FC · Aditya Hubbly · Samay Chetan · Shaun · Farhan</div>
</div></div>
<div id="df-steps">
  <div class="s"><b><span class="n">1</span>Measure</b>noise, holes and texture risk of the input</div>
  <div class="s"><b><span class="n">2</span>Classical</b>denoise → hole fill → joint bilateral → guided filter</div>
  <div class="s"><b><span class="n">3</span>Learn</b>gated RGB-guided CNN predicts a sharp residual</div>
  <div class="s"><b><span class="n">4</span>Fuse</b>adaptive hybrid picks and combines the outputs</div>
</div>
"""

FOOT_HTML = """<div id="df-foot">DepthForge · Team Tiger FC ·
<a href="https://github.com/Aditya-stack-droid/depthforge" target="_blank">GitHub</a> ·
trained and evaluated on NYU Depth V2 (unseen test split) · images are processed in memory and not stored</div>"""

with gr.Blocks(title="DepthForge · depth upsampling") as demo:
    gr.HTML(HERO_HTML)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            with gr.Group(elem_classes="df-card"):
                sample = gr.Dropdown(list_samples(), label="1 · Pick a sample (or upload your own below)",
                                     value=(list_samples() or [None])[0])
                rgb_in = gr.Image(label="RGB guide image", type="numpy", height=220)
                depth_in = gr.File(label="Depth map (.png 16-bit mm / 8-bit / .npy m)",
                                   file_types=[".png", ".npy", ".tif", ".tiff"])
                kind = gr.Radio(["Ground truth depth (simulate sensor, get metrics)", "Low-res depth (real input)"],
                                value="Ground truth depth (simulate sensor, get metrics)", label="Depth input type")
            with gr.Group(elem_classes="df-card"):
                with gr.Row():
                    scale = gr.Radio(["2", "4", "8"], value="4", label="2 · Upsampling factor")
                    cond = gr.Dropdown(list(degrade.CONDITIONS), value="hard", label="Sensor degradation")
                method = gr.Radio(pipeline.METHODS, value="hybrid", label="3 · Method")
                stages = gr.CheckboxGroup(list(STAGE_LABELS.values()),
                                          value=[v for k, v in STAGE_LABELS.items() if k != "snap"],
                                          label="Pipeline switches: turn stages off to see what each adds")
            go = gr.Button("▶  Run DepthForge", variant="primary", elem_classes="df-run", size="lg")
            cmp_btn = gr.Button("Compare all 6 methods", variant="secondary")
            gr.Markdown(f"<span style='color:#64748b;font-size:12px'>CNN: {pipeline.model_status()} · "
                        "first run after idle may take ~10 s</span>")
        with gr.Column(scale=2, min_width=340):
            with gr.Group(elem_classes="df-card"):
                result = gr.Image(label="Result depth (warm = near, cool = far)", height=340)
            with gr.Group(elem_classes="df-card"):
                gallery = gr.Gallery(label="Every stage, step by step", columns=4, height=330)
            with gr.Row():
                with gr.Group(elem_classes="df-card"):
                    met = gr.JSON(label="Metrics")
                with gr.Group(elem_classes="df-card"):
                    par = gr.JSON(label="What the adaptive analysis decided")
            dl = gr.File(label="Download depth (16-bit PNG, millimetres)")
    with gr.Accordion("Method comparison: nearest · bilinear · bicubic · classical · DL · hybrid", open=False) as acc:
        table = gr.Dataframe(headers=["method", "RMSE", "MAE", "PSNR", "SSIM", "Edge-RMSE", "Boundary-F1", "ms"])
        cmp_gal = gr.Gallery(columns=7, height=220)
    gr.HTML(FOOT_HTML)

    sample.change(load_sample, sample, [rgb_in, depth_in])
    demo.load(load_sample, sample, [rgb_in, depth_in])  # judges can press Run immediately
    go.click(run_one, [rgb_in, depth_in, kind, scale, cond, method, stages], [result, gallery, met, par, dl])
    cmp_btn.click(compare_all, [rgb_in, depth_in, kind, scale, cond], [table, cmp_gal]).then(
        lambda: gr.Accordion(open=True), None, acc)

if __name__ == "__main__":
    FORCE_DARK = "() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }"
    THEME = gr.themes.Base(primary_hue="orange", secondary_hue="cyan", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
        body_background_fill="#0b0f17", body_background_fill_dark="#0b0f17",
        block_background_fill="#121826", block_background_fill_dark="#121826",
        block_border_color="#232b3b", block_border_color_dark="#232b3b",
        body_text_color="#e2e8f0", body_text_color_dark="#e2e8f0",
        block_label_text_color="#94a3b8", block_label_text_color_dark="#94a3b8",
        input_background_fill="#0f1522", input_background_fill_dark="#0f1522",
        border_color_primary="#232b3b", border_color_primary_dark="#232b3b")
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   css=CSS, theme=THEME, js=FORCE_DARK)
