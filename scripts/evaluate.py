"""Comparative evaluation + ablation + failure-case mining.

  python scripts/evaluate.py --data nyu --mat nyu_depth_v2_labeled.mat --n 449 \
      --team T01 --usns "1RV23CS001,1RV23CS002"

Writes:
  results/eval_P4_<TEAM>.md   (submission file; first line = USNs)
  results/results.json         (machine-readable; read by the Switch judge agent)
  results/failures/*.png       (worst cases for the failure analysis)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from depthforge import data, degrade, metrics, pipeline, viz  # noqa: E402

COLS = ["rmse", "mae", "psnr", "ssim", "edge_rmse", "boundary_f1", "delta1", "delta105", "latency_ms", "peak_mem_mb"]
HEAD = "| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |"
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"
GROUP = {"nearest": "Interpolation", "bilinear": "Interpolation", "bicubic": "Interpolation",
         "classical": "Classical refinement", "dl": "Deep learning", "hybrid": "Hybrid (ours)"}


def get_pairs(a):
    if a.data == "nyu":
        # n test images spread evenly over the 449-image test split (all of them if n >= 449)
        test = list(data.NYU_TEST)
        idx = [test[int(round(j))] for j in np.linspace(0, len(test) - 1, min(a.n, len(test)))]
        return [(f"nyu_{i}", r, d) for i, (r, d) in zip(idx, data.load_nyu(a.mat, idx))]
    if a.data == "folder":
        return list(data.load_folder(a.folder))[: a.n]
    return [(f"synth_{s}", *data.synth_scene(240, 320, s)) for s in range(9000, 9000 + a.n)]  # unseen seeds


def row(name, m):
    return "| " + name + " | " + " | ".join(
        f"{m[c]:.4f}" if c in ("rmse", "mae", "edge_rmse") else f"{m[c]:.3f}" if c in ("ssim", "boundary_f1")
        else f"{m[c]:.2f}" if c in ("psnr", "delta1", "delta105") else f"{m[c]:.1f}" for c in COLS) + " |"


def mean(lst):
    return {c: float(np.nanmean([x[c] for x in lst])) for c in COLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["nyu", "synth", "folder"], default="synth")
    ap.add_argument("--mat", default="nyu_depth_v2_labeled.mat")
    ap.add_argument("--folder")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--scales", default="2,4,8")
    ap.add_argument("--conditions", default="clean,hard")
    ap.add_argument("--methods", default=",".join(pipeline.METHODS))
    ap.add_argument("--team", default="T01")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    out = Path(a.out); (out / "failures").mkdir(parents=True, exist_ok=True)
    pairs = get_pairs(a)
    scales = [int(s) for s in a.scales.split(",")]
    conds = a.conditions.split(",")
    methods = [m for m in a.methods.split(",") if m]
    if pipeline.load_model() is None:
        print("! CNN unavailable:", pipeline.model_status(), "-> skipping 'dl'")
        methods = [m for m in methods if m != "dl"]
    print(f"{len(pairs)} images x {scales} x {conds} x {methods}")

    res = defaultdict(list)          # (cond, scale, method) -> [metrics]
    worst = []                       # (edge_rmse gain vs bicubic, ...)
    t0 = time.time()
    for idx, (name, rgb, gt) in enumerate(pairs):
        for cond in conds:
            for s in scales:
                rgb_c, gt_c = degrade.crop_to_scale(rgb, gt, s)
                lr = degrade.make_lr(gt_c, s, *degrade.CONDITIONS[cond], seed=idx)
                per = {}
                for m in methods:
                    r = pipeline.run(m, lr, rgb_c)
                    mm = metrics.all_metrics(r["depth"], gt_c)
                    mm.update(latency_ms=r["latency_ms"], peak_mem_mb=r["peak_mem_mb"])
                    res[(cond, s, m)].append(mm); per[m] = (mm, r["depth"])
                if "hybrid" in per:
                    best_other = min(v[0]["rmse"] for k, v in per.items() if k != "hybrid")
                    worst.append((per["hybrid"][0]["rmse"] - best_other, per["hybrid"][0]["rmse"],
                                  name, cond, s, rgb_c, gt_c, lr, per["hybrid"][1]))
        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    # ---------------- ablation (hybrid, 4x, hardest condition) -------------- #
    abl_cond = conds[-1]
    abl_rows = []
    abl_cfgs = [("Full hybrid", {})] + [(f"− {k}", {k: False}) for k in pipeline.HYBRID_STAGES if k != "snap"] \
        + [("+ snap (opt-in)", {"snap": True})]
    if pipeline.load_model() is None:
        abl_cfgs = [c for c in abl_cfgs if c[0] not in ("− cnn", "− fusion")]
    for label, st in abl_cfgs:
        lst = []
        for idx, (name, rgb, gt) in enumerate(pairs[: min(len(pairs), 30)]):
            rgb_c, gt_c = degrade.crop_to_scale(rgb, gt, 4)
            lr = degrade.make_lr(gt_c, 4, *degrade.CONDITIONS[abl_cond], seed=idx)
            r = pipeline.run("hybrid", lr, rgb_c, st)
            mm = metrics.all_metrics(r["depth"], gt_c); mm.update(latency_ms=r["latency_ms"], peak_mem_mb=r["peak_mem_mb"])
            lst.append(mm)
        abl_rows.append((label, mean(lst)))

    # ---------------- failure cases ---------------- #
    worst.sort(key=lambda t: (-t[0], -t[1]))
    fails = []
    picked, seen = [], set()
    for w in worst:                      # 3 distinct images
        if w[2] not in seen:
            picked.append(w); seen.add(w[2])
        if len(picked) == 3:
            break
    for k, (gain, e, name, cond, s, rgb_c, gt_c, lr, pred) in enumerate(picked):
        panel = np.concatenate([rgb_c, viz.nn_upscale(viz.colorize(lr), gt_c.shape),
                                viz.colorize(pred), viz.colorize(gt_c), viz.error_map(pred, gt_c)], 1)
        fn = out / "failures" / f"fail{k+1}_{name}_{cond}_x{s}.png"
        cv2.imwrite(str(fn), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        fails.append(dict(file=fn.name, image=name, condition=cond, scale=s, rmse=e, rmse_minus_bicubic=gain))


    # ---------------- write markdown ---------------- #
    L = [a.usns, "", f"# Evaluation — P4 Edge-Preserving Depth Map Upsampling — Team {a.team}", "",
         f"Dataset: **{a.data}** ({len(pairs)} unseen test images"
         + (", evenly sampled from the NYU v2 standard test split 1000-1448, never used for training or tuning" if a.data == "nyu" else "") + "). "
         f"Degradations: {', '.join(conds)} (see `depthforge/degrade.py`). Latency: 2-core CPU, per image, tiled 128² CNN inference in FP32, no test-time augmentation.", ""]
    L += ["## 1. Depth accuracy, edge preservation, latency & memory (per scale)", ""]
    for cond in conds:
        for s in scales:
            L += [f"### {s}× upsampling — condition `{cond}`", "", HEAD, SEP]
            for m in methods:
                L.append(row(f"{m} ({GROUP[m]})", mean(res[(cond, s, m)])))
            L.append("")
    L += ["## 2. Performance across upsampling factors (RMSE, condition `%s`)" % conds[0], "",
          "| Method | " + " | ".join(f"{s}×" for s in scales) + " |", "|---|" + "---|" * len(scales)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{mean(res[(conds[0], s, m)])['rmse']:.4f}" for s in scales) + " |")
    L += ["", f"## 3. Ablation — hybrid pipeline, 4×, condition `{abl_cond}`", "", HEAD, SEP]
    L += [row(lbl, m) for lbl, m in abl_rows]
    L += ["", "## 4. Failure cases (largest hybrid losses vs the best other method on the same input)", "",
          "Panels: RGB | LR input | hybrid | ground truth | |error|.", ""]
    for f in fails:
        L.append(f"- `{f['file']}` — {f['image']}, {f['condition']}, {f['scale']}×, RMSE {f['rmse']:.4f} "
                 f"({f['rmse_minus_bicubic']:+.4f} vs best other method). Analysis: see docs/FAILURE_ANALYSIS.md")
    md = out / f"eval_P4_{a.team}.md"
    md.write_text("\n".join(L) + "\n")
    js = {f"{c}|{s}|{m}": mean(v) for (c, s, m), v in res.items()}
    json.dump(dict(summary=js, ablation=dict(abl_rows), failures=fails, model=pipeline.model_status()),
              open(out / "results.json", "w"), indent=1)
    print("\n".join(L[:40]))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
