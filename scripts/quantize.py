"""Model optimisation study: FP32 vs FP16-storage vs BF16 inference vs 30% L1 pruning.
Measures checkpoint size, params, latency and accuracy (hybrid pipeline, 4x, hard).

  python scripts/quantize.py --n 10
"""
from __future__ import annotations

import argparse
import copy
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from depthforge import classical as C, data, degrade, metrics, pipeline  # noqa: E402
from depthforge.model import count_params  # noqa: E402


def size_mb(sd):
    b = io.BytesIO(); torch.save(sd, b); return b.tell() / 2 ** 20


def evaluate(net, pairs, dtype=torch.float32):
    """RMSE + CNN latency of `net` inside the real (tiled) inference path, 4x hard."""
    errs, times = [], []
    pipeline._MODEL = net
    for i, (rgb, gt) in enumerate(pairs):
        rgb, gt = degrade.crop_to_scale(rgb, gt, 4)
        lr = degrade.make_lr(gt, 4, *degrade.CONDITIONS["hard"], seed=i)
        bic = C.interpolate(C.fill_holes(lr), gt.shape, "bicubic")
        t = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            out = pipeline.run_cnn(bic, rgb)
        times.append((time.perf_counter() - t) * 1000)
        errs.append(metrics.rmse(out, gt))
    return float(np.mean(errs)), float(np.median(times))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mat", default=None, help="NYU .mat -> evaluate on NYU val images 900-999")
    a = ap.parse_args()
    net = pipeline.load_model()
    if net is None:
        sys.exit("no checkpoint: " + pipeline.model_status())
    if a.mat:
        pairs = list(data.load_nyu(a.mat, list(range(900, 1000, max(1, 100 // a.n)))[: a.n])); src = "NYU val"
    else:
        pairs = [data.synth_scene(240, 320, 9000 + i) for i in range(a.n)]; src = "synthetic"
    rows = []
    e, t = evaluate(net, pairs)
    rows.append(("FP32 (baseline)", count_params(net), size_mb(net.state_dict()), t, e))
    half = {k: v.half() for k, v in net.state_dict().items()}
    net16 = copy.deepcopy(net); net16.load_state_dict({k: v.float() for k, v in half.items()})
    e, t = evaluate(net16, pairs)
    rows.append(("FP16 weights (storage)", count_params(net), size_mb(half), t, e))
    e, t = evaluate(net, pairs, torch.bfloat16)
    rows.append(("BF16 autocast inference", count_params(net), size_mb(half), t, e))
    pr = copy.deepcopy(net)
    for m in pr.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels > 1:
            prune.l1_unstructured(m, "weight", amount=0.3); prune.remove(m, "weight")
    nz = sum(int((p != 0).sum()) for p in pr.parameters())
    e, t = evaluate(pr, pairs)
    # dense storage; the gain is in non-zero params (a sparse kernel/format would realise it)
    rows.append(("30% L1 pruning (dense storage)", nz, size_mb(pr.state_dict()), t, e))
    L = ["| Variant | Non-zero params | Checkpoint (MB) | CNN latency (ms, CPU) | CNN RMSE, 4× hard (m) |", "|---|---|---|---|---|"]
    L += [f"| {n} | {p:,} | {s:.2f} | {t:.1f} | {e:.4f} |" for n, p, s, t, e in rows]
    out = ROOT / "results" / "optimization.md"; out.parent.mkdir(exist_ok=True)
    out.write_text(f"# Model optimisation ({src}, {len(pairs)} images, 4x, condition hard, tiled CPU inference)\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
