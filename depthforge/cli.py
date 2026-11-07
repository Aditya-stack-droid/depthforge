"""Command-line interface — this is what the Switch agents call.

  python -m depthforge.cli analyze --rgb samples/synthetic_1_rgb.png --depth samples/synthetic_1_depth.png --scale 4
  python -m depthforge.cli run     --rgb ... --depth ... --method hybrid --scale 8 --condition hard --off snap,cnn
  python -m depthforge.cli compare --rgb ... --depth ... --scale 4 --condition noisy
  python -m depthforge.cli ablate  --n 10 --scale 4 --condition hard

Every command prints ONE JSON object to stdout (easy for agents to quote in chat)
and optionally saves a side-by-side PNG with --png.
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from . import data, degrade, metrics, pipeline, viz
from .adaptive import analyze


def _load(a):
    if a.rgb and a.depth:
        rgb = cv2.cvtColor(cv2.imread(a.rgb), cv2.COLOR_BGR2RGB)
        gt = data.read_depth(a.depth)
        if gt.shape != rgb.shape[:2]:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        rgb, gt = data.synth_scene(240, 320, a.seed)
    rgb, gt = degrade.crop_to_scale(rgb, gt, a.scale)
    lr = degrade.make_lr(gt, a.scale, *degrade.CONDITIONS[a.condition], seed=a.seed)
    return rgb, gt, lr


def _round(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def cmd_analyze(a):
    rgb, gt, lr = _load(a)
    from .adaptive import choose_params
    return dict(stats=analyze(lr, rgb), params=_round(choose_params(lr, rgb)))


def cmd_run(a):
    rgb, gt, lr = _load(a)
    off = {k: False for k in (a.off.split(",") if a.off else []) if k}
    off.update({k: True for k in (a.on.split(",") if a.on else []) if k})
    r = pipeline.run(a.method, lr, rgb, off)
    m = metrics.all_metrics(r["depth"], gt)
    m.update(latency_ms=r["latency_ms"], peak_mem_mb=r["peak_mem_mb"])
    if a.png:
        panel = np.concatenate([rgb, viz.nn_upscale(viz.colorize(lr), gt.shape), viz.colorize(r["depth"]),
                                viz.colorize(gt), viz.error_map(r["depth"], gt)], 1)
        cv2.imwrite(a.png, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return dict(method=a.method, scale=a.scale, condition=a.condition, stages_off=list(off),
                metrics=_round(m), notes=r["notes"], png=a.png)


def cmd_compare(a):
    rgb, gt, lr = _load(a)
    out = {}
    for m in pipeline.METHODS:
        if m == "dl" and pipeline.load_model() is None:
            continue
        r = pipeline.run(m, lr, rgb)
        mm = metrics.all_metrics(r["depth"], gt); mm["latency_ms"] = r["latency_ms"]
        out[m] = _round(mm)
    best = min(out, key=lambda k: out[k]["rmse"])
    return dict(scale=a.scale, condition=a.condition, results=out, best_by_rmse=best)


def cmd_ablate(a):
    rows = {}
    cfgs = [("full", {})] + [(f"no_{k}", {k: False}) for k in pipeline.HYBRID_STAGES if k != "snap"] \
        + [("with_snap", {"snap": True})]
    for name, st in cfgs:
        acc = []
        for i in range(a.n):
            rgb, gt = data.synth_scene(240, 320, 9000 + i) if not a.rgb else (None, None)
            if a.rgb:
                a.seed = i
                rgb, gt, lr = _load(a)
            else:
                rgb, gt = degrade.crop_to_scale(rgb, gt, a.scale)
                lr = degrade.make_lr(gt, a.scale, *degrade.CONDITIONS[a.condition], seed=i)
            r = pipeline.run("hybrid", lr, rgb, st)
            acc.append(metrics.all_metrics(r["depth"], gt))
        rows[name] = _round({k: float(np.nanmean([x[k] for x in acc])) for k in acc[0]})
    full = rows["full"]["rmse"]
    contrib = {k.replace("no_", ""): round(v["rmse"] - full, 4) for k, v in rows.items() if k != "full"}
    return dict(scale=a.scale, condition=a.condition, n=a.n, ablation=rows,
                rmse_increase_when_removed=contrib, cnn=pipeline.model_status())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="depthforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "run", "compare", "ablate"):
        p = sub.add_parser(name)
        p.add_argument("--rgb"); p.add_argument("--depth")
        p.add_argument("--scale", type=int, default=4, choices=[2, 4, 8, 16])
        p.add_argument("--condition", default="hard", choices=list(degrade.CONDITIONS))
        p.add_argument("--seed", type=int, default=0)
        if name == "run":
            p.add_argument("--method", default="hybrid", choices=pipeline.METHODS)
            p.add_argument("--off", default="", help="comma list of stages to disable: " + ",".join(pipeline.HYBRID_STAGES))
            p.add_argument("--on", default="", help="comma list of opt-in stages to enable (e.g. snap)")
            p.add_argument("--png")
        if name == "ablate":
            p.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        res = dict(analyze=cmd_analyze, run=cmd_run, compare=cmd_compare, ablate=cmd_ablate)[a.cmd](a)
        print(json.dumps(res, indent=1, default=float))
    except Exception as e:
        print(json.dumps(dict(error=str(e), cmd=a.cmd)))
        sys.exit(1)


if __name__ == "__main__":
    main()
