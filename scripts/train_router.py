"""Fit the learned per-image router (depthforge/router.py).

  python scripts/train_router.py --data nyu --mat nyu_depth_v2_labeled.mat --n 100

Uses the NYU *validation* images 900-999 (never the test split 1000-1448). For every
image x scale x condition it runs all six candidate paths once, records the RMSE of each,
and fits a ridge regressor from input statistics to log-RMSE. Ridge strength and the
safety margin are chosen by 5-fold cross-validation *grouped by image*; the CV report
(router vs rules vs always-CNN vs oracle) goes to results/router_report.md.
Requires the trained CNN checkpoint. Raw per-sample results are cached in
results/router_samples.npz so you can re-fit without recomputing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from depthforge import data, degrade, metrics, pipeline, router as R  # noqa: E402
from depthforge.adaptive import choose_params  # noqa: E402


def collect(a):
    if a.data == "nyu":
        idx = list(range(900, 900 + a.n))           # validation split only
        pairs = list(data.load_nyu(a.mat, idx))
    else:
        pairs = [data.synth_scene(240, 320, s) for s in range(7000, 7000 + a.n)]
    rows = []
    for i, (rgb, gt) in enumerate(pairs):
        for cond in degrade.CONDITIONS:
            for s in (2, 4, 8):
                rgb_c, gt_c = degrade.crop_to_scale(rgb, gt, s)
                lr = degrade.make_lr(gt_c, s, *degrade.CONDITIONS[cond], seed=10_000 + i)
                p = choose_params(lr, rgb_c)
                cands = pipeline.all_candidates(lr, rgb_c, p)
                err = [metrics.rmse(cands[c], gt_c) for c in R.CANDIDATES]
                rows.append((i, s, list(degrade.CONDITIONS).index(cond), R.features(p),
                             R.CANDIDATES.index(R.rule_candidate(p)), err))
        print(f"  image {i + 1}/{len(pairs)}", flush=True)
    img = np.array([r[0] for r in rows]); sc = np.array([r[1] for r in rows]); cd = np.array([r[2] for r in rows])
    X = np.stack([r[3] for r in rows]); rule = np.array([r[4] for r in rows]); E = np.array([r[5] for r in rows])
    return img, sc, cd, X, rule, E


def choose(pred, rule, margin):
    best = pred.argmin(1)
    r = pred[np.arange(len(pred)), rule]
    return np.where(pred[np.arange(len(pred)), best] > r - margin, rule, best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["nyu", "synth"], default="synth")
    ap.add_argument("--mat", default="nyu_depth_v2_labeled.mat")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cache", default=str(ROOT / "results" / "router_samples.npz"))
    ap.add_argument("--out", default=str(ROOT / "weights" / "router.json"))
    ap.add_argument("--refit", action="store_true", help="reuse cached samples")
    a = ap.parse_args()
    if pipeline.load_model() is None:
        sys.exit("CNN unavailable: " + pipeline.model_status())
    cache = Path(a.cache)
    if a.refit and cache.exists():
        z = np.load(cache); img, sc, cd, X, rule, E = (z[k] for k in ("img", "sc", "cd", "X", "rule", "E"))
    else:
        img, sc, cd, X, rule, E = collect(a)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, img=img, sc=sc, cd=cd, X=X, rule=rule, E=E)
    n = len(E)
    Y = np.log(E) - np.log(E).mean(1, keepdims=True)
    folds = img % 5
    best = (1e9, None)
    for lam in (0.1, 1, 10, 100):
        for margin in (0.0, 0.002, 0.005, 0.01, 0.02):
            tot = np.zeros(n)
            for f in range(5):
                tr, te = folds != f, folds == f
                mu, sd, W = R.fit(X[tr], Y[tr], lam)
                pred = ((X[te] - mu) / sd) @ W
                pick = choose(pred, rule[te], margin)
                tot[te] = E[te][np.arange(te.sum()), pick]
            if tot.mean() < best[0]:
                best = (tot.mean(), (lam, margin, tot.copy()))
    lam, margin, cv = best[1]
    rules_e = E[np.arange(n), rule]
    L = ["# Router cross-validation (NYU val 900-999, grouped 5-fold by image)", "",
         f"samples: {n}   ridge lambda: {lam}   margin: {margin}", "",
         "| Strategy | mean RMSE (m) |", "|---|---|"]
    for name, e in [("always CNN(bicubic) 'b'", E[:, 0]), ("hand-set rules (previous hybrid)", rules_e),
                    ("**learned router**", cv), ("oracle (best path per sample)", E.min(1))]:
        L.append(f"| {name} | {e.mean():.5f} |")
    L += ["", "| scale | condition | rules | router | oracle |", "|---|---|---|---|---|"]
    cn = list(degrade.CONDITIONS)
    for s in (2, 4, 8):
        for c in range(len(cn)):
            m = (sc == s) & (cd == c)
            if m.any():
                L.append(f"| {s}x | {cn[c]} | {rules_e[m].mean():.4f} | {cv[m].mean():.4f} | {E[m].min(1).mean():.4f} |")
    mu, sd, W = R.fit(X, Y, lam)
    R.save(a.out, mu, sd, W, margin, meta=dict(data=a.data, n_samples=int(n), cv_rmse=float(cv.mean()),
                                              rules_rmse=float(rules_e.mean())))
    rep = ROOT / "results" / "router_report.md"
    rep.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("\nwrote", a.out, "and", rep)
    if cv.mean() >= rules_e.mean():
        print("! router is NOT better than the rules in CV; delete weights/router.json to keep the rule-based hybrid")


if __name__ == "__main__":
    main()
