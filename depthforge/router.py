"""Learned per-image router: picks which hybrid path to run.

The hand-set rules in `adaptive.choose_params` (dual_path if holes >= 20 %, refine if
scale >= 8 or holes > 2 %) are replaced by a small ridge regressor that predicts, from
cheap input statistics, the log-RMSE of each candidate path. The router picks the
predicted-best candidate, but only when it beats the rule-based default by a margin
(so it can't be much worse than the rules), and it runs only the CNN passes that the
chosen path needs (so it is also faster than always running both).

Candidates (b = CNN on bicubic prior, c = CNN on classical JBU prior):
    b, c, b+c averaged, and each of those followed by a light guided-filter refinement.
Trained by scripts/train_router.py on NYU *validation* images (never the test split);
weights live in weights/router.json. No scikit-learn needed at deploy time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CANDIDATES = ["b", "c", "bc", "b_ref", "c_ref", "bc_ref"]
FEATURES = ["bias", "log2_scale", "log2_scale_sq", "holes", "holes_sq", "log_noise", "texture_risk",
            "edge_density", "lr_grad", "holes_x_scale", "noise_x_scale"]

_PATH = Path(os.environ.get("DEPTHFORGE_ROUTER", Path(__file__).resolve().parent.parent / "weights" / "router.json"))
_ROUTER = None
_LOADED = False


def features(p: dict) -> np.ndarray:
    ls = np.log2(max(p["scale"], 1))
    h, n = p["holes"], p["noise"]
    return np.array([1.0, ls, ls * ls, h, h * h, np.log(n + 1e-3), p["texture_risk"],
                     p.get("edge_density", 0.0), p.get("lr_grad", 0.0), h * ls, n * ls], np.float64)


def rule_candidate(p: dict) -> str:
    """The original hand-set strategy, expressed as a candidate name."""
    return ("bc" if p["dual_path"] else "b") + ("_ref" if p["refine"] else "")


def needs(cand: str) -> tuple[bool, bool, bool]:
    """-> (needs CNN on bicubic, needs CNN on classical, refine)"""
    base = cand.replace("_ref", "")
    return "b" in base, "c" in base, cand.endswith("_ref")


class Router:
    def __init__(self, d: dict):
        self.cands = d["candidates"]
        self.mu, self.sd = np.array(d["mean"]), np.array(d["std"])
        self.W = np.array(d["weights"])            # n_feat x n_cand
        self.margin = float(d.get("margin", 0.0))

    def predict(self, p: dict) -> np.ndarray:
        x = (features(p) - self.mu) / self.sd
        return x @ self.W

    def choose(self, p: dict) -> str:
        pred = self.predict(p)
        default = rule_candidate(p)
        best = int(np.argmin(pred))
        if default in self.cands and pred[best] > pred[self.cands.index(default)] - self.margin:
            return default
        return self.cands[best]


def fit(X: np.ndarray, Y: np.ndarray, lam: float):
    """Ridge on standardised features. X: n x f (col 0 = bias), Y: n x c (centred log-RMSE)."""
    mu, sd = X.mean(0), X.std(0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    R = lam * np.eye(Z.shape[1]); R[0, 0] = 0.0
    W = np.linalg.solve(Z.T @ Z + R, Z.T @ Y)
    return mu, sd, W


def load() -> Router | None:
    global _ROUTER, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            if _PATH.exists() and os.environ.get("DEPTHFORGE_ROUTER_OFF", "0") != "1":
                _ROUTER = Router(json.loads(_PATH.read_text()))
        except Exception:
            _ROUTER = None
    return _ROUTER


def save(path, mu, sd, W, margin, meta=None):
    json.dump(dict(candidates=CANDIDATES, features=FEATURES, mean=mu.tolist(), std=sd.tolist(),
                   weights=W.tolist(), margin=margin, meta=meta or {}), open(path, "w"), indent=1)
