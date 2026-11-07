"""Input analysis + adaptive parameter selection.

Instead of fixed constants, every run first *measures* the input (scale, noise,
missing data, how much RGB texture there is that does NOT correspond to depth
edges) and derives filter parameters from those measurements. The resulting
`params` dict is shown in the UI and posted by the Switch agents, so every
decision is explainable in the viva.
"""
from __future__ import annotations

import numpy as np

from . import classical as C

# relative-noise level above which denoising / guided filtering switch on
# (clean NYU maps estimate 0.002-0.007 because coarse sampling of real structure
# looks like noise; sensor noise of 1-2 % estimates 0.017-0.022)
NOISE_ON = 0.008


def analyze(lr: np.ndarray, rgb_hr: np.ndarray) -> dict:
    H, W = rgb_hr.shape[:2]
    h, w = lr.shape
    scale = max(1, int(round(((H / h) + (W / w)) / 2)))
    noise = C.estimate_noise(lr)
    holes = C.hole_ratio(lr)
    up = C.interpolate(C.fill_holes(lr), (H, W), "bilinear")
    rgb_e = C.rgb_edges(rgb_hr)
    dep_e = C.band(C.depth_discontinuities(up, 0.05), max(2, scale))
    edge_px = rgb_e.sum() + 1e-6
    # fraction of RGB edges with no depth edge nearby == texture that could be
    # wrongly copied into the depth map ("texture-copy risk")
    texture_risk = float((rgb_e & ~dep_e).sum() / edge_px)
    valid = lr[lr > 0]
    return dict(scale=scale, noise=round(noise, 4), holes=round(holes, 4),
                texture_risk=round(texture_risk, 3),
                depth_min=float(valid.min()) if valid.size else 0.0,
                depth_max=float(valid.max()) if valid.size else 0.0)


def choose_params(lr: np.ndarray, rgb_hr: np.ndarray, stats: dict | None = None) -> dict:
    s = stats or analyze(lr, rgb_hr)
    scale, noise, tex = s["scale"], s["noise"], s["texture_risk"]
    p = dict(s)
    # JBU: larger LR window at small scales isn't needed; spatial sigma in LR px.
    p["jbu_radius"] = 2 if scale <= 4 else 3
    # tight spatial kernel on clean input (keeps detail; tuned on NYU val),
    # wider when the LR samples are noisy so JBU also averages noise out
    p["jbu_sigma_s"] = 0.7 if noise < NOISE_ON else 1.0
    # Range sigma: high texture risk -> wider colour tolerance (trust colour less)
    p["jbu_sigma_r"] = float(np.clip(0.06 + 0.10 * tex, 0.06, 0.16))
    # Guided filter radius grows with scale; eps grows with noise and texture risk
    # Guided filter = noise remover. On clean input JBU is already edge-aware and an
    # extra colour-guided pass only re-injects texture, so it is switched off.
    p["use_guided"] = bool(noise >= NOISE_ON)
    p["gf_radius"] = int(np.clip(scale // 2 + 1, 2, 5))
    p["gf_eps"] = float(np.clip(1e-4 * (1 + 40 * noise) * (1 + 4 * tex), 1e-4, 1e-2))
    # Discontinuity detection & snapping band
    p["disc_thresh"] = float(np.clip(0.03 + 1.5 * noise, 0.03, 0.12))
    p["band"] = int(np.clip(scale // 2, 1, 4))
    p["snap_k"] = int(np.clip(2 * (scale // 2) + 3, 3, 9))
    # Hybrid strategy (selected on NYU *validation*, see docs/DESIGN.md §3.4):
    #  - many missing samples -> run the CNN on BOTH priors (bicubic and classical JBU)
    #    and average them: the two priors fail differently inside holes
    #  - large factor or any holes -> final edge-aware guided refinement of the CNN output
    p["dual_path"] = bool(s["holes"] >= 0.2)
    p["refine"] = bool(scale >= 8 or s["holes"] > 0.02)
    return p
