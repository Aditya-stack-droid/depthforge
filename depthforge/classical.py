"""Classical computer-vision building blocks for depth upsampling.

Everything here is pure NumPy/OpenCV (no learned weights) so it runs anywhere
and gives the interpretable baselines the judges ask for:

  * nearest / bilinear / bicubic interpolation baselines
  * hole filling (normalized convolution pyramid)
  * adaptive noise removal (median, strength picked from a noise estimate)
  * Joint Bilateral Upsampling (Kopf et al. 2007) with RGB guidance
  * Colour guided filter (He et al. 2010)
  * Edge detection (RGB Canny) and depth discontinuity detection
  * Boundary-aware refinement ("min/max snapping" in discontinuity bands)
"""
from __future__ import annotations

import cv2
import numpy as np

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


# --------------------------------------------------------------------------- #
# Interpolation baselines
# --------------------------------------------------------------------------- #
def interpolate(lr: np.ndarray, out_hw: tuple[int, int], mode: str = "bicubic") -> np.ndarray:
    h, w = out_hw
    up = cv2.resize(lr.astype(np.float32), (w, h), interpolation=INTERP[mode])
    return np.clip(up, 0, None)


# --------------------------------------------------------------------------- #
# Analysis helpers (used by the adaptive parameter selection)
# --------------------------------------------------------------------------- #
def hole_ratio(depth: np.ndarray) -> float:
    return float((depth <= 0).mean())


def estimate_noise(depth: np.ndarray) -> float:
    """Robust relative noise estimate: MAD of the Laplacian on valid pixels,
    divided by median depth. ~0 for clean maps, ~0.01-0.05 for noisy sensors."""
    valid = depth > 0
    if valid.sum() < 16:
        return 0.0
    d = fill_holes(depth) if (~valid).any() else depth
    lap = cv2.Laplacian(d.astype(np.float32), cv2.CV_32F, ksize=1)
    lap = lap[valid]
    mad = np.median(np.abs(lap - np.median(lap)))
    # ignore the high-magnitude tail (real discontinuities) by using the MAD
    sigma = 1.4826 * mad / np.sqrt(20.0)  # Laplacian(ksize=1) noise gain = sqrt(20)
    return float(sigma / (np.median(depth[valid]) + 1e-6))


# --------------------------------------------------------------------------- #
# Hole filling & denoising
# --------------------------------------------------------------------------- #
def fill_holes(depth: np.ndarray, max_iter: int = 12) -> np.ndarray:
    """Normalized convolution with a growing kernel. Only invalid pixels change."""
    d = depth.astype(np.float32).copy()
    valid = (d > 0).astype(np.float32)
    if valid.all() or valid.sum() == 0:
        return d
    out = d.copy()
    missing = valid == 0
    k = 3
    for _ in range(max_iter):
        num = cv2.GaussianBlur(d * valid, (k, k), 0)
        den = cv2.GaussianBlur(valid, (k, k), 0)
        fillable = missing & (den > 1e-3)
        out[fillable] = num[fillable] / den[fillable]
        missing = missing & ~fillable
        if not missing.any():
            break
        k = k * 2 + 1
        if k > 255:
            k = 255
    if missing.any():  # still empty (huge holes) -> global median
        out[missing] = np.median(d[valid > 0])
    return out


def fill_large_holes(depth: np.ndarray, radius: int = 2) -> np.ndarray:
    """Fill only holes that the JBU window cannot bridge (no valid sample within
    `radius`). Small holes are left at 0 because JBU skips invalid samples and
    re-estimates them with RGB guidance, which is more accurate than blind filling."""
    valid = (depth > 0).astype(np.uint8)
    if valid.all():
        return depth
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    reachable = cv2.dilate(valid, k) > 0
    out = depth.copy()
    big = ~reachable
    if big.any():
        out[big] = fill_holes(depth)[big]
    return out


def denoise_depth(depth: np.ndarray, noise: float) -> np.ndarray:
    """Adaptive edge-preserving denoise of the LR map (skipped when the estimated
    relative noise is negligible)."""
    from .adaptive import NOISE_ON
    if noise < NOISE_ON:
        return depth
    valid = depth > 0
    # Bilateral on depth itself: range sigma = 3x the estimated noise, so random
    # noise is averaged out but real depth steps (>> noise) are left intact.
    sig_r = 3.0 * noise * float(np.median(depth[valid]))
    sm = cv2.bilateralFilter(depth.astype(np.float32), 5, sig_r, 1.5)
    out = depth.copy()
    out[valid] = sm[valid]
    return out


# --------------------------------------------------------------------------- #
# Joint bilateral upsampling
# --------------------------------------------------------------------------- #
def joint_bilateral_upsample(lr: np.ndarray, rgb_hr: np.ndarray, radius: int = 2,
                             sigma_s: float = 1.0, sigma_r: float = 0.08) -> np.ndarray:
    """JBU: every HR pixel is a weighted mean of nearby LR depth samples; weights =
    spatial Gaussian (in LR grid units) x range Gaussian on the *HR* guide colour.
    Missing LR samples get zero weight, so this also fills small holes."""
    H, W = rgb_hr.shape[:2]
    h, w = lr.shape
    sy, sx = H / h, W / w
    guide = rgb_hr.astype(np.float32) / 255.0
    guide_lr = cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ly = (yy + 0.5) / sy - 0.5  # HR pixel position in LR coordinates
    lx = (xx + 0.5) / sx - 0.5
    by, bx = np.floor(ly).astype(np.int32), np.floor(lx).astype(np.int32)

    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    for dy in range(-radius + 1, radius + 1):
        for dx in range(-radius + 1, radius + 1):
            qy = np.clip(by + dy, 0, h - 1)
            qx = np.clip(bx + dx, 0, w - 1)
            dq = lr[qy, qx]
            ws = np.exp(-((qy - ly) ** 2 + (qx - lx) ** 2) / (2 * sigma_s ** 2))
            diff = guide - guide_lr[qy, qx]
            wr = np.exp(-(diff ** 2).sum(-1) / (2 * sigma_r ** 2))
            wgt = ws * wr * (dq > 0)
            num += wgt * dq
            den += wgt
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0).astype(np.float32)
    if (out <= 0).any():
        out = fill_holes(out)
    return out


# --------------------------------------------------------------------------- #
# Guided filter (colour guide)
# --------------------------------------------------------------------------- #
def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide_rgb: np.ndarray, p: np.ndarray, r: int = 4, eps: float = 1e-3) -> np.ndarray:
    """He et al. colour guided filter. p is filtered with the local linear model
    q = a^T I + b; edges present in I *and* p are kept, flat areas smoothed."""
    I = guide_rgb.astype(np.float32) / 255.0
    p = p.astype(np.float32)
    scale = float(p.max()) + 1e-6
    p = p / scale
    mI = [_box(I[..., c], r) for c in range(3)]
    mp = _box(p, r)
    cov_Ip = np.stack([_box(I[..., c] * p, r) - mI[c] * mp for c in range(3)], -1)
    S = np.empty(p.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(I[..., i] * I[..., j], r) - mI[i] * mI[j]
            S[..., i, j] = v
            S[..., j, i] = v
    S += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(S, cov_Ip[..., None])[..., 0]
    b = mp - (a * np.stack(mI, -1)).sum(-1)
    ma = np.stack([_box(a[..., c], r) for c in range(3)], -1)
    mb = _box(b, r)
    q = (ma * I).sum(-1) + mb
    return np.clip(q * scale, 0, None).astype(np.float32)


# --------------------------------------------------------------------------- #
# Edges & discontinuities
# --------------------------------------------------------------------------- #
def rgb_edges(rgb: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, lo, hi) > 0


def depth_discontinuities(depth: np.ndarray, rel_thresh: float = 0.05) -> np.ndarray:
    """Pixels where depth jumps by more than rel_thresh (relative) to a neighbour.
    Relative (log-space) so a 10 cm step matters more at 1 m than at 8 m."""
    d = np.log(np.maximum(depth, 1e-3)).astype(np.float32)
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)) / 4
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 4
    return np.maximum(gx, gy) > rel_thresh


def band(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def snap_discontinuities(depth: np.ndarray, region: np.ndarray, k: int = 5) -> np.ndarray:
    """Boundary-aware refinement: inside discontinuity bands, upsampling leaves
    'mixed' depths that belong to neither object (flying pixels). Snap each such
    pixel to whichever of the local foreground (min) / background (max) it is
    closer to. Outside the band nothing changes."""
    kernel = np.ones((k, k), np.uint8)
    lo = cv2.erode(depth, kernel)
    hi = cv2.dilate(depth, kernel)
    snapped = np.where(np.abs(depth - lo) < np.abs(hi - depth), lo, hi)
    out = depth.copy()
    out[region] = snapped[region]
    return out


# --------------------------------------------------------------------------- #
# Full classical pipeline
# --------------------------------------------------------------------------- #
def classical_pipeline(lr: np.ndarray, rgb_hr: np.ndarray, params: dict | None = None,
                       stages: dict | None = None, viz: dict | None = None) -> np.ndarray:
    """Adaptive classical refinement. `stages` toggles individual components (used
    by the switchboard UI and the ablation study)."""
    from .adaptive import choose_params  # local import to avoid a cycle
    H, W = rgb_hr.shape[:2]
    st = dict(denoise=True, hole_fill=True, jbu=True, guided=True, snap=False)
    st.update(stages or {})
    p = params or choose_params(lr, rgb_hr)

    d = lr
    if st["denoise"]:  # range-weighted, so zero (missing) samples get ~0 weight
        d = denoise_depth(d, p["noise"])
    if st["hole_fill"]:
        d = fill_large_holes(d, p["jbu_radius"]) if st["jbu"] else fill_holes(d)
    if viz is not None:
        viz["lr_prepared"] = d

    if st["jbu"]:
        up = joint_bilateral_upsample(d, rgb_hr, radius=p["jbu_radius"],
                                      sigma_s=p["jbu_sigma_s"], sigma_r=p["jbu_sigma_r"])
    else:
        up = interpolate(fill_holes(d) if not st["hole_fill"] else d, (H, W), "bicubic")
    if viz is not None:
        viz["upsampled"] = up

    if st["guided"] and p.get("use_guided", True):
        up = guided_filter(rgb_hr, up, r=p["gf_radius"], eps=p["gf_eps"])
    if st["snap"]:
        disc = depth_discontinuities(up, p["disc_thresh"])
        region = band(disc, p["band"])
        up = snap_discontinuities(up, region, k=p["snap_k"])
        if viz is not None:
            viz["discontinuities"] = disc
    return up.astype(np.float32)
