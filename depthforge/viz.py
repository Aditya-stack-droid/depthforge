from __future__ import annotations

import cv2
import numpy as np


def colorize(depth: np.ndarray, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Depth -> RGB uint8 (near = warm). Missing pixels shown black."""
    d = depth.astype(np.float32)
    valid = d > 0
    if not valid.any():
        return np.zeros(d.shape + (3,), np.uint8)
    lo = np.percentile(d[valid], 1) if vmin is None else vmin
    hi = np.percentile(d[valid], 99) if vmax is None else vmax
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap(((1 - n) * 255).astype(np.uint8), cmap)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[~valid] = 0
    return img


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40)) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    out[mask] = color
    return out


def error_map(pred, gt, vmax=None):
    err = np.abs(pred - gt)
    err[gt <= 0] = 0
    vmax = vmax or max(np.percentile(err, 99), 1e-6)
    n = np.clip(err / vmax, 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def weight_map(w):
    img = cv2.applyColorMap((np.clip(w, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nn_upscale(img, hw):
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
