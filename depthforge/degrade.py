"""Degradation model: turns a clean high-resolution depth map into a realistic
low-resolution / sparse / noisy sensor measurement.

Convention used everywhere in DepthForge:
    depth  -> float32 array, metres, 0 == missing / invalid
    rgb    -> uint8 HxWx3 (RGB order)
"""
from __future__ import annotations

import cv2
import numpy as np


def make_lr(depth_hr: np.ndarray,
            scale: int,
            noise_std: float = 0.0,
            hole_ratio: float = 0.0,
            sparse_keep: float = 1.0,
            seed: int | None = None) -> np.ndarray:
    """Simulate a low-resolution depth sensor.

    scale       : 2, 4, 8 ... (HR size must be divisible; we crop otherwise)
    noise_std   : depth-proportional Gaussian noise (0.01 == 1% of depth)
    hole_ratio  : fraction of LR pixels dropped in blobs (sensor dropouts)
    sparse_keep : fraction of LR pixels kept at random (LiDAR-like sparsity)
    """
    rng = np.random.default_rng(seed)
    h, w = depth_hr.shape
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    d = depth_hr[:h2, :w2]
    # Nearest decimation (centre sample) — a real sensor samples, it does not average
    # across object boundaries (averaging would create fake "flying pixel" depths).
    off = scale // 2
    lr = d[off::scale, off::scale].astype(np.float32).copy()

    if noise_std > 0:
        valid = lr > 0
        lr[valid] += rng.normal(0, noise_std, valid.sum()).astype(np.float32) * lr[valid]
        lr = np.clip(lr, 0, None)

    if hole_ratio > 0:
        lh, lw = lr.shape
        mask = np.zeros((lh, lw), np.uint8)
        target = hole_ratio * lh * lw
        while mask.sum() < target:
            cy, cx = rng.integers(0, lh), rng.integers(0, lw)
            r = int(rng.integers(1, max(2, min(lh, lw) // 10)))
            cv2.circle(mask, (int(cx), int(cy)), r, 1, -1)
        lr[mask > 0] = 0

    if sparse_keep < 1.0:
        keep = rng.random(lr.shape) < sparse_keep
        lr[~keep] = 0
    return lr


def crop_to_scale(rgb: np.ndarray, depth: np.ndarray, scale: int):
    h, w = depth.shape
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    return rgb[:h2, :w2], depth[:h2, :w2]


CONDITIONS = {
    # name: (noise_std, hole_ratio, sparse_keep)
    "clean": (0.0, 0.0, 1.0),
    "noisy": (0.02, 0.0, 1.0),
    "holes": (0.0, 0.08, 1.0),
    "sparse": (0.0, 0.0, 0.5),
    "hard": (0.02, 0.08, 0.7),
}
