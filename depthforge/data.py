"""Data: NYU Depth V2 (labeled .mat), Middlebury-style folders, a procedural
synthetic RGB-D generator (for tests / no-download demos) and file I/O."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
# Standard guided-DSR split (Hui et al. / DJF / FDSR...): first 1000 train, last 449 test
NYU_TRAIN = range(0, 1000)
NYU_TEST = range(1000, 1449)


def load_nyu(mat_path: str | Path, indices, crop: int = 6):
    """Yield (rgb uint8 HxWx3, depth float32 metres). The 6-px border is the
    standard crop of NYU's invalid frame edge."""
    import h5py
    with h5py.File(mat_path, "r") as f:
        imgs, deps = f["images"], f["depths"]
        for i in indices:
            rgb = np.transpose(imgs[i], (2, 1, 0))
            dep = np.transpose(deps[i], (1, 0)).astype(np.float32)
            if crop:
                rgb, dep = rgb[crop:-crop, crop:-crop], dep[crop:-crop, crop:-crop]
            yield np.ascontiguousarray(rgb), np.ascontiguousarray(dep)


def load_folder(folder: str | Path):
    """Folder with pairs  xxx_rgb.png + xxx_depth.png (16-bit mm) or .npy (metres)."""
    folder = Path(folder)
    for rp in sorted(folder.glob("*_rgb.png")):
        stem = rp.name[:-8]
        rgb = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        npy = folder / f"{stem}_depth.npy"
        if npy.exists():
            dep = np.load(npy).astype(np.float32)
        else:
            dep = read_depth(folder / f"{stem}_depth.png")
        yield stem, rgb, dep


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #
def read_depth(path) -> np.ndarray:
    """Read depth from .npy (metres) or 16-bit PNG (millimetres). 8-bit PNGs are
    treated as relative depth scaled to 0..10 m."""
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError(f"could not read depth image {path}")
    if d.ndim == 3:
        d = d[..., 0]
    if d.dtype == np.uint16:
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32) / 255.0 * 10.0


def write_depth_png(path, depth):
    cv2.imwrite(str(path), np.clip(depth * 1000, 0, 65535).astype(np.uint16))


# --------------------------------------------------------------------------- #
# Synthetic RGB-D scenes
# --------------------------------------------------------------------------- #
def synth_scene(h=240, w=320, seed=None):
    """Piecewise-planar indoor-like scene with *decorrelated* RGB texture:
    textures on flat surfaces (texture-copy traps) and objects whose colour
    matches the background (weak-RGB-edge traps)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # back wall / floor slanted plane
    depth = (4.0 + rng.uniform(-1, 1) * xx / w + rng.uniform(0.5, 2.0) * (1 - yy / h)).astype(np.float32)
    bg = rng.integers(60, 200, 3)
    rgb = np.ones((h, w, 3), np.float32) * bg
    # wall texture (no depth edge!)
    if rng.random() < 0.8:
        period = rng.integers(6, 20)
        stripes = ((xx // period + yy // period) % 2) if rng.random() < 0.5 else ((xx // period) % 2)
        rgb += (stripes[..., None] * rng.integers(20, 60))
    n = rng.integers(3, 7)
    for _ in range(n):
        mask = np.zeros((h, w), np.uint8)
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        if rng.random() < 0.5:
            ax, ay = rng.integers(w // 12, w // 4), rng.integers(h // 12, h // 4)
            cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        else:
            pts = rng.integers(0, [w, h], size=(rng.integers(3, 6), 2)).astype(np.int32)
            pts = ((pts - pts.mean(0)) * 0.4 + [cx, cy]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        m = mask > 0
        z0 = rng.uniform(1.0, 3.5)
        plane = z0 + rng.uniform(-0.4, 0.4) * (xx - cx) / w + rng.uniform(-0.4, 0.4) * (yy - cy) / h
        depth[m] = np.minimum(depth[m], plane[m])
        col = bg + rng.integers(-15, 15, 3) if rng.random() < 0.2 else rng.integers(0, 255, 3)
        rgb[m] = col
        if rng.random() < 0.4:  # printed texture on object
            rgb[m] += (((xx[m] + yy[m]) // rng.integers(3, 8)) % 2)[:, None] * 40
    rgb += rng.normal(0, 4, rgb.shape)
    # soft shading so colours vary inside surfaces
    rgb *= (0.75 + 0.25 * (1 - depth / depth.max()))[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8), depth.astype(np.float32)
