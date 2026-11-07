"""Evaluation metrics required by P4: RMSE, MAE, PSNR, SSIM, edge/boundary
accuracy, latency & memory (latency/memory are measured in pipeline.run)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from . import classical as C


def _valid(gt):
    return gt > 0


def rmse(pred, gt):
    v = _valid(gt)
    return float(np.sqrt(np.mean((pred[v] - gt[v]) ** 2)))


def mae(pred, gt):
    v = _valid(gt)
    return float(np.mean(np.abs(pred[v] - gt[v])))


def psnr(pred, gt):
    v = _valid(gt)
    mse = np.mean((pred[v] - gt[v]) ** 2)
    peak = gt[v].max()
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-12)))


def ssim(pred, gt):
    rng = float(gt.max() - gt[gt > 0].min()) or 1.0
    return float(structural_similarity(gt, pred.astype(gt.dtype), data_range=rng))


def boundary_masks(gt, width=2, thresh=0.05):
    disc = C.depth_discontinuities(gt, thresh)
    return disc, C.band(disc, width)


def edge_rmse(pred, gt, width=2):
    """RMSE restricted to a band around ground-truth depth discontinuities —
    the place where interpolation blurs and where P4 is judged."""
    _, b = boundary_masks(gt, width)
    b &= _valid(gt)
    if b.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[b] - gt[b]) ** 2)))


def boundary_f1(pred, gt, tol=2, thresh=0.05):
    """F1 between predicted and GT discontinuity maps with a `tol`-pixel tolerance."""
    pe = C.depth_discontinuities(np.maximum(pred, 1e-3), thresh)
    ge = C.depth_discontinuities(gt, thresh)
    if ge.sum() == 0 and pe.sum() == 0:
        return 1.0
    prec = (pe & C.band(ge, tol)).sum() / max(pe.sum(), 1)
    rec = (ge & C.band(pe, tol)).sum() / max(ge.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


def delta_acc(pred, gt, thr=1.25):
    """Standard depth 'accuracy': % of pixels with max(pred/gt, gt/pred) < thr.
    thr=1.25 is the usual delta1; 1.05 is a much stricter 5 % band."""
    v = _valid(gt)
    r = np.maximum(pred[v] / gt[v], gt[v] / np.maximum(pred[v], 1e-6))
    return float((r < thr).mean() * 100)


def all_metrics(pred, gt) -> dict:
    return dict(rmse=rmse(pred, gt), mae=mae(pred, gt), psnr=psnr(pred, gt),
                ssim=ssim(pred, gt), edge_rmse=edge_rmse(pred, gt),
                boundary_f1=boundary_f1(pred, gt),
                delta1=delta_acc(pred, gt, 1.25), delta105=delta_acc(pred, gt, 1.05))
