# Failure-case analysis (NYU Depth V2, unseen test images)

`scripts/evaluate.py` mines failures automatically. For every test input it finds where the hybrid
loses most to the *best other method* on that same input. The panels in `results/failures/` are:
RGB | LR input (black = missing) | hybrid | ground truth | |error| (bright = worse).

## F1: Large hole straddling a depth discontinuity (`fail1_nyu_1389_hard_x8.png`)
**What:** a big dropout covers the boundary between a cabinet (near) and the far wall seen through
a doorway. The reconstruction bleeds the near depth into the far region. That is the bright blob
in the error map.
**Why:** inside the hole there are *no* depth samples on either side of the edge. The CNN and JBU
must guess where the boundary is from RGB alone, and at 8× the nearest valid samples are 30+ HR
pixels away.
**Mitigation:** feed the validity mask as an extra CNN channel so the net knows when it is
extrapolating; use a monocular-depth prior inside large holes; flag hole regions as
low-confidence in the UI.

## F2: Thin structures at 8× (`fail2_nyu_1371_hard_x8.png`)
**What:** chair spindles, chair backs and a picture-frame edge are thickened or partly lost. The
error map lights up along every thin edge.
**Why:** at 8× a 4-px chair leg is smaller than one LR sample, and with about 30 % of samples
missing ("hard") many legs were never measured. RGB shows them, but the network is deliberately
cautious about turning RGB edges into depth edges (anti texture-copy). That caution trades
thin-structure recall for robustness.
**Mitigation:** a boundary-weighted loss (higher weight near GT discontinuities) and 8×-specific
training. Boundary snapping is *not* the fix: it hurt on NYU (see below).

## F3: Glass windows, foliage and unreliable ground truth (`fail3_nyu_1272_hard_x2.png`)
**What:** errors concentrate on the window (outdoor scene through glass) and on the potted plant.
**Why:** (a) The Kinect cannot measure through glass. NYU's "ground truth" there was in-painted
by the dataset's colourisation step, so it is itself noisy and partly made up. (b) Foliage has
depth changing almost every pixel, and the large dropout over the plant removes all evidence. At
2× classical JBU follows the remaining samples more literally and wins slightly.
**Mitigation:** exclude in-painted GT pixels (NYU raw vs filled) when evaluating; enable
dual-path fusion at lower hole ratios for 2×.

## What the ablation says (4×, `hard`, 30 NYU test images)
* **The CNN is the dominant component.** Removing it raises RMSE from 0.058 to 0.065 m (+13 %)
  and drops Boundary-F1 from 0.83 to 0.76.
* **Classical stages inside the hybrid contribute little on NYU** (each within ±0.5 %). Once the CNN
  runs, the classical branch matters only in dual-path mode for very sparse input. On their own,
  classical methods still beat bicubic by 8–15 % RMSE on degraded input.
* **Boundary snapping hurts on real data** (0.058 to 0.070 m, +22 %). It helped on our
  piecewise-planar synthetic scenes, but slanted floors and curved furniture create false
  "discontinuities" that snapping turns into steps. It is therefore opt-in and off by default.
* **Flip test-time augmentation hurt slightly** on NYU validation, so it is off by default.
* **Whole-image CNN inference was worse than the input.** Tiled 128² inference, matching the
  training crop and therefore the GroupNorm statistics, was our biggest single fix (val 4× clean
  RMSE 0.0467 → 0.0307 m).

## Lessons
1. Synthetic data misled two design decisions (snapping, per-pixel confidence fusion). Every
   decision was re-checked on NYU **validation** images (900–999). The test split (1000–1448) was
   used only for the final numbers.
2. Train/test mismatches, such as normalisation statistics or crop size, can hide a good model.
   Always measure the model inside the real inference path.
