# Demo script and viva prep: DepthForge (P4)

## 3-minute demo (on the live URL)
| Time | Who | Do | Say |
|---|---|---|---|
| 0:00–0:20 | Member 1 | Open the app; the `nyu_test_1` sample is preloaded | "DepthForge turns low-res, noisy, holey depth into sharp HR depth using the RGB image as a guide, without copying RGB texture into the geometry." |
| 0:20–0:50 | Member 1 | 4×, `hard`, hybrid → **Run** | "The sensor here has noise, holes and 30 % missing samples. Look at the adaptive parameters: it measured the noise and hole ratio and decided to use dual-path fusion and edge refinement." |
| 0:50–1:20 | Member 2 | Scroll the intermediate gallery | "Low-res input → JBU → CNN output → result → ground truth → error map. Errors sit on thin edges, the hardest case at 4×." |
| 1:20–1:50 | Member 3 | Untick **DepthForge-Net (CNN)** → Run | "Without the network RMSE goes up about 13 % and Boundary-F1 drops. The CNN is our biggest contributor." |
| 1:50–2:20 | Member 3 | Tick **Boundary snapping** → Run | "Snapping helped on synthetic data but hurt on real NYU, so we made it opt-in. We let validation data decide, not intuition." |
| 2:20–2:50 | Member 4 | **Compare all methods** (8×, clean) | "Nearest → bilinear → bicubic → classical → DL → hybrid. Hybrid cuts bicubic RMSE by 31 % at 8×." |
| 2:50–3:00 | Member 4 | Upload the judge's image, if given | Use **Low-res depth (real input)** if they give an LR depth map. |

If the judge gives an RGB image *without* depth, say so honestly: P4 is depth *upsampling*, so it needs a measured LR depth map. We can show it on a ground-truth depth map with a simulated sensor.

## Numbers to remember (NYU test, 100 unseen images)
| | 2× | 4× | 8× |
|---|---|---|---|
| Bicubic RMSE (clean) | 0.0395 m | 0.0561 m | 0.0855 m |
| **Hybrid RMSE (clean)** | **0.0249 m** | **0.0372 m** | **0.0586 m** |
| Improvement | −37 % | −34 % | −31 % |
| δ<1.25 (clean) | 99.98 % | 99.95 % | 99.87 % |
| δ<1.05 (clean / hard) | 99.5 / 97.5 % | 99.1 / 97.1 % | 98.0 / 95.3 % |

* Model: 0.50 M parameters, 1.9 MB (0.97 MB in FP16).
* Data: NYU v2 labeled, split 900 train / 100 val / 449 test (images 1000–1448).
* Training: about 100 min on CPU, about 9 k steps of 128² crops, warm-started from synthetic.
* Latency: about 1.4 s per CNN pass on 2 cloud CPU cores (on a laptop it can be about 5× slower).

## Who explains what
**Member 1: problem, data and metrics**
* Degradation model: centre-sample decimation, depth-proportional noise, blob holes and random sparsity (`degrade.py`). There are five conditions.
* Metrics: RMSE/MAE (metres), PSNR, SSIM, **Edge-RMSE** (error within 2 px of GT discontinuities), **Boundary-F1**, δ accuracies, latency and memory.
* Why the NYU split: it's the standard guided depth super-resolution protocol. The test set was never used for tuning.

**Member 2: classical pipeline** (`classical.py`, `adaptive.py`)
* Input analysis: noise (MAD of the Laplacian), hole ratio, texture-copy risk (RGB edges with no nearby depth edge).
* Range-adaptive bilateral denoise, with σ_r = 3 × the estimated noise.
* Joint Bilateral Upsampling: spatial Gaussian on the LR grid × colour Gaussian on the HR guide, skipping invalid samples.
* Guided filter (He et al.): a local linear model, switched on only when the input is noisy.
* Why fill only *large* holes: JBU re-estimates small holes better with RGB guidance.

**Member 3: the network** (`model.py`, `pipeline.run_cnn`)
* Two streams (depth and RGB), with RGB injected through a learned sigmoid **gate**. The net can suppress RGB edges that have no depth edge (anti texture-copy).
* Predicts a **residual** on top of the coarse input, with a zero-initialised output layer. One model covers 2×, 4× and 8×.
* Loss: L1 + 0.5 × gradient L1. GroupNorm instead of BatchNorm (small batches).
* **Tiled inference:** GroupNorm statistics depend on image size, and whole-image inference was worse than the input. 128² tiles matching the training crops fixed it. This was our biggest single improvement.

**Member 4: hybrid, ablation, failures and optimisation**
* Hybrid: CNN on the bicubic prior. When ≥ 20 % of samples are missing it adds CNN on the classical prior and averages them (**dual-path**). At 8× or with holes it adds a guided refinement. These decisions come from the measured input statistics.
* Ablation: CNN −13 %, snapping +22 % worse, other classical stages ±0.5 % inside the hybrid.
* Failure cases:
  1. a large hole across a depth edge
  2. thin chair legs at 8×
  3. glass, where NYU's own GT is in-painted
* Optimisation: BF16 is 2.5× faster on CPUs with native bfloat16 (+0.4 % RMSE). FP16 storage halves the size. 30 % pruning without fine-tuning costs +21 % RMSE.

## Likely questions, with short answers
1. **Why not just use the CNN?** On clean input the hybrid *is* the CNN, because the adaptive rule chooses it. With many missing samples the dual-path average is better (4×/8× hard), because the two priors fail differently inside holes.
2. **What is texture copying?** RGB edges (stripes, posters, prints) wrongly appearing as depth edges. We fight it with the texture-risk measure, the learned RGB gate, and fusion choices.
3. **Why RMSE in metres, not %?** That's the standard for depth super-resolution. We also report δ accuracies as percentages.
4. **How do you know it generalises?** The test images were never used for training or tuning, and our numbers are on those.
5. **Why is the classical method barely better than bicubic on clean NYU?** NYU depth is smooth and the RGB is highly textured. JBU's colour weighting helps on noisy input (−15 %) but gains little on clean, smooth maps.
6. **Why GroupNorm?** Batch size 8 on CPU gave a train/eval mismatch with BatchNorm.
7. **What would you do with more time?** Train on GPU for hours, use a boundary-weighted loss, give the CNN the validity mask as input, and add per-scale models.
8. **Biggest mistake?** Trusting synthetic results. Snapping and per-pixel fusion looked great there and failed on NYU, so we re-validated everything on NYU val.
9. **Latency?** About 2–3 s per image on a cloud CPU, under the 10 s target. Hybrid is slower when dual-path runs (two CNN passes).
10. **Ethics?** Indoor depth maps reveal home layouts. The app stores nothing, and uploads live only in temp files.

## Switch agents (optional demo, 30 s)
In the `depthforge-lab` room: `@forge-judge run the ablation at 4x hard`. The judge runs `python -m depthforge.cli ablate` and posts which stage matters most. Pitch: "our AI team ran experiments, and the judge agent re-verified every claimed improvement; humans approve training and pushes."
