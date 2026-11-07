# Design document — DepthForge (P4: Edge-Preserving Depth Map Upsampling and Refinement)

## 1. Problem & goals
Turn sparse, noisy or low-resolution depth (2×/4×/8× below the RGB resolution) into dense HR
depth that keeps **object boundaries and depth discontinuities sharp**, using the RGB image as
a guide but **without copying RGB texture into the geometry**.

Success criteria: lower RMSE/MAE and Edge-RMSE than bicubic at every factor; higher boundary
F1; < 10 s per request on a free CPU host; robust to holes, noise and sparsity.

## 2. Data
| Use | Data | Notes |
|---|---|---|
| Train / val | NYU Depth V2 labeled, images 0–899 / 900–999 | random 256² crops, flips |
| Test (unseen) | NYU Depth V2 labeled, images 1000–1448 | standard guided-DSR split |
| Smoke tests / demo | procedural synthetic RGB-D (`data.synth_scene`) | piecewise-planar depth with *decorrelated* RGB texture: stripes on flat walls (texture-copy traps), objects coloured like the background (weak-edge traps) |
| Optional | Middlebury / KITTI via `--data folder` | `xxx_rgb.png` + `xxx_depth.png/.npy` pairs |

**Degradation model** (`degrade.py`): centre-sample decimation (no averaging across edges, like a
real sensor), depth-proportional Gaussian noise, blob-shaped dropouts, random sparsity. Five
named conditions: `clean, noisy, holes, sparse, hard` (= all combined).

## 3. Method
### 3.1 Input analysis → adaptive parameters (`adaptive.py`)
Nothing is a fixed constant. Every input is measured first:
* **noise** — robust MAD of the Laplacian / median depth
* **holes** — invalid-pixel ratio
* **texture risk** — fraction of RGB Canny edges that have *no* depth discontinuity nearby

These drive: JBU window and range σ (higher texture risk means trusting colour less), guided-filter
radius ∝ scale and ε ∝ noise·texture, discontinuity threshold ∝ noise, snapping band ∝ scale,
and the CNN-vs-classical blend weight. The chosen values are shown in the UI and the CLI output.

### 3.2 Classical pipeline (`classical.py`)
1. **Range-adaptive bilateral denoise** on LR depth. σ_r = 3 × estimated noise, so steps larger than
   the noise survive. Missing (0) samples get ~zero weight automatically.
2. **Large-hole filling** by normalized convolution. Only holes that the JBU window cannot reach are
   filled; small holes are left for JBU, which re-estimates them with RGB guidance (the ablation
   showed that blind pre-filling hurt).
3. **Joint Bilateral Upsampling** (Kopf 2007): spatial Gaussian in LR coordinates × colour Gaussian
   on the HR guide; invalid samples are skipped.
4. **Colour guided filter** (He 2010): local linear model, edge-aware smoothing.
5. **Depth-discontinuity detection** in log-depth, then optional **min/max boundary snapping**.
   Pixels in the discontinuity band take the closer of the local foreground/background depth,
   which removes "flying pixels". Snapping helped on piecewise-planar synthetic scenes but
   **hurt on real NYU depth** (slanted and curved surfaces get false "edges"), so it is now
   *opt-in* (UI switch / `--on snap`) and off by default.

Tuned on the 100 NYU **validation** images (never the test split): JBU spatial σ = 0.7 LR px on
clean input (1.0 when noisy). Denoise and the guided filter only switch on above a relative-noise
estimate of 0.008. Clean NYU maps estimate 0.002–0.007, and 1–2 % sensor noise estimates 0.017–0.022.

Baselines: nearest, bilinear, bicubic (`classical.interpolate`).

### 3.3 Deep model — DepthForge-Net (`model.py`, 0.50 M params)
* Input: coarse depth already at HR size, plus HR RGB. Upsampling happens *before* the net, so one
  model serves 2×/4×/8×.
* Two-stream encoder. RGB features enter through a **learned sigmoid gate computed from depth+RGB
  features**, so the net can suppress RGB edges with no geometric counterpart (anti texture-copy).
* U-Net decoder predicts a **residual**. The output layer is zero-initialised, so training starts
  from the input and the classical prior is never lost.
* Loss: L1 + 0.5·gradient-L1 (rewards sharp, correctly placed edges).
* Training mix: random scale ∈ {2,4,8}, random condition, and the coarse input is 50 % classical
  output and 50 % bicubic. One checkpoint therefore serves both the **DL-only** and the **hybrid** modes.
* GroupNorm (not BatchNorm), because small batches gave a train/eval mismatch.
* Training: warm-started from the synthetic checkpoint, then fine-tuned on NYU train
  (900 images, random 128² crops). The best checkpoint is picked by val L1 on the 100 NYU val images.
* **Test-time augmentation** (flip self-ensemble) is implemented (`DEPTHFORGE_TTA=1`) but off by
  default: on NYU val it slightly hurt (4× clean 0.0302 → 0.0337 m), so reported numbers do not use it.

### 3.4 Hybrid: adaptive selection and combination (`pipeline.py`)
We first prototyped a per-pixel confidence blend: CNN near edges, classical on textured flat areas.
It won on synthetic scenes but **not on real NYU**. On NYU validation the CNN applied to the plain
bicubic prior was the best single output, so the hybrid was redesigned around what the
validation data showed:

1. **Always:** CNN on the bicubic prior (best single estimator on NYU val).
2. **Dual-path fusion, when ≥ 20 % of LR samples are missing** (sparse / hard sensors): the CNN also
   runs on the *classical* JBU prior, and the two predictions are averaged. The priors fail
   differently inside holes, so averaging cancels part of the error (val, 4× sparse: 0.0486 → 0.0460 m).
3. **Edge-aware refinement, when the factor is ≥ 8× or there are holes:** a small colour-guided
   filter (r = 2, ε = 1e-4) cleans residual ringing (val, 8× hard: 0.0831 → 0.0822 m).
4. Boundary snapping stays opt-in.

The decision (`dual_path`, `refine`) is computed from the measured input statistics and shown in
the UI, the CLI and the agents' messages. This is the brief's "adaptive mechanism to select or combine
enhancement outputs".

**Tiled inference.** The net is trained on 128² crops. GroupNorm statistics depend on the spatial
extent, so whole-image inference made the CNN *worse* than its input (val 4× clean
0.0445 → 0.0467 m). Running on overlapping 128² tiles blended with a Hann window, the same size as
training, fixed it (→ 0.0307 m). It also keeps memory flat for any image size.

### 3.5 Comparison required by the brief
| Family | Methods |
|---|---|
| Interpolation | nearest, bilinear, bicubic |
| Classical refinement | adaptive denoise + hole fill + JBU + guided filter + snapping |
| Deep learning only | DepthForge-Net on bicubic input |
| Hybrid | CNN(bicubic) + adaptive dual-path fusion with CNN(classical) + edge-aware refinement |

## 4. Evaluation
Metrics (`metrics.py`): RMSE, MAE, PSNR, SSIM, **Edge-RMSE** (RMSE in a 2-px band around GT
discontinuities), **Boundary-F1** (2-px tolerance), latency (ms), peak memory (tracemalloc), all
at 2×/4×/8× and per degradation condition. The **ablation** removes one hybrid stage at a time.
**Failure mining** picks the cases where hybrid loses most to bicubic. Output goes to
`results/eval_P4_<TEAM>.md`.

## 5. Optimisation (`scripts/quantize.py`)
Measured on NYU val (4×, `hard`, tiled CPU inference):

| Variant | Size | CNN latency | RMSE |
|---|---|---|---|
| FP32 baseline | 1.93 MB | 1409 ms | 0.0704 m |
| FP16 weight storage | 0.97 MB | ≈ same | 0.0704 m |
| **BF16 autocast inference** | 0.97 MB | **556 ms (2.5× faster)** | 0.0707 m (+0.4 %) |
| 30 % L1 pruning (no fine-tune) | 70 % non-zero | ≈ same | 0.0854 m (+21 %) |

BF16 helps only on CPUs with native bfloat16 support (e.g. cloud Xeons with AVX512-BF16/AMX); on ordinary laptops it is much slower, so it is off by default (`DEPTHFORGE_BF16=1` enables it). Pruning without fine-tuning hurts,
and unstructured sparsity gives no CPU speed-up without sparse kernels.

## 6. Deployment & UX
* **Serving runtime:** the CNN is exported to ONNX (`scripts/export_onnx.py`, max abs diff vs PyTorch 2e-7). The
  deploy image installs only ONNX Runtime, which cuts the image by about 800 MB and peak RAM to about 450 MB, and is about 20 % faster than
  PyTorch on CPU. Tiles run in mini-batches of 4 to bound memory.
* `app.py` (Gradio): sample loader, upload of RGB plus depth (16-bit mm PNG, 8-bit, `.npy`), a "ground
  truth → simulate sensor" mode with metrics, and a "real LR depth" mode.
* Pipeline **switches** for every stage, every intermediate visualisation, the chosen parameters,
  a 16-bit PNG download, and a "compare all methods" table.
* Input validation: missing files, unreadable or empty depth, NaNs, aspect mismatch (warning), and
  over-size outputs. Large inputs are downscaled to ≤ 960 px to stay under 10 s on CPU.
* `Dockerfile` (CPU torch, non-root, healthcheck) → Hugging Face Spaces / Render / local
  `docker compose up`.

## 7. Agentic workflow (Switch)
`switch/depthforge-lab.template.yaml` creates a Switch room with four agents: lead, classical,
neural and judge. They run the JSON CLI and post results in Discord/Telegram. A human must reply
"approve" before training, commits or pushes. Any claimed improvement must be re-checked by the
judge on unseen seeds. This is how the team ran the ablations and the parameter sweeps.

## 8. Limitations & ethics
* Trained on CPU for ~100 min (≈9 k steps of 128² crops), far less than published guided-DSR models
  (hours on GPUs). Longer GPU training (`notebooks/train_nyu_colab.ipynb`) should improve 8×.
* NYU covers indoor scenes at 0.5–10 m. Outdoor LiDAR (KITTI) sparsity patterns are different and
  untested.
* Strong RGB edges that coincide with tiny depth steps can still be over-sharpened by snapping.
* Depth maps of homes can reveal private layouts. The app keeps no uploads (temp files only)
  and logs nothing.
