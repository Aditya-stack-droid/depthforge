1RUA24CSE0021,1RUA24CSE0401,1RUA24CSE0427,1RUA24CSE0490

# Evaluation — P4 Edge-Preserving Depth Map Upsampling — Team TigerFC

Dataset: **nyu** (100 unseen test images, evenly sampled from the NYU v2 standard test split 1000-1448, never used for training or tuning). Degradations: clean, hard (see `depthforge/degrade.py`). Latency: 2-core CPU, per image, tiled 128² CNN inference in FP32, no test-time augmentation.

## 1. Depth accuracy, edge preservation, latency & memory (per scale)

### 2× upsampling — condition `clean`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.0478 | 0.0099 | 42.40 | 0.986 | 0.1535 | 0.988 | 99.89 | 99.03 | 6.8 | 6.2 |
| bilinear (Interpolation) | 0.0387 | 0.0101 | 44.25 | 0.991 | 0.1245 | 0.961 | 99.95 | 98.99 | 6.1 | 6.2 |
| bicubic (Interpolation) | 0.0395 | 0.0109 | 44.08 | 0.991 | 0.1270 | 0.983 | 99.95 | 98.95 | 6.0 | 6.2 |
| classical (Classical refinement) | 0.0366 | 0.0102 | 44.63 | 0.991 | 0.1175 | 0.957 | 99.96 | 99.08 | 178.6 | 32.8 |
| dl (Deep learning) | 0.0249 | 0.0076 | 47.93 | 0.995 | 0.0787 | 0.977 | 99.98 | 99.50 | 1425.5 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0249 | 0.0076 | 47.93 | 0.995 | 0.0787 | 0.977 | 99.98 | 99.50 | 1404.2 | 16.5 |

### 4× upsampling — condition `clean`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.0749 | 0.0186 | 38.50 | 0.966 | 0.2407 | 0.946 | 99.73 | 98.09 | 5.8 | 6.2 |
| bilinear (Interpolation) | 0.0555 | 0.0158 | 41.15 | 0.982 | 0.1790 | 0.869 | 99.89 | 98.12 | 4.5 | 6.2 |
| bicubic (Interpolation) | 0.0561 | 0.0170 | 41.07 | 0.980 | 0.1783 | 0.890 | 99.88 | 98.08 | 4.6 | 6.2 |
| classical (Classical refinement) | 0.0515 | 0.0161 | 41.68 | 0.982 | 0.1628 | 0.887 | 99.90 | 98.33 | 180.6 | 33.4 |
| dl (Deep learning) | 0.0372 | 0.0116 | 44.51 | 0.990 | 0.1173 | 0.956 | 99.95 | 99.09 | 1410.0 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0372 | 0.0116 | 44.51 | 0.990 | 0.1173 | 0.956 | 99.95 | 99.09 | 1403.2 | 16.5 |

### 8× upsampling — condition `clean`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.1154 | 0.0338 | 34.73 | 0.925 | 0.3692 | 0.684 | 99.36 | 96.53 | 6.4 | 6.1 |
| bilinear (Interpolation) | 0.0842 | 0.0277 | 37.52 | 0.963 | 0.2632 | 0.625 | 99.70 | 96.18 | 4.6 | 6.1 |
| bicubic (Interpolation) | 0.0855 | 0.0304 | 37.40 | 0.958 | 0.2612 | 0.666 | 99.69 | 95.93 | 4.8 | 6.1 |
| classical (Classical refinement) | 0.0848 | 0.0313 | 37.60 | 0.961 | 0.2477 | 0.761 | 99.70 | 96.04 | 386.2 | 34.8 |
| dl (Deep learning) | 0.0592 | 0.0207 | 40.47 | 0.977 | 0.1856 | 0.888 | 99.86 | 98.07 | 1411.2 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0586 | 0.0207 | 40.58 | 0.977 | 0.1832 | 0.882 | 99.87 | 98.02 | 1488.8 | 58.5 |

### 2× upsampling — condition `hard`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.1022 | 0.0524 | 35.36 | 0.853 | 0.2248 | 0.676 | 99.63 | 95.95 | 15.0 | 6.2 |
| bilinear (Interpolation) | 0.0895 | 0.0417 | 36.71 | 0.921 | 0.2038 | 0.897 | 99.68 | 97.01 | 14.1 | 6.2 |
| bicubic (Interpolation) | 0.0956 | 0.0488 | 36.00 | 0.881 | 0.2081 | 0.798 | 99.67 | 96.65 | 14.0 | 6.2 |
| classical (Classical refinement) | 0.0804 | 0.0295 | 37.88 | 0.973 | 0.1938 | 0.854 | 99.70 | 96.94 | 267.0 | 58.6 |
| dl (Deep learning) | 0.0743 | 0.0256 | 38.74 | 0.979 | 0.1697 | 0.900 | 99.72 | 97.82 | 1441.0 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0753 | 0.0258 | 38.57 | 0.978 | 0.1763 | 0.880 | 99.72 | 97.52 | 3139.8 | 62.8 |

### 4× upsampling — condition `hard`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.1216 | 0.0595 | 33.95 | 0.859 | 0.2948 | 0.597 | 99.43 | 94.52 | 7.0 | 6.2 |
| bilinear (Interpolation) | 0.1044 | 0.0483 | 35.42 | 0.927 | 0.2544 | 0.775 | 99.57 | 95.74 | 6.1 | 6.2 |
| bicubic (Interpolation) | 0.1098 | 0.0544 | 34.88 | 0.899 | 0.2583 | 0.801 | 99.55 | 95.39 | 6.2 | 6.2 |
| classical (Classical refinement) | 0.0934 | 0.0377 | 36.52 | 0.964 | 0.2316 | 0.786 | 99.62 | 95.83 | 262.9 | 58.4 |
| dl (Deep learning) | 0.0856 | 0.0322 | 37.49 | 0.972 | 0.2056 | 0.860 | 99.67 | 97.28 | 1409.6 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0847 | 0.0318 | 37.55 | 0.973 | 0.2033 | 0.844 | 99.68 | 97.05 | 3092.9 | 62.8 |

### 8× upsampling — condition `hard`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| nearest (Interpolation) | 0.1530 | 0.0733 | 32.01 | 0.858 | 0.3956 | 0.451 | 99.02 | 91.65 | 6.7 | 6.1 |
| bilinear (Interpolation) | 0.1277 | 0.0604 | 33.66 | 0.932 | 0.3274 | 0.516 | 99.32 | 93.07 | 5.0 | 6.1 |
| bicubic (Interpolation) | 0.1334 | 0.0661 | 33.21 | 0.916 | 0.3346 | 0.566 | 99.30 | 92.59 | 5.1 | 6.1 |
| classical (Classical refinement) | 0.1189 | 0.0548 | 34.42 | 0.947 | 0.2976 | 0.682 | 99.44 | 92.47 | 461.0 | 57.5 |
| dl (Deep learning) | 0.1023 | 0.0440 | 35.69 | 0.961 | 0.2640 | 0.770 | 99.57 | 95.71 | 1427.0 | 16.5 |
| hybrid (Hybrid (ours)) | 0.0984 | 0.0435 | 36.04 | 0.963 | 0.2539 | 0.764 | 99.63 | 95.30 | 3313.4 | 61.9 |

## 2. Performance across upsampling factors (RMSE, condition `clean`)

| Method | 2× | 4× | 8× |
|---|---|---|---|
| nearest | 0.0478 | 0.0749 | 0.1154 |
| bilinear | 0.0387 | 0.0555 | 0.0842 |
| bicubic | 0.0395 | 0.0561 | 0.0855 |
| classical | 0.0366 | 0.0515 | 0.0848 |
| dl | 0.0249 | 0.0372 | 0.0592 |
| hybrid | 0.0249 | 0.0372 | 0.0586 |

## 3. Ablation — hybrid pipeline, 4×, condition `hard`

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge-RMSE (m) ↓ | Boundary-F1 ↑ | δ<1.25 acc (%) ↑ | δ<1.05 acc (%) ↑ | Latency (ms) ↓ | Peak mem (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Full hybrid | 0.0577 | 0.0228 | 38.18 | 0.974 | 0.1553 | 0.827 | 99.84 | 97.74 | 3082.8 | 62.8 |
| − denoise | 0.0574 | 0.0228 | 38.23 | 0.974 | 0.1546 | 0.837 | 99.84 | 97.78 | 3010.1 | 62.8 |
| − hole_fill | 0.0579 | 0.0229 | 38.14 | 0.974 | 0.1566 | 0.823 | 99.83 | 97.74 | 3136.5 | 62.8 |
| − jbu | 0.0581 | 0.0230 | 38.10 | 0.974 | 0.1564 | 0.827 | 99.83 | 97.73 | 2916.8 | 62.8 |
| − guided | 0.0577 | 0.0229 | 38.18 | 0.974 | 0.1546 | 0.828 | 99.84 | 97.75 | 3094.9 | 62.8 |
| − cnn | 0.0653 | 0.0275 | 37.10 | 0.965 | 0.1805 | 0.755 | 99.80 | 96.80 | 259.5 | 59.5 |
| − fusion | 0.0575 | 0.0231 | 38.21 | 0.974 | 0.1547 | 0.834 | 99.84 | 97.81 | 1520.3 | 59.4 |
| − refine | 0.0576 | 0.0226 | 38.20 | 0.975 | 0.1539 | 0.837 | 99.83 | 97.86 | 2993.5 | 60.6 |
| + snap (opt-in) | 0.0702 | 0.0246 | 36.62 | 0.967 | 0.2074 | 0.820 | 99.70 | 97.52 | 3133.6 | 62.8 |

## 4. Failure cases (largest hybrid losses vs the best other method on the same input)

Panels: RGB | LR input | hybrid | ground truth | |error|.

- `fail1_nyu_1389_hard_x8.png` — nyu_1389, hard, 8×, RMSE 0.2634 (+0.0258 vs best other method). Analysis: see docs/FAILURE_ANALYSIS.md
- `fail2_nyu_1371_hard_x8.png` — nyu_1371, hard, 8×, RMSE 0.1286 (+0.0111 vs best other method). Analysis: see docs/FAILURE_ANALYSIS.md
- `fail3_nyu_1272_hard_x2.png` — nyu_1272, hard, 2×, RMSE 0.0777 (+0.0097 vs best other method). Analysis: see docs/FAILURE_ANALYSIS.md
