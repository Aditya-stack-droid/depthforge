# Report guide: DepthForge (Team Tiger FC, P4)

Everything for the report is in this folder. Suggested structure, with where to get each part:

| # | Report section | Source in this folder |
|---|---|---|
| 1 | Title, team, links | Team Tiger FC: Aditya Hubbly (1RUA24CSE0021), Samay Chetan (1RUA24CSE0401), Shaun (1RUA24CSE0427), Farhan (1RUA24CSE0490). Live: https://depthforge-731810864279.us-central1.run.app · Code: https://github.com/Aditya-stack-droid/depthforge |
| 2 | Abstract | Summary below |
| 3 | Problem statement (P4) | Hackathon PDF, P4 section; `docs/DESIGN.md` §1 |
| 4 | Dataset and degradation model | `docs/DESIGN.md` §2 |
| 5 | Methodology: classical, deep learning, hybrid, adaptive parameters | `docs/DESIGN.md` §3 |
| 6 | System architecture | `docs/architecture.png`, `docs/ARCHITECTURE.md` |
| 7 | Experimental results (RMSE, MAE, PSNR, SSIM, edge accuracy, δ accuracy, latency, memory at 2×/4×/8×) | `results/eval_P4_TigerFC.md` |
| 8 | Ablation study | `results/eval_P4_TigerFC.md` §3, `docs/FAILURE_ANALYSIS.md` |
| 9 | Failure case analysis (with images) | `docs/FAILURE_ANALYSIS.md`, `docs/screenshots/fail*.png` |
| 10 | Model optimisation (FP16/BF16/pruning, ONNX) | `docs/DESIGN.md` §5–6, `results/optimization.md` |
| 11 | Deployment and UI (screenshots) | `docs/DEPLOY.md`, `docs/screenshots/app_desktop.png`, `app_mobile.png` |
| 12 | Limitations, ethics, future work | `docs/DESIGN.md` §8, `docs/FAILURE_ANALYSIS.md` "Lessons" |
| 13 | Conclusion | Summary below |
| 14 | References | List below |

## Abstract / conclusion summary
Low-cost depth sensors produce low-resolution, noisy and incomplete depth maps, while the RGB camera
on the same device is sharp. DepthForge reconstructs high-resolution depth at 2×, 4× and 8× using
the RGB image as a guide, while avoiding texture-copying artifacts. It combines (1) an adaptive
classical pipeline (bilateral denoising, hole filling, joint bilateral upsampling, guided filtering),
(2) DepthForge-Net, a 0.5 M-parameter two-stream residual CNN with gated RGB fusion, and (3) an adaptive
hybrid that selects and combines outputs based on measured input statistics. On 100 unseen NYU Depth V2
test images, the hybrid reduces RMSE against bicubic interpolation by 37 %, 34 % and 31 % at 2×, 4× and
8× (0.0249 / 0.0372 / 0.0586 m on clean input). It keeps 99.6–99.98 % of pixels within 25 % of the true
depth, and 95–99.5 % within 5 %. The system runs on a CPU in a few seconds and is deployed as a public
web app on Google Cloud Run.

## Key numbers (NYU test, hybrid)
| | 2× | 4× | 8× |
|---|---|---|---|
| RMSE clean (m), bicubic → ours | 0.0395 → 0.0249 | 0.0561 → 0.0372 | 0.0855 → 0.0586 |
| RMSE hard (m), bicubic → ours | 0.0956 → 0.0753 | 0.1098 → 0.0847 | 0.1334 → 0.0984 |
| δ<1.25 clean | 99.98 % | 99.95 % | 99.87 % |
| δ<1.05 clean / hard | 99.50 / 97.52 % | 99.09 / 97.05 % | 98.02 / 95.30 % |
| PSNR / SSIM clean | 47.9 dB / 0.995 | 44.5 dB / 0.990 | 40.6 dB / 0.977 |

## References
1. Silberman, Hoiem, Kohli, Fergus. *Indoor Segmentation and Support Inference from RGBD Images* (NYU Depth V2), ECCV 2012.
2. Kopf, Cohen, Lischinski, Uyttendaele. *Joint Bilateral Upsampling*, ACM TOG (SIGGRAPH) 2007.
3. He, Sun, Tang. *Guided Image Filtering*, ECCV 2010 / TPAMI 2013.
4. Tomasi, Manduchi. *Bilateral Filtering for Gray and Color Images*, ICCV 1998.
5. Ronneberger, Fischer, Brox. *U-Net*, MICCAI 2015.
6. Wu, He. *Group Normalization*, ECCV 2018.
7. Hui, Loy, Tang. *Depth Map Super-Resolution by Deep Multi-Scale Guidance* (MSG-Net), ECCV 2016.
8. Wang, Bovik, Sheikh, Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity* (SSIM), IEEE TIP 2004.

## Honesty notes for the report
* Numbers are on 100 test images evenly sampled from the 449-image NYU test split, never used for training or tuning.
* Our degradation (sensor-style decimation + noise + holes) differs from most papers (bicubic downsampling), so absolute numbers are not directly comparable to published results.
* Training was about 100 min on CPU. Published models are much larger and GPU-trained, and reach lower error.
