# CLAUDE.md: project context for Claude Code

## What this is
DepthForge: CV Hackathon 2026, problem P4 (edge-preserving depth map upsampling 2×/4×/8×).
Team Tiger FC: Aditya Hubbly 1RUA24CSE0021, Samay Chetan 1RUA24CSE0401, Shaun 1RUA24CSE0427, Farhan 1RUA24CSE0490.
- Live app: https://depthforge-731810864279.us-central1.run.app (Google Cloud Run, project `depthforge-514004`, region `us-central1`)
- Repo: https://github.com/Aditya-stack-droid/depthforge (public)

## Layout
- `depthforge/`: library
  - `classical.py`: interpolation, hole filling, denoise, JBU, guided filter, discontinuities, snapping
  - `adaptive.py`: input analysis → parameters (`NOISE_ON`, `dual_path`, `refine`)
  - `model.py`: DepthForge-Net (0.5 M params, GroupNorm, residual)
  - `pipeline.py`: `run(method, lr, rgb, stages)`. The CNN runs on 128² tiles; backend is torch or ONNX (`DEPTHFORGE_BACKEND`).
  - `router.py`: learned per-image router (ridge on input stats picks the hybrid path; `weights/router.json`, optional, falls back to the hand-set rules)
  - `viewer3d.py`: offline WebGL2 3D viewer embedded in the app
  - `metrics.py`, `degrade.py`, `data.py` (NYU loader, synthetic scenes), `viz.py`, `cli.py` (JSON CLI)
- `app.py`: FastAPI server: serves the website in `web/` (index.html, style.css, app.js, light theme, CSS variables in `:root`) and the JSON API (`/api/config`, `/api/run`, `/api/compare`, `/api/results`, `/api/thumb/<name>`). `depthforge/service.py` holds the request logic. `app_gradio.py` is the old Gradio UI, kept as a fallback.
- `scripts/`:
  - `train.py` (`--amp --ema`), `train_router.py`, `evaluate.py` (writes `results/eval_P4_<team>.md`)
  - `quantize.py`, `export_onnx.py`, `set_team.py`
- `weights/`: `depthforge_net.pt` (NYU-trained), `depthforge_net.onnx` (deploy), `depthforge_net_synth.pt`
- `docs/`: DESIGN, ARCHITECTURE (+png), FAILURE_ANALYSIS, DEPLOY, VIVA_PREP, REPORT_GUIDE, `screenshots/`
- `switch/`: Switch Console room template (optional, not tested in Console)

## Commands
```bash
pip install -r requirements.txt            # dev: torch + onnxruntime
python app.py                              # UI on :7860
python -m pytest -q tests                  # 23 tests
python -m depthforge.cli compare --scale 4 --condition hard
python scripts/train.py --data nyu --mat nyu_depth_v2_labeled.mat --minutes 60 --crop 128 --init weights/depthforge_net.pt
python scripts/train_router.py --data nyu --mat nyu_depth_v2_labeled.mat --n 100   # after retraining; fits on val 900-999, writes weights/router.json + results/router_report.md
python scripts/export_onnx.py              # after retraining; deploy uses the ONNX file
python scripts/evaluate.py --data nyu --mat nyu_depth_v2_labeled.mat --n 100 --team TigerFC --usns "1RUA24CSE0021,1RUA24CSE0401,1RUA24CSE0427,1RUA24CSE0490"
```
The NYU dataset (`nyu_depth_v2_labeled.mat`, 2.8 GB) is NOT in the repo. Download it from
https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

## Redeploy (Cloud Shell, bump the tag each time)
Cloud Build's default service account lacks IAM permissions, so build the image locally in Cloud Shell:
```bash
cd ~/depthforge && git pull && I=us-central1-docker.pkg.dev/depthforge-514004/depthforge/app:v3 && \
docker build -t $I . && docker push $I && gcloud run deploy depthforge --image $I --region us-central1 --quiet
```

## Key decisions (keep these)
- Report only NYU **test** numbers. Tune on val (900–999), never on test (1000–1448).
- The CNN must run tiled at the training crop size. Whole-image inference hurt because of GroupNorm statistics.
- Boundary snapping and flip TTA are off by default because both hurt on NYU val.
- Router: keep `weights/router.json` only if `results/router_report.md` shows it beats the rules in CV on val. Re-fit it after every retrain (it is specific to the checkpoint). Disable with `DEPTHFORGE_ROUTER_OFF=1`.
- BF16 is off by default: it's slow on CPUs without native bf16.
- The deploy image is ONNX-only (`requirements-deploy.txt`) and stays under 512 MB RAM.

## Results (NYU test, 100 imgs, hybrid, clean): RMSE 0.0249 / 0.0372 / 0.0586 m at 2/4/8× (−37/−34/−31 % vs bicubic)
