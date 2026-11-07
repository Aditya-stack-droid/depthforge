# AGENTS.md — how Switch agents work in this repo

You are an agent in the **depthforge-lab** Switch room. Humans talk to you from
Discord / Telegram / Slack. The code is a P4 hackathon project: RGB-guided depth map
upsampling (2×/4×/8×) with classical, deep-learning and hybrid pipelines.

## Map of the repo
| Path | What |
|---|---|
| `depthforge/classical.py` | interpolation, hole filling, denoise, JBU, guided filter, discontinuities, boundary snapping |
| `depthforge/adaptive.py` | measures each input (noise, holes, texture risk) → filter parameters |
| `depthforge/model.py` | DepthForge-Net (gated two-stream residual U-Net, ~0.5 M params) |
| `depthforge/pipeline.py` | `run(method, lr, rgb, stages)` — single entry point |
| `depthforge/cli.py` | JSON CLI — **use this, do not write ad-hoc scripts** |
| `scripts/train.py`, `scripts/evaluate.py`, `scripts/quantize.py` | training / submission eval / BF16-FP16-pruning study |
| `app.py` | Gradio switchboard (deployed URL) |

## Commands (all print one JSON object)
```bash
python -m depthforge.cli analyze --scale 4 --condition hard
python -m depthforge.cli run --method hybrid --scale 8 --condition noisy --on snap --png /tmp/x.png
python -m depthforge.cli compare --scale 4 --condition holes --seed 3
python -m depthforge.cli ablate --n 10 --scale 4 --condition hard
python scripts/evaluate.py --data nyu --mat nyu_depth_v2_labeled.mat --n 100 --team T01  # or --data synth
```
Methods: `nearest bilinear bicubic classical dl hybrid`.
Stages: `denoise hole_fill jbu guided cnn fusion` (on by default; disable with `--off`) and
`snap` (opt-in, enable with `--on snap`).
Conditions: `clean noisy holes sparse hard`.

## Rules
1. Post short messages: one-line verdict + small table. Attach PNGs when useful.
2. Training, commits, pushes and anything touching `weights/` need a human **approve**.
3. Improvement claims need ≥5 unseen seeds and must be re-checked by `forge-judge`.
4. Never commit secrets, datasets or `.switch/` folders.
