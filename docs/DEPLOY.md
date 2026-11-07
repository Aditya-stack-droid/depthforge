# Deploying DepthForge

## A. Google Cloud Run (free tier, real CPUs): recommended
The app image uses ONNX Runtime instead of PyTorch. It is small, needs < 512 MB RAM, and gives
results identical to PyTorch. Free tier: 180,000 vCPU-seconds and 360,000 GiB-seconds a month,
which is thousands of demo runs. Google asks for a card to verify the account, but usage inside
the free tier is not charged. Cloud Build or Artifact Registry storage may cost a few cents.

1. Put the code on GitHub first (section B).
2. Go to https://console.cloud.google.com, create a project (e.g. `depthforge`), and attach a billing account.
3. Open **Cloud Shell** (the `>_` icon, top right) and run:
   ```bash
   git clone https://github.com/<you>/depthforge.git && cd depthforge
   gcloud run deploy depthforge --source . --region us-central1 \
     --allow-unauthenticated --cpu 2 --memory 1Gi --max-instances 1 \
     --timeout 120 --cpu-boost
   ```
   Answer **Y** to enabling APIs and creating the Artifact Registry repo. The build takes about 5 minutes.
4. The command prints `Service URL: https://depthforge-xxxx.run.app`. That is your live URL.
5. Before judging, open the URL once. It scales to zero when idle, and the first request takes about
   10–20 s to start. `--max-instances 1` caps cost.

## A2. Hugging Face Space (needs PRO for Docker since 2026)
Docker and Gradio Spaces now require a paid plan; only static Spaces are free. With PRO, create a
Space with the **Docker** SDK and upload this folder. The YAML header in `README.md` configures it.

## A3. Render (free, no card, but slow)
Render free = 0.1 CPU / 512 MB, so expect 30–60 s per run and about 1 min to wake after 15 min idle.
New → Web Service → connect the GitHub repo → Runtime **Docker** → Instance type **Free**.

## B. GitHub repository
1. github.com, then **New repository** `depthforge`, **Public**, then create.
2. **uploading an existing file**: drag in the same files and folders, then commit.
3. GitHub Actions (`.github/workflows/tests.yml`) runs the unit tests on every push.
4. Paste the repo URL into the README and the submission form.

## C. Local (any OS)
* Docker: `docker compose up --build` (first build only; afterwards `docker compose up`), then open http://localhost:7860
* Windows one-click: `run_windows.bat`
* Python: `pip install -r requirements.txt && python app.py`

## D. Team ID and USNs
```
python scripts/set_team.py --team T07 --usns "USN1,USN2,USN3,USN4"
```
This renames `results/eval_P4_<team>.md` and writes the USNs on its first line (required by the rubric).

## Notes
* Do **not** upload `nyu_depth_v2_labeled.mat` (2.8 GB); `.gitignore` excludes it.
* The deploy image runs the CNN with ONNX Runtime (`weights/depthforge_net.onnx`, made by `scripts/export_onnx.py`).
  Training and evaluation use PyTorch (`requirements.txt`). Both give identical outputs (max diff 2e-7).
* `DEPTHFORGE_BF16=1` (PyTorch backend only) is faster only on CPUs with native bfloat16.
