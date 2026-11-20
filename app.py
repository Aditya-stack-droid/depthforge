"""DepthForge web app: a FastAPI server that serves the website (web/) and a small JSON API.

  python app.py          # http://localhost:7860

The previous Gradio UI is kept in app_gradio.py.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from depthforge import data, service

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD = 25 * 2 ** 20

app = FastAPI(title="DepthForge", docs_url="/api/docs", redoc_url=None)


@app.exception_handler(service.RequestError)
async def _bad_request(_, e: service.RequestError):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def api_config():
    return service.config()


@app.get("/api/results")
def api_results():
    return service.results_summary()


@app.get("/api/thumb/{name}")
def api_thumb(name: str):
    return Response(service.thumbnail(name), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


async def _read_inputs(sample, rgb, depth):
    if sample:
        return service.load_sample(sample)
    if rgb is None or depth is None:
        raise service.RequestError("Choose a sample, or upload both an RGB image and a depth map.")
    rb, db = await rgb.read(), await depth.read()
    if len(rb) > MAX_UPLOAD or len(db) > MAX_UPLOAD:
        raise service.RequestError("Files must be smaller than 25 MB.")
    img = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise service.RequestError("Could not read the RGB image (use PNG or JPEG).")
    suffix = Path(depth.filename or "d.png").suffix.lower() or ".png"
    if suffix not in (".png", ".npy", ".tif", ".tiff"):
        raise service.RequestError("Depth must be a .png (16-bit mm or 8-bit), .tif or .npy (metres) file.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(db)
    try:
        d = data.read_depth(f.name)
    except Exception as e:
        raise service.RequestError(f"Could not read the depth file: {e}")
    finally:
        os.unlink(f.name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d


@app.post("/api/run")
async def api_run(sample: str = Form(None), rgb: UploadFile = File(None), depth: UploadFile = File(None),
                  kind: str = Form("gt"), scale: int = Form(4), condition: str = Form("hard"),
                  method: str = Form("hybrid"), off: str = Form(""), on: str = Form("")):
    img, d = await _read_inputs(sample, rgb, depth)
    split = lambda s: [x for x in s.split(",") if x]
    return await run_in_threadpool(service.run_request, img, d, kind == "gt", scale, condition, method,
                                   split(off), split(on))


@app.post("/api/compare")
async def api_compare(sample: str = Form(None), rgb: UploadFile = File(None), depth: UploadFile = File(None),
                      scale: int = Form(4), condition: str = Form("hard")):
    img, d = await _read_inputs(sample, rgb, depth)
    return await run_in_threadpool(service.compare_request, img, d, scale, condition)


@app.get("/")
def index():
    # cache-bust the static files by mtime so a redeploy is picked up immediately
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for f in ("style.css", "app.js"):
        html = html.replace(f"/static/{f}", f"/static/{f}?v={int((WEB / f).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=WEB), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), log_level="warning")
