# DepthForge web app: lightweight CPU image (ONNX Runtime, no PyTorch).
# Runs on Google Cloud Run, Render, Railway, Fly.io, HF Spaces (Docker) or locally.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False PORT=7860 \
    DEPTHFORGE_BACKEND=onnx MALLOC_ARENA_MAX=2
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# non-root user
RUN useradd -m -u 1000 app
WORKDIR /home/app/depthforge

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY --chown=app:app . .
USER app
EXPOSE 7860
# Cloud Run / Render inject $PORT; app.py reads it
CMD ["python", "app.py"]
