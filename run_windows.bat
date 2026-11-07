@echo off
REM DepthForge one-click run on Windows (needs Python 3.10+ or Docker Desktop)
where docker >nul 2>nul
if %errorlevel%==0 (
  echo Starting with Docker...  open http://localhost:7860
  docker compose up
) else (
  echo Docker not found - using Python venv
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements.txt
  python app.py
)
