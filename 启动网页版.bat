@echo off
chcp 65001 >nul
title Emotion Agent - Web (backend + UI)
cd /d d:/aiqg/emotion_agent
set HF_HOME=d:/aiqg/hf_cache
set PIP_CACHE_DIR=d:/aiqg/pip_cache
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONPATH=d:/aiqg/emotion_agent
set HF_HUB_VERBOSITY=error
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
set EMOTION_OPEN_BROWSER=1
echo.
echo ============================================================
echo  Emotion Agent - backend + web UI (same origin)
echo  This window IS the backend. Close it to stop the service.
echo  Browser opens http://127.0.0.1:8000/ automatically.
echo ============================================================
echo.
rem [045] listen on 0.0.0.0 so phones in the same LAN can connect
d:/aiqg/venv/Scripts/python.exe -m uvicorn server_api:app --host 0.0.0.0 --port 8000
echo.
echo Backend stopped. Press any key to close.
pause >nul
