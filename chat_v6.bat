@echo off
chcp 65001 >nul
title Emotion Agent v6 - Chat
cd /d d:\aiqg\emotion_agent
set HF_HOME=d:\aiqg\hf_cache
set PIP_CACHE_DIR=d:\aiqg\pip_cache
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONPATH=d:\aiqg\emotion_agent
rem silence HF Hub anon warning + model loading progress bars
set HF_HUB_VERBOSITY=error
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
d:\aiqg\venv\Scripts\python.exe scripts\chat_cli_v6.py
echo.
echo [session ended] press any key to close...
pause >nul
