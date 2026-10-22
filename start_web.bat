@echo off
chcp 65001 >nul
title Emotion Agent v6 - Web
cd /d d:\aiqg\emotion_agent
set HF_HOME=d:\aiqg\hf_cache
set PIP_CACHE_DIR=d:\aiqg\pip_cache
set HF_ENDPOINT=https://hf-mirror.com
rem [024] model cached locally: force offline load so a stale/dead system
rem proxy cannot break startup (023.5 proxy residue class of failure)
set HF_HUB_OFFLINE=1
set PYTHONPATH=d:\aiqg\emotion_agent
set HF_HUB_VERBOSITY=error
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
call d:\aiqg\venv\Scripts\activate
streamlit run app.py
pause
