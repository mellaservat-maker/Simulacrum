@echo off
cd /d d:\aiqg\emotion_agent
set HF_HOME=d:\aiqg\hf_cache
set PIP_CACHE_DIR=d:\aiqg\pip_cache
set HF_ENDPOINT=https://hf-mirror.com
rem [024] model cached locally: force offline load so a stale/dead system
rem proxy cannot break startup (023.5 proxy residue class of failure)
set HF_HUB_OFFLINE=1
d:\aiqg\venv\Scripts\python.exe %*