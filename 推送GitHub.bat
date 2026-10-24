@echo off
chcp 65001 >nul
title Push emotion_agent (Vesper) to GitHub
cd /d d:\aiqg\emotion_agent
echo.
echo ============================================================
echo  STEP 1  A browser page will open: https://github.com/new
echo          Repository name : Simulacrum
echo          Visibility      : Public
echo          IMPORTANT       : do NOT tick Add a README file,
echo                            .gitignore or license
echo          Then click Create repository.
echo ============================================================
echo.
start "" "https://github.com/new"
echo.
set /p REPO=STEP 2  Paste the repo URL (ends with .git) and press Enter: 
if "%REPO%"=="" goto cancelled
git remote remove origin 2>nul
git remote add origin %REPO%
git branch -M main
echo.
echo ============================================================
echo  STEP 3  Pushing... a login window may pop up (authorize once).
echo          If network error: turn on proxy and retry, or run
echo          git config --global http.proxy http://127.0.0.1:PORT
echo ============================================================
echo.
git push -u origin main
echo.
echo ------------------------------------------------------------
echo  OK if you see branch main set up to track origin/main
echo  Repository: %REPO%
echo ------------------------------------------------------------
pause
goto :eof
:cancelled
echo Cancelled (no URL entered).
pause
