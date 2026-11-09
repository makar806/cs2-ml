@echo off
cd /d "%~dp0"
set LABEL_STUDIO_BASE_DATA_DIR=%~dp0.label-studio
set LABEL_STUDIO_DISABLE_ANALYTICS=true
set LATEST_VERSION_CHECK=false
"%~dp0.venv-labeling\Scripts\label-studio.exe" start --host http://127.0.0.1:8080 --internal-host 127.0.0.1 --port 8080
pause
