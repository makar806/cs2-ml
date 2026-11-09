@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" scripts\train_pose.py %*
pause
