@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" scripts\check_environment.py
pause
