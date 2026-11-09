@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\profile_stutter.py --suite capture-trace --seconds 25
pause
