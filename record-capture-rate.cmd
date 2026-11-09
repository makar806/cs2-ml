@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\profile_stutter.py --suite capture-rate --seconds 25
pause
