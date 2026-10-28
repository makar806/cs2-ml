@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\live_mvp.py --mode local-bots --death-source manual --autostart
pause
