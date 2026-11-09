@echo off
cd /d "%~dp0"
start "" /b ".venv\Scripts\pythonw.exe" scripts\mvp_simulator.py
start "" /b ".venv\Scripts\pythonw.exe" scripts\live_mvp.py --mode simulator --sensitivity 1 --autostart
