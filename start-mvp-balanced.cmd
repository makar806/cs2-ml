@echo off
cd /d "%~dp0"
start "" /b ".venv\Scripts\pythonw.exe" scripts\live_mvp.py --mode local-bots --sensitivity 0.97 --imgsz 960 --hz 30 --blocking-gpu-wait --autostart
