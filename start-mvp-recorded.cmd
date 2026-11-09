@echo off
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" scripts\live_mvp.py --mode local-bots --sensitivity 0.97 --aim-profile configs\aim-profile.json --death-source visual --ak-spray --autostart
