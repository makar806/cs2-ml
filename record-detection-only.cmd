@echo off
cd /d "%~dp0"
echo Close the ordinary MVP panel first. This records detections only; no aiming or firing.
"%~dp0.venv\Scripts\python.exe" scripts\record_localization.py --scenario both-motion --duration 90
pause
