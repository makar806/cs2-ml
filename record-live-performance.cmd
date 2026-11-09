@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\profile_stutter.py --suite live --seconds 45
echo.
echo Results are in runs\stutter-profile. Recorder does not control the mouse.
pause
