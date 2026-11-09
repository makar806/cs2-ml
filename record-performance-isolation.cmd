@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\profile_stutter.py --suite isolation
echo.
echo Results are in runs\stutter-profile. This recorder never controls the mouse.
pause
