@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo VR-FBT environment is missing. Run setup.bat first.
    pause
    exit /b 1
)

start "VR-FBT" ".venv\Scripts\pythonw.exe" Main.py
