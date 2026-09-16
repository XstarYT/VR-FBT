@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo This project needs 64-bit Python 3.12. The existing .venv has not been changed.
    goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" Main.py --check
if errorlevel 1 goto failed
echo Setup complete. Start VR-FBT with run.bat.
pause
exit /b 0
:failed
echo Setup failed. Review the error above. Install 64-bit Python 3.12 and Git for Windows if missing.
pause
exit /b 1
