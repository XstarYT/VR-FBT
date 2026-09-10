@echo off

cls

echo Restarting...

timeout /t 5 /nobreak >nul

cls

if "%1"=="py" call run.bat

if "%1"=="exe" start "VR-FBT" Main.exe
