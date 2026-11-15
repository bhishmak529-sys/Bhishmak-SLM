@echo off
title Bhishmak AI Enterprise Core
color 0A

:: Purane atke hue Python instances ko clean karega
taskkill /F /IM python.exe >nul 2>&1

echo ==================================================
echo    STARTING BHISHMAK AI DESKTOP ENVIRONMENT...
echo ==================================================
echo.

echo [1/3] Ensuring Ollama Engine is active...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running.
) else (
    start "" /B ollama serve >nul 2>&1
    echo Waking up engine, waiting 6 seconds...
    timeout /t 6 /nobreak >nul
)

echo [2/3] Verifying Ollama port readiness...
timeout /t 2 /nobreak >nul

echo [3/3] Launching Bhishmak AI Workspace...
cd /d "D:\Local_BCA_Bot"
python app.py

pause