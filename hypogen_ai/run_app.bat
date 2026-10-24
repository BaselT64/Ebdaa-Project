@echo off
title HypoGen-AI Server
cd /d "C:\Users\ns143\.gemini\antigravity\scratch\hypogen_ai"

echo ========================================================
echo   HypoGen-AI Platform (De-Novo Allergen Neutralization)
echo   ISEF / Ibdaa Research Framework
echo ========================================================

set PYTHON_EXE=C:\Users\ns143\AppData\Local\Python\pythoncore-3.14-64\python.exe

if not exist "%PYTHON_EXE%" (
    echo [INFO] Specific Python 3.14 path not found, using system PATH python...
    set PYTHON_EXE=python
)

echo [OK] Using Python: %PYTHON_EXE%
echo [OK] Opening browser at http://localhost:8501 ...

start http://localhost:8501
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501 --server.headless true

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
