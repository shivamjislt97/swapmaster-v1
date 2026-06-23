@echo off
REM =============================================================================
REM SwapMaster V1 - Run Script (Windows)
REM =============================================================================

echo ============================================
echo   SwapMaster V1 - Starting...
echo ============================================

cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] No virtual environment found. Run: setup.bat
    pause
    exit /b 1
)

REM Check .env
if not exist ".env" (
    echo [ERROR] .env file not found. Copy .env.example to .env and configure it.
    pause
    exit /b 1
)

REM Start the application
echo [START] Starting SwapMaster...
python app\startup.py
pause
