@echo off
REM =============================================================================
REM SwapMaster V1 - Native Setup Script (Windows)
REM =============================================================================

echo ============================================
echo   SwapMaster V1 - Native Installation
echo ============================================
echo.

cd /d "%~dp0"

REM 1. Check Python
echo [1/6] Checking Python...
python --version 2>nul
if %errorlevel% neq 0 (
    python3 --version 2>nul
    if %errorlevel% neq 0 (
        echo ERROR: Python not found. Install Python 3.10+ first.
        echo   Download: https://www.python.org/downloads/
        echo   Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

REM 2. Create virtual environment
echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created: venv\
) else (
    echo   Virtual environment already exists: venv\
)

REM 3. Activate and install dependencies
echo [3/6] Installing Python dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo   Dependencies installed

REM 4. Check system dependencies
echo [4/6] Checking system dependencies...

where ffmpeg >nul 2>nul
if %errorlevel% equ 0 (
    echo   [OK] ffmpeg found
) else (
    echo   [WARN] ffmpeg not found. Install it:
    echo     Download: https://ffmpeg.org/download.html
    echo     Or use: winget install ffmpeg
)

where rclone >nul 2>nul
if %errorlevel% equ 0 (
    echo   [OK] rclone found
) else (
    echo   [WARN] rclone not found. Install it:
    echo     Download: https://rclone.org/downloads/
)

where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo   [OK] nvidia-smi found
) else (
    echo   [INFO] nvidia-smi not found ^(CPU-only mode^)
)

REM 5. Setup directories
echo [5/6] Setting up directories...
if not exist "pipeline\logs" mkdir pipeline\logs
if not exist "pipeline\workspace\output" mkdir pipeline\workspace\output
if not exist "pipeline\workspace\temp" mkdir pipeline\workspace\temp
if not exist "pipeline\downloads\video" mkdir pipeline\downloads\video
if not exist "pipeline\downloads\face" mkdir pipeline\downloads\face
if not exist "pipeline\dashboard_sessions" mkdir pipeline\dashboard_sessions
if not exist "persistent\faces" mkdir persistent\faces
if not exist ".config\rclone" mkdir .config\rclone
echo   Directories created

REM 6. Check rclone config
echo [6/6] Checking rclone configuration...
if exist ".config\rclone\rclone.conf" (
    echo   [OK] rclone.conf found
) else (
    echo   [WARN] rclone.conf not found at .config\rclone\rclone.conf
    echo   Run: rclone config
    echo   Create a remote named 'gdrive' with Google Drive
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To start SwapMaster:
echo   run.bat
echo.
echo Or manually:
echo   venv\Scripts\activate.bat
echo   python app\startup.py
echo.
pause
