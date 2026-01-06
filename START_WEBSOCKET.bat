@echo off
REM Quick Start Script for Pose Estimation WebSocket System
echo.
echo ========================================
echo   Pose Estimation WebSocket System
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if virtual environment exists
if exist "pose_venv\Scripts\activate.bat" (
    echo [INFO] Using existing virtual environment
    call pose_venv\Scripts\activate.bat
) else (
    echo [INFO] Virtual environment not found, using system Python
)

REM Check if dependencies are installed
echo [INFO] Checking dependencies...
python -c "import websockets" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] websockets not found
    echo [INFO] Installing dependencies...
    cd websocket_backend
    pip install -r requirements.txt
    cd ..
)

echo.
echo ========================================
echo   Starting Backend Server
echo ========================================
echo.
echo Server will start on ws://localhost:8765
echo.
echo To stop the server, press Ctrl+C
echo.
echo After server starts:
echo   1. Open websocket_frontend\index.html in browser
echo   2. Click "Connect to Server"
echo   3. Click "Start Camera"
echo.
echo ========================================
echo.

cd websocket_backend
python server.py

pause
