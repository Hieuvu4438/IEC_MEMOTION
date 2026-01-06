#!/bin/bash
# Quick Start Script for Pose Estimation WebSocket System

echo ""
echo "========================================"
echo "  Pose Estimation WebSocket System"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "[OK] Python found"
echo ""

# Check if virtual environment exists
if [ -d "pose_venv" ]; then
    echo "[INFO] Using existing virtual environment"
    source pose_venv/bin/activate
else
    echo "[INFO] Virtual environment not found, using system Python"
fi

# Check if dependencies are installed
echo "[INFO] Checking dependencies..."
python3 -c "import websockets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] websockets not found"
    echo "[INFO] Installing dependencies..."
    cd websocket_backend
    pip3 install -r requirements.txt
    cd ..
fi

echo ""
echo "========================================"
echo "  Starting Backend Server"
echo "========================================"
echo ""
echo "Server will start on ws://localhost:8765"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""
echo "After server starts:"
echo "  1. Open websocket_frontend/index.html in browser"
echo "  2. Click 'Connect to Server'"
echo "  3. Click 'Start Camera'"
echo ""
echo "========================================"
echo ""

cd websocket_backend
python3 server.py
