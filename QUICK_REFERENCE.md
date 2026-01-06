# ⚡ Quick Reference Card

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install
cd websocket_backend
pip install -r requirements.txt

# 2. Run Backend
python server.py

# 3. Open Frontend
# Open websocket_frontend/index.html in browser
# Click "Connect" → Click "Start Camera"
```

## 📂 Project Structure

```
websocket_backend/    → Python WebSocket Server + AI
websocket_frontend/   → HTML/CSS/JS Web Interface
```

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `websocket_backend/server.py` | Main backend server |
| `websocket_backend/requirements.txt` | Python dependencies |
| `websocket_frontend/index.html` | Web interface |
| `websocket_frontend/app.js` | WebSocket client logic |
| `websocket_frontend/style.css` | Styling |
| `START_WEBSOCKET.bat` | Windows startup script |
| `WEBSOCKET_README.md` | Full documentation |

## 🎮 Common Commands

### Backend

```bash
# Start server
python server.py

# Install dependencies
pip install -r requirements.txt

# Test rtmlib
python -c "from rtmlib import Wholebody; print('OK')"

# Check port
netstat -ano | findstr :8765  # Windows
lsof -i :8765                 # Linux/Mac
```

### Frontend

```bash
# Open directly
# Double-click index.html

# Or use local server
python -m http.server 8000
# Then: http://localhost:8000
```

## 🔧 Configuration

### Backend (server.py)

```python
host = 'localhost'      # Server address
port = 8765            # Port number
mode = 'balanced'      # Model: lightweight/balanced/performance
device = 'cpu'         # Device: cpu/cuda
```

### Frontend (app.js)

```javascript
defaultServerUrl: 'ws://localhost:8765'  // Backend URL
defaultFPS: 15                           // Frames per second
```

## 📊 Performance Tips

| Issue | Solution |
|-------|----------|
| High latency | Lower FPS to 10-15 |
| Low FPS | Close other apps, use GPU |
| No detection | Better lighting, face camera |
| High CPU | Reduce FPS, use lightweight mode |

## 🎯 Keypoints (COCO-17)

```
Head:     0-nose, 1/2-eyes, 3/4-ears
Arms:     5/6-shoulders, 7/8-elbows, 9/10-wrists
Torso:    5/6-shoulders, 11/12-hips
Legs:     11/12-hips, 13/14-knees, 15/16-ankles
```

## 📐 Joint Angles (8 Joints)

- Left/Right Elbow
- Left/Right Shoulder
- Left/Right Hip
- Left/Right Knee

## 🔌 WebSocket Protocol

### Send Frame (Client → Server)

```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,..."
}
```

### Receive Result (Server → Client)

```json
{
  "type": "result",
  "frame": "data:image/jpeg;base64,..with_skeleton",
  "keypoints": [{"name": "nose", "x": 320, "y": 180, "confidence": 0.95}, ...],
  "angles": {"left_elbow": 145.2, ...}
}
```

## 🐛 Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `rtmlib not found` | `pip install rtmlib` |
| `Port 8765 in use` | Change port in server.py |
| `Cannot connect` | Check backend is running |
| `Camera not working` | Allow browser permissions |
| `No pose detected` | Better lighting, face camera |
| `High latency` | Lower FPS to 10-15 |

## 📖 Documentation

- [WEBSOCKET_README.md](WEBSOCKET_README.md) - Overview
- [websocket_backend/README.md](websocket_backend/README.md) - Backend docs
- [websocket_frontend/README.md](websocket_frontend/README.md) - Frontend docs
- [COMPARISON.md](COMPARISON.md) - test.py comparison
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) - Setup guide
- [SUMMARY_VI.md](SUMMARY_VI.md) - Vietnamese summary

## 🎨 Color Scheme

| Color | Usage |
|-------|-------|
| Green (0,255,0) | Skeleton, success |
| Blue (255,0,0) | Keypoints |
| Red (0,0,255) | Errors |

## 📏 Typical Values

| Metric | Typical Range |
|--------|---------------|
| Latency | 50-100ms |
| FPS | 15-30 |
| Frame Size | 50-100 KB |
| Keypoints Detected | 10-17 |
| Processing Time | 30-50ms (CPU) |

## 🔐 Default Settings

- **Server URL:** ws://localhost:8765
- **FPS:** 15
- **Model Mode:** balanced
- **Device:** CPU
- **Confidence Threshold:** 0.3

## 🎛️ Frontend Controls

| Button | Action |
|--------|--------|
| Connect | Connect/disconnect WebSocket |
| Start Camera | Start webcam capture |
| Stop Camera | Stop webcam |
| Clear Log | Clear console |
| FPS Input | Change frame rate (1-30) |
| Server URL | Change backend address |

## 📱 Browser Support

| Browser | Status |
|---------|--------|
| Chrome 90+ | ✅ Full support |
| Firefox 88+ | ✅ Full support |
| Edge 90+ | ✅ Full support |
| Safari 14+ | ✅ Works (may need HTTPS) |
| Mobile | ⚠️ Partial (portrait only) |

## 🔄 Startup Order

1. ✅ Start Backend (python server.py)
2. ✅ Open Frontend (index.html)
3. ✅ Connect to Server
4. ✅ Start Camera
5. ✅ Watch real-time pose estimation!

## 💾 Dependencies

### Backend

- Python 3.8+
- websockets 12.0
- rtmlib 1.0.1
- opencv-python 4.9.0
- numpy 1.24.3

### Frontend

- Modern browser
- WebSocket support
- MediaDevices API
- Canvas API

## 📞 Quick Help Commands

```bash
# Check Python version
python --version

# Check if port is free
netstat -ano | findstr :8765

# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8765

# Install all dependencies
pip install websockets rtmlib opencv-python numpy

# Check installed packages
pip list | grep -E "websockets|rtmlib|opencv|numpy"
```

## ⚠️ Common Errors

| Error | Meaning | Fix |
|-------|---------|-----|
| `ModuleNotFoundError: No module named 'rtmlib'` | rtmlib not installed | `pip install rtmlib` |
| `[WinError 10048] Only one usage...` | Port in use | Change port or kill process |
| `OSError: [Errno 98] Address already in use` | Port in use (Linux) | Kill process on port 8765 |
| `WebSocket connection failed` | Backend not running | Start server.py |
| `NotAllowedError: Permission denied` | Camera permission | Allow in browser |

## 🌟 Features at a Glance

✅ Real-time pose estimation
✅ 17 keypoint detection
✅ 8 joint angle calculation
✅ Skeleton visualization
✅ Confidence scoring
✅ WebSocket communication
✅ Modern web UI
✅ Statistics dashboard
✅ Console logging
✅ FPS control
✅ Multi-client support

---

**Need more details? Check [WEBSOCKET_README.md](WEBSOCKET_README.md)**
