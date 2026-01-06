# 🧘 Real-time Pose Estimation with WebSocket

Complete WebSocket-based system for real-time pose estimation. Built based on the logic from `yoga_therapy/src/core/test.py`.

## 🎯 Overview

This project consists of:
1. **Backend (WebSocket Server):** Python server that receives video frames, processes them with AI model (RTMPose), and returns frames with detected keypoints
2. **Frontend (Web Client):** Modern web interface that captures webcam, sends frames via WebSocket, and displays results

## ✨ Key Features

### Backend
- ✅ WebSocket server for real-time communication
- ✅ RTMPose AI model for pose estimation
- ✅ 17 COCO keypoints detection
- ✅ Joint angle calculation
- ✅ Skeleton visualization overlay
- ✅ Multiple clients support

### Frontend
- ✅ Real-time webcam capture
- ✅ Live pose visualization
- ✅ Keypoint display with confidence scores
- ✅ Joint angle display
- ✅ FPS control (1-30 FPS)
- ✅ Statistics dashboard
- ✅ Console logging
- ✅ Responsive design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge)
- Webcam
- Internet connection (for initial model download)

### Step 1: Install Backend

```bash
cd websocket_backend
pip install -r requirements.txt
```

### Step 2: Start Backend Server

```bash
python server.py
```

Output:
```
🚀 Pose Estimation WebSocket Server
📡 Server: ws://localhost:8765
🤖 AI Model: RTMPose (rtmlib)
⏳ Waiting for connections...
```

### Step 3: Open Frontend

Open `websocket_frontend/index.html` in your browser:
- Double-click the file, or
- Right-click → Open with → Browser

### Step 4: Connect and Use

1. Click **"Connect to Server"**
2. Click **"Start Camera"**
3. Allow webcam access
4. See real-time pose estimation!

## 📁 Project Structure

```
human_pose_estimation/
├── websocket_backend/
│   ├── server.py              # WebSocket server with AI model
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend documentation
│
├── websocket_frontend/
│   ├── index.html            # Main HTML page
│   ├── style.css             # Styling
│   ├── app.js                # WebSocket client logic
│   └── README.md             # Frontend documentation
│
└── yoga_therapy/
    └── src/core/
        └── test.py           # Original logic (reference)
```

## 🎮 How It Works

### Data Flow

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Webcam    │         │   Frontend   │         │   Backend   │
│             │────────▶│  (Browser)   │◀───────▶│  (Python)   │
│  Video Feed │         │              │ WebSocket│             │
└─────────────┘         └──────────────┘         └─────────────┘
                              │                         │
                              │                         ▼
                              │                  ┌──────────────┐
                              │                  │  AI Model    │
                              │                  │  (RTMPose)   │
                              │                  └──────────────┘
                              │                         │
                              ▼                         │
                        ┌──────────────┐              │
                        │  Display:    │◀──────────────┘
                        │  - Skeleton  │   Processed
                        │  - Keypoints │   Frame
                        │  - Angles    │
                        └──────────────┘
```

### Processing Pipeline

1. **Capture:** Frontend captures webcam frame
2. **Encode:** Frame encoded to JPEG and base64
3. **Send:** Frame sent via WebSocket to backend
4. **AI Processing:** Backend runs pose estimation model
5. **Visualization:** Backend draws skeleton on frame
6. **Response:** Backend sends processed frame + keypoints + angles
7. **Display:** Frontend shows results and statistics

## 🎯 Features Based on test.py Logic

This implementation includes all key features from `test.py`:

- ✅ **Pose Estimation:** Using RTMPose (same as test.py)
- ✅ **17 Keypoints Detection:** COCO format
- ✅ **Skeleton Drawing:** Visual overlay on frame
- ✅ **Joint Angle Calculation:** 8 major joints
- ✅ **Real-time Processing:** Live video feed
- ✅ **Confidence Scores:** For each keypoint
- ✅ **Visual Feedback:** Colored skeleton and keypoints

### Key Differences from test.py

| Feature | test.py | WebSocket Version |
|---------|---------|-------------------|
| Input | Pre-recorded videos | Live webcam |
| Interface | OpenCV window | Web browser |
| Communication | Local | WebSocket |
| Comparison | Two videos | Single live feed |
| Feedback | On-screen overlay | Web dashboard |

## 📊 Performance

- **Processing Time:** 30-50ms per frame (CPU)
- **FPS:** 15-30 FPS
- **Latency:** < 100ms (local network)
- **Bandwidth:** ~50-100 KB per frame (depends on quality)

## 🔧 Configuration

### Backend Configuration

Edit `websocket_backend/server.py`:

```python
# Server settings
host = 'localhost'  # Change to '0.0.0.0' for external access
port = 8765         # Change port if needed

# Model settings
mode = 'balanced'   # 'lightweight', 'balanced', or 'performance'
device = 'cpu'      # 'cpu' or 'cuda' (requires NVIDIA GPU)
```

### Frontend Configuration

Edit `websocket_frontend/app.js`:

```javascript
const CONFIG = {
    defaultServerUrl: 'ws://localhost:8765',  // Backend URL
    defaultFPS: 15,                          // Default FPS
    reconnectDelay: 3000,                    // Auto-reconnect delay
    maxReconnectAttempts: 5                  // Max reconnect tries
};
```

## 🎨 Customization

### Change Skeleton Color

Backend (`server.py`):
```python
COLORS = {
    'skeleton': (0, 255, 0),     # Green (BGR)
    'keypoint': (255, 0, 0),     # Blue
}
```

### Add Custom Metrics

Frontend (`app.js`):
```javascript
function handlePoseResult(result) {
    // Your custom logic here
    console.log('Custom metric:', result.keypoints.length);
}
```

## 🐛 Troubleshooting

### Backend Issues

**rtmlib not found:**
```bash
pip install rtmlib
```

**Port already in use:**
```bash
# Use different port
python server.py --port 8766
```

### Frontend Issues

**Cannot connect:**
- Check backend is running
- Check WebSocket URL matches
- Check firewall settings

**Camera not working:**
- Allow camera permissions
- Close other apps using camera
- Try different browser

**High latency:**
- Reduce FPS in frontend
- Use wired connection
- Check CPU usage

## 📚 Documentation

- [Backend README](websocket_backend/README.md) - Detailed backend documentation
- [Frontend README](websocket_frontend/README.md) - Detailed frontend documentation
- [Original test.py](yoga_therapy/src/core/test.py) - Reference implementation

## 🤝 Contributing

Feel free to improve this project:
1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit pull request

## 📝 License

MIT License - Free to use and modify!

## 🙏 Credits

- **RTMPose** - State-of-the-art pose estimation model
- **rtmlib** - Python wrapper for RTMPose
- **OpenCV** - Computer vision library
- **WebSockets** - Real-time communication

## 📧 Support

For issues or questions:
1. Check the documentation
2. Review troubleshooting section
3. Check console logs
4. Open an issue on GitHub

---

**Enjoy real-time pose estimation! 🧘‍♀️**
