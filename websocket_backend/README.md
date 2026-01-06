# 🧘 Pose Estimation WebSocket Backend

Real-time pose estimation server using WebSocket. Receives video frames from clients, processes them with AI model (RTMPose), and returns frames with detected keypoints and skeleton overlay.

## ✨ Features

- ✅ **Real-time WebSocket communication**
- ✅ **AI-powered pose estimation** (RTMPose via rtmlib)
- ✅ **17 keypoints detection** (COCO format)
- ✅ **Joint angle calculation**
- ✅ **Skeleton visualization**
- ✅ **Multiple clients support**
- ✅ **Low latency processing**

## 📋 Requirements

- Python 3.8+
- Webcam or video source
- Internet connection (for initial model download)

## 🚀 Installation

### 1. Create Virtual Environment (Optional but recommended)

```bash
cd websocket_backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The first time you run the server, rtmlib will automatically download the RTMPose model (~10-20 MB).

## 🎮 Usage

### Start the Server

```bash
python server.py
```

The server will start on `ws://localhost:8765` by default.

You should see:
```
🚀 Pose Estimation WebSocket Server
📡 Server: ws://localhost:8765
🤖 AI Model: RTMPose (rtmlib)
⏳ Waiting for connections...
```

### Server Configuration

You can modify these settings in `server.py`:

```python
# Server settings
host = 'localhost'  # Change to '0.0.0.0' for external access
port = 8765

# Model settings
mode = 'balanced'  # Options: 'lightweight', 'balanced', 'performance'
device = 'cpu'     # Options: 'cpu', 'cuda'
```

## 📡 WebSocket Protocol

### Client → Server

**Send Frame:**
```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Health Check:**
```json
{
  "type": "ping"
}
```

### Server → Client

**Pose Result:**
```json
{
  "type": "result",
  "success": true,
  "frame": "data:image/jpeg;base64,...",
  "keypoints": [
    {
      "id": 0,
      "name": "nose",
      "x": 320.5,
      "y": 180.2,
      "confidence": 0.95
    },
    ...
  ],
  "angles": {
    "left_elbow": 145.2,
    "right_elbow": 142.8,
    ...
  },
  "frame_count": 1234,
  "timestamp": "2026-01-06T12:34:56.789"
}
```

**Error:**
```json
{
  "type": "error",
  "success": false,
  "message": "Error description",
  "timestamp": "2026-01-06T12:34:56.789"
}
```

## 🎯 Detected Keypoints (COCO-17)

1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle

## 📐 Joint Angles

The server calculates angles for these joints:
- Left/Right Elbow
- Left/Right Shoulder
- Left/Right Hip
- Left/Right Knee

## 🔧 Troubleshooting

### Model Download Failed
```bash
# Manually download model
python -c "from rtmlib import Wholebody; Wholebody()"
```

### Port Already in Use
Change the port in `server.py`:
```python
port = 8766  # Use different port
```

### Low Performance
- Use GPU: Change `device='cpu'` to `device='cuda'` (requires CUDA)
- Use lightweight mode: Change `mode='balanced'` to `mode='lightweight'`
- Reduce client FPS

## 📊 Performance

- **Processing Time:** 30-50ms per frame (CPU)
- **FPS:** 15-30 FPS
- **Latency:** < 100ms (local network)

## 🤝 Integration with Frontend

This backend is designed to work with the WebSocket frontend in `../websocket_frontend/`.

Start both:
1. Backend: `python server.py` (this folder)
2. Frontend: Open `../websocket_frontend/index.html` in browser

## 📝 License

MIT License - Feel free to use in your projects!

## 🙏 Credits

- **RTMPose** - Real-time pose estimation model
- **rtmlib** - Python wrapper for RTMPose
- **OpenCV** - Computer vision library
