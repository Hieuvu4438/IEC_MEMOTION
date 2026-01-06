# 🏗️ System Architecture Diagram

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT SIDE (Browser)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐                                                │
│  │   Webcam     │                                                │
│  │   📹         │                                                │
│  └──────┬───────┘                                                │
│         │ Video Stream                                           │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              JavaScript (app.js)                         │  │
│  │                                                          │  │
│  │  1. Capture frame from webcam                          │  │
│  │  2. Draw to canvas                                     │  │
│  │  3. Convert to JPEG base64                             │  │
│  │  4. Send via WebSocket                                 │  │
│  │  5. Receive processed frame + data                     │  │
│  │  6. Update UI                                          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        │ WebSocket (ws://localhost:8765)
                        │ JSON Protocol
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVER SIDE (Python)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │           WebSocket Server (server.py)                 │    │
│  │                                                        │    │
│  │  1. Receive base64 frame                             │    │
│  │  2. Decode to numpy array                            │    │
│  │  3. Pass to AI model                                 │    │
│  │  4. Draw skeleton on frame                           │    │
│  │  5. Encode to base64                                 │    │
│  │  6. Send back with keypoints + angles                │    │
│  └───────────────────┬────────────────────────────────────┘    │
│                      │                                           │
│                      ▼                                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │          AI Model (RTMPose via rtmlib)                │    │
│  │                                                        │    │
│  │  • Detect 17 keypoints (COCO format)                 │    │
│  │  • Calculate confidence scores                       │    │
│  │  • Return (x, y) coordinates                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow (Step by Step)

```
STEP 1: Capture
┌──────────────┐
│   Webcam     │ ─────▶ getUserMedia() API
└──────────────┘        │
                        ▼
                   <video> element


STEP 2: Encode
<video> ────▶ Canvas ────▶ toDataURL() ────▶ Base64 String
              (draw)       (JPEG)


STEP 3: Send (Client → Server)
{
  "type": "frame",
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
            │
            │ WebSocket
            ▼
    Python Server


STEP 4: Process (Server)
Base64 ──▶ Decode ──▶ NumPy Array ──▶ AI Model
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  RTMPose     │
                                   │  Detection   │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   Keypoints (17 x 2)
                                   Scores (17)


STEP 5: Visualize (Server)
Frame ──▶ Draw Skeleton ──▶ Draw Keypoints ──▶ Encode to Base64


STEP 6: Calculate (Server)
Keypoints ──▶ Calculate Angles ──▶ 8 Joint Angles


STEP 7: Send Back (Server → Client)
{
  "type": "result",
  "frame": "base64_with_skeleton",
  "keypoints": [...],
  "angles": {...}
}
            │
            │ WebSocket
            ▼
    Browser (app.js)


STEP 8: Display (Client)
Response ──▶ Parse JSON ──▶ Update <img> src
                          ──▶ Update keypoints list
                          ──▶ Update angles list
                          ──▶ Update statistics
```

## Component Breakdown

### Frontend (websocket_frontend/)

```
┌─────────────────────────────────────────────────────────────┐
│                       index.html                            │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Header    │  │  Video       │  │  Info Panel        │ │
│  │  Controls  │  │  Display     │  │  - Statistics      │ │
│  │            │  │  - Webcam    │  │  - Keypoints       │ │
│  │  - Connect │  │  - Output    │  │  - Angles          │ │
│  │  - Start   │  │              │  │                    │ │
│  │  - Stop    │  │              │  │                    │ │
│  └────────────┘  └──────────────┘  └────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Console Log                            │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    style.css             app.js              WebSocket
```

### Backend (websocket_backend/)

```
┌───────────────────────────────────────────────────────────┐
│                       server.py                           │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │         PoseEstimationServer                    │    │
│  │                                                 │    │
│  │  • WebSocket Handler                          │    │
│  │  • Client Management                          │    │
│  │  • Frame Processing Pipeline                  │    │
│  └───────────────┬─────────────────────────────────┘    │
│                  │                                        │
│  ┌───────────────▼─────────────────────────────────┐    │
│  │         PoseEstimator                           │    │
│  │                                                 │    │
│  │  • RTMPose Model (rtmlib)                     │    │
│  │  • Keypoint Detection                         │    │
│  │  • Confidence Scoring                         │    │
│  └───────────────┬─────────────────────────────────┘    │
│                  │                                        │
│  ┌───────────────▼─────────────────────────────────┐    │
│  │      Math & Visualization Utils                 │    │
│  │                                                 │    │
│  │  • calculate_angle()                          │    │
│  │  • compute_joint_angles()                     │    │
│  │  • draw_skeleton_on_frame()                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Message Protocol

### Client → Server Messages

```json
┌─────────────────────────────────────┐
│ Frame Message                       │
├─────────────────────────────────────┤
│ {                                   │
│   "type": "frame",                  │
│   "image": "data:image/jpeg;..."    │
│ }                                   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Ping Message                        │
├─────────────────────────────────────┤
│ {                                   │
│   "type": "ping"                    │
│ }                                   │
└─────────────────────────────────────┘
```

### Server → Client Messages

```json
┌──────────────────────────────────────────────────────┐
│ Success Result                                       │
├──────────────────────────────────────────────────────┤
│ {                                                    │
│   "type": "result",                                  │
│   "success": true,                                   │
│   "frame": "data:image/jpeg;base64,...",            │
│   "keypoints": [                                     │
│     {"id": 0, "name": "nose", "x": 320, "y": 180,  │
│      "confidence": 0.95},                           │
│     ...                                              │
│   ],                                                 │
│   "angles": {                                        │
│     "left_elbow": 145.2,                            │
│     "right_elbow": 142.8,                           │
│     ...                                              │
│   },                                                 │
│   "frame_count": 1234,                              │
│   "timestamp": "2026-01-06T12:34:56.789"           │
│ }                                                    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐
│ Error Result                         │
├──────────────────────────────────────┤
│ {                                    │
│   "type": "error",                   │
│   "success": false,                  │
│   "message": "Error description",    │
│   "timestamp": "2026-01-06..."      │
│ }                                    │
└──────────────────────────────────────┘
```

## Technology Stack

```
┌────────────────────────────────────────────────────────┐
│                    FRONTEND                            │
├────────────────────────────────────────────────────────┤
│  HTML5           │  Structure & Layout                │
│  CSS3            │  Styling & Responsive Design       │
│  JavaScript ES6  │  WebSocket Client & Logic          │
│  WebSocket API   │  Real-time Communication           │
│  MediaDevices    │  Webcam Access                     │
│  Canvas API      │  Frame Capture & Encoding          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    BACKEND                             │
├────────────────────────────────────────────────────────┤
│  Python 3.8+     │  Programming Language              │
│  websockets      │  WebSocket Server                  │
│  rtmlib          │  RTMPose Model Wrapper             │
│  OpenCV          │  Computer Vision (cv2)             │
│  NumPy           │  Numerical Operations              │
│  asyncio         │  Async Programming                 │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    AI MODEL                            │
├────────────────────────────────────────────────────────┤
│  RTMPose         │  Real-Time Pose Estimation         │
│  ONNX Runtime    │  Model Inference Engine            │
│  COCO Format     │  17 Keypoints Standard             │
└────────────────────────────────────────────────────────┘
```

## Deployment Options

```
┌──────────────────────────────────────────────────────────┐
│                  LOCAL DEVELOPMENT                       │
│  Backend:  localhost:8765 (Python)                      │
│  Frontend: file:///.../index.html (Browser)            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  LOCAL NETWORK                           │
│  Backend:  0.0.0.0:8765 → 192.168.1.x:8765             │
│  Frontend: http://192.168.1.x/index.html                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  CLOUD DEPLOYMENT                        │
│  Backend:  Cloud VM (AWS/GCP/Azure) + SSL              │
│  Frontend: Static hosting (GitHub Pages, Netlify)       │
│  Protocol: wss:// (WebSocket Secure)                    │
└──────────────────────────────────────────────────────────┘
```

## Performance Metrics

```
┌─────────────────────────────────────────────────┐
│          Typical Performance (CPU)              │
├─────────────────────────────────────────────────┤
│  Frame Processing:    30-50 ms                  │
│  Network Latency:     5-20 ms (local)          │
│  Total Roundtrip:     50-100 ms                │
│  Achievable FPS:      15-30 FPS                │
│  Frame Size:          ~50-100 KB (JPEG)        │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│          Performance with GPU                   │
├─────────────────────────────────────────────────┤
│  Frame Processing:    10-20 ms                  │
│  Total Roundtrip:     20-50 ms                 │
│  Achievable FPS:      30-60 FPS                │
└─────────────────────────────────────────────────┘
```

## Scalability

```
┌────────────────────────────────────────────────────┐
│  Current: Single Server, Multiple Clients          │
│                                                    │
│  Client 1 ──┐                                      │
│  Client 2 ──┼──▶ WebSocket Server ──▶ AI Model   │
│  Client 3 ──┘                                      │
│                                                    │
│  Note: Sequential processing per frame             │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Future: Load Balanced, Multi-Server               │
│                                                    │
│  Clients ──▶ Load Balancer ──▶ Server 1 ──▶ GPU 1│
│                            ├──▶ Server 2 ──▶ GPU 2│
│                            └──▶ Server 3 ──▶ GPU 3│
└────────────────────────────────────────────────────┘
```
