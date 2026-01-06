# 🧘 Pose Estimation WebSocket Frontend

Modern web interface for real-time pose estimation. Captures webcam video, sends frames to WebSocket backend, and displays the results with skeleton overlay and keypoint information.

## ✨ Features

- ✅ **Real-time webcam capture**
- ✅ **WebSocket communication**
- ✅ **Live pose visualization**
- ✅ **Keypoint display with confidence scores**
- ✅ **Joint angle display**
- ✅ **FPS control (1-30 FPS)**
- ✅ **Statistics dashboard**
- ✅ **Console logging**
- ✅ **Responsive design**
- ✅ **No build tools required** - Pure HTML/CSS/JS

## 📋 Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Webcam access
- Running WebSocket backend server

## 🚀 Usage

### 1. Start Backend Server

First, make sure the backend server is running:

```bash
cd ../websocket_backend
python server.py
```

### 2. Open Frontend

Simply open `index.html` in your web browser:

- **Double-click** `index.html`, or
- **Right-click** → Open with → Browser, or
- Use a local server (optional):

```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

### 3. Connect and Start

1. Click **"Connect to Server"** button
2. Wait for "Connected" status (green indicator)
3. Click **"Start Camera"** button
4. Allow webcam access when prompted
5. Watch real-time pose estimation!

## 🎮 Controls

### Connection Panel

- **Server URL:** WebSocket server address (default: `ws://localhost:8765`)
- **FPS Limit:** Control frame rate (1-30 FPS)
- **Connect/Disconnect:** Connect to backend server
- **Start/Stop Camera:** Control webcam

### Display Sections

- **📹 Webcam Input:** Raw camera feed
- **🤖 AI Processed Output:** Frame with skeleton overlay
- **📊 Statistics:** Real-time metrics
  - Frames Processed
  - FPS (Frames Per Second)
  - Latency (milliseconds)
  - Keypoints Detected
- **🎯 Detected Keypoints:** List of detected body points with coordinates and confidence
- **📐 Joint Angles:** Calculated angles for major joints

## ⚙️ Configuration

### Change Server URL

If your backend is running on a different machine or port:

```javascript
// In the web interface, change the Server URL input to:
ws://192.168.1.100:8765  // Remote server
ws://localhost:8766      // Different port
```

### Adjust FPS

Lower FPS = Less bandwidth, better for slow connections
Higher FPS = Smoother video, more responsive

Recommended:
- Local testing: 15-20 FPS
- Slow network: 5-10 FPS
- Fast network: 20-30 FPS

### Modify Appearance

Edit `style.css` to customize:
- Colors
- Layout
- Font sizes
- Component spacing

## 📊 Understanding the Output

### Keypoint Confidence

- **Green badge (>70%):** High confidence, reliable detection
- **Yellow badge (40-70%):** Medium confidence
- **Red badge (<40%):** Low confidence, may be inaccurate

### Joint Angles

Angles are calculated for:
- **Elbows:** Angle between shoulder-elbow-wrist
- **Shoulders:** Angle between elbow-shoulder-hip
- **Hips:** Angle between shoulder-hip-knee
- **Knees:** Angle between hip-knee-ankle

Normal ranges:
- Extended limb: ~170-180°
- Right angle: ~90°
- Fully bent: <45°

## 🔧 Troubleshooting

### Camera Not Working

1. **Check permissions:** Browser needs webcam access
2. **HTTPS required:** Some browsers require HTTPS for camera (use local server)
3. **Camera in use:** Close other apps using the camera

### Cannot Connect to Server

1. **Check backend is running:** `python server.py`
2. **Check server URL:** Should be `ws://localhost:8765`
3. **Check firewall:** Allow connections on port 8765
4. **Check network:** Backend and frontend on same network

### High Latency

1. **Reduce FPS:** Lower to 10-15 FPS
2. **Check network:** Ensure good connection
3. **Backend overload:** Only one client at a time for best performance

### No Pose Detected

1. **Check lighting:** Ensure good visibility
2. **Full body in frame:** Show upper body at minimum
3. **Face camera:** Front-facing works best
4. **Check console:** Look for error messages

## 🌐 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Full | Recommended |
| Firefox | ✅ Full | Works well |
| Edge | ✅ Full | Chromium-based |
| Safari | ✅ Full | May need HTTPS |
| Mobile | ⚠️ Partial | Portrait mode only |

## 📱 Mobile Usage

The interface is responsive and works on mobile devices:
- Use rear camera for better results
- Portrait orientation recommended
- May need HTTPS (use ngrok or similar)

## 🎨 Customization Examples

### Change Skeleton Color

In `style.css`:
```css
/* Default green skeleton */
.skeleton-line {
    stroke: #00ff00;  /* Change to your color */
}
```

### Add Custom Metrics

In `app.js`, add to `handlePoseResult()`:
```javascript
// Calculate custom metrics
const avgConfidence = keypoints.reduce((sum, kp) => sum + kp.confidence, 0) / keypoints.length;
console.log('Average confidence:', avgConfidence);
```

## 🔗 Integration

### Embed in Your Web App

```html
<iframe src="path/to/index.html" width="100%" height="800px"></iframe>
```

### Use as Component

Extract the JavaScript logic from `app.js` and integrate into your framework (React, Vue, Angular).

## 📝 File Structure

```
websocket_frontend/
├── index.html     # Main HTML structure
├── style.css      # Styling and layout
├── app.js         # WebSocket logic and UI updates
└── README.md      # This file
```

## 🤝 Works With

- `../websocket_backend/` - Pose estimation backend
- Any WebSocket server that follows the protocol

## 📈 Performance Tips

1. **Reduce resolution:** Smaller camera resolution = faster processing
2. **Lower FPS:** 15 FPS is usually sufficient
3. **Close other tabs:** Free up browser resources
4. **Use wired connection:** More stable than WiFi

## 🙏 Credits

- WebSocket API
- MediaDevices API (Webcam access)
- Canvas API (Image processing)

## 📄 License

MIT License - Free to use and modify!
