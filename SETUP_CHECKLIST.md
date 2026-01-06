# ✅ Setup Checklist

Use this checklist to set up and verify the WebSocket Pose Estimation system.

## 📋 Pre-Installation Checklist

- [ ] **Python 3.8+** installed
  ```bash
  python --version
  # Should show: Python 3.8.x or higher
  ```

- [ ] **pip** available
  ```bash
  pip --version
  # Should show pip version
  ```

- [ ] **Modern Browser** (Chrome/Firefox/Edge)
  - [ ] Chrome 90+
  - [ ] Firefox 88+
  - [ ] Edge 90+

- [ ] **Webcam** connected and working
  - [ ] Test in native camera app first

- [ ] **Internet connection** (for initial model download)

## 📦 Backend Installation Checklist

- [ ] Navigate to backend folder
  ```bash
  cd websocket_backend
  ```

- [ ] (Optional) Create virtual environment
  ```bash
  python -m venv venv
  
  # Windows
  venv\Scripts\activate
  
  # Linux/Mac
  source venv/bin/activate
  ```

- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Verify installations
  ```bash
  python -c "import websockets; print('✓ websockets')"
  python -c "import cv2; print('✓ opencv-python')"
  python -c "import numpy; print('✓ numpy')"
  ```

- [ ] Install rtmlib (may take a few minutes)
  ```bash
  pip install rtmlib
  ```

- [ ] Verify rtmlib
  ```bash
  python -c "from rtmlib import Wholebody; print('✓ rtmlib')"
  ```

## 🚀 Backend Startup Checklist

- [ ] Start the server
  ```bash
  python server.py
  ```

- [ ] Verify server is running
  - [ ] Should see: `🚀 Pose Estimation WebSocket Server`
  - [ ] Should see: `📡 Server: ws://localhost:8765`
  - [ ] Should see: `⏳ Waiting for connections...`

- [ ] Check for errors
  - [ ] No red error messages
  - [ ] Model loads successfully (first run may download ~20MB)

- [ ] Test WebSocket (optional)
  ```bash
  # In another terminal
  python -c "import asyncio; import websockets; asyncio.run(websockets.connect('ws://localhost:8765'))"
  ```

## 🌐 Frontend Setup Checklist

- [ ] Navigate to frontend folder
  ```bash
  cd websocket_frontend
  ```

- [ ] Verify files exist
  - [ ] index.html
  - [ ] style.css
  - [ ] app.js

- [ ] Open in browser
  - **Option 1:** Double-click `index.html`
  - **Option 2:** Right-click → Open with → Browser
  - **Option 3:** Use local server
    ```bash
    # Python 3
    python -m http.server 8000
    # Then open: http://localhost:8000
    ```

## 🎮 Frontend Usage Checklist

- [ ] **Page loads correctly**
  - [ ] Header visible
  - [ ] Buttons visible
  - [ ] Status shows "Disconnected" (red)

- [ ] **Connect to server**
  - [ ] Click "Connect to Server" button
  - [ ] Status changes to "Connected" (green)
  - [ ] Console log shows "✅ Kết nối WebSocket thành công!"
  - [ ] "Start Camera" button becomes enabled

- [ ] **Start camera**
  - [ ] Click "Start Camera" button
  - [ ] Browser asks for camera permission
  - [ ] Allow camera access
  - [ ] Webcam video appears in left panel
  - [ ] Console log shows "✅ Camera đã sẵn sàng!"

- [ ] **Verify processing**
  - [ ] Processed frames appear in right panel
  - [ ] Skeleton overlay visible on processed frames
  - [ ] Keypoints (colored dots) visible
  - [ ] Frame count increases
  - [ ] FPS shows realistic value (10-30)
  - [ ] Latency shows < 200ms

- [ ] **Check data display**
  - [ ] Keypoints list shows detected points
  - [ ] Each keypoint has coordinates
  - [ ] Confidence scores displayed
  - [ ] Joint angles list shows values
  - [ ] Angles in reasonable range (0-180°)

## 🔍 Troubleshooting Checklist

### Backend Issues

- [ ] **rtmlib not found**
  ```bash
  pip install rtmlib
  ```

- [ ] **Port already in use**
  - [ ] Check if another server is running
  - [ ] Change port in server.py
  - [ ] Or kill existing process:
    ```bash
    # Windows
    netstat -ano | findstr :8765
    taskkill /PID <PID> /F
    
    # Linux/Mac
    lsof -i :8765
    kill -9 <PID>
    ```

- [ ] **Model download fails**
  - [ ] Check internet connection
  - [ ] Try manual download:
    ```bash
    python -c "from rtmlib import Wholebody; Wholebody()"
    ```

- [ ] **Low performance**
  - [ ] Close other applications
  - [ ] Reduce frontend FPS
  - [ ] Consider using GPU (if available)

### Frontend Issues

- [ ] **Cannot connect to server**
  - [ ] Verify backend is running
  - [ ] Check server URL: `ws://localhost:8765`
  - [ ] Check firewall settings
  - [ ] Try different port

- [ ] **Camera not working**
  - [ ] Check browser permissions
  - [ ] Close other apps using camera
  - [ ] Try different browser
  - [ ] For HTTPS: use local server

- [ ] **No processed frames**
  - [ ] Check backend console for errors
  - [ ] Check browser console (F12)
  - [ ] Verify WebSocket connection is active
  - [ ] Try reconnecting

- [ ] **High latency (> 200ms)**
  - [ ] Reduce FPS to 10-15
  - [ ] Check CPU usage
  - [ ] Close other applications
  - [ ] Check network (if remote)

- [ ] **Pose not detected**
  - [ ] Ensure good lighting
  - [ ] Show full upper body in frame
  - [ ] Face the camera
  - [ ] Reduce background clutter

## ✨ Feature Testing Checklist

- [ ] **Connection Management**
  - [ ] Connect works
  - [ ] Disconnect works
  - [ ] Reconnect works
  - [ ] Status indicator updates

- [ ] **Camera Control**
  - [ ] Start camera works
  - [ ] Stop camera works
  - [ ] Restart camera works
  - [ ] Camera permission handled

- [ ] **FPS Control**
  - [ ] Can change FPS value
  - [ ] FPS updates take effect
  - [ ] Lower FPS = less bandwidth
  - [ ] Higher FPS = smoother

- [ ] **Display Features**
  - [ ] Webcam feed displays
  - [ ] Processed frame displays
  - [ ] Skeleton overlay works
  - [ ] Keypoints visible
  - [ ] Colors correct

- [ ] **Data Display**
  - [ ] Statistics update
  - [ ] Frame count increases
  - [ ] FPS calculated correctly
  - [ ] Latency shows
  - [ ] Keypoint count shows

- [ ] **Keypoints Panel**
  - [ ] List shows detected keypoints
  - [ ] Coordinates display
  - [ ] Confidence scores show
  - [ ] Color coding works (green/yellow/red)
  - [ ] Scrollable if many points

- [ ] **Angles Panel**
  - [ ] 8 joint angles display
  - [ ] Vietnamese names shown
  - [ ] Values in degrees
  - [ ] Updates in real-time

- [ ] **Console Log**
  - [ ] Messages appear
  - [ ] Color coding works
  - [ ] Timestamps show
  - [ ] Clear button works
  - [ ] Scrollable

## 🎯 Performance Verification Checklist

- [ ] **Latency**
  - [ ] < 100ms (excellent)
  - [ ] 100-200ms (good)
  - [ ] > 200ms (check troubleshooting)

- [ ] **FPS**
  - [ ] 20-30 FPS (excellent)
  - [ ] 15-20 FPS (good)
  - [ ] 10-15 FPS (acceptable)
  - [ ] < 10 FPS (needs optimization)

- [ ] **Detection Quality**
  - [ ] All 17 keypoints detected (ideal)
  - [ ] At least 10 keypoints (good)
  - [ ] Confidence > 0.7 for most points
  - [ ] Skeleton looks correct

- [ ] **Stability**
  - [ ] No crashes after 5 minutes
  - [ ] No memory leaks
  - [ ] WebSocket stays connected
  - [ ] Camera stays active

## 📊 Final Verification

- [ ] **Backend Status**
  - [ ] Server running without errors
  - [ ] Processing frames successfully
  - [ ] Logs show frame counts
  - [ ] No memory issues

- [ ] **Frontend Status**
  - [ ] Connected to server
  - [ ] Camera active
  - [ ] Receiving processed frames
  - [ ] UI responsive

- [ ] **Overall System**
  - [ ] Real-time processing working
  - [ ] Pose detection accurate
  - [ ] No significant lag
  - [ ] All features functional

## 🎉 Success Criteria

Your system is working correctly if:

✅ Backend server starts without errors
✅ Frontend connects successfully
✅ Camera captures video
✅ Frames are processed and returned
✅ Skeleton overlay is visible
✅ Keypoints are detected (at least 10/17)
✅ Joint angles are calculated
✅ Latency is < 200ms
✅ FPS is > 10
✅ System runs stable for 5+ minutes

## 📝 Notes

- First run may be slower (model download)
- Lighting affects detection quality
- CPU usage will be higher during processing
- WebSocket connection should be stable
- Frontend works best in Chrome

## 🆘 Getting Help

If something doesn't work:

1. Check this checklist again
2. Review error messages (backend console + browser console F12)
3. Check [WEBSOCKET_README.md](WEBSOCKET_README.md)
4. Check [Backend README](websocket_backend/README.md)
5. Check [Frontend README](websocket_frontend/README.md)
6. Review [COMPARISON.md](COMPARISON.md) for logic understanding

---

**Good luck! 🚀🧘**
