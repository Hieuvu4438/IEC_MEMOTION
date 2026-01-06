// ==================== CONFIGURATION ====================
const CONFIG = {
    defaultServerUrl: 'ws://localhost:8765',
    defaultFPS: 15,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5
};

// ==================== STATE ====================
let state = {
    websocket: null,
    webcamStream: null,
    isConnected: false,
    isCameraActive: false,
    frameCount: 0,
    lastFrameTime: 0,
    fps: 0,
    latency: 0,
    reconnectAttempts: 0,
    frameInterval: null
};

// ==================== DOM ELEMENTS ====================
const elements = {
    // Buttons
    connectBtn: document.getElementById('connectBtn'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    clearLogBtn: document.getElementById('clearLogBtn'),
    
    // Status
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    
    // Settings
    serverUrl: document.getElementById('serverUrl'),
    fpsLimit: document.getElementById('fpsLimit'),
    
    // Video
    webcam: document.getElementById('webcam'),
    inputCanvas: document.getElementById('inputCanvas'),
    outputImage: document.getElementById('outputImage'),
    noOutput: document.getElementById('noOutput'),
    
    // Stats
    frameCount: document.getElementById('frameCount'),
    fps: document.getElementById('fps'),
    latency: document.getElementById('latency'),
    keypointCount: document.getElementById('keypointCount'),
    
    // Info
    keypointsList: document.getElementById('keypointsList'),
    anglesList: document.getElementById('anglesList'),
    
    // Console
    consoleLog: document.getElementById('consoleLog')
};

// ==================== LOGGER ====================
function log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString('vi-VN');
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span><span>${message}</span>`;
    
    elements.consoleLog.appendChild(entry);
    elements.consoleLog.scrollTop = elements.consoleLog.scrollHeight;
    
    // Also log to browser console
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// ==================== WEBSOCKET FUNCTIONS ====================
function connectWebSocket() {
    const serverUrl = elements.serverUrl.value || CONFIG.defaultServerUrl;
    
    log(`Đang kết nối đến ${serverUrl}...`, 'info');
    
    try {
        state.websocket = new WebSocket(serverUrl);
        
        state.websocket.onopen = () => {
            state.isConnected = true;
            state.reconnectAttempts = 0;
            updateConnectionStatus(true);
            log('✅ Kết nối WebSocket thành công!', 'success');
            
            elements.connectBtn.textContent = 'Disconnect';
            elements.connectBtn.classList.remove('btn-primary');
            elements.connectBtn.classList.add('btn-danger');
            elements.startBtn.disabled = false;
        };
        
        state.websocket.onmessage = (event) => {
            handleServerMessage(event.data);
        };
        
        state.websocket.onerror = (error) => {
            log('❌ Lỗi WebSocket: ' + error.message, 'error');
        };
        
        state.websocket.onclose = () => {
            state.isConnected = false;
            updateConnectionStatus(false);
            log('⚠️ Kết nối WebSocket đã đóng', 'warning');
            
            elements.connectBtn.textContent = 'Connect to Server';
            elements.connectBtn.classList.remove('btn-danger');
            elements.connectBtn.classList.add('btn-primary');
            elements.startBtn.disabled = true;
            
            // Auto reconnect
            if (state.reconnectAttempts < CONFIG.maxReconnectAttempts) {
                state.reconnectAttempts++;
                log(`Đang thử kết nối lại... (${state.reconnectAttempts}/${CONFIG.maxReconnectAttempts})`, 'info');
                setTimeout(connectWebSocket, CONFIG.reconnectDelay);
            }
        };
        
    } catch (error) {
        log('❌ Không thể kết nối: ' + error.message, 'error');
    }
}

function disconnectWebSocket() {
    if (state.websocket) {
        state.reconnectAttempts = CONFIG.maxReconnectAttempts; // Prevent auto reconnect
        state.websocket.close();
        state.websocket = null;
        log('🔌 Đã ngắt kết nối', 'info');
    }
}

function handleServerMessage(data) {
    try {
        const message = JSON.parse(data);
        
        switch (message.type) {
            case 'result':
                handlePoseResult(message);
                break;
                
            case 'pong':
                log('📡 Ping response received', 'info');
                break;
                
            case 'error':
                log('❌ Server error: ' + message.message, 'error');
                break;
                
            default:
                log('⚠️ Unknown message type: ' + message.type, 'warning');
        }
        
    } catch (error) {
        log('❌ Lỗi parse message: ' + error.message, 'error');
    }
}

function handlePoseResult(result) {
    if (!result.success) {
        log('❌ Xử lý frame thất bại: ' + result.message, 'error');
        return;
    }
    
    // Update processed frame
    elements.outputImage.src = result.frame;
    elements.outputImage.style.display = 'block';
    elements.noOutput.style.display = 'none';
    
    // Update stats
    state.frameCount = result.frame_count;
    elements.frameCount.textContent = state.frameCount;
    
    // Calculate latency
    const now = Date.now();
    state.latency = now - state.lastFrameTime;
    elements.latency.textContent = state.latency + ' ms';
    
    // Update keypoints
    updateKeypoints(result.keypoints);
    
    // Update angles
    updateAngles(result.angles);
    
    // Calculate FPS
    updateFPS();
}

function updateKeypoints(keypoints) {
    if (!keypoints || keypoints.length === 0) {
        elements.keypointsList.innerHTML = '<p class="text-muted">Không phát hiện keypoints</p>';
        elements.keypointCount.textContent = '0';
        return;
    }
    
    // Count valid keypoints (confidence > 0.3)
    const validKeypoints = keypoints.filter(kp => kp.confidence > 0.3);
    elements.keypointCount.textContent = validKeypoints.length;
    
    // Display keypoints
    elements.keypointsList.innerHTML = '';
    keypoints.forEach(kp => {
        if (kp.confidence > 0.2) {
            const item = document.createElement('div');
            item.className = 'keypoint-item';
            
            const confClass = kp.confidence > 0.7 ? '' : kp.confidence > 0.4 ? 'low' : 'very-low';
            
            item.innerHTML = `
                <span class="keypoint-name">${kp.name}</span>
                <span>
                    <span class="keypoint-coords">(${kp.x.toFixed(0)}, ${kp.y.toFixed(0)})</span>
                    <span class="confidence ${confClass}">${(kp.confidence * 100).toFixed(0)}%</span>
                </span>
            `;
            
            elements.keypointsList.appendChild(item);
        }
    });
}

function updateAngles(angles) {
    if (!angles || Object.keys(angles).length === 0) {
        elements.anglesList.innerHTML = '<p class="text-muted">Chưa tính toán góc</p>';
        return;
    }
    
    // Joint name translations
    const jointNames = {
        'left_elbow': 'Khuỷu trái',
        'right_elbow': 'Khuỷu phải',
        'left_shoulder': 'Vai trái',
        'right_shoulder': 'Vai phải',
        'left_hip': 'Hông trái',
        'right_hip': 'Hông phải',
        'left_knee': 'Gối trái',
        'right_knee': 'Gối phải'
    };
    
    elements.anglesList.innerHTML = '';
    Object.entries(angles).forEach(([joint, angle]) => {
        const item = document.createElement('div');
        item.className = 'angle-item';
        
        const vietnameseName = jointNames[joint] || joint;
        
        item.innerHTML = `
            <span class="angle-name">${vietnameseName}</span>
            <span class="angle-value">${angle.toFixed(1)}°</span>
        `;
        
        elements.anglesList.appendChild(item);
    });
}

function updateFPS() {
    const now = Date.now();
    if (state.lastFrameTime > 0) {
        const delta = now - state.lastFrameTime;
        state.fps = (1000 / delta).toFixed(1);
        elements.fps.textContent = state.fps;
    }
    state.lastFrameTime = now;
}

function updateConnectionStatus(connected) {
    if (connected) {
        elements.statusIndicator.classList.add('connected');
        elements.statusText.textContent = 'Connected';
        elements.statusText.style.color = '#28a745';
    } else {
        elements.statusIndicator.classList.remove('connected');
        elements.statusText.textContent = 'Disconnected';
        elements.statusText.style.color = '#dc3545';
    }
}

// ==================== CAMERA FUNCTIONS ====================
async function startCamera() {
    try {
        log('📷 Đang khởi động camera...', 'info');
        
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };
        
        state.webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.webcam.srcObject = state.webcamStream;
        
        state.isCameraActive = true;
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        
        log('✅ Camera đã sẵn sàng!', 'success');
        
        // Start sending frames
        startSendingFrames();
        
    } catch (error) {
        log('❌ Không thể truy cập camera: ' + error.message, 'error');
    }
}

function stopCamera() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
        elements.webcam.srcObject = null;
        
        state.isCameraActive = false;
        elements.startBtn.disabled = false;
        elements.stopBtn.disabled = true;
        
        log('📷 Camera đã dừng', 'info');
        
        // Stop sending frames
        stopSendingFrames();
    }
}

function startSendingFrames() {
    const fps = parseInt(elements.fpsLimit.value) || CONFIG.defaultFPS;
    const interval = 1000 / fps;
    
    log(`🎬 Bắt đầu gửi frames (${fps} FPS)`, 'info');
    
    state.frameInterval = setInterval(() => {
        if (state.isConnected && state.isCameraActive) {
            captureAndSendFrame();
        }
    }, interval);
}

function stopSendingFrames() {
    if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
        log('⏸️ Đã dừng gửi frames', 'info');
    }
}

function captureAndSendFrame() {
    const canvas = elements.inputCanvas;
    const context = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = elements.webcam.videoWidth;
    canvas.height = elements.webcam.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(elements.webcam, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send to server
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        const message = {
            type: 'frame',
            image: imageData,
            timestamp: Date.now()
        };
        
        state.websocket.send(JSON.stringify(message));
    }
}

// ==================== EVENT LISTENERS ====================
elements.connectBtn.addEventListener('click', () => {
    if (state.isConnected) {
        disconnectWebSocket();
        if (state.isCameraActive) {
            stopCamera();
        }
    } else {
        connectWebSocket();
    }
});

elements.startBtn.addEventListener('click', () => {
    startCamera();
});

elements.stopBtn.addEventListener('click', () => {
    stopCamera();
});

elements.clearLogBtn.addEventListener('click', () => {
    elements.consoleLog.innerHTML = '';
    log('🗑️ Console đã được xóa', 'info');
});

// Update FPS when changed
elements.fpsLimit.addEventListener('change', () => {
    if (state.isCameraActive) {
        stopSendingFrames();
        startSendingFrames();
        log(`⚙️ FPS đã được cập nhật: ${elements.fpsLimit.value}`, 'info');
    }
});

// ==================== INITIALIZATION ====================
window.addEventListener('load', () => {
    log('🚀 Ứng dụng đã sẵn sàng!', 'success');
    log('👉 Nhấn "Connect to Server" để bắt đầu', 'info');
    
    // Set default server URL
    elements.serverUrl.value = CONFIG.defaultServerUrl;
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (state.isCameraActive) {
        stopCamera();
    }
    if (state.isConnected) {
        disconnectWebSocket();
    }
});
