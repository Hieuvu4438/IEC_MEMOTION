# 🔄 So sánh: test.py vs WebSocket Version

## Tổng quan

| Tiêu chí | test.py | WebSocket Version |
|----------|---------|-------------------|
| **Input** | 2 video files (user + reference) | Webcam trực tiếp |
| **Output** | Video so sánh + điểm số | Frame real-time + keypoints |
| **Giao diện** | OpenCV window | Web browser |
| **Chạy trên** | Local machine | Client-Server |
| **Mục đích** | So sánh 2 video yoga | Real-time pose detection |

## Chi tiết chức năng

### 1. Pose Detection (Phát hiện tư thế)

#### test.py
```python
class PoseEstimator:
    def __init__(self, mode='balanced', device='cpu'):
        self.pose_tracker = Wholebody(...)
    
    def estimate(self, frame):
        keypoints, scores = self.pose_tracker(frame)
        return kps, scrs
```

#### WebSocket Version
```python
class PoseEstimator:
    def __init__(self, mode='balanced', device='cpu'):
        self.pose_tracker = Wholebody(...)
    
    def estimate(self, frame):
        keypoints, scores = self.pose_tracker(frame)
        return kps, scrs
```

✅ **Giống nhau:** Cùng sử dụng RTMPose model, cùng cách detect

### 2. Keypoints Detection (17 điểm COCO)

#### test.py
```python
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

#### WebSocket Version
```python
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

✅ **Giống nhau:** Cùng 17 keypoints COCO format

### 3. Skeleton Drawing (Vẽ xương)

#### test.py
```python
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

def draw_skeleton(image, keypoints, scores, color, thickness):
    # Vẽ các đường nối giữa keypoints
    for idx1, idx2 in SKELETON_CONNECTIONS:
        cv2.line(img, pt1, pt2, color, thickness)
```

#### WebSocket Version
```python
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

def draw_skeleton_on_frame(frame, keypoints, scores):
    # Vẽ các đường nối giữa keypoints
    for idx1, idx2 in SKELETON_CONNECTIONS:
        cv2.line(img, pt1, pt2, COLORS['skeleton'], 3)
```

✅ **Giống nhau:** Cùng cách vẽ skeleton

### 4. Joint Angles (Góc khớp)

#### test.py
```python
JOINT_ANGLES = {
    'left_elbow': (5, 7, 9),
    'right_elbow': (6, 8, 10),
    'left_shoulder': (7, 5, 11),
    'right_shoulder': (8, 6, 12),
    'left_hip': (5, 11, 13),
    'right_hip': (6, 12, 14),
    'left_knee': (11, 13, 15),
    'right_knee': (12, 14, 16),
}

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
```

#### WebSocket Version
```python
JOINT_ANGLES = {
    'left_elbow': (5, 7, 9),
    'right_elbow': (6, 8, 10),
    'left_shoulder': (7, 5, 11),
    'right_shoulder': (8, 6, 12),
    'left_hip': (5, 11, 13),
    'right_hip': (6, 12, 14),
    'left_knee': (11, 13, 15),
    'right_knee': (12, 14, 16),
}

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def compute_joint_angles(keypoints):
    angles = {}
    for joint_name, (idx_a, idx_b, idx_c) in JOINT_ANGLES.items():
        angle = calculate_angle(keypoints[idx_a], keypoints[idx_b], keypoints[idx_c])
        angles[joint_name] = float(angle)
    return angles
```

✅ **Giống nhau:** Cùng logic tính góc

## Điểm khác biệt

### 1. Input Source

**test.py:**
- Đọc 2 video files từ ổ đĩa
- So sánh user video với reference video
- Xử lý offline

**WebSocket:**
- Webcam real-time
- Không có reference video
- Xử lý online

### 2. Chức năng chính

**test.py:**
- **Mục đích:** So sánh 2 video để đánh giá yoga pose
- **Output:** Video với 2 màn hình (user + reference)
- **Scoring:** Tính điểm similarity giữa 2 poses
- **Feedback:** Hiển thị lỗi góc, gợi ý cải thiện

**WebSocket:**
- **Mục đích:** Detect pose từ webcam real-time
- **Output:** Frame với skeleton overlay
- **Data:** Trả về keypoints + angles
- **Feedback:** Hiển thị trên web dashboard

### 3. Giao diện

**test.py:**
```python
# OpenCV window
cv2.imshow('Yoga Pose Comparison', viz_frame)
key = cv2.waitKey(wait_time)

# Controls
if key == ord('q'):  # Quit
if key == ord(' '):  # Pause
if key == ord('r'):  # Restart
```

**WebSocket:**
```html
<!-- Web browser -->
<button id="connectBtn">Connect</button>
<button id="startBtn">Start Camera</button>
<video id="webcam"></video>
<img id="outputImage">
```

### 4. Communication

**test.py:**
- Không có network communication
- All-in-one script
- Local processing

**WebSocket:**
- Client-Server architecture
- WebSocket protocol
- Network-based processing

### 5. Visualization

**test.py:**
```python
# Side-by-side comparison
combined = np.hstack([user_viz, ref_viz])

# Feedback panel on the right
draw_feedback_panel(combined, score, angle_diffs)

# Progress bar
cv2.rectangle(combined, (0, h-5), (int(w * progress), h), color, -1)
```

**WebSocket:**
```javascript
// Single frame with skeleton
<img id="outputImage" src="processed_frame">

// Separate info panels
<div class="keypoints-list">...</div>
<div class="angles-list">...</div>
<div class="stats">...</div>
```

## Code Structure

### test.py Flow

```
1. Load 2 videos
   ↓
2. Read frame by frame
   ↓
3. Detect poses in both frames
   ↓
4. Calculate similarity score
   ↓
5. Calculate angle differences
   ↓
6. Draw visualization
   ↓
7. Display comparison
   ↓
8. Save output video
```

### WebSocket Flow

```
Frontend:                Backend:
1. Capture webcam       
   ↓
2. Encode to base64
   ↓
3. Send via WebSocket ──→ 4. Receive frame
                           ↓
                         5. Decode image
                           ↓
                         6. Run pose estimation
                           ↓
                         7. Calculate angles
                           ↓
                         8. Draw skeleton
                           ↓
9. Display results  ←──── 9. Send response
   ↓
10. Update UI
```

## Điểm mạnh

### test.py
✅ So sánh 2 video chi tiết
✅ Đánh giá chất lượng yoga pose
✅ Feedback cụ thể về lỗi
✅ Lưu video kết quả
✅ Không cần server

### WebSocket
✅ Real-time từ webcam
✅ Web-based, dễ access
✅ Client-Server scalable
✅ Có thể remote access
✅ Modern UI/UX
✅ Multi-platform (browser)

## Kết luận

Cả 2 version đều:
- Sử dụng **cùng AI model** (RTMPose)
- Detect **cùng 17 keypoints** (COCO)
- Vẽ **cùng skeleton structure**
- Tính **cùng joint angles**
- Có **cùng core logic**

Khác biệt chính:
- **test.py:** Offline video comparison tool
- **WebSocket:** Real-time webcam detection system

WebSocket version là **adaptation** của test.py logic cho real-time use case!
