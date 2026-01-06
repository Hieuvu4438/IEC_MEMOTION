# 🎯 test.py Logic Implementation Guide

## Overview

This document explains how the logic from `yoga_therapy/src/core/test.py` was implemented in the WebSocket version.

## Core Components from test.py

### 1. PoseEstimator Class

**test.py:**
```python
class PoseEstimator:
    """Wrapper cho rtmlib Wholebody."""
    
    def __init__(self, mode='balanced', device='cpu'):
        self.pose_tracker = Wholebody(
            to_openpose=False,
            mode=mode,
            backend='onnxruntime',
            device=device
        )
    
    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pose từ frame, trả về keypoints và scores."""
        keypoints, scores = self.pose_tracker(frame)
        
        if len(keypoints) > 0:
            kps = keypoints[0][:17]
            scrs = scores[0][:17] if len(scores) > 0 else np.ones(17)
            return kps, scrs.flatten()[:17]
        
        return np.zeros((17, 2)), np.zeros(17)
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
class PoseEstimator:
    """AI Model wrapper for pose estimation using rtmlib."""
    
    def __init__(self, mode='balanced', device='cpu'):
        self.pose_tracker = Wholebody(
            to_openpose=False,
            mode=mode,
            backend='onnxruntime',
            device=device
        )
    
    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pose from frame."""
        keypoints, scores = self.pose_tracker(frame)
        
        if len(keypoints) > 0:
            kps = keypoints[0][:17]
            scrs = scores[0][:17] if len(scores) > 0 else np.ones(17)
            return kps, scrs.flatten()[:17]
        
        return np.zeros((17, 2)), np.zeros(17)
```

**Status:** ✅ **IDENTICAL** - Same implementation

---

### 2. Keypoint Definitions

**test.py:**
```python
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
```

**Status:** ✅ **IDENTICAL** - Same 17 keypoints

---

### 3. Skeleton Connections

**test.py:**
```python
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]
```

**Status:** ✅ **IDENTICAL** - Same skeleton structure

---

### 4. Joint Angles Definition

**test.py:**
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
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
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
```

**Status:** ✅ **IDENTICAL** - Same 8 joint definitions

---

### 5. Angle Calculation

**test.py:**
```python
def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Tính góc tại p2 giữa p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 between p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
```

**Status:** ✅ **IDENTICAL** - Same mathematical formula

---

### 6. Computing All Joint Angles

**test.py:**
```python
def compute_angle_differences(user_kps: np.ndarray, ref_kps: np.ndarray) -> Dict[str, Dict]:
    """Tính sự khác biệt góc giữa các khớp."""
    angle_diffs = {}
    
    for joint_name, (idx_a, idx_b, idx_c) in JOINT_ANGLES.items():
        user_angle = calculate_angle(user_kps[idx_a], user_kps[idx_b], user_kps[idx_c])
        ref_angle = calculate_angle(ref_kps[idx_a], ref_kps[idx_b], ref_kps[idx_c])
        
        diff = abs(user_angle - ref_angle)
        
        angle_diffs[joint_name] = {
            'user': user_angle,
            'reference': ref_angle,
            'difference': diff,
            'is_error': diff > 15
        }
    
    return angle_diffs
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
def compute_joint_angles(keypoints: np.ndarray) -> Dict[str, float]:
    """Compute angles for all joints."""
    angles = {}
    for joint_name, (idx_a, idx_b, idx_c) in JOINT_ANGLES.items():
        angle = calculate_angle(
            keypoints[idx_a], 
            keypoints[idx_b], 
            keypoints[idx_c]
        )
        angles[joint_name] = float(angle)
    return angles
```

**Status:** ⚡ **ADAPTED** - Same logic, but returns angles only (no comparison)

---

### 7. Skeleton Drawing

**test.py:**
```python
def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    conf_threshold: float = 0.3
) -> np.ndarray:
    """Vẽ skeleton lên ảnh."""
    img = image.copy()
    
    # Draw skeleton lines
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if scores[idx1] < conf_threshold or scores[idx2] < conf_threshold:
            continue
        
        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
        
        if pt1[0] < 0 or pt1[1] < 0 or pt2[0] < 0 or pt2[1] < 0:
            continue
        
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score < conf_threshold:
            continue
        x, y = int(pt[0]), int(pt[1])
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), 4, color, -1)
    
    return img
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float = 0.3
) -> np.ndarray:
    """Draw skeleton and keypoints on frame."""
    img = frame.copy()
    
    # Draw skeleton lines
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if scores[idx1] < conf_threshold or scores[idx2] < conf_threshold:
            continue
        
        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
        
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(img, pt1, pt2, COLORS['skeleton'], 3)
    
    # Draw keypoints
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score < conf_threshold:
            continue
        x, y = int(pt[0]), int(pt[1])
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), 5, COLORS['keypoint'], -1)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    
    return img
```

**Status:** ✅ **NEARLY IDENTICAL** - Same logic, slightly different styling

---

### 8. Color Definitions

**test.py:**
```python
COLORS = {
    'user_skeleton': (0, 255, 0),       # Green
    'ref_skeleton': (255, 165, 0),      # Orange
    'keypoint': (255, 0, 0),            # Blue
    'error': (0, 0, 255),               # Red
    'correct': (0, 255, 0),             # Green
}
```

**WebSocket Implementation:**
```python
# websocket_backend/server.py
COLORS = {
    'skeleton': (0, 255, 0),      # Green
    'keypoint': (255, 0, 0),      # Blue
    'error_joint': (0, 0, 255),   # Red
}
```

**Status:** ✅ **SIMILAR** - Same color scheme, adapted for single pose

---

## Workflow Comparison

### test.py Workflow:

```
1. Load two videos (user + reference)
   ↓
2. Loop through frames
   ↓
3. Detect poses in both frames
   ↓
4. Normalize keypoints
   ↓
5. Calculate similarity score
   ↓
6. Calculate angle differences
   ↓
7. Draw both skeletons side-by-side
   ↓
8. Draw feedback panel
   ↓
9. Display and save result
```

### WebSocket Workflow:

```
1. Receive frame from webcam (via WebSocket)
   ↓
2. Decode base64 to image
   ↓
3. Detect pose
   ↓
4. Calculate joint angles
   ↓
5. Draw skeleton on frame
   ↓
6. Encode to base64
   ↓
7. Send frame + keypoints + angles
   ↓
8. Display on web interface
```

---

## Implementation Summary

| Component | test.py | WebSocket | Status |
|-----------|---------|-----------|--------|
| **PoseEstimator** | ✅ | ✅ | IDENTICAL |
| **17 COCO Keypoints** | ✅ | ✅ | IDENTICAL |
| **Skeleton Connections** | ✅ | ✅ | IDENTICAL |
| **8 Joint Angles** | ✅ | ✅ | IDENTICAL |
| **Angle Calculation** | ✅ | ✅ | IDENTICAL |
| **Skeleton Drawing** | ✅ | ✅ | NEARLY IDENTICAL |
| **Color Scheme** | ✅ | ✅ | SIMILAR |
| **Input Source** | 2 Videos | Webcam | DIFFERENT |
| **Output Format** | Video File | WebSocket JSON | DIFFERENT |
| **User Interface** | OpenCV Window | Web Browser | DIFFERENT |
| **Comparison Logic** | Yes (2 poses) | No (single pose) | DIFFERENT |

---

## Key Differences Explained

### 1. **Input Source**

**test.py:**
- Reads pre-recorded video files
- Compares two videos simultaneously
- Offline processing

**WebSocket:**
- Real-time webcam feed
- Single video stream
- Online processing

### 2. **Pose Comparison**

**test.py:**
- Compares user pose vs reference pose
- Calculates difference in angles
- Provides feedback on errors

**WebSocket:**
- Single pose detection only
- Returns absolute angle values
- No comparison logic (can be added later)

### 3. **Output Format**

**test.py:**
- Saves output as video file
- Side-by-side visualization
- Feedback panel overlay

**WebSocket:**
- Sends JSON over WebSocket
- Single frame display
- Separate UI panels for info

### 4. **Visualization**

**test.py:**
```python
# Side by side comparison
combined = np.hstack([user_viz, ref_viz])
combined = draw_feedback_panel(combined, score, angle_diffs)
```

**WebSocket:**
```python
# Single frame with skeleton
viz_frame = draw_skeleton_on_frame(frame, keypoints, scores)
# Send as base64 + separate JSON data
```

---

## What Was Kept from test.py

✅ **Core AI Logic:**
- PoseEstimator class
- RTMPose model usage
- 17 keypoint detection
- COCO format

✅ **Mathematical Functions:**
- calculate_angle()
- Same formula for angle calculation
- Same joint definitions

✅ **Visualization:**
- Skeleton drawing logic
- Keypoint rendering
- Color scheme
- Confidence thresholds

✅ **Data Structures:**
- COCO_KEYPOINTS list
- SKELETON_CONNECTIONS list
- JOINT_ANGLES dictionary
- COLORS dictionary

---

## What Was Adapted

⚡ **Architecture:**
- Converted to client-server model
- Added WebSocket communication
- Split into backend/frontend

⚡ **Input/Output:**
- Changed from file I/O to network I/O
- Added base64 encoding/decoding
- Changed from video to streaming

⚡ **Functionality:**
- Removed comparison logic (for now)
- Removed normalization (not needed for single pose)
- Removed feedback panel (moved to web UI)

⚡ **User Interface:**
- Changed from OpenCV window to web browser
- Added HTML/CSS/JavaScript
- Added interactive controls

---

## Conclusion

The WebSocket version **preserves 100% of the core pose estimation logic** from test.py:

✅ Same AI model (RTMPose)
✅ Same keypoint detection (17 COCO)
✅ Same skeleton structure
✅ Same angle calculation
✅ Same visualization approach

The main differences are in:
- Input source (files → webcam)
- Architecture (monolithic → client-server)
- UI (OpenCV → Web browser)
- Functionality (comparison → single detection)

**The core pose estimation logic is identical!**
