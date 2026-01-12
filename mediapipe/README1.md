# MEMOTION - Rehabilitation Support System

> Hệ thống hỗ trợ phục hồi chức năng cho người già sử dụng Computer Vision

## 🎯 Tổng quan

MEMOTION sử dụng MediaPipe Tasks API để:
1. **Pose Detection**: Nhận diện 33 điểm khung xương
2. **Face Detection**: Nhận diện 478 điểm khuôn mặt (cho FACS analysis)
3. **Procrustes Analysis**: Chuẩn hóa skeleton loại bỏ ảnh hưởng của khoảng cách, kích thước

## 📁 Cấu trúc dự án

```
memotion/
├── core/
│   ├── __init__.py
│   ├── data_types.py      # Data Classes & Type Definitions
│   ├── detector.py        # VisionDetector wrapper
│   └── procrustes.py      # Procrustes Analysis
├── modules/               # (Phase 2+)
│   ├── calibration.py
│   ├── pain_detection.py
│   └── scoring.py
├── utils/
│   ├── logger.py
│   └── visualization.py
├── models/                # Model files (.task)
├── tests/
├── main_test.py          # Test script
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt

### 1. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Tải model files

Tạo thư mục `models/` và tải các file sau:

**Pose Landmarker** (bắt buộc):
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

**Face Landmarker** (tùy chọn, cho FACS):
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 📖 Sử dụng

### Test với Webcam
```bash
python main_test.py --source webcam
```

### Test với Video
```bash
python main_test.py --source path/to/video.mp4
```

### Test với Ảnh
```bash
python main_test.py --source path/to/image.jpg --mode image
```

### Chạy Unit Tests
```bash
python main_test.py --mode test
```

## 🧪 API Reference

### VisionDetector

```python
from core import VisionDetector, DetectorConfig

config = DetectorConfig(
    pose_model_path="models/pose_landmarker_lite.task",
    min_pose_detection_confidence=0.5,
)

with VisionDetector(config) as detector:
    result = detector.process_frame(frame, timestamp_ms=0)
    
    if result.has_pose():
        # Lấy numpy array (33, 3)
        skeleton = result.pose_landmarks.to_numpy()
```

### Procrustes Analysis

```python
from core import (
    compute_procrustes_distance,
    compute_procrustes_similarity,
    align_skeleton_to_reference,
)

# So sánh hai tư thế
disparity = compute_procrustes_distance(skeleton_a, skeleton_b)
similarity = compute_procrustes_similarity(skeleton_a, skeleton_b)

# Căn chỉnh skeleton
result = align_skeleton_to_reference(target, reference)
aligned = result.aligned_skeleton.landmarks
```

## 🔑 Key Controls (Webcam Test)

| Key | Action |
|-----|--------|
| `c` | Capture reference pose |
| `r` | Reset reference pose |
| `q` | Quit |

## 📊 Output Metrics

- **Disparity**: Khoảng cách Procrustes (0 = khớp hoàn toàn)
- **Similarity**: Độ tương đồng (0-100%)

## 🔮 Roadmap

- [x] Phase 1: Pose Detection + Procrustes
- [ ] Phase 2: Motion Synchronization (DTW)
- [ ] Phase 3: Pain Detection (FACS)
- [ ] Phase 4: Scoring System
- [ ] Phase 5: Flutter Integration

## 📝 License

MIT License