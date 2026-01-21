# 📁 MEMOTION Backend Engine (`mediapipe_be`)

## 🎯 Tổng Quan

**`mediapipe_be`** là phiên bản **Backend-ready** của hệ thống MEMOTION, được thiết kế để tích hợp dễ dàng với các framework backend như **FastAPI**, **Flask**, **WebSocket Server**, etc.

### Điểm khác biệt với `mediapipe/`

| Aspect | `mediapipe/` (Original) | `mediapipe_be/` (Backend) |
|--------|------------------------|---------------------------|
| **UI** | Có UI (OpenCV window) | ❌ Không UI |
| **Output** | Hiển thị trực tiếp | JSON-serializable |
| **Entry Point** | `main_v2.py` | `EngineService` class |
| **Mục đích** | Demo/Test | Production Backend |

---

## 📂 Cấu Trúc Thư Mục

```
mediapipe_be/
│
├── 🔌 service/                  # LỚP GIAO TIẾP BACKEND (MỚI)
│   ├── __init__.py              # Export các class/function
│   ├── engine_service.py        # ⭐ Class chính - thay thế main_v2.py
│   └── schemas.py               # Định nghĩa cấu trúc JSON output
│
├── 🧠 core/                     # Logic cốt lõi (giữ nguyên từ mediapipe/)
│   ├── data_types.py            # Kiểu dữ liệu
│   ├── detector.py              # MediaPipe detector
│   ├── kinematics.py            # Tính góc khớp
│   ├── procrustes.py            # Chuẩn hóa skeleton
│   ├── synchronizer.py          # FSM đồng bộ
│   └── dtw_analysis.py          # DTW analysis
│
├── 🔧 modules/                  # Business logic (giữ nguyên)
│   ├── calibration.py           # Calibration
│   ├── pain_detection.py        # Phát hiện đau
│   ├── scoring.py               # Chấm điểm
│   ├── target_generator.py      # Cá nhân hóa mục tiêu
│   └── video_engine.py          # Video player
│
├── 🛠️ utils/                    # Tiện ích (giữ nguyên)
│   ├── logger.py                # Ghi log
│   └── visualization.py         # Vẽ (không dùng trong backend)
│
├── 📦 models/                   # Model files MediaPipe
│   ├── face_landmarker.task
│   ├── pose_landmarker_full.task
│   └── pose_landmarker_lite.task
│
├── 🎬 assets/                   # Video mẫu, config
│
├── 📖 bridge_example.py         # ⭐ File mẫu hướng dẫn tích hợp
└── folder_structure.txt         # Mô tả cấu trúc
```

---

## 🔌 Folder `service/` - Lớp Giao Tiếp Backend

### 1. `engine_service.py` - Class Chính

```python
"""
EngineService - Xử lý frame và quản lý trạng thái.

Backend chỉ cần:
1. Khởi tạo class MỘT LẦN
2. Gọi process_frame() trong mỗi vòng lặp
3. Nhận kết quả JSON-serializable
"""
```

#### Các Class/Function Chính:

| Class/Function | Mô tả |
|----------------|-------|
| `EngineConfig` | Cấu hình engine (models_dir, log_dir, etc.) |
| `EngineState` | Trạng thái nội bộ (phase, scores, angles, etc.) |
| `EngineService` | **Class chính** - xử lý frame và trả JSON |
| `AppPhase` | Enum các phase (PHASE1_DETECTION → COMPLETED) |

#### Cách Sử Dụng:

```python
from service import EngineService, EngineConfig

# 1. Khởi tạo (1 lần khi server start)
config = EngineConfig(
    models_dir="./models",
    log_dir="./data/logs",
    default_joint="left_shoulder",
)
engine = EngineService(config)
engine.initialize()

# 2. Xử lý frame (mỗi frame từ client)
result = engine.process_frame(frame, timestamp_ms)

# 3. Convert to JSON và gửi về Frontend
json_data = result.to_dict()
send_to_frontend(json_data)
```

---

### 2. `schemas.py` - Cấu Trúc JSON Output

Định nghĩa các dataclass để chuẩn hóa dữ liệu trả về. **Tất cả đều JSON-serializable**.

#### Output theo Phase:

```
┌─────────────────────────────────────────────────────────────┐
│                      EngineOutput                           │
│  {                                                          │
│    "current_phase": 1-4,                                    │
│    "phase_name": "detection|calibration|sync|scoring",      │
│    "detection": DetectionOutput,    // Phase 1              │
│    "calibration": CalibrationOutput, // Phase 2             │
│    "sync": SyncOutput,              // Phase 3              │
│    "final_report": FinalReportOutput // Phase 4             │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

#### Phase 1: `DetectionOutput` - Nhận Diện Tư Thế

```json
{
  "pose_detected": true,
  "stable_count": 25,
  "progress": 0.83,
  "countdown_remaining": 2.5,
  "status": "countdown",
  "message": "Chuan bi... 3 giay"
}
```

| Field | Type | Mô tả |
|-------|------|-------|
| `pose_detected` | bool | Đã phát hiện pose ổn định (30 frames) |
| `stable_count` | int | Số frame ổn định (0-30) |
| `progress` | float | Tiến trình (0.0-1.0) |
| `countdown_remaining` | float? | Countdown còn lại (giây) |
| `status` | string | `idle`, `detecting`, `countdown`, `transitioning` |
| `message` | string | Thông báo cho user |

---

#### Phase 2: `CalibrationOutput` - Đo Giới Hạn Vận Động

```json
{
  "current_joint": "left_shoulder",
  "current_joint_name": "Vai trai",
  "queue_index": 0,
  "total_joints": 6,
  "progress": 0.65,
  "overall_progress": 0.11,
  "current_angle": 95.5,
  "user_max_angle": 102.3,
  "countdown_remaining": 3.2,
  "status": "collecting",
  "position_instruction": "Moi ba dung NGANG",
  "joints_status": [
    {"joint_name": "Vai trai", "joint_type": "left_shoulder", "max_angle": 102.3, "status": "collecting"},
    {"joint_name": "Vai phai", "joint_type": "right_shoulder", "max_angle": null, "status": "pending"},
    ...
  ],
  "message": "Dang do khop Vai trai..."
}
```

| Field | Type | Mô tả |
|-------|------|-------|
| `current_joint` | string | Khớp đang đo (e.g., `left_shoulder`) |
| `current_joint_name` | string | Tên tiếng Việt (e.g., `Vai trai`) |
| `queue_index` | int | Vị trí trong queue (0-5) |
| `total_joints` | int | Tổng số khớp (6) |
| `progress` | float | Tiến trình khớp hiện tại (0.0-1.0) |
| `overall_progress` | float | Tiến trình tổng thể (0.0-1.0) |
| `current_angle` | float | Góc hiện tại |
| `user_max_angle` | float | Góc max đã ghi nhận |
| `countdown_remaining` | float? | Countdown còn lại |
| `status` | string | `preparing`, `collecting`, `complete`, `all_complete` |
| `position_instruction` | string | Hướng dẫn tư thế |
| `joints_status` | array | Danh sách trạng thái 6 khớp |

---

#### Phase 3: `SyncOutput` - Đồng Bộ Chuyển Động

```json
{
  "user_angle": 85.5,
  "target_angle": 90.0,
  "error": 4.5,
  "current_score": 92.5,
  "average_score": 88.3,
  "motion_phase": "hold",
  "rep_count": 3,
  "video_progress": 0.45,
  "video_paused": false,
  "pain_level": "NONE",
  "fatigue_level": "FRESH",
  "joint_errors": [
    {
      "joint_name": "Vai trai",
      "joint_type": "left_shoulder",
      "user_angle": 85.5,
      "target_angle": 90.0,
      "error": 4.5,
      "error_percent": 5.0,
      "score": 92.5,
      "direction_hint": "raise",
      "weight": 0.8
    }
  ],
  "active_joints_count": 6,
  "feedback_text": "TOT!",
  "direction_hint": "raise",
  "warning": null,
  "status": "syncing"
}
```

| Field | Type | Mô tả |
|-------|------|-------|
| `user_angle` | float | Góc hiện tại của primary joint |
| `target_angle` | float | Góc mục tiêu |
| `error` | float | Sai số (độ) |
| `current_score` | float | Điểm hiện tại (0-100) |
| `average_score` | float | Điểm trung bình |
| `motion_phase` | string | `idle`, `eccentric`, `hold`, `concentric` |
| `rep_count` | int | Số rep đã hoàn thành |
| `pain_level` | string | `NONE`, `MILD`, `MODERATE`, `SEVERE` |
| `fatigue_level` | string | `FRESH`, `LIGHT`, `MODERATE`, `HEAVY` |
| `joint_errors` | array | Chi tiết sai số từng khớp |
| `direction_hint` | string | `raise`, `lower`, `hold`, `ok` |
| `feedback_text` | string | `TUYET VOI!`, `TOT!`, `KHA`, `DIEU CHINH!` |

---

#### Phase 4: `FinalReportOutput` - Báo Cáo Cuối

```json
{
  "session_id": "session_20260121_154500",
  "exercise_name": "Arm Raise",
  "duration_seconds": 180,
  "total_score": 85.5,
  "rom_score": 88.0,
  "stability_score": 82.0,
  "flow_score": 86.5,
  "grade": "XUAT SAC",
  "grade_color": "green",
  "total_reps": 10,
  "fatigue_level": "LIGHT",
  "calibrated_joints": [
    {"joint_name": "Vai trai", "joint_type": "left_shoulder", "max_angle": 102.3},
    ...
  ],
  "primary_joint": "left_shoulder",
  "primary_max_angle": 102.3,
  "rep_scores": [
    {"rep_number": 1, "rom_score": 90.0, "stability_score": 85.0, "flow_score": 88.0, "total_score": 87.7, "duration_ms": 5000},
    ...
  ],
  "recommendations": [
    "Tiếp tục duy trì tư thế tốt",
    "Có thể tăng nhẹ số rep trong buổi sau"
  ],
  "start_time": "2026-01-21T15:45:00",
  "end_time": "2026-01-21T15:48:00"
}
```

---

## 🔗 Tích Hợp Backend

### Luồng Dữ Liệu

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Camera    │────▶│  WebSocket/API  │────▶│  EngineService  │
│  (Client)   │     │    (Server)     │     │                 │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Frontend   │◀────│  JSON Response  │◀────│  EngineOutput   │
│    (UI)     │     │                 │     │  .to_dict()     │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

### Ví Dụ FastAPI + WebSocket

```python
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64

from service import EngineService, EngineConfig

app = FastAPI()
engine: EngineService = None

@app.on_event("startup")
async def startup():
    global engine
    config = EngineConfig(models_dir="./models")
    engine = EngineService(config)
    engine.initialize()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Nhận frame từ client (base64)
            data = await websocket.receive_json()
            frame_b64 = data["frame"]
            timestamp_ms = data["timestamp_ms"]
            
            # Decode frame
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Xử lý
            result = engine.process_frame(frame, timestamp_ms)
            
            # Gửi JSON về client
            await websocket.send_json(result.to_dict())
            
    except Exception as e:
        print(f"WebSocket error: {e}")
```

---

## 📖 Hướng Dẫn Đọc Code

### Thứ Tự Đọc

```
1. bridge_example.py          → Xem cách sử dụng EngineService
2. service/schemas.py         → Hiểu cấu trúc JSON output
3. service/engine_service.py  → Hiểu logic xử lý từng phase
4. core/*, modules/*          → Chi tiết thuật toán (nếu cần)
```

### Mapping Phase → Method

| Phase | Method trong `EngineService` |
|-------|------------------------------|
| Phase 1 | `_process_phase1()` |
| Phase 2 | `_process_phase2()` |
| Phase 3 | `_process_phase3()` |
| Phase 4 | `_process_phase4()`, `_generate_final_report()` |

---

## 🚀 Quick Start

### 1. Chạy File Mẫu

```bash
cd mediapipe_be
python bridge_example.py
```

### 2. Các Chế Độ Chạy

```python
# Simulation mode (có sẵn trong bridge_example.py)
simulate_video_processing()

# Test với webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = engine.process_frame(frame, timestamp_ms)
    print(result.to_dict())
```

---

## 📝 Ghi Chú

### Helper Functions trong `schemas.py`

```python
# Xác định hướng điều chỉnh
get_direction_hint(user_angle, target_angle, tolerance=10.0)
# Returns: "raise" | "lower" | "hold" | "ok"

# Lấy feedback text
get_feedback_text(error, target_angle)
# Returns: "TUYET VOI!" | "TOT!" | "KHA" | "DIEU CHINH!"

# Lấy grade
get_grade(score)
# Returns: ("XUAT SAC", "green") | ("KHA", "yellow") | ("CAN CO GANG", "red")

# Lấy tên khớp tiếng Việt
get_joint_name_vi("left_shoulder")
# Returns: "Vai trai"
```

### Constants Quan Trọng

```python
# Thứ tự calibration 6 khớp
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,
    JointType.RIGHT_SHOULDER,
    JointType.LEFT_ELBOW,
    JointType.RIGHT_ELBOW,
    JointType.LEFT_KNEE,
    JointType.RIGHT_KNEE,
]

# Hướng dẫn tư thế
JOINT_POSITION_INSTRUCTIONS = {
    JointType.LEFT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    ...
}

# Timing
PHASE1_COUNTDOWN_DURATION = 3.0  # giây
CALIBRATION_COUNTDOWN_DURATION = 5.0  # giây
```

---

## 🔧 Cấu Hình

```python
@dataclass
class EngineConfig:
    models_dir: str = "./models"           # Đường dẫn models
    log_dir: str = "./data/logs"           # Đường dẫn logs
    ref_video_path: str = None             # Video mẫu (Phase 3)
    default_joint: str = "left_shoulder"   # Khớp mặc định
    detection_stable_threshold: int = 30   # Số frame stable (Phase 1)
    calibration_duration_ms: int = 5000    # Thời gian đo mỗi khớp (Phase 2)
```

---

## 📞 Liên Hệ

**MEMOTION Team**  
Version: 1.0.0 (Backend Ready)
