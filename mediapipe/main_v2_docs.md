# 📖 MEMOTION V2.0 - TÀI LIỆU CHI TIẾT

> **File**: `main_v2.py`  
> **Version**: 2.0.0  
> **Author**: MEMOTION Team  
> **Last Updated**: 2026-01-21  
> **Total Lines**: 1788

---

## 📋 MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc ứng dụng](#2-kiến-trúc-ứng-dụng)
3. [4 Giai đoạn hoạt động](#3-bốn-giai-đoạn-hoạt-động)
4. [Data Structures](#4-data-structures)
5. [Class MemotionAppV2](#5-class-memotionappv2)
6. [Luồng hoạt động chi tiết](#6-luồng-hoạt-động-chi-tiết)
7. [Phím điều khiển](#7-phím-điều-khiển)
8. [Công thức tính điểm](#8-công-thức-tính-điểm)
9. [Hướng dẫn sử dụng](#9-hướng-dẫn-sử-dụng)
10. [Unit Tests](#10-unit-tests)

---

## 1. TỔNG QUAN

### 1.1 Mục đích
`main_v2.py` là **phiên bản nâng cấp** của ứng dụng MEMOTION, cung cấp:
- Giao diện người dùng (UI) rõ ràng hơn
- Luồng 4 phase được tách biệt với **AUTO TRANSITION**
- **Automated Calibration** - tự động đo 6 khớp
- **Multi-joint tracking** - theo dõi và tính điểm nhiều khớp cùng lúc
- Real-time scoring với visual feedback
- Interpolated target angle cho tracking mượt mà hơn

### 1.2 Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| **4 Phases** | Pose Detection → Calibration → Motion Sync → Scoring |
| **Auto Transition** | Tự động chuyển phase không cần nhấn ENTER |
| **Automated Calibration** | Tự động đo 6 khớp theo thứ tự định sẵn |
| **Multi-joint Tracking** | Theo dõi và tính điểm tất cả khớp đã calibrate |
| **Weighted Scoring** | Điểm có trọng số theo loại bài tập |
| **Real-time Scoring** | Tính điểm ngay lập tức dựa trên sai số góc |
| **Visual Feedback** | Hiển thị màu sắc và text phản hồi |
| **Target Interpolation** | Target angle liên tục thay vì từng checkpoint |
| **Vietnamese UI** | Giao diện tiếng Việt (không dấu) |

### 1.3 Dependencies

```python
# Core dependencies
import argparse, sys, os, time, threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from queue import Queue
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2

# Internal modules
from core import (
    VisionDetector, DetectorConfig, JointType, JOINT_DEFINITIONS,
    calculate_joint_angle, MotionPhase, SyncStatus, SyncState,
    MotionSyncController, create_arm_raise_exercise, create_elbow_flex_exercise,
    compute_single_joint_dtw, PoseLandmarkIndex, create_exercise_weights,
)
from modules import (
    VideoEngine, PlaybackState, PainDetector, PainLevel,
    HealthScorer, FatigueLevel, SafeMaxCalibrator, CalibrationState,
    UserProfile,
)
from utils import (
    SessionLogger, put_vietnamese_text, draw_skeleton, draw_panel,
    draw_progress_bar, draw_phase_indicator, COLORS, draw_angle_arc,
    combine_frames_horizontal,
)
```

---

## 2. KIẾN TRÚC ỨNG DỤNG

### 2.1 Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MEMOTION V2.0                                    │
│                           (main_v2.py)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AppState (Dataclass)                         │   │
│  │  - current_phase: AppPhase                                           │   │
│  │  - is_running, is_paused                                             │   │
│  │  - Phase 1-4 specific states                                         │   │
│  │  - user_angle, target_angle, scores, etc.                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MemotionAppV2 (Class)                         │   │
│  │                                                                       │   │
│  │   Components:                    Methods:                             │   │
│  │   ├─ VisionDetector             ├─ run()                             │   │
│  │   ├─ VideoEngine                ├─ _run_phase1()                     │   │
│  │   ├─ MotionSyncController       ├─ _run_phase2()                     │   │
│  │   ├─ SafeMaxCalibrator          ├─ _run_phase3()                     │   │
│  │   ├─ PainDetector               ├─ _run_phase4()                     │   │
│  │   ├─ HealthScorer               ├─ _handle_key()                     │   │
│  │   └─ SessionLogger              └─ _transition_to_phaseX()           │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 State Machine - AppPhase

```
┌──────────────────────────────────────────────────────────────────────┐
│                    APPLICATION PHASES (AUTO TRANSITION)              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌───────────────┐                                                  │
│   │    PHASE1     │   Pose Detection                                 │
│   │   DETECTION   │   - Nhận diện skeleton                           │
│   │               │   - Đợi stable 30 frames                         │
│   └───────┬───────┘                                                  │
│           │ [AUTO] Countdown 3 giây khi pose_detected                │
│           ▼                                                          │
│   ┌───────────────┐                                                  │
│   │    PHASE2     │   Automated Calibration                          │
│   │  CALIBRATION  │   - Tự động đo 6 khớp theo thứ tự                │
│   │  (AUTOMATED)  │   - Countdown 5 giây cho mỗi khớp                │
│   └───────┬───────┘                                                  │
│           │ [AUTO] 2 giây sau khi đo xong 6 khớp                     │
│           ▼                                                          │
│   ┌───────────────┐                                                  │
│   │    PHASE3     │   Motion Sync (Multi-joint)                      │
│   │     SYNC      │   - Đồng bộ với video mẫu                        │
│   │ (MULTI-JOINT) │   - Tính điểm real-time cho TẤT CẢ khớp          │
│   └───────┬───────┘                                                  │
│           │ [AUTO] Khi video kết thúc / SyncStatus.COMPLETE          │
│           ▼                                                          │
│   ┌───────────────┐                                                  │
│   │    PHASE4     │   Scoring                                        │
│   │   SCORING     │   - Hiển thị kết quả                             │
│   │               │   - Lưu báo cáo                                  │
│   └───────┬───────┘                                                  │
│           │ [Q] hoặc [R]                                             │
│           ▼                                                          │
│   ┌───────────────┐                                                  │
│   │   COMPLETED   │   Kết thúc                                       │
│   └───────────────┘                                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. BỐN GIAI ĐOẠN HOẠT ĐỘNG

### 3.1 PHASE 1: Pose Detection (AUTO TRANSITION)

**Mục đích**: Nhận diện tư thế người dùng, đảm bảo MediaPipe detect được skeleton ổn định.

**Logic hoạt động**:
```python
PHASE1_COUNTDOWN_DURATION = 3.0  # 3 giây countdown

def _run_phase1(self, frame, result):
    # 1. Hiển thị hướng dẫn
    # 2. Kiểm tra result.has_pose()
    # 3. Nếu có pose:
    #    - Vẽ skeleton
    #    - Tăng detection_stable_count
    #    - Nếu đủ 30 frames → pose_detected = True
    #    - Bắt đầu countdown 3 giây
    #    - Khi countdown kết thúc → TỰ ĐỘNG _transition_to_phase2()
    # 4. Nếu mất pose: reset countdown
```

**UI Elements**:
```
┌────────────────────────────────────────────┐
│ GIAI DOAN 1: NHAN DIEN TU THE              │
│                                            │
│   Hay dung truoc camera...                 │
│   Dam bao toan than...                     │
│   Dung yen cho den khi...                  │
│   He thong se tu dong chuyen sang Phase 2  │
│                                            │
│   [========>          ] 45%                │
│   Dang xac nhan...                         │
│                                            │
│   (Khi pose_detected):                     │
│           ┌───┐                            │
│           │ 3 │  <-- Countdown lớn         │
│           └───┘                            │
│   Dung yen, chuan bi do gioi han...        │
└────────────────────────────────────────────┘
```

**Điều kiện chuyển Phase 2**:
- `pose_detected == True` (sau 30 frames stable)
- **TỰ ĐỘNG** sau countdown 3 giây
- Hoặc nhấn `ENTER` để bỏ qua countdown (manual override)

---

### 3.2 PHASE 2: Automated Calibration (6 khớp)

**Mục đích**: Tự động đo giới hạn vận động (Range of Motion) an toàn của 6 khớp.

**Calibration Queue** (thứ tự đo):
```python
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,   # 1. Vai trái
    JointType.RIGHT_SHOULDER,  # 2. Vai phải
    JointType.LEFT_ELBOW,      # 3. Khuỷu tay trái
    JointType.RIGHT_ELBOW,     # 4. Khuỷu tay phải
    JointType.LEFT_KNEE,       # 5. Đầu gối trái
    JointType.RIGHT_KNEE,      # 6. Đầu gối phải
]

CALIBRATION_COUNTDOWN_DURATION = 5.0  # 5 giây chuẩn bị mỗi khớp

# Hướng dẫn tư thế theo loại khớp
JOINT_POSITION_INSTRUCTIONS = {
    JointType.LEFT_SHOULDER: "Moi ba dung NGANG",
    JointType.RIGHT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_ELBOW: "Moi ba dung NGANG",
    JointType.RIGHT_ELBOW: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    JointType.RIGHT_KNEE: "Moi ba dung DOC",
}
```

**Logic hoạt động**:
```python
def _run_phase2(self, frame, result, timestamp_ms):
    # 1. Lấy khớp hiện tại từ CALIBRATION_QUEUE[queue_index]
    # 2. Nếu all_joints_calibrated:
    #    - Hiển thị kết quả
    #    - Tự động chuyển Phase 3 sau 2 giây
    # 3. Nếu chưa bắt đầu countdown cho khớp:
    #    - Bắt đầu countdown 5 giây
    # 4. Nếu đang countdown:
    #    - Hiển thị hướng dẫn tư thế
    #    - Hiển thị số đếm ngược
    #    - Khi countdown hết → bắt đầu đo
    # 5. Nếu đang đo (is_calibrating_joint):
    #    - Thu thập góc từ mỗi frame
    #    - Khi calibrator COMPLETED → lưu và chuyển khớp tiếp
```

**Quy trình tự động cho MỖI khớp**:
```
[Khớp N trong queue]
        │
        ▼
┌─────────────────────────────────┐
│ COUNTDOWN 5 giây                │
│ - Hiển thị tên khớp             │
│ - Hiển thị hướng dẫn tư thế     │
│ - "Bat dau sau: 5... 4... 3..." │
└────────────┬────────────────────┘
             │ Auto
             ▼
┌─────────────────────────────────┐
│ CalibrationState.COLLECTING     │
│ - Thu thập góc 5 giây           │
│ - Median filter loại nhiễu      │
│ - Progress bar hiển thị         │
└────────────┬────────────────────┘
             │ Auto complete
             ▼
┌─────────────────────────────────┐
│ Lưu kết quả vào calibrated_joints │
│ calibration_queue_index += 1    │
│ → Chuyển sang khớp tiếp theo    │
└─────────────────────────────────┘
```

**UI hiển thị**:
```
┌────────────────────────────────────────────────────────────┐
│ GIAI DOAN 2: DO GIOI HAN VAN DONG (TU DONG)                │
│                                                            │
│ Tien do: 2/6 khop                                          │
│ [===========>                    ] 33%                     │
│                                                            │
│ Danh sach khop:                                            │
│   [OK] Vai trai: 145.3 do                                  │
│   [OK] Vai phai: 142.8 do                                  │
│   >>> Khuyu tay trai (dang do)                             │
│       Khuyu tay phai                                       │
│       Dau goi trai                                         │
│       Dau goi phai                                         │
│                                                            │
│   Moi ba dung NGANG            <-- Huong dan tu the        │
│   Bat dau sau: 3 giay          <-- Countdown               │
│   [==============>     ]       <-- Progress                │
└────────────────────────────────────────────────────────────┘
```

**Điều kiện chuyển Phase 3**:
- `all_joints_calibrated == True` (đã đo xong 6 khớp)
- **TỰ ĐỘNG** sau 2 giây
- Profile được lưu vào `./data/user_profiles/`

---

### 3.3 PHASE 3: Motion Sync (MULTI-JOINT)

**Mục đích**: Đồng bộ chuyển động người dùng với video mẫu, tính điểm real-time cho **TẤT CẢ các khớp đã calibrated**.

**Multi-joint Tracking Flow**:
```python
def _run_phase3(self, user_frame, ref_frame, result, timestamp):
    # === MULTI-JOINT ANGLE CALCULATION ===
    # 1. Tính góc cho TẤT CẢ các khớp đang hoạt động
    self._state.user_angles_dict = self._calculate_all_joint_angles(landmarks)
    
    # === MULTI-JOINT TARGET CALCULATION ===
    # 2. Tính target cho TẤT CẢ các khớp
    self._state.target_angles_dict = self._interpolate_all_joint_targets(
        current_frame, total_frames
    )
    
    # === MULTI-JOINT SCORING ===
    # 3. Tính điểm có trọng số cho từng khớp
    multi_joint_score = self._calculate_multi_joint_score()
    
    # 4. Smooth score
    self._state.current_score = 0.7 * current_score + 0.3 * multi_joint_score
```

**Layout hiển thị (3 panels)**:
```
┌───────────────────┬───────────────────┬─────────────────────┐
│    USER VIEW      │   REFERENCE VIEW  │  DASHBOARD (320px)  │
│                   │                   │                     │
│   [Skeleton]      │   [Skeleton]      │ GIAI DOAN 3: DONG BO│
│   Goc: 85.3       │   VIDEO MAU       │ ● HOLD | Rep: 3     │
│   Muc tieu: 90    │   ● GIU           │ ○ ○ ● ○             │
│   Sai so: 4.7     │                   │                     │
│   Diem: 82        │   [=======>    ]  │ DIEM TONG: 82/100   │
│                   │                   │ [=============>   ] │
│   ┌────────────┐  │   || CHO          │                     │
│   │DAT MUC TIEU│  │                   │ CHI TIET KHOP (6):  │
│   └────────────┘  │                   │ Vai trai: 85/90|92pt│
│                   │                   │ Vai phai: 82/88|85pt│
│                   │                   │ Khuyu T: 78/85|75pt │
│                   │                   │ ...                 │
│                   │                   │                     │
│                   │                   │ KHOP CHINH: Vai trai│
│                   │                   │ ^ Nang cao hon!     │
│                   │                   │ Met moi: FRESH      │
└───────────────────┴───────────────────┴─────────────────────┘
```

**Exercise Type Detection**:
```python
# Xác định loại bài tập từ primary joint
if primary_joint in (LEFT_ELBOW, RIGHT_ELBOW):
    exercise_type = "bicep_curl"
elif primary_joint in (LEFT_KNEE, RIGHT_KNEE):
    exercise_type = "squat"
else:
    exercise_type = "arm_raise"

# Lấy trọng số cho từng khớp
joint_weights = create_exercise_weights(exercise_type)
```

**Motion Phase FSM**:
```
IDLE ──► ECCENTRIC ──► HOLD ──► CONCENTRIC ──► IDLE
 │                                              │
 └──────────────────────────────────────────────┘
                    (1 rep complete)
```

**Target Angle Interpolation (per joint)**:
```python
def _interpolate_target_angle(self, current_frame, total_frames, joint_type):
    """
    Tính target angle cho MỘT khớp cụ thể.
    Scale target dựa trên user_max_angle đã calibrated.
    """
    # Tìm checkpoint trước và sau
    # Interpolate: target = prev + progress * (next - prev)
    # Scale theo user_max nếu cần

def _interpolate_all_joint_targets(self, current_frame, total_frames):
    """Tính target cho TẤT CẢ các khớp đang hoạt động."""
    targets = {}
    for joint_type in active_joints:
        targets[joint_type] = self._interpolate_target_angle(
            current_frame, total_frames, joint_type
        )
    return targets
```

**Multi-joint Score Calculation**:
```python
def _calculate_multi_joint_score(self):
    """Tính điểm trung bình có trọng số."""
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for joint_type in active_joints:
        user_angle = user_angles_dict[joint_type]
        target_angle = target_angles_dict[joint_type]
        weight = joint_weights[joint_type]
        
        joint_score = _calculate_realtime_score(user_angle, target_angle)
        joint_scores_dict[joint_type] = joint_score
        
        total_weighted_score += joint_score * weight
        total_weight += weight
    
    return total_weighted_score / total_weight
```

**Điều kiện chuyển Phase 4**:
- `sync_status == SyncStatus.COMPLETE`
- `PlaybackState.FINISHED`

---

### 3.4 PHASE 4: Scoring & Results

**Mục đích**: Hiển thị kết quả buổi tập, lưu báo cáo.

**UI Elements**:
```
┌────────────────────────────────────────────────────────────────┐
│                  GIAI DOAN 4: KET QUA BUOI TAP                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Tong so hiep: 5                                              │
│                                                                │
│   Diem trung binh: 82/100                                      │
│   Danh gia: XUAT SAC                                           │
│                                                                │
│   Chi tiet diem:                                               │
│     ROM (bien do): 85                                          │
│     Stability (on dinh): 78                                    │
│     Flow (mu mut): 80                                          │
│                                                                │
│   Goc toi da (calibrated): 145.3                               │
│   Muc do met moi: FRESH                                        │
│                                                                │
│   Khuyen nghi:                                                 │
│     - Tiep tuc tap luyen deu dan moi ngay                      │
│     - Tang dan cuong do theo tung tuan                         │
│     - Nghi ngoi day du giua cac buoi tap                       │
│                                                                │
│   Ket qua da duoc luu vao log                                  │
│                                                                │
│              [R] Tap lai tu dau | [Q] Thoat                    │
└────────────────────────────────────────────────────────────────┘
```

**Grade System**:
```python
if score >= 80: grade = "XUAT SAC"   # Xuất sắc
if score >= 60: grade = "KHA"        # Khá
else:           grade = "CAN CO GANG" # Cần cố gắng
```

---

## 4. DATA STRUCTURES

### 4.1 AppPhase (Enum)

```python
class AppPhase(Enum):
    PHASE1_DETECTION = "phase1"    # Pose Detection
    PHASE2_CALIBRATION = "phase2"  # Safe-Max Calibration
    PHASE3_SYNC = "phase3"         # Motion Sync
    PHASE4_SCORING = "phase4"      # Scoring & Analysis
    COMPLETED = "completed"        # Hoàn thành
```

### 4.2 AppState (Dataclass)

```python
@dataclass
class AppState:
    # === Application State ===
    current_phase: AppPhase = AppPhase.PHASE1_DETECTION
    is_running: bool = True
    is_paused: bool = False
    
    # === Phase 1: Detection (AUTO TRANSITION) ===
    pose_detected: bool = False
    detection_stable_count: int = 0  # Đếm frames stable
    phase1_countdown_start: float = 0.0  # Thời điểm bắt đầu countdown 3 giây
    phase1_countdown_active: bool = False  # Đang trong countdown chuyển phase
    
    # === Phase 2: Automated Calibration ===
    selected_joint: Optional[JointType] = None
    calibration_complete: bool = False
    user_max_angle: float = 0.0
    # Automated calibration queue
    calibration_queue_index: int = 0  # Vị trí hiện tại trong queue
    calibration_countdown_start: float = 0.0  # Thời điểm bắt đầu countdown
    is_countdown_active: bool = False  # Đang countdown chuẩn bị
    is_calibrating_joint: bool = False  # Đang đo khớp hiện tại
    calibrated_joints: Dict = field(default_factory=dict)  # Dict[JointType, float]
    all_joints_calibrated: bool = False  # Đã đo xong tất cả 6 khớp
    
    # === Phase 3: Sync (MULTI-JOINT) ===
    sync_state: Optional[SyncState] = None
    motion_phase: str = "idle"           # idle/eccentric/hold/concentric
    last_motion_phase: Optional[MotionPhase] = None
    
    # Multi-joint tracking
    user_angles_dict: Dict = field(default_factory=dict)  # Dict[JointType, float]
    target_angles_dict: Dict = field(default_factory=dict)  # Dict[JointType, float]
    joint_scores_dict: Dict = field(default_factory=dict)  # Dict[JointType, float]
    joint_weights: Dict = field(default_factory=dict)  # Dict[JointType, float]
    active_joints: List = field(default_factory=list)  # Danh sách các khớp hoạt động
    
    # === Phase 4: Scoring ===
    rep_count: int = 0
    current_score: float = 0.0
    average_score: float = 0.0
    
    # === Common (backward compatible) ===
    user_angle: float = 0.0  # Primary joint angle
    target_angle: float = 0.0  # Primary joint target
    pain_level: str = "NONE"
    fatigue_level: str = "FRESH"
    message: str = ""
    warning: str = ""
```

### 4.3 Constants & Mappings

```python
# Phím số → JointType (cho manual selection - không dùng nữa trong auto mode)
JOINT_KEY_MAPPING = {
    ord('1'): JointType.LEFT_SHOULDER,
    ord('2'): JointType.RIGHT_SHOULDER,
    ord('3'): JointType.LEFT_ELBOW,
    ord('4'): JointType.RIGHT_ELBOW,
    ord('5'): JointType.LEFT_KNEE,
    ord('6'): JointType.RIGHT_KNEE,
}

# Tên tiếng Việt của khớp
JOINT_NAMES = {
    JointType.LEFT_SHOULDER: "Vai trai",
    JointType.RIGHT_SHOULDER: "Vai phai",
    JointType.LEFT_ELBOW: "Khuyu tay trai",
    JointType.RIGHT_ELBOW: "Khuyu tay phai",
    JointType.LEFT_KNEE: "Dau goi trai",
    JointType.RIGHT_KNEE: "Dau goi phai",
}

# Calibration Queue - thứ tự tự động đo 6 khớp
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,
    JointType.RIGHT_SHOULDER,
    JointType.LEFT_ELBOW,
    JointType.RIGHT_ELBOW,
    JointType.LEFT_KNEE,
    JointType.RIGHT_KNEE,
]

# Hướng dẫn tư thế cho từng loại khớp
JOINT_POSITION_INSTRUCTIONS = {
    JointType.LEFT_SHOULDER: "Moi ba dung NGANG",
    JointType.RIGHT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_ELBOW: "Moi ba dung NGANG",
    JointType.RIGHT_ELBOW: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    JointType.RIGHT_KNEE: "Moi ba dung DOC",
}

# Countdown durations
CALIBRATION_COUNTDOWN_DURATION = 5.0  # giây
PHASE1_COUNTDOWN_DURATION = 3.0  # giây

# Màu sắc motion phase
PHASE_COLORS = {
    "idle": (128, 128, 128),       # Gray
    "eccentric": (0, 255, 255),    # Yellow
    "hold": (0, 255, 0),           # Green
    "concentric": (255, 255, 0),   # Cyan
}

# Tên tiếng Việt của phase
PHASE_NAMES_VI = {
    "idle": "Nghi",
    "eccentric": "Duoi co",
    "hold": "Giu",
    "concentric": "Co co",
}
```

---

## 5. CLASS MemotionAppV2

### 5.1 Constructor

```python
class MemotionAppV2:
    DETECTION_STABLE_THRESHOLD = 30  # Frames cần stable
    PHASE1_COUNTDOWN_DURATION = 3.0  # 3 giây countdown Phase 1 → 2
    WINDOW_NAME = "MEMOTION - He thong ho tro phuc hoi chuc nang"
    
    def __init__(
        self,
        detector: VisionDetector,           # MediaPipe detector
        ref_video_path: Optional[str],      # Đường dẫn video mẫu
        default_joint: JointType,           # Khớp mặc định (primary)
        log_dir: str = "./data/logs",       # Thư mục logs
        models_dir: str = "./models"        # Thư mục models
    ):
        # State
        self._state = AppState()
        
        # Components
        self._video_engine: Optional[VideoEngine]
        self._sync_controller: Optional[MotionSyncController]
        self._calibrator = SafeMaxCalibrator(duration_ms=5000)
        self._pain_detector = PainDetector()
        self._scorer = HealthScorer()
        self._logger = SessionLogger(log_dir)
        self._user_profile: Optional[UserProfile] = None
        
        # Reference video detector
        self._ref_detector: Optional[VisionDetector] = None
        
        # Data tracking
        self._user_angles: List[float] = []
        self._ref_angles: List[float] = []
        self._score_history: List[float] = []
        self._current_landmarks: Optional[np.ndarray] = None
        self._ref_landmarks: Optional[np.ndarray] = None
        
        # Analysis queue (for async pain detection)
        self._analysis_queue = Queue(maxsize=5)
        
        # Interpolated target
        self._last_target_angle: float = 0.0
```

### 5.2 Core Methods

| Method | Mô tả |
|--------|-------|
| `run(user_source, display)` | Main loop chính |
| `_run_phase1(frame, result)` | Phase 1 với auto transition |
| `_run_phase2(frame, result, timestamp_ms)` | Phase 2 automated calibration |
| `_run_phase3(user_frame, ref_frame, result, timestamp)` | Phase 3 multi-joint |
| `_run_phase4(frame)` | Phase 4 kết quả |
| `_create_phase3_dashboard(height)` | Tạo dashboard multi-joint |
| `_handle_key(key)` | Xử lý phím nhấn |
| `_advance_phase()` | Manual override chuyển phase |
| `_transition_to_phase2/3/4()` | Chuyển đổi phase |
| `_start_calibration_for_joint(joint_type)` | Bắt đầu đo một khớp |
| `_finish_calibration_for_joint(joint_type)` | Hoàn thành đo một khớp |
| `_save_calibration_to_profile()` | Lưu profile calibration |
| `_on_rep_complete()` | Khi hoàn thành 1 rep |
| `_restart()` | Reset về Phase 1 |
| `cleanup()` | Dọn dẹp tài nguyên |

### 5.3 Multi-joint Helper Methods

```python
# Tính target angle cho MỘT khớp (với scaling)
def _interpolate_target_angle(
    self, current_frame, total_frames, joint_type=None
) -> float

# Tính target cho TẤT CẢ các khớp
def _interpolate_all_joint_targets(
    self, current_frame, total_frames
) -> Dict[JointType, float]

# Tính điểm real-time cho MỘT khớp
def _calculate_realtime_score(
    self, user_angle, target_angle
) -> float

# Tính điểm trung bình có trọng số
def _calculate_multi_joint_score(self) -> float

# Tính góc TẤT CẢ các khớp từ landmarks
def _calculate_all_joint_angles(
    self, landmarks
) -> Dict[JointType, float]

# Lấy tọa độ pixel của 3 điểm góc
def _get_joint_pixel_coords(
    self, landmarks, joint_type, frame_shape
) -> Tuple

# Khởi tạo detector cho video mẫu
def _init_ref_detector(self) -> None

# Xử lý pain detection
def _process_pain(self) -> None

# Tạo báo cáo cuối
def _generate_report(self) -> Dict
```

---

## 6. LUỒNG HOẠT ĐỘNG CHI TIẾT

### 6.1 Main Loop Flow (AUTO TRANSITION)

```python
def run(self, user_source="webcam", display=True):
    # 1. Mở camera/video
    cap = cv2.VideoCapture(...)
    
    # 2. Init reference detector
    self._init_ref_detector()
    
    # 3. Print banner (chế độ tự động)
    print("CHE DO TU DONG - Khong can nhan ENTER")
    print("  1. Nhan dien tu the -> Tu dong chuyen sau 3 giay")
    print("  2. Do gioi han 6 khop -> Tu dong chuyen sau 2 giay")
    print("  3. Dong bo video mau -> Tu dong chuyen khi hoan tat")
    print("  4. Cham diem va phan tich")
    
    # 4. Main loop
    while self._state.is_running:
        ret, frame = cap.read()
        if user_source == "webcam":
            frame = cv2.flip(frame, 1)  # Mirror
        
        # Process detection
        result = self._detector.process_frame(frame, timestamp_ms)
        
        # Handle current phase (AUTO TRANSITIONS inside each phase)
        if current_phase == PHASE1_DETECTION:
            display_frame = self._run_phase1(frame, result)
            # → Auto transition sau 3 giây countdown
        
        elif current_phase == PHASE2_CALIBRATION:
            display_frame = self._run_phase2(frame, result, timestamp_ms)
            # → Auto transition sau khi đo xong 6 khớp + 2 giây
        
        elif current_phase == PHASE3_SYNC:
            # Get reference frame
            ref_frame, ref_status = self._video_engine.get_frame()
            
            # Check rep completion
            if last_phase == CONCENTRIC and current_phase == IDLE:
                self._on_rep_complete()
            
            # Check video finished
            if ref_status.state == PlaybackState.FINISHED:
                self._transition_to_phase4()  # Auto transition
            
            display_frame = self._run_phase3(frame, ref_frame, result, timestamp)
        
        elif current_phase == PHASE4_SCORING:
            display_frame = self._run_phase4(frame)
        
        # Display & handle key
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)
    
    # Cleanup
    cap.release()
    return self._generate_report()
```

### 6.2 Transition to Phase 3 (Multi-joint Setup)

```python
def _transition_to_phase3(self):
    # 1. Check video mẫu exists
    if not ref_video_path:
        _transition_to_phase4()
        return
    
    # 2. Setup video engine
    self._video_engine = VideoEngine(ref_video_path)
    
    # 3. Xác định loại bài tập
    if primary_joint in (LEFT_ELBOW, RIGHT_ELBOW):
        exercise_type = "bicep_curl"
    elif primary_joint in (LEFT_KNEE, RIGHT_KNEE):
        exercise_type = "squat"
    else:
        exercise_type = "arm_raise"
    
    # 4. Lấy trọng số cho từng khớp
    joint_weights = create_exercise_weights(exercise_type)
    
    # 5. Xác định active joints (từ calibrated_joints)
    active_joints = list(calibrated_joints.keys())
    
    # 6. Khởi tạo dictionaries
    user_angles_dict = {jt: 0.0 for jt in active_joints}
    target_angles_dict = {jt: 0.0 for jt in active_joints}
    joint_scores_dict = {jt: 0.0 for jt in active_joints}
    
    # 7. Create exercise với max_angle từ calibration
    exercise = create_arm_raise_exercise(total_frames, fps, max_angle)
    
    # 8. Setup sync controller
    sync_controller = MotionSyncController(exercise, user_max_angle=max_angle)
    
    # 9. Start session
    logger.start_session(session_id, exercise.name)
    scorer.start_session(exercise.name, session_id)
    
    # 10. Print setup info
    print(f"[SETUP] Active joints ({len(active_joints)}):")
    for jt in active_joints:
        print(f"  - {JOINT_NAMES[jt]}: max={angle:.1f}do, weight={weight:.2f}")
```

### 6.3 Rep Completion Flow

```python
def _on_rep_complete(self):
    # 1. Compute DTW nếu đủ data
    if len(user_angles) > 20 and len(ref_angles) > 20:
        dtw_result = compute_single_joint_dtw(
            user_angles[-50:],
            ref_angles[-50:]
        )
    
    # 2. Complete rep trong scorer
    rep_score = self._scorer.complete_rep(target, dtw_result)
    
    # 3. Log kết quả
    self._logger.log_rep(
        rep_score.rep_number,
        {rom, stability, flow, total},
        jerk_value,
        duration_ms
    )
    
    # 4. Print console
    print(f"[REP {rep_number}] Score: {total_score}")
```

### 6.4 Key Handling Flow (Simplified for Auto Mode)

```python
def _handle_key(self, key):
    if key == 'q' or ESC:
        is_running = False
    
    elif key == ENTER:
        # Manual override - bỏ qua countdown
        _advance_phase()
    
    elif key == SPACE:
        # Phase 3: Pause/Resume video
        if PHASE3_SYNC:
            is_paused = toggle
    
    elif key == 'r':
        _restart()  # Reset về Phase 1
    
    # Note: Phím 1-6 không còn dùng trong auto calibration mode
```

---

## 7. PHÍM ĐIỀU KHIỂN

| Phím | Phase | Chức năng |
|------|-------|-----------|
| `ENTER` | 1 | Manual override - bỏ qua countdown 3 giây |
| `ENTER` | 2 | Không dùng (auto calibration) |
| `SPACE` | 3 | Pause/Resume video |
| `R` | All | Restart về Phase 1 |
| `Q` / `ESC` | All | Thoát ứng dụng |

**Lưu ý**: Trong chế độ AUTO TRANSITION:
- Phase 1 → 2: Tự động sau 3 giây khi pose_detected
- Phase 2 → 3: Tự động sau 2 giây khi đo xong 6 khớp
- Phase 3 → 4: Tự động khi video kết thúc

---

## 8. CÔNG THỨC TÍNH ĐIỂM

### 8.1 Real-time Score (Single Joint)

```python
def _calculate_realtime_score(self, user_angle, target_angle):
    if target_angle <= 0:
        return current_score  # Giữ nguyên
    
    error = abs(user_angle - target_angle)
    error_percent = (error / target_angle) * 100
    
    if error_percent < 5:
        score = 100.0
    elif error_percent < 10:
        score = 95.0 - (error_percent - 5) * 1.0   # 95-90
    elif error_percent < 15:
        score = 90.0 - (error_percent - 10) * 2.0  # 90-80
    elif error_percent < 25:
        score = 80.0 - (error_percent - 15) * 1.5  # 80-65
    elif error_percent < 40:
        score = 65.0 - (error_percent - 25) * 1.0  # 65-50
    else:
        score = max(0, 50.0 - (error_percent - 40) * 0.5)
    
    return max(0, min(100, score))
```

**Score mapping table**:
```
┌──────────────────┬───────────────┬────────────┐
│ Error Percent    │ Score Range   │ Feedback   │
├──────────────────┼───────────────┼────────────┤
│ < 5%             │ 100           │ TUYET VOI! │
│ 5% - 10%         │ 95 - 90       │ TOT!       │
│ 10% - 15%        │ 90 - 80       │ TOT!       │
│ 15% - 25%        │ 80 - 65       │ KHA        │
│ 25% - 40%        │ 65 - 50       │ DIEU CHINH │
│ > 40%            │ < 50          │ DIEU CHINH │
└──────────────────┴───────────────┴────────────┘
```

### 8.2 Multi-joint Weighted Score

```python
def _calculate_multi_joint_score(self):
    """Tính điểm trung bình có trọng số."""
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for joint_type in active_joints:
        user_angle = user_angles_dict[joint_type]
        target_angle = target_angles_dict[joint_type]
        weight = joint_weights.get(joint_type, 0.5)
        
        joint_score = _calculate_realtime_score(user_angle, target_angle)
        joint_scores_dict[joint_type] = joint_score
        
        total_weighted_score += joint_score * weight
        total_weight += weight
    
    if total_weight > 0:
        return total_weighted_score / total_weight
    return current_score

# Weighted Score = Σ(joint_score × weight) / Σ(weight)
```

### 8.3 Score Smoothing

```python
# Để tránh score nhảy quá nhanh
current_score = 0.7 * current_score + 0.3 * realtime_score
```

### 8.4 Final Score (từ HealthScorer)

```python
Total Score = weighted_sum(
    ROM Score × 0.30,
    Stability Score × 0.20,
    Flow Score × 0.20,
    Symmetry Score × 0.15,
    Compensation Score × 0.15
)
```

---

## 9. HƯỚNG DẪN SỬ DỤNG

### 9.1 Command Line Arguments

```bash
python main_v2.py [OPTIONS]

Options:
  --source      Input source (default: "webcam")
                Có thể là: webcam, path/to/video.mp4
  
  --ref-video   Đường dẫn video mẫu
                Bắt buộc cho Phase 3
  
  --joint       Khớp mặc định để theo dõi
                Choices: left_shoulder, right_shoulder,
                         left_elbow, right_elbow,
                         left_knee, right_knee
                Default: left_shoulder
  
  --mode        Chế độ chạy
                Choices: run, test
                Default: run
  
  --headless    Chạy không hiển thị UI
  
  --models-dir  Thư mục chứa model files
                Default: ./models
  
  --log-dir     Thư mục lưu logs
                Default: ./data/logs
```

### 9.2 Ví dụ sử dụng

```bash
# Chạy với webcam và video mẫu
python main_v2.py --source webcam --ref-video videos/arm_raise.mp4

# Chạy với video input
python main_v2.py --source path/to/user.mp4 --ref-video videos/arm_raise.mp4

# Chạy với khớp khuỷu tay
python main_v2.py --source webcam --ref-video videos/elbow.mp4 --joint left_elbow

# Chạy test mode
python main_v2.py --mode test

# Chạy headless (không UI)
python main_v2.py --source webcam --ref-video videos/arm_raise.mp4 --headless
```

### 9.3 Yêu cầu model files

Đảm bảo có các file trong `./models/`:
```
models/
├── pose_landmarker_lite.task    # Bắt buộc
└── face_landmarker.task         # Tùy chọn (cho pain detection)
```

---

## 10. UNIT TESTS

### 10.1 Chạy tests

```bash
python main_v2.py --mode test
```

### 10.2 Test Cases

```python
def run_unit_tests():
    # TEST 1: Visualization
    # - put_vietnamese_text
    # - draw_skeleton
    
    # TEST 2: SafeMaxCalibrator
    # - Khởi tạo state = IDLE
    
    # TEST 3: PainDetector
    # - Khởi tạo thành công
    
    # TEST 4: HealthScorer
    # - start_session
    # - add_frame (20 frames)
    # - complete_rep
    # - Verify score
    
    # TEST 5: MotionSyncController
    # - create_arm_raise_exercise
    # - update()
    # - Verify phase
```

### 10.3 Expected Output

```
============================================================
UNIT TESTS - MEMOTION v2.0
============================================================

[TEST 1] Visualization...
  OK - Vietnamese text

[TEST 2] SafeMaxCalibrator...
  OK - Calibrator

[TEST 3] PainDetector...
  OK - PainDetector

[TEST 4] HealthScorer...
  OK - Score: 85.2

[TEST 5] MotionSyncController...
  OK - Phase: eccentric

============================================================
ALL TESTS PASSED!
============================================================
```

---

## 📝 CHANGELOG

### Version 2.0.0 (Current)
- **AUTO TRANSITION**: Tự động chuyển phase không cần nhấn ENTER
  - Phase 1 → 2: Countdown 3 giây
  - Phase 2 → 3: 2 giây sau khi đo xong
  - Phase 3 → 4: Khi video kết thúc
- **Automated Calibration**: Tự động đo 6 khớp theo thứ tự
  - Countdown 5 giây chuẩn bị mỗi khớp
  - Hướng dẫn tư thế theo loại khớp
  - Lưu profile vào `./data/user_profiles/`
- **Multi-joint Tracking**: Theo dõi và tính điểm tất cả khớp
  - Weighted scoring theo loại bài tập
  - Chi tiết điểm từng khớp trên dashboard
- UI rõ ràng hơn với panels và progress bars
- Real-time scoring với visual feedback
- Target angle interpolation (per joint)
- Score smoothing để tránh nhảy
- Direction hints (nâng cao/hạ thấp)
- Vietnamese feedback text

### So sánh với main_final.py

| Feature | main_final.py | main_v2.py |
|---------|---------------|------------|
| Phase transition | Manual (ENTER) | **Auto** với countdown |
| Calibration | Chọn 1 khớp | **Tự động 6 khớp** |
| Joint tracking | Single joint | **Multi-joint** |
| Scoring | Single joint | **Weighted multi-joint** |
| Phase separation | Basic | Clear UI cho mỗi phase |
| Real-time score | Từ scorer | Interpolated + smoothed |
| Target angle | Từ checkpoint | Interpolated per joint |
| Visual feedback | Basic | Colors + text + banners |
| Vietnamese UI | Partial | Full |

---

## 🔗 REFERENCES

- [main_context.md](main_context.md) - Tài liệu tổng quan hệ thống
- [README1.md](README1.md) - Hướng dẫn cơ bản
- [core/](core/) - Module cốt lõi
- [modules/](modules/) - Module chức năng
- [utils/](utils/) - Tiện ích

---

> **Note**: Tài liệu này mô tả chi tiết file `main_v2.py` (1788 lines). Xem `main_context.md` để hiểu tổng quan về toàn bộ hệ thống MEMOTION.
