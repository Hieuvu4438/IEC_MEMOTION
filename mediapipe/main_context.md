# 📚 MEMOTION - MEDIAPIPE MODULE CONTEXT

> **Tài liệu chi tiết về hệ thống hỗ trợ phục hồi chức năng cho người già**
> 
> Author: MEMOTION Team  
> Version: 2.0.0  
> Last Updated: 2026-01-21

---

## 📋 MỤC LỤC

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Module Core](#4-module-core)
5. [Module Modules](#5-module-modules)
6. [Module Utils](#6-module-utils)
7. [Luồng hoạt động chính](#7-luồng-hoạt-động-chính)
8. [File Main Entry Points](#8-file-main-entry-points)
9. [Data Flow](#9-data-flow)
10. [Công thức toán học](#10-công-thức-toán-học)
11. [Hướng dẫn sử dụng](#11-hướng-dẫn-sử-dụng)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Mục đích
MEMOTION là hệ thống hỗ trợ phục hồi chức năng cho **người già** sử dụng Computer Vision. Hệ thống tập trung vào:

- **An toàn**: Không ép người dùng vượt quá giới hạn vận động
- **Cá nhân hóa**: Điều chỉnh mục tiêu theo khả năng từng người
- **Theo dõi đau đớn**: Tự động phát hiện khi người dùng đau
- **Khuyến khích**: Đưa ra phản hồi tích cực, không phán xét

### 1.2 Công nghệ sử dụng
- **MediaPipe Tasks API**: Nhận diện pose (33 landmarks) và face (478 landmarks)
- **OpenCV**: Xử lý video và hiển thị
- **NumPy/SciPy**: Tính toán khoa học
- **FastDTW**: So sánh nhịp điệu chuyển động

### 1.3 Bốn giai đoạn chính (4 Phases)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMOTION WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PHASE 1          PHASE 2           PHASE 3          PHASE 4      │
│   ────────         ────────          ────────         ────────     │
│   Pose             Safe-Max          Motion           Scoring &    │
│   Detection        Calibration       Sync             Analysis     │
│                                                                     │
│   ↓ Nhận diện      ↓ Đo giới hạn     ↓ Đồng bộ        ↓ Chấm điểm  │
│     tư thế           vận động          video            đa chiều   │
│     skeleton         an toàn           mẫu                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. KIẾN TRÚC TỔNG THỂ

### 2.1 Sơ đồ kiến trúc

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              MEMOTION APP                                │
│                            (main_final.py)                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐           │
│  │   INPUT LAYER  │   │  PROCESS LAYER │   │  OUTPUT LAYER  │           │
│  │                │   │                │   │                │           │
│  │  • Webcam      │──▶│  • Detector    │──▶│  • Display     │           │
│  │  • Video file  │   │  • Calibrator  │   │  • Dashboard   │           │
│  │  • Ref video   │   │  • Synchronizer│   │  • Logger      │           │
│  │                │   │  • Scorer      │   │  • Reports     │           │
│  └────────────────┘   │  • PainDetector│   └────────────────┘           │
│                       └────────────────┘                                │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                           MODULES LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  CORE              │  MODULES            │  UTILS               │   │
│  │  ─────             │  ───────            │  ─────               │   │
│  │  • data_types      │  • calibration      │  • logger            │   │
│  │  • detector        │  • pain_detection   │  • visualization     │   │
│  │  • kinematics      │  • scoring          │                      │   │
│  │  • procrustes      │  • target_generator │                      │   │
│  │  • synchronizer    │  • video_engine     │                      │   │
│  │  • dtw_analysis    │                     │                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dependency Graph

```
main_final.py
    │
    ├── core/
    │   ├── data_types.py (Base - không phụ thuộc)
    │   ├── detector.py → data_types
    │   ├── kinematics.py → data_types
    │   ├── procrustes.py → data_types
    │   ├── synchronizer.py → kinematics
    │   └── dtw_analysis.py → kinematics
    │
    ├── modules/
    │   ├── calibration.py → core.kinematics, core.data_types
    │   ├── pain_detection.py → core.data_types
    │   ├── scoring.py → core.kinematics, core.synchronizer, core.dtw_analysis
    │   ├── target_generator.py → core.kinematics, modules.calibration
    │   └── video_engine.py (standalone)
    │
    └── utils/
        ├── logger.py (standalone)
        └── visualization.py → core.data_types
```

---

## 3. CẤU TRÚC THƯ MỤC

```
mediapipe/
│
├── 📁 core/                        # Thành phần cốt lõi
│   ├── __init__.py                 # Export public APIs
│   ├── data_types.py               # Data classes chuẩn hóa
│   ├── detector.py                 # MediaPipe wrapper
│   ├── dtw_analysis.py             # Dynamic Time Warping
│   ├── kinematics.py               # Tính toán góc khớp
│   ├── procrustes.py               # Chuẩn hóa skeleton
│   └── synchronizer.py             # FSM đồng bộ video
│
├── 📁 modules/                     # Module chức năng
│   ├── __init__.py
│   ├── calibration.py              # Safe-Max Calibration
│   ├── pain_detection.py           # Nhận diện đau (FACS)
│   ├── scoring.py                  # Chấm điểm đa chiều
│   ├── target_generator.py         # Cá nhân hóa mục tiêu
│   └── video_engine.py             # Smart Video Player
│
├── 📁 utils/                       # Tiện ích
│   ├── __init__.py
│   ├── logger.py                   # Ghi nhật ký
│   └── visualization.py            # Vẽ UI, skeleton
│
├── 📁 models/                      # MediaPipe models
│   ├── face_landmarker.task
│   ├── pose_landmarker_full.task
│   └── pose_landmarker_lite.task
│
├── 📁 data/                        # Dữ liệu runtime
│   ├── logs/                       # Session logs (JSON, CSV)
│   └── user_profiles/              # Profiles người dùng
│
├── 📁 test_logs/                   # Test session logs
├── 📁 videos/                      # Video mẫu
├── 📁 scripts/                     # Scripts hỗ trợ
│
├── main_final.py                   # Entry point chính
├── main_test.py                    # Test suite
├── main_v2.py                      # Version 2
├── main_sync_test.py               # Test đồng bộ
├── test_calibration.py             # Test calibration
├── comprehensive_audit.py          # Audit toàn diện
├── requirements.txt                # Dependencies
└── README1.md                      # Hướng dẫn cơ bản
```

---

## 4. MODULE CORE

### 4.1 `data_types.py` - Data Classes

**Vai trò**: Định nghĩa các cấu trúc dữ liệu chuẩn hóa, dễ dàng chuyển đổi sang Flutter/Dart sau này.

#### Classes chính:

```python
@dataclass(frozen=True)
class Point3D:
    """Điểm trong không gian 3D"""
    x: float
    y: float
    z: float
    visibility: Optional[float] = None  # Độ tin cậy 0-1
    presence: Optional[float] = None    # Xác suất tồn tại

@dataclass
class LandmarkSet:
    """Tập hợp landmarks của một loại (pose/face/hand)"""
    landmarks: List[Point3D]
    landmark_type: LandmarkType
    timestamp_ms: int = 0

@dataclass
class DetectionResult:
    """Kết quả detection từ một frame"""
    pose_landmarks: Optional[LandmarkSet]
    face_landmarks: Optional[LandmarkSet]
    pose_world_landmarks: Optional[LandmarkSet]
    frame_width: int
    frame_height: int
    timestamp_ms: int
    is_valid: bool

@dataclass
class NormalizedSkeleton:
    """Skeleton đã chuẩn hóa qua Procrustes"""
    landmarks: np.ndarray
    centroid: np.ndarray
    scale: float
    rotation_matrix: np.ndarray

class PoseLandmarkIndex:
    """Chỉ số 33 landmarks MediaPipe Pose"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_HIP = 23
    RIGHT_HIP = 24
    # ... và nhiều hơn
```

---

### 4.2 `detector.py` - Vision Detector

**Vai trò**: Wrapper cho MediaPipe Tasks API, cung cấp interface thống nhất.

#### Class chính:

```python
@dataclass
class DetectorConfig:
    """Cấu hình detector"""
    pose_model_path: Optional[str] = None
    face_model_path: Optional[str] = None
    min_pose_detection_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5
    running_mode: str = "VIDEO"  # IMAGE, VIDEO, LIVE_STREAM

class VisionDetector:
    """Wrapper cho MediaPipe"""
    
    def __init__(self, config: DetectorConfig):
        # Khởi tạo PoseLandmarker và FaceLandmarker
        
    def process_frame(self, image: np.ndarray, timestamp_ms: int) -> DetectionResult:
        """Xử lý một frame, trả về DetectionResult"""
```

#### Logic hoạt động:

```
Input Frame (BGR)
      │
      ▼
┌─────────────────┐
│ Convert to RGB  │
│ MediaPipe Image │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Pose  │ │ Face  │
│Landmrk│ │Landmrk│
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────┐
│ DetectionResult │
│ - pose_landmarks│
│ - face_landmarks│
│ - is_valid      │
└─────────────────┘
```

---

### 4.3 `kinematics.py` - Tính toán góc khớp

**Vai trò**: Tính góc giữa các khớp cơ thể từ pose landmarks.

#### Công thức toán học:

```
Góc giữa 3 điểm A, B, C (B là đỉnh góc):

    Vector BA = A - B
    Vector BC = C - B
    
    cos(θ) = (BA · BC) / (|BA| × |BC|)
    θ = arccos(cos(θ))
```

#### Enum và Definitions:

```python
class JointType(Enum):
    """Các khớp cần theo dõi"""
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"

@dataclass
class JointDefinition:
    """Định nghĩa một khớp"""
    proximal: int   # Điểm gần thân (vd: vai)
    vertex: int     # Đỉnh góc (vd: khuỷu tay)
    distal: int     # Điểm xa thân (vd: cổ tay)
    name: str
    normal_range: Tuple[float, float]

# Ví dụ: Khuỷu tay trái
# Vai → Khuỷu → Cổ tay
JointType.LEFT_ELBOW: JointDefinition(
    proximal=LEFT_SHOULDER,
    vertex=LEFT_ELBOW,
    distal=LEFT_WRIST,
    name="Khuỷu tay trái",
    normal_range=(0.0, 145.0)
)
```

#### Functions:

```python
def calculate_angle(point_a, point_b, point_c, use_3d=True) -> float:
    """Tính góc giữa 3 điểm, B là đỉnh"""
    
def calculate_joint_angle(landmarks, joint_type, use_3d=True) -> float:
    """Tính góc của một khớp cụ thể từ landmarks"""
    
def calculate_all_joint_angles(landmarks) -> Dict[JointType, float]:
    """Tính tất cả góc khớp"""
```

---

### 4.4 `synchronizer.py` - Motion Synchronizer

**Vai trò**: Điều khiển video mẫu đồng bộ với chuyển động người dùng bằng FSM (Finite State Machine).

#### Mô hình FSM:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   IDLE ──► ECCENTRIC ──► HOLD ──► CONCENTRIC ──┐    │
│     ▲                                          │    │
│     └──────────────────────────────────────────┘    │
│                                                      │
└──────────────────────────────────────────────────────┘

Giải thích:
- IDLE: Tư thế nghỉ, chuẩn bị
- ECCENTRIC: Pha duỗi cơ (vd: hạ người squat)
- HOLD: Giữ tại điểm cao trào
- CONCENTRIC: Pha co cơ (vd: đứng lên)
```

#### Classes:

```python
class MotionPhase(Enum):
    IDLE = "idle"
    ECCENTRIC = "eccentric"
    HOLD = "hold"
    CONCENTRIC = "concentric"

class SyncStatus(Enum):
    PLAY = "play"        # Video chạy bình thường
    PAUSE = "pause"      # Video dừng chờ user
    LOOP = "loop"        # Video lặp đoạn
    SKIP = "skip"        # Bỏ qua
    COMPLETE = "complete"

@dataclass
class PhaseCheckpoint:
    """Điểm mốc trong video"""
    frame_index: int
    phase_start: MotionPhase
    target_angle: float
    tolerance: float = 10.0

@dataclass
class SyncState:
    """Trạng thái đồng bộ"""
    current_phase: MotionPhase
    sync_status: SyncStatus
    ref_frame: int
    user_angle: float
    target_angle: float
    rep_count: int

class MotionSyncController:
    """Bộ điều khiển đồng bộ"""
    
    def update(self, user_angle, ref_frame, current_time) -> SyncState:
        """Cập nhật và trả về trạng thái mới"""
```

#### Nguyên tắc "Wait-for-User":

```
1. Video chạy bình thường đến checkpoint
2. Tại checkpoint: kiểm tra user đã đạt ngưỡng chưa
3. Nếu chưa → PAUSE/LOOP cho đến khi đạt
4. Khi đạt → tiếp tục đến checkpoint tiếp theo
```

---

### 4.5 `procrustes.py` - Procrustes Analysis

**Vai trò**: Chuẩn hóa skeleton, loại bỏ sự khác biệt về vị trí, kích thước, hướng quay.

#### Thuật toán 3 bước:

```
Step 1: TRANSLATION
┌─────────────────────┐
│ Dịch centroid về    │
│ gốc tọa độ (0,0,0)  │
└─────────────────────┘
         │
         ▼
Step 2: SCALING
┌─────────────────────┐
│ Chuẩn hóa kích      │
│ thước về unit norm  │
└─────────────────────┘
         │
         ▼
Step 3: ROTATION
┌─────────────────────┐
│ Xoay để minimize    │
│ khoảng cách với ref │
└─────────────────────┘
```

#### Functions:

```python
def normalize_skeleton(skeleton, use_core_landmarks=True) -> NormalizedSkeleton:
    """Chuẩn hóa (Translation + Scaling)"""

def align_skeleton_to_reference(target, reference) -> ProcrustesResult:
    """Căn chỉnh target theo reference (full Procrustes)"""

def compute_procrustes_distance(s1, s2) -> float:
    """Tính khoảng cách Procrustes (0 = khớp hoàn toàn)"""
```

---

### 4.6 `dtw_analysis.py` - Dynamic Time Warping

**Vai trò**: So sánh nhịp điệu chuyển động giữa user và video mẫu.

#### Tại sao cần DTW?

```
Người già di chuyển với tốc độ khác nhau, có thể dừng giữa chừng.
DTW "kéo giãn" thời gian để tìm sự tương đồng tối ưu.

User:  ──●───●─────●───────●───●──
          \   \     \       \ /
           \   \     \       X
            \   \     \     / \
Ref:   ──●───●───●───●───●───●───●──
```

#### Weighted DTW:

```python
@dataclass
class DTWResult:
    distance: float           # Khoảng cách DTW
    normalized_distance: float
    path: List[Tuple[int, int]]
    similarity_score: float   # 0-100%
    rhythm_quality: str       # "excellent", "good", "fair", "poor"

def compute_weighted_dtw(
    user_sequences: Dict[JointType, List[float]],
    ref_sequences: Dict[JointType, List[float]],
    weights: Dict[JointType, float]
) -> DTWResult:
    """
    Weighted DTW cho nhiều khớp
    
    Ví dụ weights cho bài giơ tay:
    - Vai: 1.0 (quan trọng nhất)
    - Khuỷu: 0.7
    - Đầu gối: 0.1 (không liên quan)
    """
```

---

## 5. MODULE MODULES

### 5.1 `calibration.py` - Safe-Max Calibration

**Vai trò**: Xác định giới hạn vận động (ROM) an toàn của từng người.

#### Ý nghĩa nhân văn:

```
Mỗi người già có giới hạn khác nhau do:
- Tuổi tác và sức khỏe
- Tiền sử chấn thương  
- Bệnh mãn tính (viêm khớp, thoái hóa...)

Calibration giúp:
- Đặt mục tiêu AN TOÀN, không gây đau
- Giảm áp lực tâm lý
- Theo dõi tiến triển khách quan
```

#### Classes:

```python
class CalibrationState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class JointCalibrationData:
    joint_type: str
    max_angle: float        # Góc tối đa an toàn
    min_angle: float
    raw_angles: List[float] # Góc thô từ các frame
    confidence: float       # Độ tin cậy 0-1

@dataclass
class UserProfile:
    user_id: str
    name: str
    age: int
    joint_limits: Dict[str, JointCalibrationData]
    notes: str

class SafeMaxCalibrator:
    """Bộ calibration"""
    
    def start_calibration(self, joint_type: JointType):
        """Bắt đầu thu thập"""
        
    def add_frame(self, landmarks, timestamp_ms):
        """Thêm frame vào bộ thu thập"""
        
    def finish_calibration(self) -> JointCalibrationData:
        """Hoàn thành và tính max angle"""
```

#### Quy trình Calibration:

```
1. Hướng dẫn user thực hiện động tác "hết khả năng" (KHÔNG GÂY ĐAU)
2. Thu thập góc khớp trong 5-10 giây
3. Áp dụng Median Filter loại bỏ outliers
4. Trích xuất max ổn định làm θ_user_max
```

---

### 5.2 `pain_detection.py` - Pain Detection

**Vai trò**: Nhận diện đau qua biểu cảm khuôn mặt sử dụng FACS (Facial Action Coding System).

#### Action Units liên quan đến đau:

```
AU4:  Cau mày (Brow Lowerer)
AU6:  Nheo má (Cheek Raiser)
AU7:  Căng mí mắt (Lid Tightener)
AU9:  Nhăn mũi (Nose Wrinkler)
AU10: Nâng môi trên (Upper Lip Raiser)
AU43: Nhắm mắt (Eye Closure)
```

#### Classes:

```python
class PainLevel(Enum):
    NONE = 0      # Không đau
    MILD = 1      # Nhẹ - có thể tiếp tục
    MODERATE = 2  # Trung bình - cần chú ý
    SEVERE = 3    # Nặng - nên dừng

@dataclass
class PainAnalysisResult:
    pain_level: PainLevel
    pain_score: float           # 0-100
    au_activations: Dict[str, float]
    is_pain_detected: bool
    confidence: float
    message: str

class PainDetector:
    # Ngưỡng phát hiện AU
    AU_THRESHOLDS = {
        "AU4": 0.15,   # Cau mày
        "AU6": 0.12,   # Nheo má
        "AU7": 0.15,   # Căng mí
        "AU43": 0.40,  # Nhắm mắt
    }
    
    def analyze(self, face_landmarks) -> PainAnalysisResult:
        """Phân tích biểu cảm và trả về kết quả"""
```

#### Thuật toán:

```
1. Tính các tỷ lệ khoảng cách giữa landmarks
2. So sánh với baseline (trạng thái bình thường)
3. Nếu nhiều AU kích hoạt đồng thời → đau
4. Theo dõi thời gian (>500ms mới tính)
```

---

### 5.3 `scoring.py` - Health Scoring

**Vai trò**: Chấm điểm đa chiều đánh giá chất lượng tập luyện.

#### 5 chỉ số đánh giá:

```
┌─────────────────────────────────────────────────────────────┐
│                     SCORING MATRIX                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ROM Score (30%)                                         │
│     So sánh góc đạt được với mục tiêu                       │
│     100% nếu đạt hoặc vượt mục tiêu                         │
│                                                             │
│  2. Stability Score (20%)                                   │
│     Độ rung lắc trong pha HOLD                              │
│     Dựa trên std deviation của góc                          │
│                                                             │
│  3. Flow Score (20%)                                        │
│     Từ kết quả DTW                                          │
│     Độ mượt mà của chuyển động                              │
│                                                             │
│  4. Symmetry Score (15%)                                    │
│     Cân bằng trái-phải                                      │
│                                                             │
│  5. Compensation Score (15%)                                │
│     Trừ điểm nếu có động tác bù trừ                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Phát hiện mệt mỏi qua Jerk:

```
Jerk = d³x/dt³ (đạo hàm bậc 3 của vị trí)

- Jerk thấp = chuyển động mượt mà
- Jerk cao = chuyển động giật
- Jerk tăng dần qua các rep = dấu hiệu mệt mỏi
```

#### Classes:

```python
class FatigueLevel(Enum):
    FRESH = 0       # Khỏe
    LIGHT = 1       # Hơi mệt
    MODERATE = 2    # Mệt vừa
    HEAVY = 3       # Rất mệt

@dataclass
class RepScore:
    rep_number: int
    rom_score: float
    stability_score: float
    flow_score: float
    symmetry_score: float
    compensation_score: float
    total_score: float
    jerk_value: float
    compensation_detected: List[str]

@dataclass
class SessionReport:
    session_id: str
    total_reps: int
    rep_scores: List[RepScore]
    average_scores: Dict[str, float]
    fatigue_analysis: Dict
    recommendations: List[str]

class HealthScorer:
    SCORE_WEIGHTS = {
        "rom": 0.30,
        "stability": 0.20,
        "flow": 0.20,
        "symmetry": 0.15,
        "compensation": 0.15,
    }
```

---

### 5.4 `target_generator.py` - Target Generator

**Vai trò**: Cá nhân hóa mục tiêu bằng cách co giãn video mẫu phù hợp với giới hạn vận động người dùng.

#### Công thức chính:

```
θ_target(t) = θ_ref(t) × (θ_user_max / max(θ_ref)) × (1 + α)

Trong đó:
- θ_ref(t): Góc trong video mẫu tại thời điểm t
- θ_user_max: Góc tối đa an toàn từ calibration
- max(θ_ref): Góc lớn nhất trong video mẫu
- α: Challenge Factor (mặc định 5%)
```

#### Ý nghĩa công thức:

```
Ví dụ: 
- Người già gập khuỷu tối đa 90°
- Video mẫu gập 120°
- Tỷ lệ = 90/120 = 0.75 (giảm 25%)
- Với α = 0.05: scale = 0.75 × 1.05 = 0.7875

Mục tiêu mới = góc_mẫu × 0.7875

Đảm bảo:
1. KHÔNG vượt quá khả năng người già
2. Có chút thử thách (5%) để khuyến khích tiến bộ
3. Tỷ lệ động tác được bảo toàn
```

---

### 5.5 `video_engine.py` - Video Engine

**Vai trò**: Smart Video Player với khả năng đồng bộ.

#### Tính năng:

```
- Tạm dừng tại checkpoint chờ người dùng
- Lặp lại đoạn video khi cần
- Nhảy đến frame cụ thể
- Điều khiển tốc độ phát
```

#### Classes:

```python
class PlaybackState(Enum):
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    LOOPING = auto()
    SEEKING = auto()
    FINISHED = auto()

@dataclass
class PlaybackStatus:
    state: PlaybackState
    current_frame: int
    progress: float
    is_at_checkpoint: bool
    loop_count: int

class VideoEngine:
    def set_checkpoints(self, frames: List[int]):
        """Đặt các điểm dừng"""
        
    def set_speed(self, factor: float):
        """Điều chỉnh tốc độ"""
        
    def get_frame(self) -> Tuple[np.ndarray, PlaybackStatus]:
        """Lấy frame tiếp theo"""
```

---

## 6. MODULE UTILS

### 6.1 `logger.py` - Session Logger

**Vai trò**: Ghi nhật ký chi tiết cho buổi tập.

#### Định dạng output:

```
- JSON: Cấu trúc đầy đủ cho phân tích
- CSV: Dễ mở bằng Excel cho bác sĩ
- Console: Real-time monitoring
```

#### Classes:

```python
class LogCategory(Enum):
    SESSION = "session"
    REP = "rep"
    PAIN = "pain"
    FATIGUE = "fatigue"
    SAFETY = "safety"
    SYNC = "sync"

class SessionLogger:
    def start_session(self, session_id, exercise_name):
        """Bắt đầu logging"""
        
    def log_rep(self, rep_number, scores, jerk, duration):
        """Log kết quả một rep"""
        
    def log_pain(self, level, score, au_scores, message):
        """Log cảnh báo đau"""
        
    def end_session(self, report) -> str:
        """Kết thúc và trả về path file log"""
```

---

### 6.2 `visualization.py` - Visualization

**Vai trò**: Các hàm vẽ UI, skeleton, text tiếng Việt.

#### Tính năng chính:

```python
def put_vietnamese_text(frame, text, position, color, font_size):
    """Vẽ text tiếng Việt (sử dụng PIL)"""

def draw_skeleton(frame, landmarks, color):
    """Vẽ skeleton lên frame"""

def draw_angle_arc(frame, center, angle, radius, color):
    """Vẽ cung thể hiện góc"""

def draw_panel(frame, position, size, title, content):
    """Vẽ panel thông tin"""

def draw_progress_bar(frame, position, width, progress, color):
    """Vẽ thanh progress"""

def draw_phase_indicator(frame, position, current_phase):
    """Vẽ indicator pha hiện tại"""

def combine_frames_horizontal(frames):
    """Ghép nhiều frame ngang"""
```

#### Color scheme:

```python
COLORS = {
    'skeleton': (0, 255, 0),        # Green
    'skeleton_ref': (0, 200, 255),  # Orange
    'highlight': (0, 0, 255),       # Red
    'success': (0, 255, 0),         # Green
    'warning': (0, 165, 255),       # Orange
    'error': (0, 0, 255),           # Red
}
```

---

## 7. LUỒNG HOẠT ĐỘNG CHÍNH

### 7.1 Main Loop Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN LOOP                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────┐                                                     │
│   │ Read User │──────────────────────────────────────┐              │
│   │   Frame   │                                      │              │
│   └─────┬─────┘                                      │              │
│         │                                            │              │
│         ▼                                            ▼              │
│   ┌───────────┐                              ┌───────────┐          │
│   │  Detect   │                              │ Read Ref  │          │
│   │   Pose    │                              │   Frame   │          │
│   └─────┬─────┘                              └─────┬─────┘          │
│         │                                          │                │
│         ▼                                          │                │
│   ┌───────────┐                                    │                │
│   │ Calculate │                                    │                │
│   │   Angle   │                                    │                │
│   └─────┬─────┘                                    │                │
│         │                                          │                │
│         ▼                                          │                │
│   ┌─────────────────────────────────────────┐      │                │
│   │           SYNCHRONIZER                  │◀─────┘                │
│   │  - Update FSM state                     │                       │
│   │  - Check if user reached checkpoint     │                       │
│   │  - Decide: PLAY / PAUSE / LOOP          │                       │
│   └─────────────────┬───────────────────────┘                       │
│                     │                                               │
│         ┌───────────┼───────────┐                                   │
│         ▼           ▼           ▼                                   │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│   │  Pain    │ │  Scorer  │ │  Logger  │                           │
│   │ Detector │ │ add_frame│ │  log     │                           │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘                           │
│        │            │            │                                  │
│        └────────────┼────────────┘                                  │
│                     │                                               │
│                     ▼                                               │
│              ┌───────────┐                                          │
│              │  Render   │                                          │
│              │  Display  │                                          │
│              └───────────┘                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Rep Complete Flow

```
Khi phát hiện: last_phase == CONCENTRIC và current_phase == IDLE

    ┌─────────────────────────────────────────┐
    │        ON REP COMPLETE                  │
    ├─────────────────────────────────────────┤
    │                                         │
    │  1. Compute DTW                         │
    │     - user_angles vs ref_angles         │
    │     - Lấy 50 frames gần nhất            │
    │                                         │
    │  2. Complete Rep in Scorer              │
    │     - Calculate ROM score               │
    │     - Calculate Stability score         │
    │     - Calculate Flow score (from DTW)   │
    │     - Calculate Symmetry score          │
    │     - Calculate Compensation score      │
    │     - Compute total score               │
    │                                         │
    │  3. Log Rep                             │
    │     - Write to JSON/CSV                 │
    │                                         │
    │  4. Check Fatigue                       │
    │     - Compare Jerk with baseline        │
    │                                         │
    │  5. Update Dashboard                    │
    │     - rep_count++                       │
    │     - Update scores display             │
    │                                         │
    └─────────────────────────────────────────┘
```

---

## 8. FILE MAIN ENTRY POINTS

### 8.1 `main_final.py` - Main Application

**Mục đích**: Entry point chính của hệ thống.

#### Class MemotionApp:

```python
class MemotionApp:
    """Ứng dụng MEMOTION hoàn chỉnh"""
    
    def __init__(self, detector, ref_video_path, joint_type, log_dir):
        self._detector = detector
        self._video_engine = None
        self._sync_controller = None
        self._pain_detector = PainDetector()
        self._scorer = HealthScorer()
        self._logger = SessionLogger(log_dir)
    
    def setup(self):
        """Khởi tạo video engine và sync controller"""
        
    def run(self, user_source, display=True):
        """Chạy main loop"""
        
    def cleanup(self):
        """Dọn dẹp tài nguyên"""
```

#### Controls:

```
SPACE: Pause/Resume hoặc Bắt đầu calibration
R:     Restart
Q:     Quit
1-6:   Chọn khớp để đo (Phase 2)
ENTER: Xác nhận/Chuyển phase tiếp theo
ESC:   Thoát
```

### 8.2 Các file test khác

| File | Mục đích |
|------|----------|
| `main_test.py` | Test các module riêng lẻ |
| `main_v2.py` | Version 2 với cải tiến |
| `main_sync_test.py` | Test đồng bộ video |
| `test_calibration.py` | Test calibration module |
| `comprehensive_audit.py` | Audit toàn diện hệ thống |

---

## 9. DATA FLOW

### 9.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │ Camera  │────▶│  Frame  │────▶│Detector │────▶│Detection│               │
│  │         │     │  BGR    │     │         │     │ Result  │               │
│  └─────────┘     └─────────┘     └─────────┘     └────┬────┘               │
│                                                       │                     │
│                                           ┌───────────┼───────────┐         │
│                                           ▼           ▼           ▼         │
│                                     ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│                                     │  Pose    │ │  Face    │ │  World   │ │
│                                     │Landmarks │ │Landmarks │ │Landmarks │ │
│                                     └────┬─────┘ └────┬─────┘ └──────────┘ │
│                                          │            │                     │
│                                          ▼            ▼                     │
│                                    ┌──────────┐ ┌──────────┐               │
│                                    │Kinematics│ │  Pain    │               │
│                                    │Calculate │ │ Detector │               │
│                                    │  Angle   │ │          │               │
│                                    └────┬─────┘ └────┬─────┘               │
│                                         │            │                      │
│                                         ▼            ▼                      │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐  ┌─────────┐                  │
│  │  Ref    │────▶│ Target  │────▶│  Sync   │  │  Pain   │                  │
│  │ Video   │     │ Angle   │     │ State   │  │  Level  │                  │
│  └─────────┘     └─────────┘     └────┬────┘  └────┬────┘                  │
│                                       │            │                        │
│                                       └──────┬─────┘                        │
│                                              ▼                              │
│                                       ┌─────────────┐                       │
│                                       │   Scorer    │                       │
│                                       │  Calculate  │                       │
│                                       │   Scores    │                       │
│                                       └──────┬──────┘                       │
│                                              │                              │
│                                              ▼                              │
│                                       ┌─────────────┐                       │
│                                       │   Logger    │                       │
│                                       │  Save to    │                       │
│                                       │  JSON/CSV   │                       │
│                                       └─────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Output Files Structure

```
data/
├── logs/
│   └── 20260121/
│       ├── session_1234567_143000.json    # Full session data
│       └── session_1234567_143000.csv     # Per-rep scores
│
└── user_profiles/
    └── user_20260121_143000.json          # Calibration data

JSON Structure:
{
  "session_id": "session_1234567_143000",
  "start_time": 1737463800.0,
  "end_time": 1737464100.0,
  "exercise_name": "arm_raise",
  "total_reps": 10,
  "rep_scores": [...],
  "average_scores": {...},
  "fatigue_analysis": {...},
  "pain_events": [...],
  "recommendations": [...]
}
```

---

## 10. CÔNG THỨC TOÁN HỌC

### 10.1 Tính góc giữa 3 điểm

```
Input: 3 điểm A, B, C (B là đỉnh góc)

Vector BA = A - B
Vector BC = C - B

cos(θ) = (BA · BC) / (|BA| × |BC|)
θ = arccos(cos(θ))

Output: θ (degrees, 0-180)
```

### 10.2 Target Scaling Formula

```
θ_target(t) = θ_ref(t) × scale_factor

scale_factor = (θ_user_max / max(θ_ref)) × (1 + α)

Với:
- θ_user_max: Góc max an toàn của user
- max(θ_ref): Góc max trong video mẫu
- α: Challenge factor (default 0.05)
```

### 10.3 DTW Distance

```
DTW(X, Y) = dtw_matrix[n, m]

dtw_matrix[i, j] = |x[i] - y[j]| + min(
    dtw_matrix[i-1, j],     # Insertion
    dtw_matrix[i, j-1],     # Deletion
    dtw_matrix[i-1, j-1]    # Match
)
```

### 10.4 Jerk Calculation

```
Jerk = d³x/dt³

Trong thực tế:
- velocity = Δposition / Δt
- acceleration = Δvelocity / Δt
- jerk = Δacceleration / Δt

Jerk value = mean(|jerk|)
```

### 10.5 Scoring Formula

```
Total Score = Σ (weight_i × score_i)

Với:
- ROM Score × 0.30
- Stability Score × 0.20
- Flow Score × 0.20
- Symmetry Score × 0.15
- Compensation Score × 0.15
```

---

## 11. HƯỚNG DẪN SỬ DỤNG

### 11.1 Cài đặt

```bash
# Tạo môi trường ảo
python -m venv med_venv
source med_venv/bin/activate  # Linux/Mac
# hoặc
med_venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 11.2 Tải model files

Download và đặt vào thư mục `models/`:

```
Pose Landmarker:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

Face Landmarker:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 11.3 Chạy ứng dụng

```bash
# Với webcam và video mẫu
python main_final.py --source webcam --ref-video exercise.mp4

# Test mode
python main_final.py --mode test

# Với video input
python main_final.py --source path/to/user_video.mp4 --ref-video exercise.mp4
```

### 11.4 Command line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | webcam | Input source (webcam hoặc path) |
| `--ref-video` | None | Video mẫu |
| `--joint` | left_shoulder | Khớp theo dõi |
| `--mode` | run | run hoặc test |
| `--headless` | False | Chạy không hiển thị |
| `--models-dir` | ./models | Thư mục models |
| `--log-dir` | ./data/logs | Thư mục logs |

---

## 📝 CHANGELOG

### Version 2.0.0
- Tích hợp 4 phases hoàn chỉnh
- Thêm Compensation Detection
- Cải thiện DTW comparison (user vs ref)
- Multi-threaded pain analysis

### Version 1.2.0
- Thêm DTW Analysis
- Thêm Fatigue Detection via Jerk

### Version 1.0.0
- Initial release
- Pose Detection
- Basic Calibration
- Pain Detection

---

## 🔗 REFERENCES

1. MediaPipe Tasks API: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
2. FACS Manual: Ekman, P., & Friesen, W. V. (1978)
3. DTW Algorithm: https://en.wikipedia.org/wiki/Dynamic_time_warping
4. Procrustes Analysis: https://en.wikipedia.org/wiki/Procrustes_analysis

---

> **Note**: Tài liệu này được tạo tự động dựa trên phân tích mã nguồn. Cập nhật khi có thay đổi lớn trong hệ thống.
