# 📁 MEMOTION - Hướng Dẫn Đọc Code

## 🎯 Tổng Quan Dự Án

**MEMOTION** là hệ thống hỗ trợ phục hồi chức năng cho **người già** sử dụng MediaPipe để nhận diện tư thế và theo dõi chuyển động.

### Triết lý thiết kế
> *"Người già thường di chuyển chậm hơn video mẫu. Thay vì ép họ theo kịp tốc độ (gây stress và nguy hiểm), hệ thống sẽ CHỜ người dùng hoàn thành từng pha, KHÔNG PHÁN XÉT về tốc độ, và KHUYẾN KHÍCH bằng phản hồi tích cực."*

### 4 Giai Đoạn Chính

| Phase | Tên | Mô tả |
|-------|-----|-------|
| **Phase 1** | Pose Detection | Nhận diện tư thế, vẽ skeleton |
| **Phase 2** | Safe-Max Calibration | Đo giới hạn vận động cá nhân (ROM) |
| **Phase 3** | Motion Sync | Đồng bộ với video mẫu (Wait-for-User) |
| **Phase 4** | Scoring & Analysis | Chấm điểm và phân tích mệt mỏi |

---

## 📂 Cấu Trúc Thư Mục

```
mediapipe/
│
├── 🧠 core/                    # MODULE CỐT LÕI (Thuật toán)
│   ├── __init__.py             # Export các class/function
│   ├── data_types.py           # Định nghĩa kiểu dữ liệu chuẩn
│   ├── detector.py             # Wrapper MediaPipe Tasks API
│   ├── procrustes.py           # Chuẩn hóa skeleton (Procrustes Analysis)
│   ├── kinematics.py           # Tính toán góc khớp
│   ├── synchronizer.py         # FSM đồng bộ chuyển động
│   └── dtw_analysis.py         # So sánh nhịp điệu (DTW)
│
├── 🔧 modules/                 # MODULE CHỨC NĂNG (Business Logic)
│   ├── __init__.py
│   ├── calibration.py          # Safe-Max Calibration
│   ├── target_generator.py     # Cá nhân hóa mục tiêu bài tập
│   ├── video_engine.py         # Smart Video Player
│   ├── pain_detection.py       # Nhận diện đau qua FACS
│   └── scoring.py              # Chấm điểm đa chiều
│
├── 🛠️ utils/                   # TIỆN ÍCH
│   ├── __init__.py
│   ├── logger.py               # Ghi log (JSON/CSV)
│   └── visualization.py        # Vẽ skeleton, UI, text tiếng Việt
│
├── 📦 models/                  # MODEL FILES (MediaPipe)
│   ├── face_landmarker.task
│   ├── pose_landmarker_full.task
│   └── pose_landmarker_lite.task
│
├── 💾 data/                    # DỮ LIỆU
│   ├── logs/                   # Log các buổi tập (theo ngày)
│   └── user_profiles/          # Profile người dùng (calibration)
│
├── 📝 scripts/                 # Script tham khảo
├── 🧪 test_logs/               # Log test
├── 🎬 videos/                  # Video mẫu bài tập
│
├── ⭐ main_v2.py               # FILE CHÍNH - Tích hợp 4 Phase
├── main_final.py               # Phiên bản gọn
├── main_test.py                # Test Phase 1
├── main_sync_test.py           # Test Phase 3
├── test_calibration.py         # Test Phase 2
└── requirements.txt            # Dependencies
```

---

## 📚 Chi Tiết Từng File

### 🧠 Folder `core/` - Lõi Thuật Toán

#### 1. `data_types.py` - Kiểu Dữ Liệu Chuẩn
```python
# Các class chính:
Point3D          # Điểm 3D (x, y, z, visibility, presence)
LandmarkSet      # Tập hợp landmarks của pose/face
DetectionResult  # Kết quả detection từ 1 frame
NormalizedSkeleton  # Skeleton đã chuẩn hóa
PoseLandmarkIndex   # Chỉ số 33 landmarks của MediaPipe Pose
```
**Vai trò**: Foundation cho toàn hệ thống, định nghĩa cấu trúc dữ liệu chuẩn.

#### 2. `detector.py` - Nhận Diện Tư Thế
```python
# Các class chính:
DetectorConfig   # Cấu hình (model path, confidence thresholds)
VisionDetector   # Wrapper cho MediaPipe Tasks API

# Sử dụng:
detector = VisionDetector(config)
result = detector.process_frame(frame, timestamp_ms)
```
**Vai trò**: Detect Pose (33 landmarks) và Face (478 landmarks) từ video/camera.

#### 3. `kinematics.py` - Tính Góc Khớp
```python
# Công thức toán học:
# cos(θ) = (BA · BC) / (|BA| × |BC|)

# Các class/function chính:
JointType        # Enum các khớp (LEFT_ELBOW, RIGHT_KNEE, ...)
JointDefinition  # Định nghĩa khớp bằng 3 landmark indices
JOINT_DEFINITIONS  # Dict mapping JointType → JointDefinition

calculate_angle()       # Tính góc giữa 3 điểm
calculate_joint_angle() # Tính góc của 1 khớp cụ thể
```
**Vai trò**: Tính góc các khớp từ pose landmarks.

#### 4. `procrustes.py` - Chuẩn Hóa Skeleton
```python
# 3 bước Procrustes Analysis:
# 1. Translation: Dịch centroid về gốc tọa độ
# 2. Scaling: Chuẩn hóa về unit norm
# 3. Rotation: Xoay để minimize khoảng cách với reference

# Các function chính:
normalize_skeleton()           # Chuẩn hóa 1 skeleton
align_skeleton_to_reference()  # Căn chỉnh theo reference
compute_procrustes_distance()  # Tính khoảng cách sau căn chỉnh
```
**Vai trò**: Loại bỏ khác biệt vị trí/kích thước/hướng để so sánh công bằng.

#### 5. `synchronizer.py` - FSM Đồng Bộ
```python
# Mô hình FSM:
# IDLE → ECCENTRIC → HOLD → CONCENTRIC → IDLE

# Các class chính:
MotionPhase      # Enum: IDLE, ECCENTRIC, HOLD, CONCENTRIC
SyncStatus       # Enum: PLAY, PAUSE, LOOP, SKIP, COMPLETE
PhaseCheckpoint  # Điểm mốc trong video mẫu
MotionSyncController  # Bộ điều khiển đồng bộ

# Nguyên tắc "Wait-for-User":
# Video chờ tại checkpoint cho đến khi user đạt ngưỡng góc
```
**Vai trò**: Điều khiển video mẫu đồng bộ với chuyển động người dùng.

#### 6. `dtw_analysis.py` - So Sánh Nhịp Điệu
```python
# Dynamic Time Warping - "kéo giãn" thời gian để so sánh

# Các function chính:
compute_dtw_distance()   # DTW cho 2 chuỗi 1D
compute_weighted_dtw()   # Weighted DTW cho nhiều khớp
DTWResult               # Kết quả: distance, similarity_score, rhythm_quality
```
**Vai trò**: So sánh nhịp điệu chuyển động dù tốc độ khác nhau.

---

### 🔧 Folder `modules/` - Business Logic

#### 1. `calibration.py` - Đo Giới Hạn Vận Động
```python
# Quy trình Safe-Max Calibration:
# 1. Người dùng thực hiện động tác "hết khả năng" (KHÔNG GÂY ĐAU)
# 2. Thu thập góc khớp trong 5-10 giây
# 3. Áp dụng Median Filter loại bỏ nhiễu
# 4. Trích xuất θ_user_max

# Các class chính:
SafeMaxCalibrator     # Bộ calibration
UserProfile           # Profile người dùng
JointCalibrationData  # Data calibration của 1 khớp
CalibrationState      # Enum trạng thái
```
**Vai trò**: Xác định giới hạn vận động AN TOÀN của từng người.

#### 2. `target_generator.py` - Cá Nhân Hóa Mục Tiêu
```python
# Công thức chính:
# θ_target(t) = θ_ref(t) × (θ_user_max / max(θ_ref)) × (1 + α)
#
# α = Challenge Factor (mặc định 5%)

# Các function chính:
compute_scale_factor()       # Tính hệ số scale
rescale_reference_motion()   # Co giãn chuỗi góc từ video mẫu
RescaledMotion              # Kết quả rescale
```
**Vai trò**: Co giãn video mẫu phù hợp với năng lực từng người.

#### 3. `video_engine.py` - Smart Video Player
```python
# Các chế độ đặc biệt:
# - Wait-at-checkpoint: Dừng tại điểm mốc
# - Loop-segment: Lặp lại đoạn
# - Speed control: Điều chỉnh tốc độ

# Các class chính:
VideoEngine     # Smart Video Player
VideoInfo       # Thông tin video
PlaybackState   # Enum: PLAYING, PAUSED, LOOPING, ...
PlaybackStatus  # Trạng thái hiện tại
```
**Vai trò**: Phát video mẫu với khả năng tạm dừng/lặp thông minh.

#### 4. `pain_detection.py` - Nhận Diện Đau
```python
# Sử dụng FACS (Facial Action Coding System):
# - AU4: Cau mày
# - AU6/7: Nheo mắt
# - AU9/10: Nhăn mũi/môi
# - AU43: Nhắm mắt

# Các class chính:
PainDetector        # Bộ phát hiện đau
PainLevel           # Enum: NONE, MILD, MODERATE, SEVERE
PainEvent           # Ghi nhận sự kiện đau
PainAnalysisResult  # Kết quả phân tích
```
**Vai trò**: Phát hiện đau qua biểu cảm mặt để dừng bài tập kịp thời.

#### 5. `scoring.py` - Chấm Điểm Đa Chiều
```python
# 5 chỉ số đánh giá:
# 1. ROM Score: Mức độ đạt góc mục tiêu
# 2. Stability Score: Độ ổn định trong pha HOLD
# 3. Flow Score: Độ mượt mà (từ DTW)
# 4. Symmetry Score: Cân bằng trái-phải
# 5. Compensation Score: Điểm bù trừ

# Phát hiện mệt mỏi qua Jerk (đạo hàm bậc 3):
# Jerk tăng dần qua các rep = dấu hiệu mệt mỏi

# Các class chính:
HealthScorer    # Bộ chấm điểm
RepScore        # Điểm của 1 rep
SessionReport   # Báo cáo buổi tập
FatigueLevel    # Enum: FRESH, LIGHT, MODERATE, HEAVY
```
**Vai trò**: Đánh giá chất lượng tập luyện và phát hiện mệt mỏi.

---

### 🛠️ Folder `utils/` - Tiện Ích

#### 1. `logger.py` - Ghi Log
```python
# Output formats:
# - JSON: Cấu trúc đầy đủ cho phân tích
# - CSV: Dễ mở bằng Excel
# - Console: Real-time monitoring

# Các class chính:
SessionLogger  # Logger cho 1 buổi tập (thread-safe)
LogEntry       # 1 entry trong log
LogLevel       # DEBUG, INFO, WARNING, ERROR
LogCategory    # SESSION, REP, PAIN, FATIGUE, ...
```

#### 2. `visualization.py` - Vẽ UI
```python
# Các function chính:
draw_skeleton()        # Vẽ skeleton lên frame
put_vietnamese_text()  # Vẽ text tiếng Việt (dùng PIL)
draw_panel()           # Vẽ panel thông tin
draw_progress_bar()    # Vẽ progress bar
draw_angle_arc()       # Vẽ cung góc tại khớp

# Hỗ trợ font tiếng Việt qua PIL
VietnameseTextRenderer
```

---

## 📖 Thứ Tự Đọc Code Khuyến Nghị

### Cách 1: Bottom-Up (Hiểu từ nền tảng)

```
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 1: Đọc Data Types (Foundation)                    │
│  ► core/data_types.py                                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 2: Đọc Core Algorithms                            │
│  ► core/detector.py      (Nhận diện)                    │
│  ► core/kinematics.py    (Tính góc)                     │
│  ► core/procrustes.py    (Chuẩn hóa)                    │
│  ► core/synchronizer.py  (FSM)                          │
│  ► core/dtw_analysis.py  (DTW)                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 3: Đọc Business Modules                           │
│  ► modules/calibration.py      (Phase 2)                │
│  ► modules/target_generator.py (Phase 2)                │
│  ► modules/video_engine.py     (Phase 3)                │
│  ► modules/pain_detection.py   (Phase 4)                │
│  ► modules/scoring.py          (Phase 4)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 4: Đọc Utils                                      │
│  ► utils/logger.py                                      │
│  ► utils/visualization.py                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 5: Đọc Integration                                │
│  ► main_v2.py (Xem cách tích hợp tất cả)                │
└─────────────────────────────────────────────────────────┘
```

### Cách 2: Top-Down (Hiểu flow trước)

```
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 1: Đọc main_v2.py (dòng 1-200)                    │
│  → Hiểu 4 Phase, AppState, flow tổng quan               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 2: Đọc theo từng Phase                            │
│  Phase 1: core/detector.py, core/kinematics.py          │
│  Phase 2: modules/calibration.py, target_generator.py   │
│  Phase 3: core/synchronizer.py, modules/video_engine.py │
│  Phase 4: modules/scoring.py, modules/pain_detection.py │
└─────────────────────────────────────────────────────────┘
```

### Cách 3: Theo Test Files

```
1. main_test.py         → Hiểu Phase 1 (Pose Detection)
2. test_calibration.py  → Hiểu Phase 2 (Calibration)
3. main_sync_test.py    → Hiểu Phase 3 (Motion Sync)
4. main_v2.py           → Hiểu tích hợp hoàn chỉnh
```

---

## 🔄 Luồng Dữ Liệu

```
┌──────────────────┐
│  Camera/Video    │
└────────┬─────────┘
         ▼
┌──────────────────┐     ┌─────────────────────────┐
│  VisionDetector  │────▶│  DetectionResult        │
└──────────────────┘     │  ├── pose_landmarks     │
                         │  └── face_landmarks     │
                         └───────────┬─────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Kinematics     │        │  Procrustes     │        │  PainDetector   │
│  (Tính góc)     │        │  (Chuẩn hóa)    │        │  (FACS)         │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Calibration    │        │  Synchronizer   │        │  PainLevel      │
│  (θ_user_max)   │        │  (FSM)          │        │                 │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                         ┌─────────────────┐
                         │  HealthScorer   │
                         │  (Chấm điểm)    │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │  SessionReport  │
                         │  + SessionLogger│
                         └─────────────────┘
```

---

## 🚀 Chạy Thử

### Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng chính
```bash
# Với webcam
python main_v2.py --source webcam

# Với video
python main_v2.py --source video.mp4 --ref-video exercise.mp4

# Mode test
python main_v2.py --mode test
```

### Điều khiển
| Phím | Chức năng |
|------|-----------|
| `SPACE` | Pause/Resume hoặc Bắt đầu calibration |
| `1-6` | Chọn khớp để đo (Phase 2) |
| `ENTER` | Xác nhận/Chuyển phase tiếp theo |
| `R` | Restart |
| `Q/ESC` | Thoát |

---

## 📝 Tips Khi Đọc Code

1. **Đọc docstring trước** - Mỗi file có docstring chi tiết giải thích mục đích

2. **Chú ý dataclass** - `data_types.py` là foundation, hiểu nó trước

3. **Hiểu FSM** - `synchronizer.py` là trái tim của Phase 3

4. **Đọc công thức toán** - Có trong docstring của `kinematics.py`, `procrustes.py`

5. **Chạy test riêng lẻ** - Dùng `main_test.py`, `test_calibration.py` để hiểu từng phần

---

## 📞 Liên Hệ

**MEMOTION Team**  
Version: 2.0.0
