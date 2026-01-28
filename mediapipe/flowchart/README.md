# MEMOTION V2.0 - Flowcharts

Tập hợp các sơ đồ mô tả kiến trúc và luồng hoạt động của hệ thống MEMOTION V2.0.

## Danh sách sơ đồ

### 1. Kiến trúc tổng quan
**File**: `01_architecture.png`

Mô tả cấu trúc tổng thể của ứng dụng bao gồm:
- AppState (Dataclass) - Quản lý trạng thái
- MemotionAppV2 - Class chính
- Components (VisionDetector, VideoEngine, etc.)
- Input/Output sources

### 2. State Machine - AppPhase
**File**: `02_state_machine.png`

Sơ đồ trạng thái 5 phases:
- PHASE1_DETECTION → PHASE2_CALIBRATION → PHASE3_SYNC → PHASE4_SCORING → COMPLETED
- Auto transition với countdown

### 3. Phase 1: Pose Detection
**File**: `03_phase1_detection.png`

Luồng nhận diện tư thế:
- Detect skeleton với MediaPipe
- Đợi 30 frames stable
- Countdown 3 giây
- Auto chuyển Phase 2

### 4. Phase 2: Automated Calibration
**File**: `04_phase2_calibration.png`

Quy trình đo giới hạn vận động 6 khớp:
- Queue: LEFT_SHOULDER → RIGHT_SHOULDER → LEFT_ELBOW → RIGHT_ELBOW → LEFT_KNEE → RIGHT_KNEE
- Countdown 5 giây mỗi khớp
- Thu thập góc với median filter
- Auto chuyển Phase 3 sau 2 giây

### 5. Phase 3: Motion Sync
**File**: `05_phase3_sync.png`

Đồng bộ chuyển động multi-joint:
- Tính góc tất cả khớp
- Interpolate target angles
- Tính điểm có trọng số
- Score smoothing
- 3-panel layout

### 6. Phase 4: Scoring
**File**: `06_phase4_scoring.png`

Hiển thị kết quả:
- Tính tổng điểm
- Xác định grade (XUAT SAC/KHA/CAN CO GANG)
- Chi tiết điểm (ROM, Stability, Flow, Symmetry, Compensation)
- Lưu log và báo cáo

### 7. Main Loop
**File**: `07_main_loop.png`

Luồng chính của hàm `run()`:
- Init components
- Mở camera
- Loop xử lý frame
- Handle key

### 8. Motion Phase FSM
**File**: `08_motion_phase_fsm.png`

State machine cho motion phases:
- IDLE → ECCENTRIC → HOLD → CONCENTRIC → IDLE
- Mỗi cycle = 1 rep

### 9. Scoring Algorithm
**File**: `09_scoring_algorithm.png`

Thuật toán tính điểm:
- Single joint score (error percentage → score)
- Multi-joint weighted score
- Score smoothing (0.7 * old + 0.3 * new)

### 10. Data Structures
**File**: `10_data_structures.png`

Class diagram:
- AppPhase (Enum)
- AppState (Dataclass)
- JointType (Enum)
- MemotionAppV2 (Main Class)

### 11. Calibration Single Joint
**File**: `11_calibration_single_joint.png`

Quy trình đo một khớp:
- Hiển thị hướng dẫn tư thế
- Countdown 5 giây
- Collecting state
- Median filter
- Lưu max angle

### 12. Complete Flow
**File**: `12_complete_flow.png`

Tổng quan toàn bộ luồng hoạt động:
- Phase 1 → Phase 2 → Phase 3 → Phase 4
- Auto transitions
- Restart flow

## Cách sử dụng

### Xem ảnh
Mở trực tiếp các file `.png` bằng image viewer.

### Chỉnh sửa sơ đồ
1. Mở file `.mmd` tương ứng
2. Chỉnh sửa code Mermaid
3. Render lại bằng lệnh:
```bash
npx @mermaid-js/mermaid-cli -i <file>.mmd -o <file>.png -c mermaid-config.json -b white
```

### Render tất cả
```bash
for file in *.mmd; do
  npx @mermaid-js/mermaid-cli -i "$file" -o "${file%.mmd}.png" -c mermaid-config.json -b white
done
```

## Style
- Theme: Black & White
- No icons
- Professional layout
- Clear, readable text
