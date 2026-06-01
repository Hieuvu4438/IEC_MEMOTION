# MEMOTION V2.0 - TAI LIEU KY THUAT CHI TIET

> **File**: `main_v2.py` + `mediapipe_be/`
> **Version**: 2.0.0
> **Author**: MEMOTION Team
> **Last Updated**: 2026-01-22

---

## MUC LUC

1. [Tong quan](#1-tong-quan)
2. [Kien truc he thong](#2-kien-truc-he-thong)
3. [4 Giai doan hoat dong](#3-bon-giai-doan-hoat-dong)
4. [Cac thuat toan su dung](#4-cac-thuat-toan-su-dung)
5. [Backend Integration (mediapipe_be)](#5-backend-integration)
6. [Data Structures](#6-data-structures)
7. [Luong hoat dong chi tiet](#7-luong-hoat-dong-chi-tiet)
8. [Cong thuc tinh diem](#8-cong-thuc-tinh-diem)
9. [Huong dan su dung](#9-huong-dan-su-dung)

---

## 1. TONG QUAN

### 1.1 Muc dich

**MEMOTION** la he thong ho tro phuc hoi chuc nang cho nguoi gia su dung Computer Vision. He thong tap trung vao:

- **An toan**: Khong ep nguoi dung vuot qua gioi han van dong
- **Ca nhan hoa**: Dieu chinh muc tieu dua tren kha nang cua tung nguoi
- **Theo doi dau**: Tu dong phat hien khi nguoi dung dau (qua FACS)
- **Khuyen khich**: Phan hoi tich cuc, khong phan xet

### 1.2 Tinh nang chinh

| Tinh nang | Mo ta |
|-----------|-------|
| **4 Phases** | Pose Detection -> Calibration -> Motion Sync -> Scoring |
| **Auto Transition** | Tu dong chuyen phase khong can nhan ENTER |
| **Automated Calibration** | Tu dong do 6 khop theo thu tu dinh san |
| **Multi-joint Tracking** | Theo doi va tinh diem tat ca khop da calibrate |
| **Weighted Scoring** | Diem co trong so theo loai bai tap |
| **Pain Detection** | Phat hien dau qua bieu cam khuon mat (FACS) |
| **DTW Analysis** | So sanh nhip dieu chuyen dong |

### 1.3 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Ngon ngu chinh |
| **MediaPipe Tasks API** | Pose detection (33 landmarks), Face detection (478 landmarks) |
| **OpenCV** | Xu ly video & hien thi |
| **NumPy / SciPy** | Tinh toan khoa hoc, Procrustes analysis |
| **FastDTW** | Dynamic Time Warping cho so sanh nhip dieu |

---

## 2. KIEN TRUC HE THONG

### 2.1 So do tong quan

```
+-------------------------------------------------------------------------+
|                          MEMOTION SYSTEM                                |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------------+     +----------------------+                  |
|  |   main_v2.py (UI)    |     | mediapipe_be (API)   |                  |
|  |   MemotionAppV2      |     | MemotionEngine       |                  |
|  +----------+-----------+     +----------+-----------+                  |
|             |                            |                              |
|             +------------+---------------+                              |
|                          |                                              |
|             +------------v-----------+                                  |
|             |      CORE MODULES      |                                  |
|             +------------------------+                                  |
|             |                        |                                  |
|  +----------v----------+  +----------v----------+                       |
|  | core/               |  | modules/            |                       |
|  | - detector.py       |  | - calibration.py    |                       |
|  | - kinematics.py     |  | - scoring.py        |                       |
|  | - synchronizer.py   |  | - pain_detection.py |                       |
|  | - dtw_analysis.py   |  | - video_engine.py   |                       |
|  | - procrustes.py     |  | - target_generator  |                       |
|  +---------------------+  +---------------------+                       |
|                                                                         |
+-------------------------------------------------------------------------+
```

### 2.2 Hai phien ban

| Phien ban | Location | Muc dich | Co UI |
|-----------|----------|----------|-------|
| **Demo/Test** | `mediapipe/main_v2.py` | Test local voi OpenCV | Co |
| **Backend** | `mediapipe_be/service/` | Production (FastAPI, WebSocket) | Khong |

### 2.3 State Machine - AppPhase

```
+----------------------------------------------------------------------+
|                    APPLICATION PHASES (AUTO TRANSITION)               |
+----------------------------------------------------------------------+
|                                                                       |
|   +-----------------+                                                 |
|   |     PHASE 1     |   Pose Detection                                |
|   |    DETECTION    |   - Nhan dien skeleton                          |
|   |                 |   - Doi stable 30 frames                        |
|   +--------+--------+                                                 |
|            | [AUTO] Countdown 3 giay khi pose_detected                |
|            v                                                          |
|   +-----------------+                                                 |
|   |     PHASE 2     |   Automated Calibration                         |
|   |   CALIBRATION   |   - Tu dong do 6 khop theo thu tu               |
|   |   (AUTOMATED)   |   - Countdown 5 giay cho moi khop               |
|   +--------+--------+                                                 |
|            | [AUTO] 2 giay sau khi do xong 6 khop                     |
|            v                                                          |
|   +-----------------+                                                 |
|   |     PHASE 3     |   Motion Sync (Multi-joint)                     |
|   |      SYNC       |   - Dong bo voi video mau                       |
|   |  (MULTI-JOINT)  |   - Tinh diem real-time cho TAT CA khop         |
|   +--------+--------+                                                 |
|            | [AUTO] Khi video ket thuc / SyncStatus.COMPLETE          |
|            v                                                          |
|   +-----------------+                                                 |
|   |     PHASE 4     |   Scoring                                       |
|   |    SCORING      |   - Hien thi ket qua                            |
|   |                 |   - Luu bao cao                                 |
|   +-----------------+                                                 |
|                                                                       |
+----------------------------------------------------------------------+
```

---

## 3. BON GIAI DOAN HOAT DONG

### 3.1 PHASE 1: Pose Detection

**Muc dich**: Nhan dien tu the nguoi dung, dam bao MediaPipe detect duoc skeleton on dinh.

**Thuat toan**:
```python
PHASE1_STABLE_FRAMES_REQUIRED = 30
PHASE1_COUNTDOWN_DURATION = 3.0  # giay

def _run_phase1(self, frame, result):
    if result.has_pose():
        self._state.detection_stable_count += 1
        progress = detection_stable_count / PHASE1_STABLE_FRAMES_REQUIRED

        if detection_stable_count >= PHASE1_STABLE_FRAMES_REQUIRED:
            self._state.pose_detected = True

            # Bat dau countdown
            if not phase1_countdown_active:
                phase1_countdown_start = time.time()
                phase1_countdown_active = True

            # Kiem tra countdown
            elapsed = time.time() - phase1_countdown_start
            if elapsed >= PHASE1_COUNTDOWN_DURATION:
                self._transition_to_phase2()  # AUTO TRANSITION
    else:
        # Mat pose -> Reset
        detection_stable_count = 0
        phase1_countdown_active = False
```

**Dieu kien chuyen Phase 2**:
- `pose_detected == True` (sau 30 frames stable)
- **TU DONG** sau countdown 3 giay

---

### 3.2 PHASE 2: Automated Calibration

**Muc dich**: Tu dong do gioi han van dong (Range of Motion) an toan cua 6 khop.

**Calibration Queue** (thu tu do):
```python
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,   # 1. Vai trai
    JointType.RIGHT_SHOULDER,  # 2. Vai phai
    JointType.LEFT_ELBOW,      # 3. Khuyu tay trai
    JointType.RIGHT_ELBOW,     # 4. Khuyu tay phai
    JointType.LEFT_KNEE,       # 5. Dau goi trai
    JointType.RIGHT_KNEE,      # 6. Dau goi phai
]
```

**Thuat toan Safe-Max Calibration**:
```
+---------------------------+
| THU THAP GOC (5 giay)     |
| - Ghi nhan goc moi frame  |
| - raw_angles = [...]      |
+------------+--------------+
             |
             v
+---------------------------+
| MEDIAN FILTER             |
| - Lam muot chuoi goc      |
| - Loai bo nhieu           |
+------------+--------------+
             |
             v
+---------------------------+
| LOAI BO OUTLIERS          |
| - Loai bo goc ngoai       |
|   2 * standard deviation  |
+------------+--------------+
             |
             v
+---------------------------+
| TINH MAX ON DINH          |
| - Percentile 95           |
| - Khong lay max tuyet doi |
+---------------------------+
```

**Code chi tiet**:
```python
def finish_calibration(self):
    angles = np.array(collected_angles)

    # Step 1: Median Filter - lam muot
    smoothed_angles = self._median_filter(angles)

    # Step 2: Loai bo outliers (ngoai 2 std)
    mean = np.mean(smoothed_angles)
    std = np.std(smoothed_angles)
    mask = (smoothed_angles >= mean - 2*std) & (smoothed_angles <= mean + 2*std)
    filtered_angles = smoothed_angles[mask]

    # Step 3: Percentile 95 (khong lay max tuyet doi)
    max_angle = np.percentile(filtered_angles, 95)

    # Tinh do tin cay
    confidence = max(0.0, 1.0 - (std / 30.0))

    return JointCalibrationData(max_angle=max_angle, confidence=confidence)
```

**Dieu kien chuyen Phase 3**:
- `all_joints_calibrated == True` (da do xong 6 khop)
- **TU DONG** sau 2 giay

---

### 3.3 PHASE 3: Motion Sync (Multi-joint)

**Muc dich**: Dong bo chuyen dong nguoi dung voi video mau, tinh diem real-time cho **TAT CA cac khop da calibrated**.

**Finite State Machine (FSM) cho Motion Phase**:
```
+-------+     +------------+     +------+     +------------+
| IDLE  | --> | ECCENTRIC  | --> | HOLD | --> | CONCENTRIC |
+---+---+     +------------+     +------+     +-----+------+
    ^                                               |
    |                                               |
    +-----------------------------------------------+
                    (1 rep complete)
```

**Giai thich cac pha**:
| Phase | Tieng Viet | Mo ta |
|-------|------------|-------|
| IDLE | Nghi | Tu the nghi, chuan bi bat dau |
| ECCENTRIC | Duoi co | Pha "di ra" - co duoi ra (vd: ha nguoi xuong squat) |
| HOLD | Giu | Giu tai diem cao trao (vd: day squat) |
| CONCENTRIC | Co co | Pha "di ve" - co co lai (vd: dung len tu squat) |

**Multi-joint Tracking Flow**:
```python
def _run_phase3(self, user_frame, ref_frame, result, timestamp):
    # === TINH GOC CHO TAT CA KHOP ===
    user_angles_dict = {}
    for joint_type in active_joints:
        angle = calculate_joint_angle(landmarks, joint_type, use_3d=True)
        user_angles_dict[joint_type] = angle

    # === TINH TARGET CHO TAT CA KHOP ===
    target_angles_dict = {}
    for joint_type in active_joints:
        target = _interpolate_target_angle(current_frame, total_frames, joint_type)
        target_angles_dict[joint_type] = target

    # === TINH DIEM CO TRONG SO ===
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

    multi_joint_score = total_weighted_score / total_weight

    # === SMOOTH SCORE ===
    current_score = 0.7 * current_score + 0.3 * multi_joint_score
```

**Wait-for-User Logic**:
```python
def _check_sync_status(self, user_angle, ref_frame, timestamp):
    next_checkpoint = get_next_checkpoint(ref_frame)

    if next_checkpoint is None:
        if ref_frame >= total_frames - 1:
            return SyncStatus.COMPLETE
        return SyncStatus.PLAY

    # Chua den checkpoint -> chay binh thuong
    if ref_frame < next_checkpoint.frame_index:
        return SyncStatus.PLAY

    # Da den checkpoint - kiem tra user
    if next_checkpoint.is_reached(user_angle):
        return SyncStatus.PLAY  # User dat -> tiep tuc

    # User chua dat -> cho
    if wait_duration > MAX_WAIT_TIME:
        return SyncStatus.SKIP  # Cho qua lau -> bo qua

    return SyncStatus.PAUSE  # Cho user
```

---

### 3.4 PHASE 4: Scoring & Results

**Muc dich**: Hien thi ket qua buoi tap, luu bao cao.

**Grade System**:
```python
def get_grade(score: float) -> tuple:
    if score >= 80:
        return ("XUAT SAC", "green")
    elif score >= 60:
        return ("KHA", "yellow")
    else:
        return ("CAN CO GANG", "red")
```

---

## 4. CAC THUAT TOAN SU DUNG

### 4.1 Tinh goc khop (Kinematics)

**Cong thuc toan hoc**:
```
Goc giua 3 diem A, B, C (voi B la dinh goc):

Vector BA = A - B
Vector BC = C - B

cos(theta) = (BA . BC) / (|BA| x |BC|)
theta = arccos(cos(theta))
```

**Code**:
```python
def calculate_angle(point_a, point_b, point_c, use_3d=True):
    # Chuyen doi ve numpy array
    a = _to_numpy(point_a, use_3d)
    b = _to_numpy(point_b, use_3d)
    c = _to_numpy(point_c, use_3d)

    # Tao vector tu dinh goc B
    vector_ba = a - b
    vector_bc = c - b

    # Tinh do dai (norm) cua cac vector
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    # Tinh dot product
    dot_product = np.dot(vector_ba, vector_bc)

    # Tinh cosine cua goc
    cos_angle = dot_product / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Tranh loi so hoc

    # Tinh goc bang arccos
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return float(angle_degrees)
```

**Dinh nghia cac khop**:
```python
JOINT_DEFINITIONS = {
    # Khuyu tay: Vai -> Khuyu -> Co tay
    JointType.LEFT_ELBOW: JointDefinition(
        proximal=LEFT_SHOULDER,  # A
        vertex=LEFT_ELBOW,       # B (dinh goc)
        distal=LEFT_WRIST,       # C
        normal_range=(0.0, 145.0)
    ),

    # Vai: Hong -> Vai -> Khuyu (do goc dang tay)
    JointType.LEFT_SHOULDER: JointDefinition(
        proximal=LEFT_HIP,
        vertex=LEFT_SHOULDER,
        distal=LEFT_ELBOW,
        normal_range=(0.0, 180.0)
    ),

    # Dau goi: Hong -> Dau goi -> Mat ca
    JointType.LEFT_KNEE: JointDefinition(
        proximal=LEFT_HIP,
        vertex=LEFT_KNEE,
        distal=LEFT_ANKLE,
        normal_range=(0.0, 140.0)
    ),
}
```

---

### 4.2 Dynamic Time Warping (DTW)

**Muc dich**: So sanh nhip dieu chuyen dong giua nguoi dung va video mau.

**Tai sao can DTW thay vi so sanh truc tiep?**
- Nguoi gia di chuyen voi toc do khac nhau
- Co the dung lai giua chung
- DTW "keo gian" thoi gian de tim su tuong dong toi uu

**Cong thuc**:
```
DTW Matrix: D[i, j] = cost(i, j) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

Trong do:
  - cost(i, j) = |seq1[i] - seq2[j]|
  - D[0, 0] = 0
  - D[i, 0] = infinity (i > 0)
  - D[0, j] = infinity (j > 0)
```

**Code Simple DTW**:
```python
def _simple_dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)

    # Ma tran chi phi tich luy
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Dien ma tran
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # Insertion
                dtw_matrix[i, j-1],     # Deletion
                dtw_matrix[i-1, j-1]    # Match
            )

    # Backtrack de tim duong di
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        # ... backtrack logic

    return dtw_matrix[n, m], path
```

**Weighted DTW cho nhieu khop**:
```python
def compute_weighted_dtw(user_sequences, ref_sequences, weights):
    """
    Total Distance = sum(weight_i * dtw_distance_i) / sum(weight_i)

    Example weights cho bai tap gio tay:
        weights = {
            JointType.LEFT_SHOULDER: 1.0,   # Quan trong nhat
            JointType.LEFT_ELBOW: 0.7,
            JointType.LEFT_KNEE: 0.1,       # Khong lien quan
        }
    """
    total_weighted_distance = 0.0
    total_weight = 0.0

    for joint_type, user_seq in user_sequences.items():
        ref_seq = ref_sequences[joint_type]
        weight = weights[joint_type]

        # Tien xu ly: lam muot + chuan hoa
        user_processed = preprocess_sequence(user_seq)
        ref_processed = preprocess_sequence(ref_seq)

        # Tinh DTW
        distance, path = compute_dtw_distance(user_processed, ref_processed)
        normalized = distance / max(len(user_seq), len(ref_seq))

        total_weighted_distance += weight * normalized
        total_weight += weight

    final_distance = total_weighted_distance / total_weight

    # Chuyen doi sang similarity score (0-100)
    similarity_score = 100.0 * np.exp(-final_distance * 3)

    return DTWResult(
        distance=total_weighted_distance,
        normalized_distance=final_distance,
        similarity_score=similarity_score,
        rhythm_quality=_evaluate_rhythm_quality(similarity_score)
    )
```

**Tien xu ly chuoi truoc khi tinh DTW**:
```python
def preprocess_sequence(sequence, smooth_window=5, normalize=True):
    arr = np.array(sequence, dtype=np.float64)

    # 1. Lam muot bang moving average (giam nhieu)
    if smooth_window > 1:
        arr = uniform_filter1d(arr, size=smooth_window, mode='nearest')

    # 2. Chuan hoa ve [0, 1] (de so sanh cong bang)
    if normalize:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val)

    return arr
```

---

### 4.3 Jerk Analysis (Phat hien met moi)

**Dinh nghia**:
```
Jerk = d^3x/dt^3 (dao ham bac 3 cua vi tri)

Y nghia:
- Jerk thap = chuyen dong muot ma
- Jerk cao = chuyen dong giat, khong kiem soat
- Jerk tang dan qua cac rep = dau hieu met moi
```

**Code**:
```python
def _calculate_jerk(angles, timestamps):
    if len(angles) < 4:
        return 0.0

    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1e-6, dt)  # Tranh chia cho 0

    # Velocity (dao ham bac 1)
    velocity = np.diff(angles) / dt

    # Acceleration (dao ham bac 2)
    dt2 = dt[:-1]
    acceleration = np.diff(velocity) / dt2

    # Jerk (dao ham bac 3)
    dt3 = dt2[:-1]
    jerk = np.diff(acceleration) / dt3

    # Squared Jerk (chuan hoa theo thoi gian)
    total_time = timestamps[-1] - timestamps[0]
    squared_jerk = np.sum(jerk ** 2) / total_time

    return squared_jerk
```

**Phat hien muc do met moi**:
```python
JERK_THRESHOLDS = {
    FatigueLevel.LIGHT: 1.5,      # Tang 50%
    FatigueLevel.MODERATE: 2.0,   # Tang 100%
    FatigueLevel.HEAVY: 3.0,      # Tang 200%
}

def _check_fatigue(self):
    if baseline_jerk is None:
        return FatigueLevel.FRESH

    current_jerk = jerk_values[-1]
    jerk_ratio = current_jerk / baseline_jerk

    if jerk_ratio >= 3.0:
        return FatigueLevel.HEAVY
    elif jerk_ratio >= 2.0:
        return FatigueLevel.MODERATE
    elif jerk_ratio >= 1.5:
        return FatigueLevel.LIGHT
    else:
        return FatigueLevel.FRESH
```

---

### 4.4 Pain Detection (FACS)

**Facial Action Coding System (FACS)**: He thong ma hoa bieu cam khuon mat de phat hien dau.

**Cac Action Units (AU) lien quan den dau**:
```python
PAIN_RELEVANT_AUS = {
    "AU4": "Brow Lowerer",       # Chau may xuong
    "AU6": "Cheek Raiser",       # Nang ma
    "AU7": "Lid Tightener",      # Siết mi mat
    "AU9": "Nose Wrinkler",      # Nhan mui
    "AU10": "Upper Lip Raiser",  # Nang moi tren
    "AU43": "Eyes Closed",       # Nham mat
}
```

**Logic phat hien**:
```python
def analyze(self, face_landmarks):
    # Tinh cac AU tu face landmarks (478 diem)
    au_scores = self._compute_au_scores(face_landmarks)

    # Tinh pain score
    pain_score = (
        0.3 * au_scores["AU4"] +
        0.2 * au_scores["AU6"] +
        0.2 * au_scores["AU7"] +
        0.15 * au_scores["AU9"] +
        0.15 * au_scores["AU43"]
    )

    # Phan loai muc do dau
    if pain_score > 0.7:
        return PainLevel.SEVERE
    elif pain_score > 0.5:
        return PainLevel.MODERATE
    elif pain_score > 0.3:
        return PainLevel.MILD
    else:
        return PainLevel.NONE
```

---

### 4.5 Compensation Detection (Bu tru)

**Muc dich**: Phat hien khi nguoi dung su dung dong tac bu tru de "gian lan" goc.

**Cac loai bu tru phat hien**:
```python
def _calculate_compensation_score(self):
    issues = []
    penalties = []

    # 1. SHOULDER HIKING (Nhun vai)
    # Kiem tra chenh lech chieu cao vai trai/phai
    if len(shoulder_heights) >= 5:
        shoulder_diffs = [abs(left - right) for left, right in shoulder_heights]
        if max(shoulder_diffs) > 0.08:  # > 8% chieu cao frame
            issues.append("Vai khong deu (nang)")
            penalties.append(40)
        elif max(shoulder_diffs) > 0.05:
            issues.append("Vai khong deu (nhe)")
            penalties.append(20)

    # 2. TRUNK LEAN (Nghieng than)
    # Tinh goc nghieng tu mid-shoulder den mid-hip
    if len(torso_tilts) >= 5:
        max_tilt = np.max(np.abs(torso_tilts))
        if max_tilt > 20:  # > 20 do
            issues.append("Nghieng than nhieu")
            penalties.append(35)
        elif max_tilt > 15:
            issues.append("Nghieng than")
            penalties.append(20)

    # 3. HIP SHIFT (Xoay hong)
    if len(hip_positions) >= 5:
        hip_diffs = [abs(left - right) for left, right in hip_positions]
        if max(hip_diffs) > 0.08:
            issues.append("Hong khong can bang")
            penalties.append(25)

    # Tinh diem: bat dau tu 100, tru penalties
    total_penalty = sum(penalties)
    score = max(0, 100 - total_penalty)

    return score, issues
```

---

## 5. BACKEND INTEGRATION

### 5.1 MemotionEngine (Headless)

`mediapipe_be/service/engine_service.py` cung cap class `MemotionEngine` - phien ban khong UI de tich hop vao backend.

**Nguyen tac thiet ke**:
1. **STATEFUL**: Moi instance co `self._state` rieng biet
2. **HANDS-FREE**: Tu dong chuyen phase khi dat dieu kien
3. **JSON-ONLY OUTPUT**: Khong tra ve object MediaPipe/NumPy tho

**Usage (Multi-user Backend)**:
```python
from service import MemotionEngine, EngineConfig

# Moi user co 1 engine instance rieng
user_engines: Dict[str, MemotionEngine] = {}

def handle_user_frame(user_id: str, frame: np.ndarray, timestamp_ms: int):
    if user_id not in user_engines:
        user_engines[user_id] = MemotionEngine.create_instance(
            config=EngineConfig(ref_video_path="./videos/exercise.mp4")
        )

    result = user_engines[user_id].process_frame(frame, timestamp_ms)
    return result.to_dict()  # JSON-serializable
```

### 5.2 EngineOutput Schema

```python
@dataclass
class EngineOutput:
    current_phase: int           # 1-4
    phase_name: str              # "detection" | "calibration" | "sync" | "scoring"
    detection: DetectionOutput   # Phase 1 data
    calibration: CalibrationOutput  # Phase 2 data
    sync: SyncOutput             # Phase 3 data
    final_report: FinalReportOutput  # Phase 4 data
    timestamp_ms: int
    error: Optional[str]

    def to_dict(self) -> Dict:
        return {
            "phase": self.current_phase,
            "phase_name": self.phase_name,
            "detection": self.detection,
            "calibration": self.calibration,
            "sync": self.sync,
            "final_report": self.final_report,
            "error": self.error
        }
```

### 5.3 JointError Schema (Multi-joint feedback)

```python
@dataclass
class JointError:
    joint_name: str          # "Vai trai", "Vai phai", etc. (tieng Viet)
    joint_type: str          # "left_shoulder", "right_shoulder", etc.
    user_angle: float        # Goc hien tai cua nguoi dung
    target_angle: float      # Goc muc tieu
    error: float             # Sai so tuyet doi (do)
    error_percent: float     # Sai so tuong doi (%)
    direction_hint: str      # "raise" | "lower" | "hold" | "ok"
    score: float             # Diem cua khop nay (0-100)
    weight: float            # Trong so cua khop trong bai tap
```

### 5.4 Data Flow (Backend)

```
Camera -> WebSocket/API -> MemotionEngine -> process_frame()
       -> EngineOutput.to_dict() -> JSON -> Frontend
```

---

## 6. DATA STRUCTURES

### 6.1 AppPhase (Enum)

```python
class AppPhase(Enum):
    PHASE1_DETECTION = 1
    PHASE2_CALIBRATION = 2
    PHASE3_SYNC = 3
    PHASE4_SCORING = 4
    COMPLETED = 5
```

### 6.2 JointType (Enum)

```python
class JointType(Enum):
    # Chi tren
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"

    # Chi duoi
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
```

### 6.3 MotionPhase (Enum)

```python
class MotionPhase(Enum):
    IDLE = "idle"                  # Tu the nghi
    ECCENTRIC = "eccentric"        # Pha duoi co
    HOLD = "hold"                  # Giu tai diem cao trao
    CONCENTRIC = "concentric"      # Pha co co
```

### 6.4 Constants

```python
# Calibration Queue - thu tu tu dong do 6 khop
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,
    JointType.RIGHT_SHOULDER,
    JointType.LEFT_ELBOW,
    JointType.RIGHT_ELBOW,
    JointType.LEFT_KNEE,
    JointType.RIGHT_KNEE,
]

# Huong dan tu the
JOINT_POSITION_INSTRUCTIONS = {
    JointType.LEFT_SHOULDER: "Moi ba dung NGANG",
    JointType.RIGHT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_ELBOW: "Moi ba dung NGANG",
    JointType.RIGHT_ELBOW: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    JointType.RIGHT_KNEE: "Moi ba dung DOC",
}

# Timing constants
PHASE1_STABLE_FRAMES_REQUIRED = 30
PHASE1_COUNTDOWN_DURATION = 3.0  # giay
CALIBRATION_COUNTDOWN_DURATION = 5.0  # giay
PHASE2_COMPLETE_DELAY = 2.0  # giay

# Trong so cho tung loai bai tap
EXERCISE_WEIGHTS = {
    "arm_raise": {
        JointType.LEFT_SHOULDER: 1.0,
        JointType.RIGHT_SHOULDER: 1.0,
        JointType.LEFT_ELBOW: 0.6,
        JointType.RIGHT_ELBOW: 0.6,
        JointType.LEFT_KNEE: 0.1,
        JointType.RIGHT_KNEE: 0.1,
    },
    "bicep_curl": {
        JointType.LEFT_ELBOW: 1.0,
        JointType.RIGHT_ELBOW: 1.0,
        JointType.LEFT_SHOULDER: 0.5,
        JointType.RIGHT_SHOULDER: 0.5,
    },
    "squat": {
        JointType.LEFT_KNEE: 1.0,
        JointType.RIGHT_KNEE: 1.0,
        JointType.LEFT_HIP: 0.8,
        JointType.RIGHT_HIP: 0.8,
    },
}
```

---

## 7. LUONG HOAT DONG CHI TIET

### 7.1 Main Loop Flow

```python
def run(self, user_source="webcam", display=True):
    cap = cv2.VideoCapture(...)

    while is_running:
        ret, frame = cap.read()
        timestamp_ms = int(time.time() * 1000)

        # Process detection
        result = detector.process_frame(frame, timestamp_ms)

        # Routing den phase hien tai
        if current_phase == PHASE1_DETECTION:
            display_frame = self._run_phase1(frame, result)

        elif current_phase == PHASE2_CALIBRATION:
            display_frame = self._run_phase2(frame, result, timestamp_ms)

        elif current_phase == PHASE3_SYNC:
            ref_frame, ref_status = video_engine.get_frame()
            display_frame = self._run_phase3(frame, ref_frame, result, timestamp)

        elif current_phase == PHASE4_SCORING:
            display_frame = self._run_phase4(frame)

        # Display & handle key
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)

    return self._generate_report()
```

### 7.2 Rep Completion Flow

```python
def _on_rep_complete(self):
    # 1. Compute DTW neu du data
    if len(user_angles) > 20 and len(ref_angles) > 20:
        dtw_result = compute_single_joint_dtw(
            user_angles[-50:],
            ref_angles[-50:]
        )

    # 2. Complete rep trong scorer
    rep_score = scorer.complete_rep(target, dtw_result)

    # 3. Log ket qua
    logger.log_rep(
        rep_score.rep_number,
        {rom, stability, flow, total},
        jerk_value,
        duration_ms
    )
```

---

## 8. CONG THUC TINH DIEM

### 8.1 Real-time Score (Single Joint)

```python
def _calculate_realtime_score(user_angle, target_angle):
    if target_angle <= 0:
        return current_score

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

**Bang chuyen doi**:
```
+------------------+---------------+------------+
| Error Percent    | Score Range   | Feedback   |
+------------------+---------------+------------+
| < 5%             | 100           | TUYET VOI! |
| 5% - 10%         | 95 - 90       | TOT!       |
| 10% - 15%        | 90 - 80       | TOT!       |
| 15% - 25%        | 80 - 65       | KHA        |
| 25% - 40%        | 65 - 50       | DIEU CHINH |
| > 40%            | < 50          | DIEU CHINH |
+------------------+---------------+------------+
```

### 8.2 Multi-joint Weighted Score

```python
Weighted Score = sum(joint_score_i * weight_i) / sum(weight_i)
```

### 8.3 Final Score (tu HealthScorer)

```python
SCORE_WEIGHTS = {
    "rom": 0.30,           # ROM Score
    "stability": 0.20,     # Stability Score
    "flow": 0.20,          # Flow Score (tu DTW)
    "symmetry": 0.15,      # Symmetry Score
    "compensation": 0.15,  # Compensation Score (tru diem neu bu tru)
}

Total Score = sum(component_score * weight)
```

### 8.4 ROM Score (Chi tiet)

```python
def _calculate_rom_score(angles, target):
    # 1. Max angle score (40%)
    max_achieved = np.max(angles)
    max_score = min(100.0, (max_achieved / target) * 100)

    # 2. Hold time score - thoi gian giu >= 80% target (30%)
    threshold = target * 0.8
    frames_above_threshold = np.sum(angles >= threshold)
    hold_score = min(1.0, frames_above_threshold / min_frames_required) * 100

    # 3. Peak quality score - kiem tra dat goc co on dinh khong (30%)
    peak_std = np.std(peak_region)
    peak_quality_score = max(0, 100 - peak_std * 5)

    # Tong hop
    final_score = 0.40 * max_score + 0.30 * hold_score + 0.30 * peak_quality_score

    return final_score
```

### 8.5 Stability Score (Chi tiet)

```python
def _calculate_stability_score(angles, phases):
    # Loc ra cac goc trong pha HOLD
    hold_angles = [angles[i] for i in range(len(phases)) if phases[i] == HOLD]

    # 1. Standard deviation score (50%)
    std = np.std(hold_angles)
    std_score = max(0, 100 - std * 10)  # std < 2 = 100, std > 10 = 0

    # 2. Oscillation count - so lan dao dong vuot nguong (30%)
    deviations = np.abs(hold_angles - np.mean(hold_angles))
    crossings = np.sum(deviations > 3.0)  # 3 do
    oscillation_score = (1 - crossings / max_crossings) * 100

    # 3. Drift score - goc co giam dan khong (dau hieu met) (20%)
    drift = np.mean(first_half) - np.mean(second_half)
    drift_score = (1 - min(1.0, drift / 5.0)) * 100

    # Tong hop
    final_score = 0.50 * std_score + 0.30 * oscillation_score + 0.20 * drift_score

    return final_score
```

---

## 9. HUONG DAN SU DUNG

### 9.1 Command Line Arguments

```bash
python main_v2.py [OPTIONS]

Options:
  --source      Input source (default: "webcam")
                Co the la: webcam, path/to/video.mp4

  --ref-video   Duong dan video mau
                Bat buoc cho Phase 3

  --joint       Khop mac dinh de theo doi
                Choices: left_shoulder, right_shoulder,
                         left_elbow, right_elbow,
                         left_knee, right_knee
                Default: left_shoulder

  --mode        Che do chay
                Choices: run, test
                Default: run

  --headless    Chay khong hien thi UI

  --models-dir  Thu muc chua model files
                Default: ./models

  --log-dir     Thu muc luu logs
                Default: ./data/logs
```

### 9.2 Vi du su dung

```bash
# Chay voi webcam va video mau
python main_v2.py --source webcam --ref-video videos/arm_raise.mp4

# Chay voi video input
python main_v2.py --source path/to/user.mp4 --ref-video videos/arm_raise.mp4

# Chay voi khop khuyu tay
python main_v2.py --source webcam --ref-video videos/elbow.mp4 --joint left_elbow

# Chay test mode
python main_v2.py --mode test
```

### 9.3 Phim dieu khien

| Phim | Phase | Chuc nang |
|------|-------|-----------|
| `ENTER` | 1 | Manual override - bo qua countdown 3 giay |
| `SPACE` | 3 | Pause/Resume video |
| `R` | All | Restart ve Phase 1 |
| `Q` / `ESC` | All | Thoat ung dung |

---

## REFERENCES

- `PROJECT_CONTEXT.md` - Context cho Claude Code
- `.cursorrules` - Quy tac code
- `core/` - Module cot loi
- `modules/` - Module chuc nang
- `mediapipe_be/` - Backend integration

---

> **Note**: Tai lieu nay mo ta chi tiet ve luong hoat dong, thuat toan su dung, va logic code cua MEMOTION v2.0.
