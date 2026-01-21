#!/usr/bin/env python3
"""
MEMOTION - Complete Integration (Final Version 2.0)

Tích hợp hoàn chỉnh 4 giai đoạn với UI rõ ràng:
1. PHASE 1: Pose Detection - Nhận diện tư thế, vẽ skeleton
2. PHASE 2: Safe-Max Calibration - Đo giới hạn vận động
3. PHASE 3: Motion Sync - Đồng bộ với video mẫu
4. PHASE 4: Scoring & Analysis - Chấm điểm và phân tích

Usage:
    python main_v2.py --source webcam --ref-video exercise.mp4
    python main_v2.py --mode test

Controls:
    SPACE: Pause/Resume hoặc Bắt đầu calibration
    R: Restart
    Q: Quit
    1-6: Chọn khớp để đo (Phase 2)
    ENTER: Xác nhận/Chuyển phase tiếp theo
    ESC: Thoát

Author: MEMOTION Team
Version: 2.0.0
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from queue import Queue
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV required. Install: pip install opencv-python")
    sys.exit(1)

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


class AppPhase(Enum):
    """Các giai đoạn của ứng dụng."""
    PHASE1_DETECTION = "phase1"      # Pose Detection
    PHASE2_CALIBRATION = "phase2"    # Safe-Max Calibration
    PHASE3_SYNC = "phase3"           # Motion Sync
    PHASE4_SCORING = "phase4"        # Scoring & Analysis
    COMPLETED = "completed"          # Hoàn thành


# Mapping từ phím số sang JointType
JOINT_KEY_MAPPING = {
    ord('1'): JointType.LEFT_SHOULDER,
    ord('2'): JointType.RIGHT_SHOULDER,
    ord('3'): JointType.LEFT_ELBOW,
    ord('4'): JointType.RIGHT_ELBOW,
    ord('5'): JointType.LEFT_KNEE,
    ord('6'): JointType.RIGHT_KNEE,
}

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

# Countdown duration (giây)
CALIBRATION_COUNTDOWN_DURATION = 5.0

# Phase colors
PHASE_COLORS = {
    "idle": (128, 128, 128),       # Gray
    "eccentric": (0, 255, 255),    # Yellow
    "hold": (0, 255, 0),           # Green
    "concentric": (255, 255, 0),   # Cyan
}

PHASE_NAMES_VI = {
    "idle": "Nghi",
    "eccentric": "Duoi co",
    "hold": "Giu",
    "concentric": "Co co",
}


@dataclass
class AppState:
    """Trạng thái toàn cục của ứng dụng."""
    current_phase: AppPhase = AppPhase.PHASE1_DETECTION
    is_running: bool = True
    is_paused: bool = False
    
    # Phase 1 state
    pose_detected: bool = False
    detection_stable_count: int = 0
    phase1_countdown_start: float = 0.0  # Thời điểm bắt đầu countdown 3 giây
    phase1_countdown_active: bool = False  # Đang trong countdown chuyển phase
    
    # Phase 2 state - Automated Calibration
    selected_joint: Optional[JointType] = None
    calibration_complete: bool = False
    user_max_angle: float = 0.0
    # Automated calibration queue
    calibration_queue_index: int = 0  # Vị trí hiện tại trong queue
    calibration_countdown_start: float = 0.0  # Thời điểm bắt đầu countdown
    is_countdown_active: bool = False  # Đang countdown chuẩn bị
    is_calibrating_joint: bool = False  # Đang đo khớp hiện tại
    calibrated_joints: Dict = field(default_factory=dict)  # Lưu góc max của từng khớp
    all_joints_calibrated: bool = False  # Đã đo xong tất cả 6 khớp
    
    # Phase 3 state
    sync_state: Optional[SyncState] = None
    motion_phase: str = "idle"
    last_motion_phase: Optional[MotionPhase] = None
    
    # Phase 3 - Multi-joint tracking
    user_angles_dict: Dict = field(default_factory=dict)  # Dict[JointType, float] - góc hiện tại của tất cả các khớp
    target_angles_dict: Dict = field(default_factory=dict)  # Dict[JointType, float] - góc mục tiêu của tất cả các khớp
    joint_scores_dict: Dict = field(default_factory=dict)  # Dict[JointType, float] - điểm của từng khớp
    joint_weights: Dict = field(default_factory=dict)  # Dict[JointType, float] - trọng số của từng khớp
    active_joints: List = field(default_factory=list)  # Danh sách các khớp đang hoạt động
    
    # Phase 4 state
    rep_count: int = 0
    current_score: float = 0.0
    average_score: float = 0.0
    
    # Common (backward compatible - vẫn giữ cho primary joint)
    user_angle: float = 0.0
    target_angle: float = 0.0
    pain_level: str = "NONE"
    fatigue_level: str = "FRESH"
    message: str = ""
    warning: str = ""


class MemotionAppV2:
    """Ứng dụng MEMOTION hoàn chỉnh với 4 giai đoạn và UI rõ ràng."""
    
    # Constants
    DETECTION_STABLE_THRESHOLD = 30  # Số frame cần stable để confirm phase 1
    WINDOW_NAME = "MEMOTION - He thong ho tro phuc hoi chuc nang"
    
    def __init__(
        self,
        detector: VisionDetector,
        ref_video_path: Optional[str] = None,
        default_joint: JointType = JointType.LEFT_SHOULDER,
        log_dir: str = "./data/logs",
        models_dir: str = "./models"
    ):
        self._detector = detector
        self._ref_video_path = ref_video_path
        self._default_joint = default_joint
        self._log_dir = log_dir
        self._models_dir = models_dir
        
        # State
        self._state = AppState()
        self._state.selected_joint = default_joint
        
        # Components
        self._video_engine: Optional[VideoEngine] = None
        self._sync_controller: Optional[MotionSyncController] = None
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
        self._score_history: List[float] = []  # Track scores để tính average
        self._current_landmarks: Optional[np.ndarray] = None
        self._ref_landmarks: Optional[np.ndarray] = None
        
        # Analysis queue
        self._analysis_queue = Queue(maxsize=5)
        
        # Interpolated target angle for smoother tracking
        self._last_target_angle: float = 0.0
    
    def _interpolate_target_angle(self, current_frame: int, total_frames: int, joint_type: Optional[JointType] = None) -> float:
        """
        Tính target angle dựa trên vị trí frame trong video cho một khớp cụ thể.
        
        Thay vì chỉ lấy target từ checkpoint, ta interpolate
        để có target liên tục cho mọi frame.
        
        Args:
            current_frame: Frame hiện tại
            total_frames: Tổng số frames
            joint_type: Khớp cần tính target (None = primary joint)
        """
        if not self._sync_controller:
            if joint_type and joint_type in self._state.calibrated_joints:
                return self._state.calibrated_joints[joint_type]
            return self._state.user_max_angle or 150.0
        
        exercise = self._sync_controller.exercise
        checkpoints = exercise.checkpoints
        
        if not checkpoints:
            if joint_type and joint_type in self._state.calibrated_joints:
                return self._state.calibrated_joints[joint_type]
            return self._state.user_max_angle or 150.0
        
        # Lấy max angle từ calibrated_joints cho khớp này
        if joint_type and joint_type in self._state.calibrated_joints:
            user_max = self._state.calibrated_joints[joint_type]
        else:
            user_max = self._state.user_max_angle or 150.0
        
        # Tìm checkpoint trước và sau current_frame
        prev_cp = checkpoints[0]
        next_cp = checkpoints[-1]
        
        for i, cp in enumerate(checkpoints):
            if cp.frame_index <= current_frame:
                prev_cp = cp
            if cp.frame_index > current_frame:
                next_cp = cp
                break
        
        # Interpolate giữa 2 checkpoints
        if prev_cp.frame_index == next_cp.frame_index:
            base_target = prev_cp.target_angle
        else:
            progress = (current_frame - prev_cp.frame_index) / max(1, next_cp.frame_index - prev_cp.frame_index)
            progress = max(0, min(1, progress))
            base_target = prev_cp.target_angle + progress * (next_cp.target_angle - prev_cp.target_angle)
        
        # Scale target angle dựa trên user_max calibrated
        # Nếu exercise target là 150 và user_max là 120, scale xuống
        exercise_max = max(cp.target_angle for cp in checkpoints)
        if exercise_max > 0 and user_max > 0:
            scale_factor = user_max / exercise_max
            # Chỉ scale nếu user_max nhỏ hơn exercise_max
            if scale_factor < 1.0:
                base_target = base_target * scale_factor
        
        return base_target
    
    def _interpolate_all_joint_targets(self, current_frame: int, total_frames: int) -> Dict[JointType, float]:
        """
        Tính target angles cho TẤT CẢ các khớp đã calibrated.
        
        Returns:
            Dict[JointType, float]: Target angle cho từng khớp
        """
        targets = {}
        
        for joint_type in self._state.active_joints:
            if joint_type in self._state.calibrated_joints:
                target = self._interpolate_target_angle(current_frame, total_frames, joint_type)
                targets[joint_type] = target
        
        return targets
    
    def _calculate_realtime_score(self, user_angle: float, target_angle: float) -> float:
        """
        Tính điểm thời gian thực dựa trên sai số góc cho MỘT khớp.
        
        Score = 100 - (error_percentage * penalty_factor)
        - error < 10%: full score
        - error < 20%: good score (80-100)
        - error < 30%: average score (60-80)
        - error >= 30%: low score
        """
        if target_angle <= 0:
            return self._state.current_score
        
        error = abs(user_angle - target_angle)
        error_percent = (error / target_angle) * 100
        
        if error_percent < 5:
            score = 100.0
        elif error_percent < 10:
            score = 95.0 - (error_percent - 5) * 1.0  # 95-90
        elif error_percent < 15:
            score = 90.0 - (error_percent - 10) * 2.0  # 90-80
        elif error_percent < 25:
            score = 80.0 - (error_percent - 15) * 1.5  # 80-65
        elif error_percent < 40:
            score = 65.0 - (error_percent - 25) * 1.0  # 65-50
        else:
            score = max(0, 50.0 - (error_percent - 40) * 0.5)
        
        return max(0, min(100, score))
    
    def _calculate_multi_joint_score(self) -> float:
        """
        Tính điểm trung bình có trọng số của TẤT CẢ các khớp đang hoạt động.
        
        Returns:
            float: Điểm trung bình có trọng số (0-100)
        """
        if not self._state.active_joints or not self._state.joint_weights:
            return self._state.current_score
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for joint_type in self._state.active_joints:
            if joint_type not in self._state.user_angles_dict:
                continue
            if joint_type not in self._state.target_angles_dict:
                continue
            
            user_angle = self._state.user_angles_dict[joint_type]
            target_angle = self._state.target_angles_dict[joint_type]
            weight = self._state.joint_weights.get(joint_type, 0.5)
            
            # Tính điểm cho khớp này
            joint_score = self._calculate_realtime_score(user_angle, target_angle)
            self._state.joint_scores_dict[joint_type] = joint_score
            
            total_weighted_score += joint_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        
        return self._state.current_score
    
    def _calculate_all_joint_angles(self, landmarks: np.ndarray) -> Dict[JointType, float]:
        """
        Tính góc của TẤT CẢ các khớp đang hoạt động từ landmarks.
        
        Args:
            landmarks: Pose landmarks array
            
        Returns:
            Dict[JointType, float]: Góc của từng khớp
        """
        angles = {}
        
        for joint_type in self._state.active_joints:
            try:
                angle = calculate_joint_angle(landmarks, joint_type, use_3d=True)
                angles[joint_type] = angle
            except (ValueError, IndexError):
                # Giữ nguyên góc cũ nếu có lỗi
                if joint_type in self._state.user_angles_dict:
                    angles[joint_type] = self._state.user_angles_dict[joint_type]
        
        return angles
    
    def _init_ref_detector(self) -> None:
        """Khởi tạo detector cho video mẫu."""
        models_dir = Path(self._models_dir)
        pose_model = models_dir / "pose_landmarker_lite.task"
        
        if pose_model.exists():
            config = DetectorConfig(
                pose_model_path=str(pose_model),
                running_mode="VIDEO"
            )
            self._ref_detector = VisionDetector(config)
    
    def _get_joint_pixel_coords(
        self,
        landmarks: np.ndarray,
        joint_type: JointType,
        frame_shape: Tuple[int, int]
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Lấy tọa độ pixel của 3 điểm tạo góc."""
        h, w = frame_shape
        joint_def = JOINT_DEFINITIONS.get(joint_type)
        if joint_def is None or landmarks is None:
            return None, None, None
        
        try:
            p = landmarks[joint_def.proximal]
            v = landmarks[joint_def.vertex]
            d = landmarks[joint_def.distal]
            
            return (
                (int(p[0] * w), int(p[1] * h)),
                (int(v[0] * w), int(v[1] * h)),
                (int(d[0] * w), int(d[1] * h))
            )
        except IndexError:
            return None, None, None
    
    # ================== PHASE 1: POSE DETECTION (AUTO TRANSITION) ==================
    
    PHASE1_COUNTDOWN_DURATION = 3.0  # 3 giây countdown trước khi chuyển Phase 2
    
    def _run_phase1(self, frame: np.ndarray, result) -> np.ndarray:
        """Phase 1: Nhận diện Pose và vẽ skeleton (Tự động chuyển Phase 2)."""
        output = frame.copy()
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Panel hướng dẫn
        output = draw_panel(output, (10, 10), (450, 250), "")
        
        # Tiêu đề phase
        output = put_vietnamese_text(
            output, "GIAI DOAN 1: NHAN DIEN TU THE",
            (25, 40), COLORS['info'], 18
        )
        
        # Hướng dẫn
        instructions = [
            "Hay dung truoc camera de he thong nhan dien",
            "Dam bao toan than nam trong khung hinh",
            "Dung yen cho den khi thay skeleton xuat hien",
            "He thong se tu dong chuyen sang Phase 2",
        ]
        
        y_pos = 75
        for inst in instructions:
            output = put_vietnamese_text(output, f"  {inst}", (25, y_pos), COLORS['text'], 13)
            y_pos += 25
        
        # Kiểm tra pose detected
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Vẽ skeleton
            highlight = []
            if self._state.selected_joint:
                joint_def = JOINT_DEFINITIONS.get(self._state.selected_joint)
                if joint_def:
                    highlight = [joint_def.proximal, joint_def.vertex, joint_def.distal]
            
            output = draw_skeleton(
                output, self._current_landmarks,
                color=COLORS['skeleton'],
                keypoint_color=COLORS['keypoint'],
                highlight_indices=highlight,
                use_core_only=True
            )
            
            # Đếm stable frames
            self._state.detection_stable_count += 1
            progress = min(1.0, self._state.detection_stable_count / self.DETECTION_STABLE_THRESHOLD)
            
            if self._state.detection_stable_count >= self.DETECTION_STABLE_THRESHOLD:
                self._state.pose_detected = True
                
                # === AUTO TRANSITION: Bắt đầu countdown 3 giây ===
                if not self._state.phase1_countdown_active:
                    self._state.phase1_countdown_active = True
                    self._state.phase1_countdown_start = current_time
                    print("[PHASE 1] Da nhan dien thanh cong! Bat dau dem nguoc 3 giay...")
                
                # Tính thời gian countdown còn lại
                elapsed = current_time - self._state.phase1_countdown_start
                remaining = self.PHASE1_COUNTDOWN_DURATION - elapsed
                
                if remaining > 0:
                    # Hiển thị countdown
                    status_text = f"CHUAN BI... {int(remaining) + 1} giay"
                    status_color = COLORS['success']
                    
                    # Vẽ countdown lớn ở giữa màn hình
                    countdown_num = str(int(remaining) + 1)
                    cv2.putText(output, countdown_num, (w // 2 - 30, h // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    
                    output = put_vietnamese_text(
                        output, "Dung yen, chuan bi do gioi han van dong...",
                        (w // 2 - 180, h // 2 + 50), COLORS['info'], 14
                    )
                else:
                    # Countdown kết thúc - TỰ ĐỘNG chuyển Phase 2
                    status_text = "CHUYEN SANG PHASE 2..."
                    status_color = COLORS['success']
                    self._transition_to_phase2()
            else:
                status_text = f"Dang xac nhan... {int(progress * 100)}%"
                status_color = COLORS['warning']
            
            output = put_vietnamese_text(output, status_text, (25, y_pos + 10), status_color, 16)
            output = draw_progress_bar(output, (25, y_pos + 35), (400, 18), progress, status_color)
        else:
            # Reset countdown nếu mất pose
            self._state.detection_stable_count = 0
            self._state.phase1_countdown_active = False
            output = put_vietnamese_text(
                output, "Chua phat hien nguoi. Hay dung vao khung hinh.",
                (25, y_pos + 10), COLORS['error'], 16
            )
        
        # Hiển thị thông tin phase hiện tại
        phase_text = "Phase: 1/4 - Nhan dien tu the"
        output = put_vietnamese_text(output, phase_text, (w - 280, 30), COLORS['info'], 14)
        
        # Controls (auto transition, no ENTER needed)
        controls = "[Q] Thoat | Tu dong chuyen Phase 2"
        output = put_vietnamese_text(output, controls, (w - 300, h - 25), (150, 150, 150), 12)
        
        return output
    
    # ================== PHASE 2: CALIBRATION (AUTOMATED) ==================
    
    def _run_phase2(self, frame: np.ndarray, result, timestamp_ms: int) -> np.ndarray:
        """Phase 2: Safe-Max Calibration - Tự động đo 6 khớp."""
        output = frame.copy()
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Lấy khớp hiện tại từ queue
        if self._state.calibration_queue_index < len(CALIBRATION_QUEUE):
            current_joint = CALIBRATION_QUEUE[self._state.calibration_queue_index]
            self._state.selected_joint = current_joint
        else:
            # Đã đo xong tất cả khớp
            self._state.all_joints_calibrated = True
        
        # Panel chính
        output = draw_panel(output, (10, 10), (500, 380), "")
        
        # Tiêu đề
        output = put_vietnamese_text(
            output, "GIAI DOAN 2: DO GIOI HAN VAN DONG (TU DONG)",
            (25, 40), COLORS['info'], 18
        )
        
        # Hiển thị tiến trình tổng thể
        y_pos = 75
        total_joints = len(CALIBRATION_QUEUE)
        completed_joints = len(self._state.calibrated_joints)
        output = put_vietnamese_text(
            output, f"Tien do: {completed_joints}/{total_joints} khop",
            (25, y_pos), COLORS['text'], 14
        )
        y_pos += 25
        
        # Progress bar tổng thể
        overall_progress = completed_joints / total_joints
        output = draw_progress_bar(output, (25, y_pos), (450, 12), overall_progress, COLORS['info'])
        y_pos += 25
        
        # Hiển thị danh sách khớp và trạng thái
        output = put_vietnamese_text(output, "Danh sach khop:", (25, y_pos), (150, 150, 150), 12)
        y_pos += 22
        
        for i, joint_type in enumerate(CALIBRATION_QUEUE):
            joint_name = JOINT_NAMES.get(joint_type, joint_type.value)
            
            if joint_type in self._state.calibrated_joints:
                # Đã đo xong
                angle = self._state.calibrated_joints[joint_type]
                status = f"[OK] {joint_name}: {angle:.1f} do"
                color = COLORS['success']
            elif i == self._state.calibration_queue_index:
                # Đang đo
                status = f">>> {joint_name} (dang do)"
                color = COLORS['warning']
            else:
                # Chưa đo
                status = f"    {joint_name}"
                color = (120, 120, 120)
            
            output = put_vietnamese_text(output, status, (30, y_pos), color, 12)
            y_pos += 20
        
        y_pos += 15
        
        # === LOGIC TỰ ĐỘNG ===
        if self._state.all_joints_calibrated:
            # Đã đo xong tất cả - hiển thị kết quả và tự động chuyển phase
            output = put_vietnamese_text(
                output, "DA DO XONG TAT CA 6 KHOP!",
                (25, y_pos), COLORS['success'], 18
            )
            y_pos += 30
            output = put_vietnamese_text(
                output, "Dang luu ket qua va chuyen sang Phase 3...",
                (25, y_pos), COLORS['info'], 14
            )
            
            # Tự động chuyển sang Phase 3 sau 2 giây
            if not hasattr(self, '_phase2_complete_time'):
                self._phase2_complete_time = current_time
                self._save_calibration_to_profile()
            elif current_time - self._phase2_complete_time > 2.0:
                self._state.calibration_complete = True
                self._transition_to_phase3()
                delattr(self, '_phase2_complete_time')
        
        elif not self._state.is_countdown_active and not self._state.is_calibrating_joint:
            # Bắt đầu countdown cho khớp mới
            self._state.is_countdown_active = True
            self._state.calibration_countdown_start = current_time
            print(f"[CALIBRATION] Chuan bi do: {JOINT_NAMES.get(current_joint)}")
        
        elif self._state.is_countdown_active:
            # Đang countdown chuẩn bị
            elapsed = current_time - self._state.calibration_countdown_start
            remaining = CALIBRATION_COUNTDOWN_DURATION - elapsed
            
            if remaining > 0:
                # Hiển thị countdown và hướng dẫn tư thế
                position_instruction = JOINT_POSITION_INSTRUCTIONS.get(current_joint, "")
                joint_name = JOINT_NAMES.get(current_joint, "")
                
                # Hướng dẫn tư thế với font lớn
                output = put_vietnamese_text(
                    output, position_instruction,
                    (25, y_pos), COLORS['warning'], 20
                )
                y_pos += 35
                
                # Countdown số lớn
                countdown_text = f"Bat dau sau: {int(remaining) + 1} giay"
                output = put_vietnamese_text(
                    output, countdown_text,
                    (25, y_pos), COLORS['info'], 18
                )
                y_pos += 30
                
                # Hướng dẫn chi tiết
                output = put_vietnamese_text(
                    output, f"Khop: {joint_name}",
                    (25, y_pos), COLORS['text'], 14
                )
                y_pos += 22
                output = put_vietnamese_text(
                    output, "Thuc hien dong tac HET KHA NANG (khong dau)",
                    (25, y_pos), COLORS['warning'], 12
                )
                
                # Progress bar countdown
                countdown_progress = elapsed / CALIBRATION_COUNTDOWN_DURATION
                output = draw_progress_bar(
                    output, (25, y_pos + 25), (450, 15),
                    countdown_progress, COLORS['warning']
                )
            else:
                # Countdown kết thúc - bắt đầu đo
                self._state.is_countdown_active = False
                self._state.is_calibrating_joint = True
                self._start_calibration_for_joint(current_joint)
        
        elif self._state.is_calibrating_joint:
            # Đang đo khớp
            joint_name = JOINT_NAMES.get(current_joint, "")
            
            if self._calibrator.state == CalibrationState.COLLECTING:
                progress = self._calibrator.progress
                output = put_vietnamese_text(
                    output, f"Dang do {joint_name}... {int(progress * 100)}%",
                    (25, y_pos), COLORS['warning'], 16
                )
                y_pos += 28
                output = draw_progress_bar(output, (25, y_pos), (450, 18), progress, COLORS['warning'])
                y_pos += 30
                
                # Thêm frame vào calibrator
                if result.has_pose() and current_joint:
                    try:
                        landmarks = result.pose_landmarks.to_numpy()
                        angle = calculate_joint_angle(landmarks, current_joint, use_3d=True)
                        self._state.user_angle = angle
                        self._calibrator.add_frame(result.pose_landmarks, timestamp_ms)
                        
                        output = put_vietnamese_text(
                            output, f"Goc hien tai: {angle:.1f} do",
                            (25, y_pos), COLORS['info'], 18
                        )
                    except ValueError:
                        pass
                    
                    # Kiểm tra hoàn thành
                    if self._calibrator.state == CalibrationState.COMPLETED:
                        self._finish_calibration_for_joint(current_joint)
            
            elif self._calibrator.state == CalibrationState.COMPLETED:
                # Đã đo xong khớp này - sẽ được xử lý ở frame tiếp theo
                pass
        
        # Vẽ skeleton với highlight
        if result.has_pose():
            landmarks = result.pose_landmarks.to_numpy()
            
            highlight = []
            if self._state.selected_joint:
                joint_def = JOINT_DEFINITIONS.get(self._state.selected_joint)
                if joint_def:
                    highlight = [joint_def.proximal, joint_def.vertex, joint_def.distal]
            
            output = draw_skeleton(
                output, landmarks,
                highlight_indices=highlight,
                highlight_color=COLORS['highlight'],
                use_core_only=True
            )
            
            # Vẽ góc
            if self._state.selected_joint and self._state.user_angle > 0:
                p1, pv, p2 = self._get_joint_pixel_coords(
                    landmarks, self._state.selected_joint, (h, w)
                )
                if p1 and pv and p2:
                    output = draw_angle_arc(output, p1, pv, p2, self._state.user_angle)
        
        # Phase indicator
        phase_text = f"Phase: 2/4 - Calibration ({completed_joints}/{total_joints})"
        output = put_vietnamese_text(output, phase_text, (w - 300, 30), COLORS['info'], 14)
        
        # Controls
        controls = "[Q] Thoat | [R] Bat dau lai"
        output = put_vietnamese_text(output, controls, (20, h - 25), (150, 150, 150), 12)
        
        return output
    
    def _start_calibration_for_joint(self, joint_type: JointType) -> None:
        """Bắt đầu đo một khớp cụ thể."""
        if self._user_profile is None:
            self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        self._calibrator = SafeMaxCalibrator(duration_ms=5000)
        self._calibrator.start_calibration(joint_type, self._user_profile)
        print(f"[CALIBRATION] Bat dau do: {JOINT_NAMES.get(joint_type)}")
    
    def _finish_calibration_for_joint(self, joint_type: JointType) -> None:
        """Hoàn thành đo một khớp và chuyển sang khớp tiếp theo."""
        if self._user_profile and joint_type:
            max_angle = self._user_profile.get_max_angle(joint_type)
            if max_angle:
                self._state.calibrated_joints[joint_type] = max_angle
                print(f"[CALIBRATION] Hoan thanh {JOINT_NAMES.get(joint_type)}: {max_angle:.1f} do")
                
                # Nếu là khớp được chọn chính, lưu vào user_max_angle
                if joint_type == self._default_joint:
                    self._state.user_max_angle = max_angle
        
        # Chuyển sang khớp tiếp theo
        self._state.calibration_queue_index += 1
        self._state.is_calibrating_joint = False
        self._state.user_angle = 0.0
        
        # Kiểm tra đã đo xong tất cả chưa
        if self._state.calibration_queue_index >= len(CALIBRATION_QUEUE):
            self._state.all_joints_calibrated = True
            print("[CALIBRATION] Da do xong tat ca 6 khop!")
    
    def _save_calibration_to_profile(self) -> None:
        """Lưu tất cả kết quả calibration vào user profile."""
        if self._user_profile:
            # Profile đã được cập nhật trong quá trình calibration
            # Lưu vào file
            profile_dir = Path("./data/user_profiles")
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            profile_path = profile_dir / f"user_{timestamp}.json"
            
            try:
                import json
                profile_data = {
                    "user_id": self._user_profile.user_id,
                    "created_at": timestamp,
                    "calibrated_joints": {
                        jt.value: angle for jt, angle in self._state.calibrated_joints.items()
                    }
                }
                with open(profile_path, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                print(f"[CALIBRATION] Da luu profile: {profile_path}")
            except Exception as e:
                print(f"[WARNING] Khong the luu profile: {e}")
    
    def _start_calibration(self) -> None:
        """Bắt đầu calibration (backward compatible - không dùng nữa)."""
        if self._state.selected_joint:
            self._start_calibration_for_joint(self._state.selected_joint)
    
    def _finish_calibration(self) -> None:
        """Lấy kết quả calibration (backward compatible - không dùng nữa)."""
        if self._state.selected_joint:
            self._finish_calibration_for_joint(self._state.selected_joint)
    
    # ================== PHASE 3: MOTION SYNC (MULTI-JOINT) ==================
    
    def _run_phase3(
        self,
        user_frame: np.ndarray,
        ref_frame: Optional[np.ndarray],
        result,
        timestamp: float
    ) -> np.ndarray:
        """Phase 3: Đồng bộ chuyển động với video mẫu (Multi-joint tracking)."""
        h, w = user_frame.shape[:2]
        
        # === MULTI-JOINT ANGLE CALCULATION ===
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Tính góc cho TẤT CẢ các khớp đang hoạt động
            self._state.user_angles_dict = self._calculate_all_joint_angles(self._current_landmarks)
            
            # Cập nhật primary joint angle cho backward compatibility
            primary_joint = self._state.selected_joint or self._default_joint
            if primary_joint in self._state.user_angles_dict:
                self._state.user_angle = self._state.user_angles_dict[primary_joint]
        
        # === MULTI-JOINT TARGET CALCULATION ===
        if self._video_engine:
            self._state.target_angles_dict = self._interpolate_all_joint_targets(
                self._video_engine.current_frame,
                self._video_engine.total_frames
            )
            
            # Cập nhật primary joint target cho backward compatibility
            primary_joint = self._state.selected_joint or self._default_joint
            if primary_joint in self._state.target_angles_dict:
                self._state.target_angle = self._state.target_angles_dict[primary_joint]
        
        # === USER VIEW ===
        user_display = user_frame.copy()
        
        # Vẽ skeleton user với highlight tất cả khớp hoạt động
        if result.has_pose():
            # Highlight tất cả các khớp đang hoạt động
            highlight = []
            for joint_type in self._state.active_joints:
                joint_def = JOINT_DEFINITIONS.get(joint_type)
                if joint_def:
                    highlight.extend([joint_def.proximal, joint_def.vertex, joint_def.distal])
            
            user_display = draw_skeleton(
                user_display, self._current_landmarks,
                color=COLORS['skeleton'],
                highlight_indices=list(set(highlight)),  # Remove duplicates
                use_core_only=True
            )
            
            # Vẽ góc cho primary joint
            if self._state.user_angle > 0 and self._state.selected_joint:
                p1, pv, p2 = self._get_joint_pixel_coords(
                    self._current_landmarks, self._state.selected_joint, (h, w)
                )
                if p1 and pv and p2:
                    user_display = draw_angle_arc(
                        user_display, p1, pv, p2, self._state.user_angle,
                        color=COLORS['info']
                    )
        
        # Panel thông tin user (hiển thị primary joint)
        user_display = draw_panel(user_display, (10, 10), (220, 140), "")
        user_display = put_vietnamese_text(user_display, "NGUOI DUNG", (25, 35), COLORS['text'], 16)
        user_display = put_vietnamese_text(
            user_display, f"Goc: {self._state.user_angle:.1f}",
            (25, 60), COLORS['text'], 14
        )
        user_display = put_vietnamese_text(
            user_display, f"Muc tieu: {self._state.target_angle:.1f}",
            (25, 82), COLORS['info'], 14
        )
        
        # Sai số với feedback text
        error = abs(self._state.user_angle - self._state.target_angle)
        if error < 10:
            error_color = COLORS['success']
            feedback_text = "TUYET VOI!"
        elif error < 20:
            error_color = COLORS['success'] 
            feedback_text = "TOT!"
        elif error < 30:
            error_color = COLORS['warning']
            feedback_text = "KHA"
        else:
            error_color = COLORS['error']
            feedback_text = "DIEU CHINH!"
            
        user_display = put_vietnamese_text(
            user_display, f"Sai so: {error:.1f} - {feedback_text}",
            (25, 104), error_color, 14
        )
        
        # Điểm hiện tại (multi-joint weighted average)
        score_color = COLORS['success'] if self._state.current_score >= 70 else COLORS['warning']
        user_display = put_vietnamese_text(
            user_display, f"Diem: {self._state.current_score:.0f}",
            (25, 126), score_color, 14
        )
        
        # Hiển thị trạng thái đạt mục tiêu - banner lớn nếu đạt
        if error < 15 and self._state.target_angle > 0:
            cv2.rectangle(user_display, (10, h - 50), (w // 2 - 10, h - 10), (0, 100, 0), -1)
            user_display = put_vietnamese_text(
                user_display, "DAT MUC TIEU!",
                (w // 4 - 60, h - 25), COLORS['text'], 18
            )
        
        # === REFERENCE VIEW ===
        if ref_frame is not None:
            ref_display = ref_frame.copy()
            ref_h, ref_w = ref_display.shape[:2]
            
            # Detect pose từ reference
            if self._ref_detector:
                ref_timestamp = int(time.time() * 1000)
                ref_result = self._ref_detector.process_frame(ref_frame, ref_timestamp)
                
                if ref_result.has_pose():
                    self._ref_landmarks = ref_result.pose_landmarks.to_numpy()
                    ref_display = draw_skeleton(
                        ref_display, self._ref_landmarks,
                        color=COLORS['skeleton_ref'],
                        keypoint_color=COLORS['keypoint_ref'],
                        use_core_only=True
                    )
            
            # Panel thông tin reference
            ref_display = draw_panel(ref_display, (10, 10), (180, 100), "")
            ref_display = put_vietnamese_text(ref_display, "VIDEO MAU", (25, 35), COLORS['text'], 16)
            
            # Phase indicator
            phase = self._state.motion_phase.lower()
            phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
            phase_name = PHASE_NAMES_VI.get(phase, phase.upper())
            
            cv2.circle(ref_display, (35, 65), 12, phase_color, -1)
            ref_display = put_vietnamese_text(
                ref_display, phase_name.upper(),
                (55, 70), phase_color, 14
            )
            
            # Video progress
            if self._video_engine:
                progress = self._video_engine.current_frame / max(1, self._video_engine.total_frames)
                ref_display = draw_progress_bar(
                    ref_display, (10, ref_h - 25), (ref_w - 20, 12),
                    progress, COLORS['info'], show_percentage=False
                )
            
            # Waiting indicator
            if self._state.sync_state and self._state.sync_state.sync_status == SyncStatus.PAUSE:
                ref_display = put_vietnamese_text(
                    ref_display, "|| CHO",
                    (ref_w - 80, 35), COLORS['warning'], 16
                )
        else:
            ref_display = np.zeros((h, w // 2, 3), dtype=np.uint8)
            ref_display[:] = (40, 40, 40)
            ref_display = put_vietnamese_text(
                ref_display, "KHONG CO VIDEO MAU",
                (50, h // 2), COLORS['warning'], 16
            )
        
        # === DASHBOARD (MULTI-JOINT) ===
        dashboard = self._create_phase3_dashboard(h)
        
        # Combine views
        combined = combine_frames_horizontal([user_display, ref_display, dashboard], h)
        
        # === UPDATE SYNC & SCORING (MULTI-JOINT) ===
        if self._sync_controller and self._video_engine:
            # Update sync controller với primary joint
            self._state.sync_state = self._sync_controller.update(
                self._state.user_angle,
                self._video_engine.current_frame,
                timestamp
            )
            
            self._state.motion_phase = self._state.sync_state.current_phase.value
            self._state.rep_count = self._state.sync_state.rep_count
            
            # === MULTI-JOINT SCORING ===
            # Tính điểm trung bình có trọng số của TẤT CẢ các khớp
            multi_joint_score = self._calculate_multi_joint_score()
            
            # Smooth score để tránh nhảy quá nhanh
            self._state.current_score = 0.7 * self._state.current_score + 0.3 * multi_joint_score
            
            # Track score history để tính average (sử dụng multi-joint score)
            if len(self._state.target_angles_dict) > 0:
                self._score_history.append(multi_joint_score)
                if len(self._score_history) > 0:
                    self._state.average_score = sum(self._score_history) / len(self._score_history)
        
        # Track angles (primary joint cho backward compatibility)
        self._user_angles.append(self._state.user_angle)
        if self._state.target_angle > 0:
            self._ref_angles.append(self._state.target_angle)
        
        return combined
    
    def _create_phase3_dashboard(self, height: int) -> np.ndarray:
        """Tạo dashboard cho Phase 3 (Multi-joint)."""
        width = 320  # Tăng width để hiển thị nhiều khớp
        dashboard = np.zeros((height, width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)
        
        y = 20
        
        # Tiêu đề
        dashboard = put_vietnamese_text(dashboard, "GIAI DOAN 3: DONG BO", (10, y), COLORS['info'], 14)
        y += 28
        
        # Phase hiện tại
        phase = self._state.motion_phase.lower()
        phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
        phase_name = PHASE_NAMES_VI.get(phase, phase.upper())
        
        cv2.circle(dashboard, (25, y + 3), 8, phase_color, -1)
        dashboard = put_vietnamese_text(dashboard, f"{phase_name.upper()} | Rep: {self._state.rep_count}", (40, y + 8), phase_color, 12)
        y += 25
        
        # 4 phase indicators (compact)
        phases = ["idle", "eccentric", "hold", "concentric"]
        phase_x = 10
        for p in phases:
            p_color = PHASE_COLORS.get(p, (80, 80, 80))
            if p == phase:
                cv2.circle(dashboard, (phase_x + 8, y), 6, p_color, -1)
            else:
                cv2.circle(dashboard, (phase_x + 8, y), 6, p_color, 1)
            phase_x += 75
        y += 22
        
        # === MULTI-JOINT WEIGHTED SCORE ===
        score_color = COLORS['success'] if self._state.current_score >= 70 else COLORS['warning'] if self._state.current_score >= 50 else COLORS['error']
        dashboard = put_vietnamese_text(
            dashboard, f"DIEM TONG: {self._state.current_score:.0f}/100 (TB: {self._state.average_score:.0f})",
            (10, y), score_color, 14
        )
        y += 18
        
        # Score bar
        score_bar_width = width - 30
        score_progress = self._state.current_score / 100.0
        cv2.rectangle(dashboard, (10, y), (10 + score_bar_width, y + 10), (60, 60, 60), -1)
        cv2.rectangle(dashboard, (10, y), (10 + int(score_bar_width * score_progress), y + 10), score_color, -1)
        y += 22
        
        # === MULTI-JOINT DETAILS ===
        dashboard = put_vietnamese_text(dashboard, f"CHI TIET KHOP ({len(self._state.active_joints)} khop):", (10, y), (150, 150, 150), 11)
        y += 18
        
        # Hiển thị từng khớp đang hoạt động với điểm số
        for joint_type in self._state.active_joints:
            if y > height - 100:  # Tránh vẽ ra ngoài
                break
            
            joint_name = JOINT_NAMES.get(joint_type, joint_type.value)
            # Rút gọn tên khớp
            short_name = joint_name[:12] if len(joint_name) > 12 else joint_name
            
            user_ang = self._state.user_angles_dict.get(joint_type, 0)
            target_ang = self._state.target_angles_dict.get(joint_type, 0)
            joint_score = self._state.joint_scores_dict.get(joint_type, 0)
            weight = self._state.joint_weights.get(joint_type, 0.5)
            
            # Màu theo điểm số của khớp
            if joint_score >= 80:
                j_color = COLORS['success']
            elif joint_score >= 60:
                j_color = COLORS['warning']
            else:
                j_color = COLORS['error']
            
            # Hiển thị compact: Tên | Góc/Target | Điểm
            error = abs(user_ang - target_ang) if target_ang > 0 else 0
            line_text = f"{short_name}: {user_ang:.0f}/{target_ang:.0f} | {joint_score:.0f}pt"
            
            # Weight indicator (thanh nhỏ bên phải)
            dashboard = put_vietnamese_text(dashboard, line_text, (15, y), j_color, 10)
            
            # Mini progress bar cho từng khớp
            bar_width = 60
            bar_x = width - bar_width - 10
            bar_progress = min(1.0, joint_score / 100.0)
            cv2.rectangle(dashboard, (bar_x, y - 8), (bar_x + bar_width, y + 2), (50, 50, 50), -1)
            cv2.rectangle(dashboard, (bar_x, y - 8), (bar_x + int(bar_width * bar_progress), y + 2), j_color, -1)
            
            y += 16
        
        y += 8
        
        # === PRIMARY JOINT DETAIL (larger display) ===
        primary_joint = self._state.selected_joint or self._default_joint
        primary_name = JOINT_NAMES.get(primary_joint, "Primary")
        dashboard = put_vietnamese_text(dashboard, f"KHOP CHINH: {primary_name}", (10, y), COLORS['info'], 11)
        y += 18
        
        # Primary angle comparison
        dashboard = put_vietnamese_text(
            dashboard, f"  Goc: {self._state.user_angle:.1f} -> Muc tieu: {self._state.target_angle:.1f}",
            (10, y), COLORS['text'], 10
        )
        y += 16
        
        # Error display với màu sắc
        error = abs(self._state.user_angle - self._state.target_angle)
        if error < 10:
            error_color = COLORS['success']
            feedback = "TUYET VOI!"
        elif error < 20:
            error_color = COLORS['success']
            feedback = "TOT!"
        elif error < 30:
            error_color = COLORS['warning']
            feedback = "KHA"
        else:
            error_color = COLORS['error']
            feedback = "DIEU CHINH!"
        
        dashboard = put_vietnamese_text(
            dashboard, f"  Sai so: {error:.1f}do - {feedback}",
            (10, y), error_color, 10
        )
        y += 18
        
        # Direction hint
        if self._state.target_angle > 0:
            if self._state.user_angle < self._state.target_angle - 10:
                hint = "^ Nang cao hon!"
                hint_color = COLORS['info']
            elif self._state.user_angle > self._state.target_angle + 10:
                hint = "v Ha thap hon!"
                hint_color = COLORS['warning']
            else:
                hint = "= Giu nguyen!"
                hint_color = COLORS['success']
            dashboard = put_vietnamese_text(dashboard, f"  {hint}", (10, y), hint_color, 10)
            y += 20
        
        # Fatigue & Pain (compact)
        dashboard = put_vietnamese_text(
            dashboard, f"Met moi: {self._state.fatigue_level}",
            (10, y), COLORS['text'], 10
        )
        pain_color = COLORS['success'] if self._state.pain_level == "NONE" else COLORS['error']
        dashboard = put_vietnamese_text(
            dashboard, f"| Dau: {self._state.pain_level}",
            (140, y), pain_color, 10
        )
        y += 25
        
        # Warning
        if self._state.warning:
            cv2.rectangle(dashboard, (5, y), (width - 5, y + 35), (0, 0, 100), -1)
            dashboard = put_vietnamese_text(dashboard, f"CANH BAO: {self._state.warning[:30]}", (10, y + 20), COLORS['error'], 9)
        
        # Controls
        dashboard = put_vietnamese_text(dashboard, "[SPACE] Dung/Tiep | [Q] Ket thuc", (10, height - 20), (100, 100, 100), 9)
        
        return dashboard
    
    # ================== PHASE 4: SCORING ==================
    
    def _run_phase4(self, frame: np.ndarray) -> np.ndarray:
        """Phase 4: Hiển thị kết quả."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Overlay tối
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
        
        # Tiêu đề
        output = put_vietnamese_text(
            output, "GIAI DOAN 4: KET QUA BUOI TAP",
            (w // 2 - 180, 50), COLORS['info'], 22
        )
        
        y = 110
        
        # Tổng số hiệp
        output = put_vietnamese_text(
            output, f"Tong so hiep: {self._state.rep_count}",
            (100, y), COLORS['text'], 18
        )
        y += 45
        
        # Điểm trung bình
        score = self._state.average_score
        if score >= 80:
            grade = "XUAT SAC"
            grade_color = COLORS['success']
        elif score >= 60:
            grade = "KHA"
            grade_color = COLORS['warning']
        else:
            grade = "CAN CO GANG"
            grade_color = COLORS['error']
        
        output = put_vietnamese_text(
            output, f"Diem trung binh: {score:.0f}/100",
            (100, y), COLORS['text'], 18
        )
        y += 30
        
        output = put_vietnamese_text(
            output, f"Danh gia: {grade}",
            (100, y), grade_color, 20
        )
        y += 50
        
        # Chi tiết
        output = put_vietnamese_text(output, "Chi tiet diem:", (100, y), (150, 150, 150), 14)
        y += 28
        
        scorer_status = self._scorer.get_current_status()
        details = [
            ("ROM (bien do)", scorer_status.get("last_rom", 0)),
            ("Stability (on dinh)", scorer_status.get("last_stability", 0)),
            ("Flow (mu mut)", scorer_status.get("last_flow", 0)),
        ]
        
        for name, value in details:
            output = put_vietnamese_text(
                output, f"  {name}: {value:.0f}",
                (100, y), COLORS['text'], 14
            )
            y += 25
        
        y += 25
        
        # Calibration info
        output = put_vietnamese_text(
            output, f"Goc toi da (calibrated): {self._state.user_max_angle:.1f}",
            (100, y), COLORS['text'], 14
        )
        y += 35
        
        # Fatigue
        output = put_vietnamese_text(
            output, f"Muc do met moi: {self._state.fatigue_level}",
            (100, y), COLORS['text'], 14
        )
        y += 40
        
        # Khuyến nghị
        output = put_vietnamese_text(output, "Khuyen nghi:", (100, y), COLORS['info'], 16)
        y += 28
        
        recommendations = [
            "Tiep tuc tap luyen deu dan moi ngay",
            "Tang dan cuong do theo tung tuan",
            "Nghi ngoi day du giua cac buoi tap",
        ]
        
        for rec in recommendations:
            output = put_vietnamese_text(output, f"  - {rec}", (100, y), COLORS['text'], 13)
            y += 24
        
        # Lưu thông báo
        output = put_vietnamese_text(
            output, "Ket qua da duoc luu vao log",
            (100, h - 80), COLORS['success'], 14
        )
        
        # Controls
        output = put_vietnamese_text(
            output, "[R] Tap lai tu dau | [Q] Thoat",
            (w // 2 - 120, h - 40), COLORS['info'], 14
        )
        
        return output
    
    # ================== MAIN LOOP ==================
    
    def run(self, user_source: str = "webcam", display: bool = True) -> Dict:
        """Chạy ứng dụng với luồng 4 phase."""
        # Mở camera/video
        cap = cv2.VideoCapture(0 if user_source.lower() == "webcam" else user_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {user_source}")
            return {}
        
        # Init ref detector
        self._init_ref_detector()
        
        # Print banner
        print("\n" + "=" * 60)
        print("MEMOTION - He thong ho tro phuc hoi chuc nang v2.0")
        print("=" * 60)
        print("CHE DO TU DONG - Khong can nhan ENTER")
        print("Cac giai doan:")
        print("  1. Nhan dien tu the -> Tu dong chuyen sau 3 giay")
        print("  2. Do gioi han 6 khop -> Tu dong chuyen sau 2 giay")
        print("  3. Dong bo video mau -> Tu dong chuyen khi hoan tat")
        print("  4. Cham diem va phan tich")
        print("=" * 60)
        print("[Q] Thoat | [R] Restart | [SPACE] Pause (Phase 3)")
        print("=" * 60)
        print()
        
        self._state.is_running = True
        self._state.current_phase = AppPhase.PHASE1_DETECTION
        
        while self._state.is_running:
            ret, frame = cap.read()
            if not ret:
                if user_source.lower() != "webcam":
                    break
                continue
            
            if user_source.lower() == "webcam":
                frame = cv2.flip(frame, 1)
            
            timestamp_ms = int(time.time() * 1000)
            timestamp = time.time()
            
            # Process detection
            result = self._detector.process_frame(frame, timestamp_ms)
            
            # Handle phase
            if self._state.current_phase == AppPhase.PHASE1_DETECTION:
                display_frame = self._run_phase1(frame, result)
            
            elif self._state.current_phase == AppPhase.PHASE2_CALIBRATION:
                display_frame = self._run_phase2(frame, result, timestamp_ms)
            
            elif self._state.current_phase == AppPhase.PHASE3_SYNC:
                ref_frame = None
                
                if self._video_engine:
                    if not self._state.is_paused:
                        # Handle sync status
                        if self._state.sync_state:
                            if self._state.sync_state.sync_status == SyncStatus.PAUSE:
                                self._video_engine.pause()
                            elif self._state.sync_state.sync_status in (SyncStatus.PLAY, SyncStatus.SKIP):
                                if self._video_engine.state != PlaybackState.PLAYING:
                                    self._video_engine.play()
                    
                    ref_frame, ref_status = self._video_engine.get_frame()
                    
                    # Check rep completion
                    if self._state.sync_state:
                        current_mp = self._state.sync_state.current_phase
                        if (self._state.last_motion_phase == MotionPhase.CONCENTRIC and 
                            current_mp == MotionPhase.IDLE):
                            self._on_rep_complete()
                        self._state.last_motion_phase = current_mp
                    
                    # Check completion
                    if (self._state.sync_state and 
                        self._state.sync_state.sync_status == SyncStatus.COMPLETE):
                        self._transition_to_phase4()
                    
                    if ref_status.state == PlaybackState.FINISHED:
                        self._transition_to_phase4()
                
                # Update scorer
                if result.has_pose() and self._state.sync_state:
                    pose_data = result.pose_landmarks.to_numpy()
                    self._scorer.add_frame(
                        self._state.user_angle,
                        timestamp,
                        self._state.sync_state.current_phase,
                        pose_landmarks=pose_data
                    )
                
                # Update scorer status
                scorer_status = self._scorer.get_current_status()
                self._state.current_score = scorer_status.get("last_score", 0)
                self._state.average_score = scorer_status.get("average_score", 0)
                self._state.fatigue_level = scorer_status.get("fatigue_level", "FRESH")
                
                # Pain detection
                if result.has_face() and not self._analysis_queue.full():
                    self._analysis_queue.put(result.face_landmarks.to_numpy())
                    self._process_pain()
                
                display_frame = self._run_phase3(frame, ref_frame, result, timestamp)
            
            elif self._state.current_phase == AppPhase.PHASE4_SCORING:
                display_frame = self._run_phase4(frame)
            
            else:
                display_frame = frame
            
            # Display
            if display:
                cv2.imshow(self.WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)
        
        # Cleanup
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        if self._ref_detector:
            self._ref_detector.close()
        
        return self._generate_report()
    
    def _handle_key(self, key: int) -> None:
        """Xử lý phím nhấn."""
        if key == ord('q') or key == 27:
            self._state.is_running = False
        
        elif key == 13:  # ENTER - Manual override (bỏ qua countdown)
            self._advance_phase()
        
        elif key == ord(' '):
            # Phase 3: Pause/Resume video
            if self._state.current_phase == AppPhase.PHASE3_SYNC:
                self._state.is_paused = not self._state.is_paused
                if self._video_engine:
                    if self._state.is_paused:
                        self._video_engine.pause()
                    else:
                        self._video_engine.play()
        
        elif key == ord('r'):
            self._restart()
        
        # TẤT CẢ phase transitions giờ đều TỰ ĐỘNG
        # ENTER chỉ là manual override để bỏ qua countdown
    
    def _advance_phase(self) -> None:
        """Chuyển phase tiếp theo (manual override - bỏ qua countdown)."""
        if self._state.current_phase == AppPhase.PHASE1_DETECTION:
            # Manual override: cho phép bỏ qua countdown 3 giây
            if self._state.pose_detected:
                self._transition_to_phase2()
        
        # Phase 2 → 3: Tự động sau khi đo xong 6 khớp (2 giây delay)
        # Phase 3 → 4: Tự động khi video FINISHED hoặc SyncStatus.COMPLETE
    
    def _transition_to_phase2(self) -> None:
        """Chuyển sang Phase 2 - Tự động đo 6 khớp."""
        print("\n[PHASE 2] Bat dau Calibration tu dong cho 6 khop...")
        print("  Thu tu: Vai trai -> Vai phai -> Khuyu trai -> Khuyu phai -> Goi trai -> Goi phai")
        self._state.current_phase = AppPhase.PHASE2_CALIBRATION
        self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        # Reset calibration state
        self._state.calibration_queue_index = 0
        self._state.calibrated_joints = {}
        self._state.is_countdown_active = False
        self._state.is_calibrating_joint = False
        self._state.all_joints_calibrated = False
    
    def _transition_to_phase3(self) -> None:
        """Chuyển sang Phase 3 với dữ liệu calibration từ Phase 2 (Multi-joint)."""
        if not self._ref_video_path or not Path(self._ref_video_path).exists():
            print("[WARNING] Khong co video mau, chuyen sang Phase 4")
            self._transition_to_phase4()
            return
        
        print("\n[PHASE 3] Bat dau Motion Sync (Multi-joint)...")
        self._state.current_phase = AppPhase.PHASE3_SYNC
        
        # Setup video engine
        self._video_engine = VideoEngine(self._ref_video_path)
        total_frames = self._video_engine.total_frames
        fps = self._video_engine.fps
        
        # === SETUP MULTI-JOINT TRACKING ===
        # Lấy primary joint
        primary_joint = self._state.selected_joint or self._default_joint
        
        # Xác định loại bài tập dựa trên primary joint
        if primary_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise_type = "bicep_curl"
        elif primary_joint in (JointType.LEFT_KNEE, JointType.RIGHT_KNEE):
            exercise_type = "squat"
        else:
            exercise_type = "arm_raise"
        
        # Lấy trọng số cho từng khớp từ exercise type
        self._state.joint_weights = create_exercise_weights(exercise_type)
        
        # Xác định các khớp đang hoạt động (đã được calibrated)
        self._state.active_joints = list(self._state.calibrated_joints.keys())
        
        # Nếu không có khớp nào được calibrated, sử dụng primary joint
        if not self._state.active_joints:
            self._state.active_joints = [primary_joint]
            self._state.calibrated_joints[primary_joint] = 150.0  # Default
        
        # Lấy max angle của primary joint
        if primary_joint in self._state.calibrated_joints:
            max_angle = self._state.calibrated_joints[primary_joint]
            self._state.user_max_angle = max_angle
        elif self._state.user_max_angle > 0:
            max_angle = self._state.user_max_angle
        else:
            max_angle = 150
        
        # Create exercise với max_angle từ calibration
        if primary_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise = create_elbow_flex_exercise(total_frames, fps, max_angle=max_angle)
        else:
            exercise = create_arm_raise_exercise(total_frames, fps, max_angle=max_angle)
        
        # Setup sync controller
        self._sync_controller = MotionSyncController(
            exercise,
            user_max_angle=max_angle
        )
        
        # Khởi tạo dictionaries cho multi-joint tracking
        self._state.user_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.target_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.joint_scores_dict = {jt: 0.0 for jt in self._state.active_joints}
        
        # Setup checkpoints
        checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
        self._video_engine.set_checkpoints(checkpoint_frames)
        self._video_engine.set_speed(0.7)
        
        # Start session
        session_id = f"session_{int(time.time())}"
        self._logger.start_session(session_id, exercise.name)
        self._scorer.start_session(exercise.name, session_id)
        
        self._video_engine.play()
        
        print(f"[SETUP] Exercise: {exercise.name} (type: {exercise_type})")
        print(f"[SETUP] Primary joint: {JOINT_NAMES.get(primary_joint, primary_joint.value)}")
        print(f"[SETUP] Primary max angle: {max_angle:.1f}")
        
        # In ra tất cả các khớp đang hoạt động với trọng số
        print(f"[SETUP] Active joints ({len(self._state.active_joints)}):")
        for jt in self._state.active_joints:
            angle = self._state.calibrated_joints.get(jt, 0)
            weight = self._state.joint_weights.get(jt, 0.5)
            print(f"  - {JOINT_NAMES.get(jt, jt.value)}: max={angle:.1f}do, weight={weight:.2f}")
    
    def _transition_to_phase4(self) -> None:
        """Chuyển sang Phase 4."""
        print("\n[PHASE 4] Hien thi ket qua...")
        self._state.current_phase = AppPhase.PHASE4_SCORING
        
        if self._scorer:
            report = self._scorer.compute_session_report()
            self._state.average_score = report.average_scores.get('total', 0)
    
    def _on_rep_complete(self) -> None:
        """Xử lý khi hoàn thành 1 rep."""
        dtw_result = None
        if len(self._user_angles) > 20 and len(self._ref_angles) > 20:
            user_seq = self._user_angles[-50:]
            ref_seq = self._ref_angles[-50:]
            dtw_result = compute_single_joint_dtw(user_seq, ref_seq)
        
        target = self._state.target_angle or self._state.user_max_angle or 150
        rep_score = self._scorer.complete_rep(target, dtw_result)
        
        self._logger.log_rep(
            rep_score.rep_number,
            {
                "rom": rep_score.rom_score,
                "stability": rep_score.stability_score,
                "flow": rep_score.flow_score,
                "total": rep_score.total_score
            },
            rep_score.jerk_value,
            rep_score.duration_ms
        )
        
        print(f"[REP {rep_score.rep_number}] Score: {rep_score.total_score:.0f}")
    
    def _process_pain(self) -> None:
        """Xử lý pain detection."""
        try:
            face_landmarks = self._analysis_queue.get_nowait()
            result = self._pain_detector.analyze(face_landmarks)
            if result.is_pain_detected:
                self._state.pain_level = result.pain_level.name
                self._state.warning = result.message
            else:
                self._state.pain_level = "NONE"
                self._state.warning = ""
        except:
            pass
    
    def _restart(self) -> None:
        """Restart từ đầu."""
        print("\n[RESTART]...")
        self._state = AppState()
        self._state.selected_joint = self._default_joint
        self._state.calibrated_joints = {}  # Reset calibrated joints
        self._user_angles = []
        self._ref_angles = []
        self._score_history = []
        
        # Xóa phase2 complete time nếu có
        if hasattr(self, '_phase2_complete_time'):
            delattr(self, '_phase2_complete_time')
        
        if self._video_engine:
            self._video_engine.stop()
            self._video_engine = None
        
        self._sync_controller = None
        self._calibrator = SafeMaxCalibrator(duration_ms=5000)
        self._user_profile = None
    
    def _generate_report(self) -> Dict:
        """Tạo báo cáo cuối."""
        report = {}
        
        if self._scorer:
            session_report = self._scorer.compute_session_report()
            report = session_report.to_dict()
        
        report['user_max_angle'] = self._state.user_max_angle
        report['total_reps'] = self._state.rep_count
        
        if self._logger:
            self._logger.end_session(report)
        
        print("\n" + "=" * 60)
        print("KET THUC BUOI TAP")
        print("=" * 60)
        print(f"  Tong hiep: {self._state.rep_count}")
        print(f"  Diem TB: {self._state.average_score:.0f}/100")
        print("=" * 60)
        
        return report
    
    def cleanup(self) -> None:
        """Dọn dẹp."""
        self._state.is_running = False
        if self._video_engine:
            self._video_engine.release()
        if self._ref_detector:
            self._ref_detector.close()


# ================== TESTS ==================

def run_unit_tests():
    """Chạy tests."""
    print("\n" + "=" * 60)
    print("UNIT TESTS - MEMOTION v2.0")
    print("=" * 60)
    
    print("\n[TEST 1] Visualization...")
    from utils.visualization import put_vietnamese_text, draw_skeleton
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = put_vietnamese_text(test_frame, "Test tieng Viet", (10, 50), (255, 255, 255))
    assert result.shape == test_frame.shape
    print("  OK - Vietnamese text")
    
    print("\n[TEST 2] SafeMaxCalibrator...")
    calibrator = SafeMaxCalibrator()
    assert calibrator.state == CalibrationState.IDLE
    print("  OK - Calibrator")
    
    print("\n[TEST 3] PainDetector...")
    detector = PainDetector()
    print("  OK - PainDetector")
    
    print("\n[TEST 4] HealthScorer...")
    scorer = HealthScorer()
    scorer.start_session("test", "test_session")
    for i in range(20):
        scorer.add_frame(30 + i * 2, i * 0.033, MotionPhase.ECCENTRIC)
    rep = scorer.complete_rep(90)
    print(f"  OK - Score: {rep.total_score:.1f}")
    
    print("\n[TEST 5] MotionSyncController...")
    exercise = create_arm_raise_exercise(300, 30.0)
    sync = MotionSyncController(exercise)
    state = sync.update(45.0, 100)
    print(f"  OK - Phase: {state.current_phase.value}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")


# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser(description="MEMOTION v2.0")
    parser.add_argument("--source", type=str, default="webcam")
    parser.add_argument("--ref-video", type=str, default=None)
    parser.add_argument("--joint", type=str, default="left_shoulder",
                       choices=["left_shoulder", "right_shoulder",
                               "left_elbow", "right_elbow",
                               "left_knee", "right_knee"])
    parser.add_argument("--mode", type=str, choices=["run", "test"], default="run")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./data/logs")
    args = parser.parse_args()
    
    if args.mode == "test":
        run_unit_tests()
        return
    
    # Map joint
    joint_map = {
        "left_shoulder": JointType.LEFT_SHOULDER,
        "right_shoulder": JointType.RIGHT_SHOULDER,
        "left_elbow": JointType.LEFT_ELBOW,
        "right_elbow": JointType.RIGHT_ELBOW,
        "left_knee": JointType.LEFT_KNEE,
        "right_knee": JointType.RIGHT_KNEE,
    }
    default_joint = joint_map.get(args.joint, JointType.LEFT_SHOULDER)
    
    # Check model
    models_dir = Path(args.models_dir)
    pose_model = models_dir / "pose_landmarker_lite.task"
    face_model = models_dir / "face_landmarker.task"
    
    if not pose_model.exists():
        print(f"[ERROR] Model not found: {pose_model}")
        print("[INFO] Running tests...")
        run_unit_tests()
        return
    
    config = DetectorConfig(
        pose_model_path=str(pose_model),
        face_model_path=str(face_model) if face_model.exists() else None,
        running_mode="VIDEO"
    )
    
    try:
        with VisionDetector(config) as detector:
            app = MemotionAppV2(
                detector=detector,
                ref_video_path=args.ref_video,
                default_joint=default_joint,
                log_dir=args.log_dir,
                models_dir=args.models_dir
            )
            
            app.run(user_source=args.source, display=not args.headless)
            app.cleanup()
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
