#!/usr/bin/env python3
"""
MEMOTION - Complete Integration (Final Version 2.0)

Tích hợp hoàn chỉnh 4 giai đoạn với UI rõ ràng:
1. PHASE 1: Pose Detection - Nhận diện tư thế, vẽ skeleton
2. PHASE 2: Safe-Max Calibration - Đo giới hạn vận động
3. PHASE 3: Motion Sync - Đồng bộ với video mẫu
4. PHASE 4: Scoring & Analysis - Chấm điểm và phân tích

Usage:
    python main_v2.py --ref-video exercise.mp4
    python main_v2.py --user-video user.mp4 --ref-video exercise.mp4
    python main_v2.py --mode test
    python main_v2.py --phase 1                                    # Test Phase 1 (webcam)
    python main_v2.py --phase 1 --user-video user.mp4             # Test Phase 1 (video)
    python main_v2.py --phase 2 --user-video user.mp4             # Test Phase 2 (video)
    python main_v2.py --phase 3 --user-video user.mp4 --ref-video ref.mp4  # Test Phase 3
    python main_v2.py --phase 4                                    # Test Phase 4

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
        current_time = time.time()
        
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
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
            
            self._state.detection_stable_count += 1
            if self._state.detection_stable_count >= self.DETECTION_STABLE_THRESHOLD:
                self._state.pose_detected = True
                if not self._state.phase1_countdown_active:
                    self._state.phase1_countdown_active = True
                    self._state.phase1_countdown_start = current_time
                
                elapsed = current_time - self._state.phase1_countdown_start
                remaining = self.PHASE1_COUNTDOWN_DURATION - elapsed
                
                if remaining <= 0:
                    self._transition_to_phase2()
        else:
            self._state.detection_stable_count = 0
            self._state.phase1_countdown_active = False
            
        return output
    
    # ================== PHASE 2: CALIBRATION (AUTOMATED) ==================
    
    def _run_phase2(self, frame: np.ndarray, result, timestamp_ms: int) -> np.ndarray:
        """Phase 2: Safe-Max Calibration - Tự động đo 6 khớp."""
        output = frame.copy()
        h, w = frame.shape[:2]
        current_time = time.time()
        
        if self._state.calibration_queue_index < len(CALIBRATION_QUEUE):
            current_joint = CALIBRATION_QUEUE[self._state.calibration_queue_index]
            self._state.selected_joint = current_joint
        else:
            self._state.all_joints_calibrated = True

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
            
            # Vẽ góc nếu đang đo
            if self._state.selected_joint and self._state.user_angle > 0:
                p1, pv, p2 = self._get_joint_pixel_coords(
                    landmarks, self._state.selected_joint, (h, w)
                )
                if p1 and pv and p2:
                    output = draw_angle_arc(output, p1, pv, p2, self._state.user_angle)

        # HUD Progress tổng thể
        total_joints = len(CALIBRATION_QUEUE)
        completed_joints = len(self._state.calibrated_joints)
        
        output = draw_panel(output, (20, 20), (280, 100), "GIOI HAN VAN DONG")
        output = put_vietnamese_text(
            output, f"Tien do: {completed_joints}/{total_joints} khop",
            (35, 65), COLORS['text'], 14
        )
        output = draw_progress_bar(output, (35, 85), (220, 10), completed_joints / max(1, total_joints), COLORS['info'])

        if self._state.all_joints_calibrated:
            output = draw_panel(output, (w//2 - 200, h - 160), (400, 100), "HOAN THANH!")
            output = put_vietnamese_text(
                output, "Da do xong 6 khop. Chuyen sang Phase 3...",
                (w//2 - 185, h - 100), COLORS['success'], 16
            )
            if not hasattr(self, '_phase2_complete_time'):
                self._phase2_complete_time = current_time
                self._save_calibration_to_profile()
            elif current_time - self._phase2_complete_time > 2.0:
                self._state.calibration_complete = True
                self._transition_to_phase3()
                delattr(self, '_phase2_complete_time')
        
        elif not self._state.is_countdown_active and not self._state.is_calibrating_joint:
            self._state.is_countdown_active = True
            self._state.calibration_countdown_start = current_time
        
        elif self._state.is_countdown_active:
            elapsed = current_time - self._state.calibration_countdown_start
            remaining = CALIBRATION_COUNTDOWN_DURATION - elapsed
            
            # HUD Countdown
            joint_name = JOINT_NAMES.get(current_joint, "")
            output = draw_panel(output, (w//2 - 200, h - 180), (400, 140), f"KHOP: {joint_name}")
            output = put_vietnamese_text(
                output, "Thuc hien dong tac HET KHA NANG (khong dau)",
                (w//2 - 185, h - 120), COLORS['warning'], 14
            )
            output = put_vietnamese_text(
                output, f"Bat dau sau: {int(remaining) + 1} s",
                (w//2 - 185, h - 80), COLORS['info'], 20
            )
            
            if remaining <= 0:
                self._state.is_countdown_active = False
                self._state.is_calibrating_joint = True
                self._start_calibration_for_joint(current_joint)
        
        elif self._state.is_calibrating_joint:
            if self._calibrator.state == CalibrationState.COLLECTING:
                joint_name = JOINT_NAMES.get(current_joint, "")
                output = draw_panel(output, (w//2 - 200, h - 160), (400, 120), f"DANG DO: {joint_name}")
                
                progress = self._calibrator.progress
                output = put_vietnamese_text(
                    output, f"Goc hien tai: {self._state.user_angle:.1f} do",
                    (w//2 - 185, h - 110), COLORS['info'], 18
                )
                output = draw_progress_bar(output, (w//2 - 185, h - 70), (350, 12), progress, COLORS['warning'])
                
                if result.has_pose() and current_joint:
                    try:
                        landmarks = result.pose_landmarks.to_numpy()
                        angle = calculate_joint_angle(landmarks, current_joint, use_3d=True)
                        self._state.user_angle = angle
                        self._calibrator.add_frame(result.pose_landmarks, timestamp_ms)
                    except ValueError:
                        pass
                    if self._calibrator.state == CalibrationState.COMPLETED:
                        self._finish_calibration_for_joint(current_joint)
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
            highlight = []
            for joint_type in self._state.active_joints:
                joint_def = JOINT_DEFINITIONS.get(joint_type)
                if joint_def:
                    highlight.extend([joint_def.proximal, joint_def.vertex, joint_def.distal])
            
            user_display = draw_skeleton(
                user_display, self._current_landmarks,
                color=COLORS['skeleton'],
                highlight_indices=list(set(highlight)),
                use_core_only=True
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
        else:
            ref_display = np.zeros((h, w // 2, 3), dtype=np.uint8)
            ref_display[:] = (40, 40, 40)
        
        # === VẼ UI OVERLAYS LÊN VIEW ===
        # Hướng dẫn & Dữ liệu cho User View
        user_display = draw_panel(user_display, (20, 20), (220, 130), "NGUOI DUNG")
        user_display = put_vietnamese_text(user_display, f"Goc: {self._state.user_angle:.1f} do", (35, 60), COLORS['text'], 14)
        user_display = put_vietnamese_text(user_display, f"Muc tieu: {self._state.target_angle:.1f} do", (35, 85), COLORS['info'], 14)
        
        error = abs(self._state.user_angle - self._state.target_angle)
        err_color = COLORS['success'] if error < 15 else COLORS['warning'] if error < 30 else COLORS['error']
        user_display = put_vietnamese_text(user_display, f"Sai so: {error:.1f} do", (35, 110), err_color, 14)
        
        # Điểm số trên User View
        score_color = COLORS['success'] if self._state.current_score >= 70 else COLORS['warning'] if self._state.current_score >= 50 else COLORS['error']
        user_display = draw_panel(user_display, (20, h - 150), (240, 130), "DIEM SO")
        user_display = put_vietnamese_text(user_display, f"Hien tai: {self._state.current_score:.0f}", (35, h - 110), score_color, 16)
        user_display = put_vietnamese_text(user_display, f"Tr.Binh: {self._state.average_score:.0f}", (35, h - 80), COLORS['text'], 14)
        user_display = draw_progress_bar(user_display, (35, h - 55), (190, 10), self._state.current_score/100.0, score_color)
        
        # Thông tin Phase trên Ref View
        ref_display = draw_panel(ref_display, (20, 20), (220, 110), "DONG BO")
        phase = self._state.motion_phase.lower()
        phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
        phase_name = PHASE_NAMES_VI.get(phase, phase.upper())
        ref_display = put_vietnamese_text(ref_display, f"Giai doan: {phase_name}", (35, 60), phase_color, 14)
        ref_display = put_vietnamese_text(ref_display, f"Hiep tap: {self._state.rep_count}", (35, 85), COLORS['text'], 14)

        if self._video_engine:
            progress = self._video_engine.current_frame / max(1, self._video_engine.total_frames)
            ref_w_prog = ref_display.shape[1]
            ref_display = draw_progress_bar(ref_display, (20, h - 35), (ref_w_prog - 40, 10), progress, COLORS['info'])

        # === COMBINE USER + REF HORIZONTALLY ===
        target_h = h
        user_h, user_w = user_display.shape[:2]
        ref_h, ref_w = ref_display.shape[:2]

        new_user_w = int(user_w * target_h / max(1, user_h))
        user_resized = cv2.resize(user_display, (new_user_w, target_h))

        new_ref_w = int(ref_w * target_h / max(1, ref_h))
        ref_resized = cv2.resize(ref_display, (new_ref_w, target_h))

        combined = np.hstack([user_resized, ref_resized])
        
        # === UPDATE SYNC & SCORING (MULTI-JOINT) ===
        if self._sync_controller and self._video_engine:
            self._state.sync_state = self._sync_controller.update(
                self._state.user_angle,
                self._video_engine.current_frame,
                timestamp
            )
            
            self._state.motion_phase = self._state.sync_state.current_phase.value
            self._state.rep_count = self._state.sync_state.rep_count
            
            multi_joint_score = self._calculate_multi_joint_score()
            self._state.current_score = 0.7 * self._state.current_score + 0.3 * multi_joint_score
            
            if len(self._state.target_angles_dict) > 0:
                self._score_history.append(multi_joint_score)
                if len(self._score_history) > 0:
                    self._state.average_score = sum(self._score_history) / len(self._score_history)
        
        self._user_angles.append(self._state.user_angle)
        if self._state.target_angle > 0:
            self._ref_angles.append(self._state.target_angle)
            
        return combined
    


    # ================== PHASE 4: SCORING ==================
    
    def _run_phase4(self, frame: np.ndarray) -> np.ndarray:
        """Phase 4: Hiển thị kết quả."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Làm tối nền nhẹ
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        
        # HUD Panel ở giữa
        panel_w = 400
        panel_h = 420
        panel_x = w // 2 - panel_w // 2
        panel_y = h // 2 - panel_h // 2
        
        output = draw_panel(output, (panel_x, panel_y), (panel_w, panel_h), "TONG KET BUOI TAP")
        
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
            
        y_pos = panel_y + 60
        output = put_vietnamese_text(output, f"Tong so hiep: {self._state.rep_count}", (panel_x + 20, y_pos), COLORS['text'], 16)
        y_pos += 40
        output = put_vietnamese_text(output, "Diem trung binh:", (panel_x + 20, y_pos), COLORS['text'], 16)
        output = put_vietnamese_text(output, f"{score:.0f}", (panel_x + 180, y_pos - 10), score_color, 30)
        y_pos += 40
        output = put_vietnamese_text(output, f"Danh gia: {grade}", (panel_x + 20, y_pos), grade_color, 16)
        y_pos += 40
        
        output = put_vietnamese_text(output, "Chi tiet diem:", (panel_x + 20, y_pos), (150, 150, 150), 14)
        y_pos += 30
        
        scorer_status = self._scorer.get_current_status()
        details = [
            ("ROM (bien do)", scorer_status.get("last_rom", 0)),
            ("Stability (on dinh)", scorer_status.get("last_stability", 0)),
            ("Flow (mu mut)", scorer_status.get("last_flow", 0)),
        ]
        
        for name, value in details:
            output = put_vietnamese_text(output, f"  {name}:", (panel_x + 20, y_pos), COLORS['text'], 14)
            output = draw_progress_bar(output, (panel_x + 200, y_pos - 5), (150, 10), value/100, COLORS['info'])
            y_pos += 30
            
        y_pos += 10
        output = put_vietnamese_text(output, "Khuyen nghi:", (panel_x + 20, y_pos), COLORS['info'], 14)
        y_pos += 30
        output = put_vietnamese_text(output, "- Tap luyen deu dan", (panel_x + 30, y_pos), COLORS['text'], 14)
        y_pos += 25
        output = put_vietnamese_text(output, "- Nghi ngoi du sau buoi tap", (panel_x + 30, y_pos), COLORS['text'], 14)
        
        output = put_vietnamese_text(output, "[Q] Thoat | [R] Restart", (panel_x + 90, panel_y + panel_h - 30), (150, 150, 150), 14)
        
        return output
    
    # ================== MAIN LOOP ==================
    
    def setup_phase_test(self, phase: int) -> None:
        """Thiết lập trạng thái giả lập để test từng phase riêng lẻ.
        
        Khi test phase N, các phase trước đó sẽ được bỏ qua với dữ liệu mặc định.
        
        Args:
            phase: Số phase cần test (1-4)
        """
        self._test_phase = phase  # Lưu lại phase đang test
        self._lock_phase = True   # Khóa không cho tự động chuyển phase
        
        if phase == 1:
            # Phase 1: Bắt đầu bình thường từ Pose Detection
            self._state.current_phase = AppPhase.PHASE1_DETECTION
            print("[TEST] Phase 1: Nhan dien tu the")
            print("  - Dung truoc camera de he thong nhan dien")
            print("  - Phase se KHONG tu dong chuyen sang Phase 2")
        
        elif phase == 2:
            # Phase 2: Bỏ qua Phase 1, giả lập pose detected
            self._state.current_phase = AppPhase.PHASE2_CALIBRATION
            self._state.pose_detected = True
            self._state.detection_stable_count = self.DETECTION_STABLE_THRESHOLD
            self._user_profile = UserProfile(user_id=f"test_user_{int(time.time())}")
            
            # Reset calibration state
            self._state.calibration_queue_index = 0
            self._state.calibrated_joints = {}
            self._state.is_countdown_active = False
            self._state.is_calibrating_joint = False
            self._state.all_joints_calibrated = False
            
            print("[TEST] Phase 2: Calibration (Do gioi han van dong)")
            print("  - Bo qua Phase 1 (pose da duoc nhan dien)")
            print("  - Tu dong do 6 khop, KHONG chuyen sang Phase 3")
        
        elif phase == 3:
            # Phase 3: Bỏ qua Phase 1 & 2, giả lập calibration data
            self._state.pose_detected = True
            self._state.calibration_complete = True
            self._user_profile = UserProfile(user_id=f"test_user_{int(time.time())}")
            
            # Giả lập kết quả calibration mặc định cho 6 khớp
            default_angles = {
                JointType.LEFT_SHOULDER: 150.0,
                JointType.RIGHT_SHOULDER: 150.0,
                JointType.LEFT_ELBOW: 140.0,
                JointType.RIGHT_ELBOW: 140.0,
                JointType.LEFT_KNEE: 120.0,
                JointType.RIGHT_KNEE: 120.0,
            }
            self._state.calibrated_joints = default_angles
            self._state.user_max_angle = default_angles.get(
                self._default_joint, 150.0
            )
            self._state.all_joints_calibrated = True
            
            # Chuyển vào Phase 3 (cần ref video)
            if self._ref_video_path and Path(self._ref_video_path).exists():
                self._transition_to_phase3()
                print("[TEST] Phase 3: Motion Sync (Dong bo chuyen dong)")
                print("  - Bo qua Phase 1 & 2 (dung du lieu calibration mac dinh)")
                print(f"  - Calibration mac dinh: {default_angles}")
                print("  - Phase se KHONG tu dong chuyen sang Phase 4")
            else:
                print("[ERROR] Phase 3 can video mau! Su dung: --ref-video <path>")
                self._state.is_running = False
                return
        
        elif phase == 4:
            # Phase 4: Bỏ qua tất cả, giả lập kết quả scoring
            self._state.current_phase = AppPhase.PHASE4_SCORING
            self._state.pose_detected = True
            self._state.calibration_complete = True
            self._state.user_max_angle = 150.0
            
            # Giả lập calibration data
            self._state.calibrated_joints = {
                JointType.LEFT_SHOULDER: 150.0,
                JointType.RIGHT_SHOULDER: 148.0,
                JointType.LEFT_ELBOW: 138.0,
                JointType.RIGHT_ELBOW: 142.0,
                JointType.LEFT_KNEE: 118.0,
                JointType.RIGHT_KNEE: 122.0,
            }
            
            # Giả lập kết quả tập luyện
            self._state.rep_count = 5
            self._state.current_score = 78.5
            self._state.average_score = 75.0
            self._state.fatigue_level = "MODERATE"
            self._state.pain_level = "NONE"
            
            # Giả lập scorer data
            self._scorer.start_session("test_exercise", f"test_session_{int(time.time())}")
            for i in range(50):
                self._scorer.add_frame(30 + i * 2.4, i * 0.033, MotionPhase.ECCENTRIC)
            self._scorer.complete_rep(150.0)
            
            print("[TEST] Phase 4: Scoring & Analysis (Ket qua)")
            print("  - Bo qua Phase 1, 2, 3 (dung du lieu gia lap)")
            print("  - Hien thi man hinh ket qua voi du lieu mau")
        
        else:
            print(f"[ERROR] Phase khong hop le: {phase}. Chon 1-4.")
            self._state.is_running = False
    
    def run(self, user_source: str = "webcam", display: bool = True, test_phase: Optional[int] = None) -> Dict:
        """Chạy ứng dụng với luồng 4 phase.
        
        Args:
            user_source: Nguồn video ('webcam' hoặc đường dẫn file)
            display: Có hiển thị cửa sổ không
            test_phase: Nếu chỉ định (1-4), chỉ chạy phase đó để test
        """
        # Mở camera/video
        cap = cv2.VideoCapture(0 if user_source.lower() == "webcam" else user_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {user_source}")
            return {}

        # Init ref detector
        self._init_ref_detector()

        # === FULLSCREEN SETUP ===
        if display:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self._state.is_running = True
        self._lock_phase = False  # Mặc định không khóa phase
        self._test_phase = None
        
        # === SETUP TEST PHASE nếu có ===
        if test_phase is not None:
            self.setup_phase_test(test_phase)
            if not self._state.is_running:
                cap.release()
                return {}
            
            # Print banner cho test mode
            print("\n" + "=" * 60)
            print(f"MEMOTION v2.0 - TEST PHASE {test_phase}")
            print("=" * 60)
            print(f"Dang chay Phase {test_phase} doc lap (khong tu dong chuyen phase)")
            print("[Q] Thoat | [R] Restart phase hien tai")
            print("=" * 60)
            print()
        else:
            # Print banner cho normal mode
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
            
            elif self._state.current_phase == AppPhase.COMPLETED:
                # Phase test hoàn thành, hiện thông báo
                display_frame = frame.copy()
                h_f, w_f = display_frame.shape[:2]
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w_f, h_f), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                display_frame = put_vietnamese_text(
                    display_frame, f"PHASE {self._test_phase} - HOAN THANH!",
                    (w_f // 2 - 150, h_f // 2 - 30), COLORS['success'], 24
                )
                display_frame = put_vietnamese_text(
                    display_frame, "[R] Chay lai | [Q] Thoat",
                    (w_f // 2 - 100, h_f // 2 + 30), COLORS['text'], 14
                )
            
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
            if not self._lock_phase:
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
            if self._lock_phase and self._test_phase:
                # Trong test mode: restart lại phase đang test
                self._restart()
                self.setup_phase_test(self._test_phase)
                print(f"[TEST] Restart Phase {self._test_phase}")
            else:
                self._restart()
        
        # TẤT CẢ phase transitions giờ đều TỰ ĐỘNG
        # ENTER chỉ là manual override để bỏ qua countdown
    
    def _advance_phase(self) -> None:
        """Chuyển phase tiếp theo (manual override - bỏ qua countdown)."""
        # Nếu đang lock phase (test mode), không cho chuyển phase
        if self._lock_phase:
            return
        
        if self._state.current_phase == AppPhase.PHASE1_DETECTION:
            # Manual override: cho phép bỏ qua countdown 3 giây
            if self._state.pose_detected:
                self._transition_to_phase2()
        
        # Phase 2 → 3: Tự động sau khi đo xong 6 khớp (2 giây delay)
        # Phase 3 → 4: Tự động khi video FINISHED hoặc SyncStatus.COMPLETE
    
    def _transition_to_phase2(self) -> None:
        """Chuyển sang Phase 2 - Tự động đo 6 khớp."""
        # Nếu đang lock phase (test mode Phase 1), đánh dấu completed
        if self._lock_phase and self._test_phase == 1:
            print("\n[TEST] Phase 1 HOAN THANH! Khong chuyen sang Phase 2.")
            self._state.current_phase = AppPhase.COMPLETED
            return
        
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
        # Nếu đang lock phase (test mode Phase 2), đánh dấu completed
        if self._lock_phase and self._test_phase == 2:
            print("\n[TEST] Phase 2 HOAN THANH! Khong chuyen sang Phase 3.")
            print(f"[TEST] Ket qua calibration: {self._state.calibrated_joints}")
            self._state.current_phase = AppPhase.COMPLETED
            return
        
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
        # Nếu đang lock phase (test mode Phase 3), đánh dấu completed
        if self._lock_phase and self._test_phase == 3:
            print("\n[TEST] Phase 3 HOAN THANH! Khong chuyen sang Phase 4.")
            print(f"[TEST] Rep count: {self._state.rep_count}")
            print(f"[TEST] Average score: {self._state.average_score:.1f}")
            self._state.current_phase = AppPhase.COMPLETED
            return
        
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
    parser = argparse.ArgumentParser(
        description="MEMOTION v2.0 - He thong ho tro phuc hoi chuc nang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du su dung:
  # Chay binh thuong voi webcam (4 phases)
  python main_v2.py --ref-video videos/ref.mp4

  # Chay voi video user thay vi webcam
  python main_v2.py --user-video videos/user.mp4 --ref-video videos/ref.mp4

  # Test tung phase rieng le
  python main_v2.py --phase 1                                       # Pose Detection (webcam)
  python main_v2.py --phase 1 --user-video videos/user.mp4          # Pose Detection (video)
  python main_v2.py --phase 2                                       # Calibration (webcam)
  python main_v2.py --phase 2 --user-video videos/user.mp4          # Calibration (video)
  python main_v2.py --phase 3 --ref-video videos/ref.mp4            # Motion Sync (webcam)
  python main_v2.py --phase 3 --user-video videos/user.mp4 --ref-video videos/ref.mp4
  python main_v2.py --phase 4                                       # Scoring

  # Chay unit tests
  python main_v2.py --mode test
"""
    )
    parser.add_argument("--source", type=str, default="webcam",
                       help="Nguon video: 'webcam' hoac duong dan file (xem --user-video)")
    parser.add_argument("--user-video", type=str, default=None,
                       help="Duong dan video nguoi dung (thay cho webcam). "
                            "Neu chi dinh, se dung video nay thay vi webcam. "
                            "VD: --user-video videos/user.mp4")
    parser.add_argument("--ref-video", type=str, default=None,
                       help="Duong dan video mau (bat buoc cho Phase 3). "
                            "VD: --ref-video videos/ref.mp4")
    parser.add_argument("--joint", type=str, default="left_shoulder",
                       choices=["left_shoulder", "right_shoulder",
                               "left_elbow", "right_elbow",
                               "left_knee", "right_knee"],
                       help="Khop mac dinh de theo doi")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                       help="Chi chay mot phase cu the de test (1-4). "
                            "Phase 1: Pose Detection, Phase 2: Calibration, "
                            "Phase 3: Motion Sync (can --ref-video), Phase 4: Scoring")
    parser.add_argument("--mode", type=str, choices=["run", "test"], default="run",
                       help="Che do chay: 'run' (binh thuong) hoac 'test' (unit tests)")
    parser.add_argument("--headless", action="store_true",
                       help="Chay khong hien thi cua so")
    parser.add_argument("--models-dir", type=str, default="./models",
                       help="Thu muc chua models")
    parser.add_argument("--log-dir", type=str, default="./data/logs",
                       help="Thu muc luu log")
    args = parser.parse_args()
    
    # --user-video ghi de --source
    if args.user_video:
        if not Path(args.user_video).exists():
            print(f"[ERROR] Video nguoi dung khong ton tai: {args.user_video}")
            sys.exit(1)
        args.source = args.user_video
        print(f"[INFO] Su dung video nguoi dung: {args.user_video}")
    
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
            
            print(f"[INFO] Source: {args.source}")
            if args.ref_video:
                print(f"[INFO] Ref video: {args.ref_video}")
            if args.phase:
                print(f"[INFO] Test phase: {args.phase}")
            
            app.run(
                user_source=args.source,
                display=not args.headless,
                test_phase=args.phase
            )
            app.cleanup()
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
