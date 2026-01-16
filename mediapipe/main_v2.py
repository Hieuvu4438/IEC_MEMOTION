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
from typing import Optional, Dict, List, Tuple
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
    compute_single_joint_dtw, PoseLandmarkIndex,
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
    
    # Phase 2 state
    selected_joint: Optional[JointType] = None
    calibration_complete: bool = False
    user_max_angle: float = 0.0
    
    # Phase 3 state
    sync_state: Optional[SyncState] = None
    motion_phase: str = "idle"
    last_motion_phase: Optional[MotionPhase] = None
    
    # Phase 4 state
    rep_count: int = 0
    current_score: float = 0.0
    average_score: float = 0.0
    
    # Common
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
    
    def _interpolate_target_angle(self, current_frame: int, total_frames: int) -> float:
        """
        Tính target angle dựa trên vị trí frame trong video.
        
        Thay vì chỉ lấy target từ checkpoint, ta interpolate
        để có target liên tục cho mọi frame.
        """
        if not self._sync_controller:
            return self._state.user_max_angle or 150.0
        
        exercise = self._sync_controller.exercise
        checkpoints = exercise.checkpoints
        
        if not checkpoints:
            return self._state.user_max_angle or 150.0
        
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
            return prev_cp.target_angle
        
        progress = (current_frame - prev_cp.frame_index) / max(1, next_cp.frame_index - prev_cp.frame_index)
        progress = max(0, min(1, progress))
        
        interpolated = prev_cp.target_angle + progress * (next_cp.target_angle - prev_cp.target_angle)
        
        return interpolated
    
    def _calculate_realtime_score(self, user_angle: float, target_angle: float) -> float:
        """
        Tính điểm thời gian thực dựa trên sai số góc.
        
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
    
    # ================== PHASE 1: POSE DETECTION ==================
    
    def _run_phase1(self, frame: np.ndarray, result) -> np.ndarray:
        """Phase 1: Nhận diện Pose và vẽ skeleton."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel hướng dẫn
        output = draw_panel(output, (10, 10), (450, 220), "")
        
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
            "Khi xac nhan xong, nhan ENTER de tiep tuc",
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
                status_text = "DA NHAN DIEN THANH CONG! Nhan ENTER"
                status_color = COLORS['success']
            else:
                status_text = f"Dang xac nhan... {int(progress * 100)}%"
                status_color = COLORS['warning']
            
            output = put_vietnamese_text(output, status_text, (25, y_pos + 10), status_color, 16)
            output = draw_progress_bar(output, (25, y_pos + 35), (400, 18), progress, status_color)
        else:
            self._state.detection_stable_count = 0
            output = put_vietnamese_text(
                output, "Chua phat hien nguoi. Hay dung vao khung hinh.",
                (25, y_pos + 10), COLORS['error'], 16
            )
        
        # Hiển thị thông tin phase hiện tại
        phase_text = "Phase: 1/4 - Nhan dien tu the"
        output = put_vietnamese_text(output, phase_text, (w - 280, 30), COLORS['info'], 14)
        
        # Controls
        controls = "[ENTER] Tiep tuc | [Q] Thoat"
        output = put_vietnamese_text(output, controls, (w - 280, h - 25), (150, 150, 150), 12)
        
        return output
    
    # ================== PHASE 2: CALIBRATION ==================
    
    def _run_phase2(self, frame: np.ndarray, result, timestamp_ms: int) -> np.ndarray:
        """Phase 2: Safe-Max Calibration."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel chính
        output = draw_panel(output, (10, 10), (450, 320), "")
        
        # Tiêu đề
        output = put_vietnamese_text(
            output, "GIAI DOAN 2: DO GIOI HAN VAN DONG",
            (25, 40), COLORS['info'], 18
        )
        
        # Menu chọn khớp
        y_pos = 75
        output = put_vietnamese_text(output, "Chon khop can do (nhan phim so):", (25, y_pos), (180, 180, 180), 13)
        y_pos += 28
        
        for key, joint_type in JOINT_KEY_MAPPING.items():
            key_char = chr(key)
            joint_name = JOINT_NAMES.get(joint_type, joint_type.value)
            
            if self._state.selected_joint == joint_type:
                color = COLORS['success']
                prefix = ">>>"
            else:
                color = COLORS['text']
                prefix = "   "
            
            output = put_vietnamese_text(
                output, f"{prefix} [{key_char}] {joint_name}",
                (30, y_pos), color, 13
            )
            y_pos += 22
        
        y_pos += 10
        
        # Trạng thái calibration
        if self._calibrator.state == CalibrationState.COLLECTING:
            progress = self._calibrator.progress
            output = put_vietnamese_text(
                output, f"Dang thu thap du lieu... {int(progress * 100)}%",
                (25, y_pos), COLORS['warning'], 14
            )
            y_pos += 25
            output = draw_progress_bar(output, (25, y_pos), (400, 15), progress, COLORS['warning'])
            y_pos += 25
            
            # Thêm frame vào calibrator
            if result.has_pose() and self._state.selected_joint:
                try:
                    landmarks = result.pose_landmarks.to_numpy()
                    angle = calculate_joint_angle(landmarks, self._state.selected_joint, use_3d=True)
                    self._state.user_angle = angle
                    self._calibrator.add_frame(result.pose_landmarks, timestamp_ms)
                    
                    output = put_vietnamese_text(
                        output, f"Goc hien tai: {angle:.1f} do",
                        (25, y_pos), COLORS['info'], 16
                    )
                except ValueError:
                    pass
                
                # Kiểm tra hoàn thành tự động
                if self._calibrator.state == CalibrationState.COMPLETED:
                    self._finish_calibration()
        
        elif self._calibrator.state == CalibrationState.COMPLETED:
            output = put_vietnamese_text(
                output, f"HOAN THANH! Goc toi da: {self._state.user_max_angle:.1f} do",
                (25, y_pos), COLORS['success'], 16
            )
            y_pos += 25
            output = put_vietnamese_text(
                output, "Nhan ENTER de tiep tuc sang Phase 3",
                (25, y_pos), COLORS['info'], 14
            )
            self._state.calibration_complete = True
        
        elif self._state.selected_joint:
            joint_name = JOINT_NAMES.get(self._state.selected_joint, "")
            output = put_vietnamese_text(
                output, f"Da chon: {joint_name}",
                (25, y_pos), COLORS['success'], 14
            )
            y_pos += 25
            output = put_vietnamese_text(
                output, "Nhan SPACE de bat dau do",
                (25, y_pos), COLORS['info'], 14
            )
            y_pos += 22
            output = put_vietnamese_text(
                output, "Thuc hien dong tac HET KHA NANG (khong dau)",
                (25, y_pos), COLORS['warning'], 13
            )
        else:
            output = put_vietnamese_text(
                output, "Hay chon mot khop de bat dau",
                (25, y_pos), COLORS['text'], 14
            )
        
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
        phase_text = "Phase: 2/4 - Calibration"
        output = put_vietnamese_text(output, phase_text, (w - 250, 30), COLORS['info'], 14)
        
        # Controls
        controls = "[1-6] Chon | [SPACE] Bat dau | [ENTER] Tiep tuc"
        output = put_vietnamese_text(output, controls, (20, h - 25), (150, 150, 150), 12)
        
        return output
    
    def _start_calibration(self) -> None:
        """Bắt đầu calibration."""
        if self._state.selected_joint:
            if self._user_profile is None:
                self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
            
            self._calibrator = SafeMaxCalibrator(duration_ms=5000)
            self._calibrator.start_calibration(self._state.selected_joint, self._user_profile)
            print(f"[CALIBRATION] Bat dau do: {JOINT_NAMES.get(self._state.selected_joint)}")
    
    def _finish_calibration(self) -> None:
        """Lấy kết quả calibration."""
        if self._user_profile and self._state.selected_joint:
            max_angle = self._user_profile.get_max_angle(self._state.selected_joint)
            if max_angle:
                self._state.user_max_angle = max_angle
                self._state.calibration_complete = True
                print(f"[CALIBRATION] Goc toi da: {max_angle:.1f}")
    
    # ================== PHASE 3: MOTION SYNC ==================
    
    def _run_phase3(
        self,
        user_frame: np.ndarray,
        ref_frame: Optional[np.ndarray],
        result,
        timestamp: float
    ) -> np.ndarray:
        """Phase 3: Đồng bộ chuyển động với video mẫu."""
        h, w = user_frame.shape[:2]
        
        # === USER VIEW ===
        user_display = user_frame.copy()
        
        # Vẽ skeleton user
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            highlight = []
            if self._state.selected_joint:
                joint_def = JOINT_DEFINITIONS.get(self._state.selected_joint)
                if joint_def:
                    highlight = [joint_def.proximal, joint_def.vertex, joint_def.distal]
            
            user_display = draw_skeleton(
                user_display, self._current_landmarks,
                color=COLORS['skeleton'],
                highlight_indices=highlight,
                use_core_only=True
            )
            
            # Tính góc
            if self._state.selected_joint:
                try:
                    self._state.user_angle = calculate_joint_angle(
                        self._current_landmarks, self._state.selected_joint, use_3d=True
                    )
                except ValueError:
                    pass
            
            # Vẽ góc
            if self._state.user_angle > 0:
                p1, pv, p2 = self._get_joint_pixel_coords(
                    self._current_landmarks, self._state.selected_joint, (h, w)
                )
                if p1 and pv and p2:
                    user_display = draw_angle_arc(
                        user_display, p1, pv, p2, self._state.user_angle,
                        color=COLORS['info']
                    )
        
        # Panel thông tin user
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
        
        # Điểm hiện tại
        score_color = COLORS['success'] if self._state.current_score >= 70 else COLORS['warning']
        user_display = put_vietnamese_text(
            user_display, f"Diem: {self._state.current_score:.0f}",
            (25, 126), score_color, 14
        )
        
        # Hiển thị trạng thái đạt mục tiêu - banner lớn nếu đạt
        if error < 15 and self._state.target_angle > 0:
            # Draw big success banner at bottom of user view
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
        
        # === DASHBOARD ===
        dashboard = self._create_phase3_dashboard(h)
        
        # Combine views
        combined = combine_frames_horizontal([user_display, ref_display, dashboard], h)
        
        # Update sync controller
        if self._sync_controller and self._video_engine:
            self._state.sync_state = self._sync_controller.update(
                self._state.user_angle,
                self._video_engine.current_frame,
                timestamp
            )
            # Sử dụng interpolation để có target angle liên tục
            interpolated = self._interpolate_target_angle(
                self._video_engine.current_frame,
                self._video_engine.total_frames
            )
            if interpolated > 0:
                self._state.target_angle = interpolated
            else:
                self._state.target_angle = self._state.sync_state.target_angle
            
            self._state.motion_phase = self._state.sync_state.current_phase.value
            self._state.rep_count = self._state.sync_state.rep_count
            
            # Tính điểm realtime
            realtime_score = self._calculate_realtime_score(
                self._state.user_angle, 
                self._state.target_angle
            )
            # Smooth score để tránh nhảy quá nhanh
            self._state.current_score = 0.7 * self._state.current_score + 0.3 * realtime_score
            
            # Track score history để tính average
            if self._state.target_angle > 0:
                self._score_history.append(realtime_score)
                if len(self._score_history) > 0:
                    self._state.average_score = sum(self._score_history) / len(self._score_history)
        
        # Track angles
        self._user_angles.append(self._state.user_angle)
        if self._state.target_angle > 0:
            self._ref_angles.append(self._state.target_angle)
        
        return combined
    
    def _create_phase3_dashboard(self, height: int) -> np.ndarray:
        """Tạo dashboard cho Phase 3."""
        width = 280
        dashboard = np.zeros((height, width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)
        
        y = 25
        
        # Tiêu đề
        dashboard = put_vietnamese_text(dashboard, "GIAI DOAN 3: DONG BO", (15, y), COLORS['info'], 15)
        y += 35
        
        # Phase hiện tại
        dashboard = put_vietnamese_text(dashboard, "Phase dong tac:", (15, y), (150, 150, 150), 12)
        y += 22
        
        phase = self._state.motion_phase.lower()
        phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
        phase_name = PHASE_NAMES_VI.get(phase, phase.upper())
        
        cv2.circle(dashboard, (30, y + 5), 10, phase_color, -1)
        dashboard = put_vietnamese_text(dashboard, phase_name.upper(), (50, y + 10), phase_color, 14)
        y += 35
        
        # 4 phase indicators
        phases = ["idle", "eccentric", "hold", "concentric"]
        phase_x = 15
        for p in phases:
            p_color = PHASE_COLORS.get(p, (80, 80, 80))
            if p == phase:
                cv2.circle(dashboard, (phase_x + 10, y), 8, p_color, -1)
            else:
                cv2.circle(dashboard, (phase_x + 10, y), 8, p_color, 1)
            phase_x += 65
        y += 30
        
        # Rep count
        dashboard = put_vietnamese_text(dashboard, f"So hiep: {self._state.rep_count}", (15, y), COLORS['text'], 14)
        y += 30
        
        # Score với visual indicator
        score_color = COLORS['success'] if self._state.current_score >= 70 else COLORS['warning'] if self._state.current_score >= 50 else COLORS['error']
        dashboard = put_vietnamese_text(
            dashboard, f"Diem: {self._state.current_score:.0f}/100",
            (15, y), score_color, 16
        )
        # Score bar
        score_bar_width = 180
        score_progress = self._state.current_score / 100.0
        cv2.rectangle(dashboard, (130, y - 10), (130 + score_bar_width, y + 5), (60, 60, 60), -1)
        cv2.rectangle(dashboard, (130, y - 10), (130 + int(score_bar_width * score_progress), y + 5), score_color, -1)
        y += 25
        
        dashboard = put_vietnamese_text(
            dashboard, f"TB: {self._state.average_score:.0f}",
            (15, y), (150, 150, 150), 12
        )
        y += 35
        
        # Angle comparison với visual feedback
        dashboard = put_vietnamese_text(dashboard, "So sanh goc:", (15, y), (150, 150, 150), 12)
        y += 22
        dashboard = put_vietnamese_text(
            dashboard, f"  Ban: {self._state.user_angle:.1f} do",
            (15, y), COLORS['text'], 13
        )
        y += 20
        dashboard = put_vietnamese_text(
            dashboard, f"  Muc tieu: {self._state.target_angle:.1f} do",
            (15, y), COLORS['info'], 13
        )
        y += 22
        
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
            dashboard, f"  Sai so: {error:.1f} do - {feedback}",
            (15, y), error_color, 13
        )
        y += 30
        
        # Progress to target (visual bar)
        if self._state.target_angle > 0:
            progress = min(1.0, self._state.user_angle / self._state.target_angle)
            progress_color = COLORS['success'] if progress >= 0.9 else COLORS['warning'] if progress >= 0.7 else COLORS['error']
            dashboard = draw_progress_bar(
                dashboard, (15, y), (250, 15), progress, progress_color
            )
            y += 25
            
            # Direction hint
            if self._state.user_angle < self._state.target_angle - 10:
                dashboard = put_vietnamese_text(
                    dashboard, "^ Nang cao hon!",
                    (15, y), COLORS['info'], 12
                )
            elif self._state.user_angle > self._state.target_angle + 10:
                dashboard = put_vietnamese_text(
                    dashboard, "v Ha thap hon!",
                    (15, y), COLORS['warning'], 12
                )
            else:
                dashboard = put_vietnamese_text(
                    dashboard, "= Giu nguyen!",
                    (15, y), COLORS['success'], 12
                )
            y += 30
        
        # Fatigue
        dashboard = put_vietnamese_text(
            dashboard, f"Met moi: {self._state.fatigue_level}",
            (15, y), COLORS['text'], 12
        )
        y += 25
        
        # Pain
        pain_color = COLORS['success'] if self._state.pain_level == "NONE" else COLORS['error']
        dashboard = put_vietnamese_text(
            dashboard, f"Dau: {self._state.pain_level}",
            (15, y), pain_color, 12
        )
        y += 30
        
        # Warning
        if self._state.warning:
            cv2.rectangle(dashboard, (10, y), (width - 10, y + 45), (0, 0, 100), -1)
            dashboard = put_vietnamese_text(dashboard, "CANH BAO!", (20, y + 18), COLORS['error'], 12)
            dashboard = put_vietnamese_text(
                dashboard, self._state.warning[:25],
                (20, y + 36), COLORS['text'], 10
            )
        
        # Controls
        dashboard = put_vietnamese_text(dashboard, "[SPACE] Dung/Tiep", (15, height - 50), (120, 120, 120), 11)
        dashboard = put_vietnamese_text(dashboard, "[Q] Ket thuc", (15, height - 30), (120, 120, 120), 11)
        
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
        print("Cac giai doan:")
        print("  1. Nhan dien tu the (Pose Detection)")
        print("  2. Do gioi han van dong (Calibration)")
        print("  3. Dong bo video mau (Motion Sync)")
        print("  4. Cham diem va phan tich (Scoring)")
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
        
        elif key == 13:  # ENTER
            self._advance_phase()
        
        elif key == ord(' '):
            if self._state.current_phase == AppPhase.PHASE2_CALIBRATION:
                if self._calibrator.state != CalibrationState.COLLECTING:
                    self._start_calibration()
            elif self._state.current_phase == AppPhase.PHASE3_SYNC:
                self._state.is_paused = not self._state.is_paused
                if self._video_engine:
                    if self._state.is_paused:
                        self._video_engine.pause()
                    else:
                        self._video_engine.play()
        
        elif key == ord('r'):
            self._restart()
        
        elif key in JOINT_KEY_MAPPING:
            if self._state.current_phase == AppPhase.PHASE2_CALIBRATION:
                self._state.selected_joint = JOINT_KEY_MAPPING[key]
                self._calibrator = SafeMaxCalibrator(duration_ms=5000)
    
    def _advance_phase(self) -> None:
        """Chuyển phase tiếp theo."""
        if self._state.current_phase == AppPhase.PHASE1_DETECTION:
            if self._state.pose_detected:
                self._transition_to_phase2()
        
        elif self._state.current_phase == AppPhase.PHASE2_CALIBRATION:
            if self._state.calibration_complete:
                self._transition_to_phase3()
    
    def _transition_to_phase2(self) -> None:
        """Chuyển sang Phase 2."""
        print("\n[PHASE 2] Bat dau Calibration...")
        self._state.current_phase = AppPhase.PHASE2_CALIBRATION
        self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
    
    def _transition_to_phase3(self) -> None:
        """Chuyển sang Phase 3."""
        if not self._ref_video_path or not Path(self._ref_video_path).exists():
            print("[WARNING] Khong co video mau, chuyen sang Phase 4")
            self._transition_to_phase4()
            return
        
        print("\n[PHASE 3] Bat dau Motion Sync...")
        self._state.current_phase = AppPhase.PHASE3_SYNC
        
        # Setup video engine
        self._video_engine = VideoEngine(self._ref_video_path)
        total_frames = self._video_engine.total_frames
        fps = self._video_engine.fps
        
        # Create exercise
        max_angle = self._state.user_max_angle if self._state.user_max_angle > 0 else 150
        if self._state.selected_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise = create_elbow_flex_exercise(total_frames, fps, max_angle=max_angle)
        else:
            exercise = create_arm_raise_exercise(total_frames, fps, max_angle=max_angle)
        
        # Setup sync controller
        self._sync_controller = MotionSyncController(
            exercise,
            user_max_angle=self._state.user_max_angle if self._state.user_max_angle > 0 else None
        )
        
        # Setup checkpoints
        checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
        self._video_engine.set_checkpoints(checkpoint_frames)
        self._video_engine.set_speed(0.7)
        
        # Start session
        session_id = f"session_{int(time.time())}"
        self._logger.start_session(session_id, exercise.name)
        self._scorer.start_session(exercise.name, session_id)
        
        self._video_engine.play()
        
        print(f"[SETUP] Exercise: {exercise.name}")
        print(f"[SETUP] User max: {self._state.user_max_angle:.1f}")
    
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
        self._user_angles = []
        self._ref_angles = []
        
        if self._video_engine:
            self._video_engine.stop()
            self._video_engine = None
        
        self._sync_controller = None
        self._calibrator = SafeMaxCalibrator(duration_ms=5000)
    
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
