"""
MEMOTION Engine Service - Backend Integration Layer

Refactored logic từ main_v2.py, loại bỏ hoàn toàn UI.
Class này quản lý toàn bộ luồng xử lý và trạng thái nội bộ.

Usage:
    engine = EngineService(config)
    while True:
        frame = get_frame_from_camera()
        result = engine.process_frame(frame, timestamp_ms)
        send_to_frontend(result.to_dict())

Author: MEMOTION Team
Version: 1.0.0
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
import numpy as np

# Core imports
from core import (
    VisionDetector, DetectorConfig, JointType, JOINT_DEFINITIONS,
    calculate_joint_angle, MotionPhase, SyncStatus, SyncState,
    MotionSyncController, create_arm_raise_exercise, create_elbow_flex_exercise,
    compute_single_joint_dtw, create_exercise_weights,
)
from modules import (
    VideoEngine, PlaybackState, PainDetector, PainLevel,
    HealthScorer, FatigueLevel, SafeMaxCalibrator, CalibrationState,
    UserProfile,
)
from utils import SessionLogger

# Schema imports
from .schemas import (
    DetectionOutput,
    CalibrationOutput,
    JointCalibrationStatus,
    SyncOutput,
    JointError,
    FinalReportOutput,
    RepScore,
    JointCalibrationResult,
    EngineOutput,
    get_direction_hint,
    get_feedback_text,
    get_grade,
    get_joint_name_vi,
    JOINT_NAMES_VI,
)


# ==================== CONSTANTS ====================

class AppPhase(Enum):
    """Các giai đoạn của ứng dụng."""
    PHASE1_DETECTION = "phase1"
    PHASE2_CALIBRATION = "phase2"
    PHASE3_SYNC = "phase3"
    PHASE4_SCORING = "phase4"
    COMPLETED = "completed"


# Calibration Queue - thứ tự tự động đo 6 khớp
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
    JointType.RIGHT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_ELBOW: "Moi ba dung NGANG",
    JointType.RIGHT_ELBOW: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    JointType.RIGHT_KNEE: "Moi ba dung DOC",
}

# Timing constants
PHASE1_COUNTDOWN_DURATION = 3.0  # giây
CALIBRATION_COUNTDOWN_DURATION = 5.0  # giây
PHASE2_COMPLETE_DELAY = 2.0  # giây


# ==================== APP STATE ====================

@dataclass
class EngineState:
    """Trạng thái nội bộ của Engine."""
    current_phase: AppPhase = AppPhase.PHASE1_DETECTION
    is_running: bool = True
    is_paused: bool = False
    
    # Phase 1 state
    pose_detected: bool = False
    detection_stable_count: int = 0
    phase1_countdown_start: float = 0.0
    phase1_countdown_active: bool = False
    
    # Phase 2 state
    selected_joint: Optional[JointType] = None
    calibration_complete: bool = False
    user_max_angle: float = 0.0
    calibration_queue_index: int = 0
    calibration_countdown_start: float = 0.0
    is_countdown_active: bool = False
    is_calibrating_joint: bool = False
    calibrated_joints: Dict = field(default_factory=dict)
    all_joints_calibrated: bool = False
    phase2_complete_time: float = 0.0
    
    # Phase 3 state
    sync_state: Optional[SyncState] = None
    motion_phase: str = "idle"
    last_motion_phase: Optional[MotionPhase] = None
    user_angles_dict: Dict = field(default_factory=dict)
    target_angles_dict: Dict = field(default_factory=dict)
    joint_scores_dict: Dict = field(default_factory=dict)
    joint_weights: Dict = field(default_factory=dict)
    active_joints: List = field(default_factory=list)
    
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
    
    # Session info
    session_id: str = ""
    session_start_time: float = 0.0
    exercise_name: str = ""


# ==================== ENGINE CONFIG ====================

@dataclass
class EngineConfig:
    """Cấu hình cho Engine."""
    models_dir: str = "./models"
    log_dir: str = "./data/logs"
    ref_video_path: Optional[str] = None
    default_joint: str = "left_shoulder"  # String để dễ serialize
    detection_stable_threshold: int = 30
    calibration_duration_ms: int = 5000


# ==================== ENGINE SERVICE ====================

class EngineService:
    """
    MEMOTION Engine Service - Xử lý frame và quản lý trạng thái.
    
    Backend chỉ cần:
    1. Khởi tạo class một lần
    2. Gọi process_frame() trong mỗi vòng lặp
    3. Nhận kết quả JSON-serializable
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Khởi tạo Engine.
        
        Args:
            config: Cấu hình engine (None = sử dụng default)
        """
        self._config = config or EngineConfig()
        
        # Map string to JointType
        joint_map = {
            "left_shoulder": JointType.LEFT_SHOULDER,
            "right_shoulder": JointType.RIGHT_SHOULDER,
            "left_elbow": JointType.LEFT_ELBOW,
            "right_elbow": JointType.RIGHT_ELBOW,
            "left_knee": JointType.LEFT_KNEE,
            "right_knee": JointType.RIGHT_KNEE,
        }
        self._default_joint = joint_map.get(
            self._config.default_joint, 
            JointType.LEFT_SHOULDER
        )
        
        # State
        self._state = EngineState()
        self._state.selected_joint = self._default_joint
        
        # Components (lazy init)
        self._detector: Optional[VisionDetector] = None
        self._ref_detector: Optional[VisionDetector] = None
        self._video_engine: Optional[VideoEngine] = None
        self._sync_controller: Optional[MotionSyncController] = None
        self._calibrator: Optional[SafeMaxCalibrator] = None
        self._pain_detector: Optional[PainDetector] = None
        self._scorer: Optional[HealthScorer] = None
        self._logger: Optional[SessionLogger] = None
        self._user_profile: Optional[UserProfile] = None
        
        # Data tracking
        self._user_angles: List[float] = []
        self._ref_angles: List[float] = []
        self._score_history: List[float] = []
        self._rep_scores: List[Dict] = []
        self._current_landmarks: Optional[np.ndarray] = None
        
        # Analysis queue
        self._analysis_queue = Queue(maxsize=5)
        
        # Initialize
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Khởi tạo các components.
        
        Returns:
            bool: True nếu thành công
        """
        try:
            models_dir = Path(self._config.models_dir)
            pose_model = models_dir / "pose_landmarker_lite.task"
            face_model = models_dir / "face_landmarker.task"
            
            if not pose_model.exists():
                raise FileNotFoundError(f"Pose model not found: {pose_model}")
            
            # Init detector
            config = DetectorConfig(
                pose_model_path=str(pose_model),
                face_model_path=str(face_model) if face_model.exists() else None,
                running_mode="VIDEO"
            )
            self._detector = VisionDetector(config)
            
            # Init reference detector
            ref_config = DetectorConfig(
                pose_model_path=str(pose_model),
                running_mode="VIDEO"
            )
            self._ref_detector = VisionDetector(ref_config)
            
            # Init other components
            self._calibrator = SafeMaxCalibrator(
                duration_ms=self._config.calibration_duration_ms
            )
            self._pain_detector = PainDetector()
            self._scorer = HealthScorer()
            self._logger = SessionLogger(self._config.log_dir)
            
            self._initialized = True
            self._state.session_start_time = time.time()
            
            return True
            
        except Exception as e:
            self._state.message = f"Initialization error: {str(e)}"
            return False
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp_ms: int
    ) -> EngineOutput:
        """
        Xử lý một frame và trả về kết quả.
        
        Args:
            frame: Frame ảnh (BGR numpy array)
            timestamp_ms: Timestamp tính bằng milliseconds
        
        Returns:
            EngineOutput: Kết quả xử lý theo phase hiện tại
        """
        # Lazy init
        if not self._initialized:
            if not self.initialize():
                return EngineOutput(
                    current_phase=1,
                    phase_name="detection",
                    error="Engine not initialized"
                )
        
        timestamp = timestamp_ms / 1000.0
        
        # Process detection
        result = self._detector.process_frame(frame, timestamp_ms)
        
        # Route to current phase
        if self._state.current_phase == AppPhase.PHASE1_DETECTION:
            output = self._process_phase1(result, timestamp)
            return EngineOutput(
                current_phase=1,
                phase_name="detection",
                detection=output.to_dict()
            )
        
        elif self._state.current_phase == AppPhase.PHASE2_CALIBRATION:
            output = self._process_phase2(result, timestamp_ms, timestamp)
            return EngineOutput(
                current_phase=2,
                phase_name="calibration",
                calibration=output.to_dict()
            )
        
        elif self._state.current_phase == AppPhase.PHASE3_SYNC:
            output = self._process_phase3(result, timestamp)
            return EngineOutput(
                current_phase=3,
                phase_name="sync",
                sync=output.to_dict()
            )
        
        elif self._state.current_phase == AppPhase.PHASE4_SCORING:
            output = self._process_phase4()
            return EngineOutput(
                current_phase=4,
                phase_name="scoring",
                final_report=output.to_dict()
            )
        
        else:
            # Completed
            return EngineOutput(
                current_phase=4,
                phase_name="completed",
                final_report=self._generate_final_report().to_dict()
            )
    
    # ==================== PHASE 1: DETECTION ====================
    
    def _process_phase1(self, result, timestamp: float) -> DetectionOutput:
        """Xử lý Phase 1: Pose Detection với auto transition."""
        
        output = DetectionOutput()
        
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Đếm stable frames
            self._state.detection_stable_count += 1
            progress = min(1.0, self._state.detection_stable_count / 
                          self._config.detection_stable_threshold)
            
            output.stable_count = self._state.detection_stable_count
            output.progress = progress
            
            if self._state.detection_stable_count >= self._config.detection_stable_threshold:
                self._state.pose_detected = True
                output.pose_detected = True
                
                # === AUTO TRANSITION: Countdown 3 giây ===
                if not self._state.phase1_countdown_active:
                    self._state.phase1_countdown_active = True
                    self._state.phase1_countdown_start = timestamp
                
                elapsed = timestamp - self._state.phase1_countdown_start
                remaining = PHASE1_COUNTDOWN_DURATION - elapsed
                
                if remaining > 0:
                    output.countdown_remaining = remaining
                    output.status = "countdown"
                    output.message = f"Chuan bi... {int(remaining) + 1} giay"
                else:
                    # Countdown kết thúc - chuyển Phase 2
                    output.status = "transitioning"
                    output.message = "Chuyen sang Phase 2..."
                    self._transition_to_phase2()
            else:
                output.status = "detecting"
                output.message = f"Dang xac nhan... {int(progress * 100)}%"
        else:
            # Reset nếu mất pose
            self._state.detection_stable_count = 0
            self._state.phase1_countdown_active = False
            output.status = "idle"
            output.message = "Chua phat hien nguoi. Hay dung vao khung hinh."
        
        return output
    
    # ==================== PHASE 2: CALIBRATION ====================
    
    def _process_phase2(
        self, 
        result, 
        timestamp_ms: int, 
        timestamp: float
    ) -> CalibrationOutput:
        """Xử lý Phase 2: Automated Calibration."""
        
        output = CalibrationOutput()
        
        # Lấy khớp hiện tại
        if self._state.calibration_queue_index < len(CALIBRATION_QUEUE):
            current_joint = CALIBRATION_QUEUE[self._state.calibration_queue_index]
            self._state.selected_joint = current_joint
        else:
            self._state.all_joints_calibrated = True
            current_joint = None
        
        # Build joints status
        joints_status = []
        for i, jt in enumerate(CALIBRATION_QUEUE):
            status = JointCalibrationStatus(
                joint_name=get_joint_name_vi(jt.value),
                joint_type=jt.value,
                max_angle=self._state.calibrated_joints.get(jt),
                status="complete" if jt in self._state.calibrated_joints 
                       else ("collecting" if i == self._state.calibration_queue_index 
                             else "pending")
            )
            joints_status.append(status.to_dict())
        
        output.joints_status = joints_status
        output.queue_index = self._state.calibration_queue_index
        output.total_joints = len(CALIBRATION_QUEUE)
        output.overall_progress = len(self._state.calibrated_joints) / len(CALIBRATION_QUEUE)
        
        # === LOGIC TỰ ĐỘNG ===
        if self._state.all_joints_calibrated:
            output.status = "all_complete"
            output.message = "Da do xong tat ca 6 khop!"
            
            # Tự động chuyển Phase 3 sau delay
            if self._state.phase2_complete_time == 0:
                self._state.phase2_complete_time = timestamp
                self._save_calibration_to_profile()
            elif timestamp - self._state.phase2_complete_time > PHASE2_COMPLETE_DELAY:
                self._state.calibration_complete = True
                self._transition_to_phase3()
        
        elif not self._state.is_countdown_active and not self._state.is_calibrating_joint:
            # Bắt đầu countdown cho khớp mới
            self._state.is_countdown_active = True
            self._state.calibration_countdown_start = timestamp
            output.status = "preparing"
            
        elif self._state.is_countdown_active:
            # Đang countdown
            elapsed = timestamp - self._state.calibration_countdown_start
            remaining = CALIBRATION_COUNTDOWN_DURATION - elapsed
            
            if remaining > 0:
                output.current_joint = current_joint.value if current_joint else None
                output.current_joint_name = get_joint_name_vi(current_joint.value) if current_joint else None
                output.countdown_remaining = remaining
                output.position_instruction = JOINT_POSITION_INSTRUCTIONS.get(current_joint, "")
                output.status = "preparing"
                output.message = f"Bat dau sau: {int(remaining) + 1} giay"
            else:
                # Countdown kết thúc - bắt đầu đo
                self._state.is_countdown_active = False
                self._state.is_calibrating_joint = True
                self._start_calibration_for_joint(current_joint)
                output.status = "collecting"
        
        elif self._state.is_calibrating_joint and current_joint:
            output.current_joint = current_joint.value
            output.current_joint_name = get_joint_name_vi(current_joint.value)
            output.status = "collecting"
            
            if self._calibrator.state == CalibrationState.COLLECTING:
                output.progress = self._calibrator.progress
                output.message = f"Dang do {get_joint_name_vi(current_joint.value)}... {int(output.progress * 100)}%"
                
                # Thêm frame vào calibrator
                if result.has_pose():
                    try:
                        landmarks = result.pose_landmarks.to_numpy()
                        angle = calculate_joint_angle(landmarks, current_joint, use_3d=True)
                        self._state.user_angle = angle
                        output.current_angle = angle
                        self._calibrator.add_frame(result.pose_landmarks, timestamp_ms)
                        
                        # Kiểm tra hoàn thành
                        if self._calibrator.state == CalibrationState.COMPLETED:
                            self._finish_calibration_for_joint(current_joint)
                    except ValueError:
                        pass
            
            elif self._calibrator.state == CalibrationState.COMPLETED:
                output.status = "complete"
                output.user_max_angle = self._state.calibrated_joints.get(current_joint, 0)
        
        return output
    
    def _start_calibration_for_joint(self, joint_type: JointType) -> None:
        """Bắt đầu đo một khớp."""
        if self._user_profile is None:
            self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        self._calibrator = SafeMaxCalibrator(
            duration_ms=self._config.calibration_duration_ms
        )
        self._calibrator.start_calibration(joint_type, self._user_profile)
    
    def _finish_calibration_for_joint(self, joint_type: JointType) -> None:
        """Hoàn thành đo một khớp."""
        if self._user_profile and joint_type:
            max_angle = self._user_profile.get_max_angle(joint_type)
            if max_angle:
                self._state.calibrated_joints[joint_type] = max_angle
                
                if joint_type == self._default_joint:
                    self._state.user_max_angle = max_angle
        
        # Chuyển khớp tiếp
        self._state.calibration_queue_index += 1
        self._state.is_calibrating_joint = False
        self._state.user_angle = 0.0
        
        if self._state.calibration_queue_index >= len(CALIBRATION_QUEUE):
            self._state.all_joints_calibrated = True
    
    def _save_calibration_to_profile(self) -> None:
        """Lưu calibration profile."""
        if self._user_profile:
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
            except Exception:
                pass
    
    # ==================== PHASE 3: SYNC ====================
    
    def _process_phase3(self, result, timestamp: float) -> SyncOutput:
        """Xử lý Phase 3: Motion Sync với multi-joint tracking."""
        
        output = SyncOutput()
        
        # Xử lý video engine
        ref_frame = None
        if self._video_engine:
            if not self._state.is_paused:
                if self._state.sync_state:
                    if self._state.sync_state.sync_status == SyncStatus.PAUSE:
                        self._video_engine.pause()
                    elif self._state.sync_state.sync_status in (SyncStatus.PLAY, SyncStatus.SKIP):
                        if self._video_engine.state != PlaybackState.PLAYING:
                            self._video_engine.play()
                
                ref_frame, ref_status = self._video_engine.get_frame()
                
                # Video progress
                output.video_progress = (self._video_engine.current_frame / 
                                        max(1, self._video_engine.total_frames))
                
                # Check completion
                if self._state.sync_state:
                    if self._state.sync_state.sync_status == SyncStatus.COMPLETE:
                        self._transition_to_phase4()
                        output.status = "complete"
                        return output
                
                if ref_status.state == PlaybackState.FINISHED:
                    self._transition_to_phase4()
                    output.status = "complete"
                    return output
                
                # Check rep completion
                if self._state.sync_state:
                    current_mp = self._state.sync_state.current_phase
                    if (self._state.last_motion_phase == MotionPhase.CONCENTRIC and 
                        current_mp == MotionPhase.IDLE):
                        self._on_rep_complete()
                    self._state.last_motion_phase = current_mp
        
        output.video_paused = self._state.is_paused
        
        # Tính góc multi-joint
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Tính góc tất cả khớp
            self._state.user_angles_dict = self._calculate_all_joint_angles(
                self._current_landmarks
            )
            
            # Primary joint
            primary_joint = self._state.selected_joint or self._default_joint
            if primary_joint in self._state.user_angles_dict:
                self._state.user_angle = self._state.user_angles_dict[primary_joint]
            
            # Update scorer
            if self._state.sync_state and self._scorer:
                self._scorer.add_frame(
                    self._state.user_angle,
                    timestamp,
                    self._state.sync_state.current_phase,
                    pose_landmarks=self._current_landmarks
                )
            
            # Pain detection
            if result.has_face() and not self._analysis_queue.full():
                self._analysis_queue.put(result.face_landmarks.to_numpy())
                self._process_pain()
        
        # Tính target multi-joint
        if self._video_engine:
            self._state.target_angles_dict = self._interpolate_all_joint_targets(
                self._video_engine.current_frame,
                self._video_engine.total_frames
            )
            
            primary_joint = self._state.selected_joint or self._default_joint
            if primary_joint in self._state.target_angles_dict:
                self._state.target_angle = self._state.target_angles_dict[primary_joint]
        
        # Update sync controller
        if self._sync_controller and self._video_engine:
            self._state.sync_state = self._sync_controller.update(
                self._state.user_angle,
                self._video_engine.current_frame,
                timestamp
            )
            self._state.motion_phase = self._state.sync_state.current_phase.value
            self._state.rep_count = self._state.sync_state.rep_count
        
        # Tính điểm multi-joint
        multi_joint_score = self._calculate_multi_joint_score()
        self._state.current_score = 0.7 * self._state.current_score + 0.3 * multi_joint_score
        
        if len(self._state.target_angles_dict) > 0:
            self._score_history.append(multi_joint_score)
            if len(self._score_history) > 0:
                self._state.average_score = sum(self._score_history) / len(self._score_history)
        
        # Update scorer status
        if self._scorer:
            scorer_status = self._scorer.get_current_status()
            self._state.fatigue_level = scorer_status.get("fatigue_level", "FRESH")
        
        # Build joint errors list
        joint_errors = []
        for jt in self._state.active_joints:
            user_ang = self._state.user_angles_dict.get(jt, 0)
            target_ang = self._state.target_angles_dict.get(jt, 0)
            error = abs(user_ang - target_ang)
            error_percent = (error / target_ang * 100) if target_ang > 0 else 0
            score = self._state.joint_scores_dict.get(jt, 0)
            weight = self._state.joint_weights.get(jt, 0.5)
            
            joint_error = JointError(
                joint_name=get_joint_name_vi(jt.value),
                joint_type=jt.value,
                user_angle=user_ang,
                target_angle=target_ang,
                error=error,
                error_percent=error_percent,
                score=score,
                direction_hint=get_direction_hint(user_ang, target_ang),
                weight=weight
            )
            joint_errors.append(joint_error.to_dict())
        
        # Build output
        output.user_angle = self._state.user_angle
        output.target_angle = self._state.target_angle
        output.error = abs(self._state.user_angle - self._state.target_angle)
        output.current_score = self._state.current_score
        output.average_score = self._state.average_score
        output.motion_phase = self._state.motion_phase
        output.rep_count = self._state.rep_count
        output.pain_level = self._state.pain_level
        output.fatigue_level = self._state.fatigue_level
        output.joint_errors = joint_errors
        output.active_joints_count = len(self._state.active_joints)
        output.feedback_text = get_feedback_text(output.error, output.target_angle)
        output.direction_hint = get_direction_hint(output.user_angle, output.target_angle)
        output.warning = self._state.warning if self._state.warning else None
        output.status = "paused" if self._state.is_paused else "syncing"
        
        # Track angles
        self._user_angles.append(self._state.user_angle)
        if self._state.target_angle > 0:
            self._ref_angles.append(self._state.target_angle)
        
        return output
    
    def _calculate_all_joint_angles(self, landmarks: np.ndarray) -> Dict[JointType, float]:
        """Tính góc tất cả khớp."""
        angles = {}
        for joint_type in self._state.active_joints:
            try:
                angle = calculate_joint_angle(landmarks, joint_type, use_3d=True)
                angles[joint_type] = angle
            except (ValueError, IndexError):
                if joint_type in self._state.user_angles_dict:
                    angles[joint_type] = self._state.user_angles_dict[joint_type]
        return angles
    
    def _interpolate_target_angle(
        self, 
        current_frame: int, 
        total_frames: int, 
        joint_type: Optional[JointType] = None
    ) -> float:
        """Tính target angle cho một khớp."""
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
        
        if joint_type and joint_type in self._state.calibrated_joints:
            user_max = self._state.calibrated_joints[joint_type]
        else:
            user_max = self._state.user_max_angle or 150.0
        
        prev_cp = checkpoints[0]
        next_cp = checkpoints[-1]
        
        for cp in checkpoints:
            if cp.frame_index <= current_frame:
                prev_cp = cp
            if cp.frame_index > current_frame:
                next_cp = cp
                break
        
        if prev_cp.frame_index == next_cp.frame_index:
            base_target = prev_cp.target_angle
        else:
            progress = (current_frame - prev_cp.frame_index) / max(1, next_cp.frame_index - prev_cp.frame_index)
            progress = max(0, min(1, progress))
            base_target = prev_cp.target_angle + progress * (next_cp.target_angle - prev_cp.target_angle)
        
        exercise_max = max(cp.target_angle for cp in checkpoints)
        if exercise_max > 0 and user_max > 0:
            scale_factor = user_max / exercise_max
            if scale_factor < 1.0:
                base_target = base_target * scale_factor
        
        return base_target
    
    def _interpolate_all_joint_targets(
        self, 
        current_frame: int, 
        total_frames: int
    ) -> Dict[JointType, float]:
        """Tính target cho tất cả khớp."""
        targets = {}
        for joint_type in self._state.active_joints:
            if joint_type in self._state.calibrated_joints:
                target = self._interpolate_target_angle(current_frame, total_frames, joint_type)
                targets[joint_type] = target
        return targets
    
    def _calculate_realtime_score(self, user_angle: float, target_angle: float) -> float:
        """Tính điểm real-time cho một khớp."""
        if target_angle <= 0:
            return self._state.current_score
        
        error = abs(user_angle - target_angle)
        error_percent = (error / target_angle) * 100
        
        if error_percent < 5:
            score = 100.0
        elif error_percent < 10:
            score = 95.0 - (error_percent - 5) * 1.0
        elif error_percent < 15:
            score = 90.0 - (error_percent - 10) * 2.0
        elif error_percent < 25:
            score = 80.0 - (error_percent - 15) * 1.5
        elif error_percent < 40:
            score = 65.0 - (error_percent - 25) * 1.0
        else:
            score = max(0, 50.0 - (error_percent - 40) * 0.5)
        
        return max(0, min(100, score))
    
    def _calculate_multi_joint_score(self) -> float:
        """Tính điểm weighted multi-joint."""
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
            
            joint_score = self._calculate_realtime_score(user_angle, target_angle)
            self._state.joint_scores_dict[joint_type] = joint_score
            
            total_weighted_score += joint_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        
        return self._state.current_score
    
    def _on_rep_complete(self) -> None:
        """Xử lý khi hoàn thành một rep."""
        dtw_result = None
        if len(self._user_angles) > 20 and len(self._ref_angles) > 20:
            user_seq = self._user_angles[-50:]
            ref_seq = self._ref_angles[-50:]
            dtw_result = compute_single_joint_dtw(user_seq, ref_seq)
        
        target = self._state.target_angle or self._state.user_max_angle or 150
        
        if self._scorer:
            rep_score = self._scorer.complete_rep(target, dtw_result)
            
            # Track rep scores
            self._rep_scores.append({
                "rep_number": rep_score.rep_number,
                "rom_score": rep_score.rom_score,
                "stability_score": rep_score.stability_score,
                "flow_score": rep_score.flow_score,
                "total_score": rep_score.total_score,
                "duration_ms": rep_score.duration_ms
            })
            
            if self._logger:
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
    
    def _process_pain(self) -> None:
        """Xử lý pain detection."""
        try:
            face_landmarks = self._analysis_queue.get_nowait()
            if self._pain_detector:
                result = self._pain_detector.analyze(face_landmarks)
                if result.is_pain_detected:
                    self._state.pain_level = result.pain_level.name
                    self._state.warning = result.message
                else:
                    self._state.pain_level = "NONE"
                    self._state.warning = ""
        except:
            pass
    
    # ==================== PHASE 4: SCORING ====================
    
    def _process_phase4(self) -> FinalReportOutput:
        """Xử lý Phase 4: Final Report."""
        return self._generate_final_report()
    
    def _generate_final_report(self) -> FinalReportOutput:
        """Tạo báo cáo cuối."""
        
        # Get scores from scorer
        rom_score = 0.0
        stability_score = 0.0
        flow_score = 0.0
        total_score = self._state.average_score
        
        if self._scorer:
            report = self._scorer.compute_session_report()
            rom_score = report.average_scores.get('rom', 0)
            stability_score = report.average_scores.get('stability', 0)
            flow_score = report.average_scores.get('flow', 0)
            total_score = report.average_scores.get('total', self._state.average_score)
        
        # Grade
        grade, grade_color = get_grade(total_score)
        
        # Calibrated joints
        calibrated_joints = []
        for jt, angle in self._state.calibrated_joints.items():
            calibrated_joints.append(
                JointCalibrationResult(
                    get_joint_name_vi(jt.value),
                    jt.value,
                    angle
                ).to_dict()
            )
        
        # Duration
        duration = int(time.time() - self._state.session_start_time)
        
        # Primary joint
        primary_joint = self._state.selected_joint or self._default_joint
        
        output = FinalReportOutput(
            session_id=self._state.session_id or f"session_{int(time.time())}",
            exercise_name=self._state.exercise_name or "Exercise",
            duration_seconds=duration,
            total_score=total_score,
            rom_score=rom_score,
            stability_score=stability_score,
            flow_score=flow_score,
            grade=grade,
            grade_color=grade_color,
            total_reps=self._state.rep_count,
            fatigue_level=self._state.fatigue_level,
            calibrated_joints=calibrated_joints,
            primary_joint=primary_joint.value,
            primary_max_angle=self._state.user_max_angle,
            rep_scores=self._rep_scores,
            recommendations=[
                "Tiep tuc tap luyen deu dan moi ngay",
                "Tang dan cuong do theo tung tuan",
                "Nghi ngoi day du giua cac buoi tap"
            ],
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S", 
                                     time.localtime(self._state.session_start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        
        # Log final report
        if self._logger:
            self._logger.end_session(output.to_dict())
        
        return output
    
    # ==================== PHASE TRANSITIONS ====================
    
    def _transition_to_phase2(self) -> None:
        """Chuyển sang Phase 2."""
        self._state.current_phase = AppPhase.PHASE2_CALIBRATION
        self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        self._state.calibration_queue_index = 0
        self._state.calibrated_joints = {}
        self._state.is_countdown_active = False
        self._state.is_calibrating_joint = False
        self._state.all_joints_calibrated = False
        self._state.phase2_complete_time = 0
    
    def _transition_to_phase3(self) -> None:
        """Chuyển sang Phase 3."""
        if not self._config.ref_video_path or not Path(self._config.ref_video_path).exists():
            self._transition_to_phase4()
            return
        
        self._state.current_phase = AppPhase.PHASE3_SYNC
        
        # Setup video engine
        self._video_engine = VideoEngine(self._config.ref_video_path)
        total_frames = self._video_engine.total_frames
        fps = self._video_engine.fps
        
        # Setup multi-joint tracking
        primary_joint = self._state.selected_joint or self._default_joint
        
        if primary_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise_type = "bicep_curl"
        elif primary_joint in (JointType.LEFT_KNEE, JointType.RIGHT_KNEE):
            exercise_type = "squat"
        else:
            exercise_type = "arm_raise"
        
        self._state.joint_weights = create_exercise_weights(exercise_type)
        self._state.active_joints = list(self._state.calibrated_joints.keys())
        
        if not self._state.active_joints:
            self._state.active_joints = [primary_joint]
            self._state.calibrated_joints[primary_joint] = 150.0
        
        if primary_joint in self._state.calibrated_joints:
            max_angle = self._state.calibrated_joints[primary_joint]
            self._state.user_max_angle = max_angle
        else:
            max_angle = self._state.user_max_angle or 150
        
        if primary_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise = create_elbow_flex_exercise(total_frames, fps, max_angle=max_angle)
        else:
            exercise = create_arm_raise_exercise(total_frames, fps, max_angle=max_angle)
        
        self._state.exercise_name = exercise.name
        
        self._sync_controller = MotionSyncController(
            exercise,
            user_max_angle=max_angle
        )
        
        self._state.user_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.target_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.joint_scores_dict = {jt: 0.0 for jt in self._state.active_joints}
        
        checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
        self._video_engine.set_checkpoints(checkpoint_frames)
        self._video_engine.set_speed(0.7)
        
        session_id = f"session_{int(time.time())}"
        self._state.session_id = session_id
        
        if self._logger:
            self._logger.start_session(session_id, exercise.name)
        if self._scorer:
            self._scorer.start_session(exercise.name, session_id)
        
        self._video_engine.play()
    
    def _transition_to_phase4(self) -> None:
        """Chuyển sang Phase 4."""
        self._state.current_phase = AppPhase.PHASE4_SCORING
        
        if self._scorer:
            report = self._scorer.compute_session_report()
            self._state.average_score = report.average_scores.get('total', 0)
    
    # ==================== CONTROL METHODS ====================
    
    def pause(self) -> None:
        """Pause (Phase 3)."""
        if self._state.current_phase == AppPhase.PHASE3_SYNC:
            self._state.is_paused = True
            if self._video_engine:
                self._video_engine.pause()
    
    def resume(self) -> None:
        """Resume (Phase 3)."""
        if self._state.current_phase == AppPhase.PHASE3_SYNC:
            self._state.is_paused = False
            if self._video_engine:
                self._video_engine.play()
    
    def restart(self) -> None:
        """Restart từ đầu."""
        self._state = EngineState()
        self._state.selected_joint = self._default_joint
        self._state.session_start_time = time.time()
        
        self._user_angles = []
        self._ref_angles = []
        self._score_history = []
        self._rep_scores = []
        
        if self._video_engine:
            self._video_engine.stop()
            self._video_engine = None
        
        self._sync_controller = None
        self._calibrator = SafeMaxCalibrator(
            duration_ms=self._config.calibration_duration_ms
        )
    
    def get_current_phase(self) -> int:
        """Lấy phase hiện tại (1-4)."""
        phase_map = {
            AppPhase.PHASE1_DETECTION: 1,
            AppPhase.PHASE2_CALIBRATION: 2,
            AppPhase.PHASE3_SYNC: 3,
            AppPhase.PHASE4_SCORING: 4,
            AppPhase.COMPLETED: 4,
        }
        return phase_map.get(self._state.current_phase, 1)
    
    def is_complete(self) -> bool:
        """Kiểm tra đã hoàn thành chưa."""
        return self._state.current_phase in (AppPhase.PHASE4_SCORING, AppPhase.COMPLETED)
    
    def cleanup(self) -> None:
        """Dọn dẹp resources."""
        if self._video_engine:
            self._video_engine.release()
        if self._detector:
            self._detector.close()
        if self._ref_detector:
            self._ref_detector.close()
