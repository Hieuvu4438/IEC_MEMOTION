"""
MEMOTION Engine Service - Stateful Backend Integration Layer

Class MemotionEngine la core engine xu ly frame, quan ly trang thai noi bo,
va tra ve JSON-serializable output. Moi instance doc lap, ho tro multi-user.

Thiet ke theo nguyen tac:
1. STATEFUL: Moi instance co self._state rieng biet
2. HANDS-FREE: Tu dong chuyen phase khi dat dieu kien
3. JSON-ONLY OUTPUT: Khong tra ve object MediaPipe/NumPy tho

Usage (Multi-user Backend):
    # Moi user co 1 engine instance rieng
    user_engines: Dict[str, MemotionEngine] = {}
    
    def handle_user_frame(user_id: str, frame: np.ndarray, timestamp_ms: int):
        if user_id not in user_engines:
            user_engines[user_id] = MemotionEngine.create_instance(
                config=EngineConfig(ref_video_path="./videos/exercise.mp4")
            )
        
        result = user_engines[user_id].process_frame(frame, timestamp_ms)
        return result.to_dict()  # JSON-serializable

Author: MEMOTION Team
Version: 2.0.0
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
import numpy as np
import uuid

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
    """Cac giai doan cua ung dung."""
    PHASE1_DETECTION = 1
    PHASE2_CALIBRATION = 2
    PHASE3_SYNC = 3
    PHASE4_SCORING = 4
    COMPLETED = 5


# Phase name mapping
PHASE_NAMES = {
    AppPhase.PHASE1_DETECTION: "detection",
    AppPhase.PHASE2_CALIBRATION: "calibration",
    AppPhase.PHASE3_SYNC: "sync",
    AppPhase.PHASE4_SCORING: "scoring",
    AppPhase.COMPLETED: "completed",
}


# Calibration Queue - thu tu tu dong do 6 khop
CALIBRATION_QUEUE: List[JointType] = [
    JointType.LEFT_SHOULDER,
    JointType.RIGHT_SHOULDER,
    JointType.LEFT_ELBOW,
    JointType.RIGHT_ELBOW,
    JointType.LEFT_KNEE,
    JointType.RIGHT_KNEE,
]

# Huong dan tu the
JOINT_POSITION_INSTRUCTIONS: Dict[JointType, str] = {
    JointType.LEFT_SHOULDER: "Moi ba dung NGANG",
    JointType.RIGHT_SHOULDER: "Moi ba dung NGANG",
    JointType.LEFT_ELBOW: "Moi ba dung NGANG",
    JointType.RIGHT_ELBOW: "Moi ba dung NGANG",
    JointType.LEFT_KNEE: "Moi ba dung DOC",
    JointType.RIGHT_KNEE: "Moi ba dung DOC",
}

# Timing constants
PHASE1_STABLE_FRAMES_REQUIRED: int = 30  # So frame on dinh de chuyen phase
PHASE1_COUNTDOWN_DURATION: float = 3.0  # giay
CALIBRATION_COUNTDOWN_DURATION: float = 5.0  # giay
PHASE2_COMPLETE_DELAY: float = 2.0  # giay


# ==================== ENGINE STATE (Per-Instance) ====================

@dataclass
class EngineState:
    """
    Trang thai noi bo cua Engine - MOI INSTANCE CO MOT BAN RIENG.
    
    Attributes nay KHONG duoc chia se giua cac instance.
    """
    # Current phase
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
    calibrated_joints: Dict[JointType, float] = field(default_factory=dict)
    all_joints_calibrated: bool = False
    phase2_complete_time: float = 0.0
    
    # Phase 3 state
    sync_state: Optional[SyncState] = None
    motion_phase: str = "idle"
    last_motion_phase: Optional[MotionPhase] = None
    user_angles_dict: Dict[JointType, float] = field(default_factory=dict)
    target_angles_dict: Dict[JointType, float] = field(default_factory=dict)
    joint_scores_dict: Dict[JointType, float] = field(default_factory=dict)
    joint_weights: Dict[JointType, float] = field(default_factory=dict)
    active_joints: List[JointType] = field(default_factory=list)
    
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
    instance_id: str = ""
    session_id: str = ""
    session_start_time: float = 0.0
    exercise_name: str = ""


# ==================== ENGINE CONFIG ====================

@dataclass
class EngineConfig:
    """
    Cau hinh cho Engine - truyen vao khi khoi tao.
    
    Attributes:
        models_dir: Thu muc chua models MediaPipe
        log_dir: Thu muc luu log
        ref_video_path: Duong dan video mau (bat buoc cho Phase 3)
        default_joint: Khop mac dinh (string)
        detection_stable_threshold: So frame on dinh de chuyen Phase 2
        calibration_duration_ms: Thoi gian do moi khop (ms)
    """
    models_dir: str = "./models"
    log_dir: str = "./data/logs"
    ref_video_path: Optional[str] = None
    default_joint: str = "left_shoulder"
    detection_stable_threshold: int = PHASE1_STABLE_FRAMES_REQUIRED
    calibration_duration_ms: int = 5000


# ==================== MEMOTION ENGINE (MAIN CLASS) ====================

class MemotionEngine:
    """
    MEMOTION Engine - Stateful Frame Processor cho Backend.
    
    MOI INSTANCE LA DOC LAP:
    - Tu quan ly self._state rieng
    - Khong chia se bat ky state nao voi instance khac
    - An toan cho multi-threaded/multi-user environment
    
    LUONG XU LY TU DONG (Hands-free):
    - Phase 1: Detection -> Tu dong sang Phase 2 khi 30 frames on dinh
    - Phase 2: Calibration -> Tu dong sang Phase 3 khi do xong 6 khop
    - Phase 3: Sync -> Tu dong sang Phase 4 khi video ket thuc
    - Phase 4: Scoring -> Tra ve bao cao cuoi
    
    OUTPUT BAT BUOC CO "phase":
    - Moi output deu co key "phase" (int: 1-4) va "phase_name" (string)
    - Output la JSON-serializable (khong co object MediaPipe/NumPy)
    
    Example:
        engine = MemotionEngine(config)
        result = engine.process_frame(frame, timestamp_ms)
        json_output = result.to_dict()
        # json_output["phase"] = 1, 2, 3, or 4
    """
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def create_instance(
        cls, 
        config: Optional[EngineConfig] = None,
        instance_id: Optional[str] = None
    ) -> "MemotionEngine":
        """
        Factory method: Tao instance moi voi ID duy nhat.
        
        Su dung method nay de dam bao moi user co engine rieng.
        
        Args:
            config: Cau hinh engine
            instance_id: ID tuy chinh (None = tu dong tao UUID)
        
        Returns:
            MemotionEngine: Instance moi, doc lap
        
        Example:
            # Trong Backend
            user_engines = {}
            
            @app.route('/api/process_frame', methods=['POST'])
            def process_frame():
                user_id = request.json['user_id']
                if user_id not in user_engines:
                    user_engines[user_id] = MemotionEngine.create_instance()
                ...
        """
        engine = cls(config)
        engine._state.instance_id = instance_id or str(uuid.uuid4())
        return engine
    
    # ==================== INITIALIZATION ====================
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Khoi tao Engine voi config.
        
        QUAN TRONG: Moi instance tu tao self._state rieng.
        
        Args:
            config: Cau hinh engine (None = su dung default)
        """
        # Config
        self._config = config or EngineConfig()
        
        # Map string to JointType
        self._joint_map: Dict[str, JointType] = {
            "left_shoulder": JointType.LEFT_SHOULDER,
            "right_shoulder": JointType.RIGHT_SHOULDER,
            "left_elbow": JointType.LEFT_ELBOW,
            "right_elbow": JointType.RIGHT_ELBOW,
            "left_knee": JointType.LEFT_KNEE,
            "right_knee": JointType.RIGHT_KNEE,
        }
        self._default_joint = self._joint_map.get(
            self._config.default_joint, 
            JointType.LEFT_SHOULDER
        )
        
        # ====== STATEFUL: Tao state RIENG cho instance nay ======
        self._state = EngineState()
        self._state.selected_joint = self._default_joint
        self._state.instance_id = str(uuid.uuid4())
        
        # Components (lazy init - chi tao khi can)
        self._detector: Optional[VisionDetector] = None
        self._ref_detector: Optional[VisionDetector] = None
        self._video_engine: Optional[VideoEngine] = None
        self._sync_controller: Optional[MotionSyncController] = None
        self._calibrator: Optional[SafeMaxCalibrator] = None
        self._pain_detector: Optional[PainDetector] = None
        self._scorer: Optional[HealthScorer] = None
        self._logger: Optional[SessionLogger] = None
        self._user_profile: Optional[UserProfile] = None
        
        # Data tracking (per-instance)
        self._user_angles: List[float] = []
        self._ref_angles: List[float] = []
        self._score_history: List[float] = []
        self._rep_scores: List[Dict[str, Any]] = []
        self._current_landmarks: Optional[np.ndarray] = None
        
        # Analysis queue (per-instance)
        self._analysis_queue: Queue = Queue(maxsize=5)
        
        # Init flag
        self._initialized: bool = False
    
    def initialize(self) -> bool:
        """
        Khoi tao cac components (detector, calibrator, etc.).
        
        Duoc goi tu dong khi process_frame() lan dau.
        
        Returns:
            bool: True neu thanh cong
        """
        try:
            models_dir = Path(self._config.models_dir)
            pose_model = models_dir / "pose_landmarker_lite.task"
            face_model = models_dir / "face_landmarker.task"
            
            if not pose_model.exists():
                raise FileNotFoundError(f"Pose model not found: {pose_model}")
            
            # Init main detector
            config = DetectorConfig(
                pose_model_path=str(pose_model),
                face_model_path=str(face_model) if face_model.exists() else None,
                running_mode="VIDEO"
            )
            self._detector = VisionDetector(config)
            
            # Init reference detector (for video analysis)
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
    
    # ==================== MAIN ENTRY POINT ====================
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp_ms: int
    ) -> EngineOutput:
        """
        XU LY MOT FRAME - Entry point chinh cho Backend.
        
        Ham nay tu dong:
        1. Detect pose tu frame
        2. Routing den phase hien tai (_run_phase1, _run_phase2, ...)
        3. Kiem tra dieu kien chuyen phase
        4. Tra ve output JSON-serializable voi key "phase"
        
        Args:
            frame: Frame anh (BGR numpy array, shape HxWx3)
            timestamp_ms: Timestamp tinh bang milliseconds
        
        Returns:
            EngineOutput: Ket qua xu ly, BAT BUOC co:
                - current_phase (int): 1, 2, 3, hoac 4
                - phase_name (str): "detection", "calibration", "sync", "scoring"
                - [phase_data]: Du lieu cua phase tuong ung
        
        Example:
            result = engine.process_frame(frame, timestamp_ms)
            output = result.to_dict()
            # output = {
            #     "phase": 1,
            #     "phase_name": "detection",
            #     "detection": { ... }
            # }
        """
        # Lazy initialization
        if not self._initialized:
            if not self.initialize():
                return self._create_error_output("Engine not initialized")
        
        # Convert timestamp
        timestamp = timestamp_ms / 1000.0
        
        # Process detection
        result = self._detector.process_frame(frame, timestamp_ms)
        
        # ====== ROUTING DEN PHASE HIEN TAI ======
        current_phase = self._state.current_phase
        
        if current_phase == AppPhase.PHASE1_DETECTION:
            return self._run_phase1(result, timestamp)
        
        elif current_phase == AppPhase.PHASE2_CALIBRATION:
            return self._run_phase2(result, timestamp_ms, timestamp)
        
        elif current_phase == AppPhase.PHASE3_SYNC:
            return self._run_phase3(result, timestamp)
        
        elif current_phase == AppPhase.PHASE4_SCORING:
            return self._run_phase4()
        
        else:  # COMPLETED
            return self._run_phase4()
    
    def _create_error_output(self, error_msg: str) -> EngineOutput:
        """Tao output loi voi phase hien tai."""
        phase_num = self._get_phase_number()
        phase_name = PHASE_NAMES.get(self._state.current_phase, "unknown")
        
        return EngineOutput(
            current_phase=phase_num,
            phase_name=phase_name,
            error=error_msg
        )
    
    def _get_phase_number(self) -> int:
        """Lay so phase hien tai (1-4)."""
        phase_map = {
            AppPhase.PHASE1_DETECTION: 1,
            AppPhase.PHASE2_CALIBRATION: 2,
            AppPhase.PHASE3_SYNC: 3,
            AppPhase.PHASE4_SCORING: 4,
            AppPhase.COMPLETED: 4,
        }
        return phase_map.get(self._state.current_phase, 1)
    
    # ==================== PHASE 1: DETECTION ====================
    
    def _run_phase1(self, result: Any, timestamp: float) -> EngineOutput:
        """
        Phase 1: Pose Detection voi auto-transition.
        
        Logic tu dong chuyen Phase:
        1. Dem so frame phat hien pose on dinh
        2. Khi dat 30 frames -> Bat dau countdown 3 giay
        3. Khi countdown ket thuc -> Tu dong chuyen Phase 2
        
        Args:
            result: Ket qua detect tu VisionDetector
            timestamp: Thoi gian hien tai (giay)
        
        Returns:
            EngineOutput voi phase=1 va detection data
        """
        output = DetectionOutput()
        
        if result.has_pose():
            # Luu landmarks
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Dem stable frames
            self._state.detection_stable_count += 1
            progress = min(1.0, self._state.detection_stable_count / 
                          self._config.detection_stable_threshold)
            
            output.stable_count = self._state.detection_stable_count
            output.progress = progress
            
            # ====== AUTO TRANSITION: Kiem tra du 30 frames ======
            if self._state.detection_stable_count >= self._config.detection_stable_threshold:
                self._state.pose_detected = True
                output.pose_detected = True
                
                # Bat dau countdown neu chua
                if not self._state.phase1_countdown_active:
                    self._state.phase1_countdown_active = True
                    self._state.phase1_countdown_start = timestamp
                
                # Tinh thoi gian con lai
                elapsed = timestamp - self._state.phase1_countdown_start
                remaining = PHASE1_COUNTDOWN_DURATION - elapsed
                
                if remaining > 0:
                    output.countdown_remaining = remaining
                    output.status = "countdown"
                    output.message = f"Chuan bi... {int(remaining) + 1} giay"
                else:
                    # ====== CHUYEN PHASE 2 ======
                    output.status = "transitioning"
                    output.message = "Chuyen sang Phase 2: Calibration"
                    self._transition_to_phase2()
            else:
                output.status = "detecting"
                output.message = f"Dang xac nhan... {int(progress * 100)}%"
        else:
            # Mat pose -> Reset counter
            self._state.detection_stable_count = 0
            self._state.phase1_countdown_active = False
            output.status = "idle"
            output.message = "Chua phat hien nguoi. Hay dung vao khung hinh."
        
        return EngineOutput(
            current_phase=1,
            phase_name="detection",
            detection=output.to_dict()
        )
    
    # ==================== PHASE 2: CALIBRATION ====================
    
    def _run_phase2(
        self, 
        result: Any, 
        timestamp_ms: int, 
        timestamp: float
    ) -> EngineOutput:
        """
        Phase 2: Automated Safe-Max Calibration.
        
        Logic tu dong chuyen Phase:
        1. Do tung khop trong CALIBRATION_QUEUE (6 khop)
        2. Moi khop: Countdown 5s -> Do 5s -> Luu ket qua
        3. Khi do xong 6 khop -> Delay 2s -> Tu dong chuyen Phase 3
        
        Args:
            result: Ket qua detect
            timestamp_ms: Timestamp (ms)
            timestamp: Timestamp (giay)
        
        Returns:
            EngineOutput voi phase=2 va calibration data
        """
        output = CalibrationOutput()
        
        # Lay khop hien tai trong queue
        current_joint: Optional[JointType] = None
        if self._state.calibration_queue_index < len(CALIBRATION_QUEUE):
            current_joint = CALIBRATION_QUEUE[self._state.calibration_queue_index]
            self._state.selected_joint = current_joint
        else:
            self._state.all_joints_calibrated = True
        
        # Build joints status list
        joints_status = self._build_joints_status()
        output.joints_status = joints_status
        output.queue_index = self._state.calibration_queue_index
        output.total_joints = len(CALIBRATION_QUEUE)
        output.overall_progress = len(self._state.calibrated_joints) / len(CALIBRATION_QUEUE)
        
        # ====== LOGIC TU DONG ======
        
        if self._state.all_joints_calibrated:
            # Da do xong tat ca
            output.status = "all_complete"
            output.message = "Da do xong tat ca 6 khop!"
            
            # Tu dong chuyen Phase 3 sau delay
            if self._state.phase2_complete_time == 0:
                self._state.phase2_complete_time = timestamp
                self._save_calibration_to_profile()
            elif timestamp - self._state.phase2_complete_time > PHASE2_COMPLETE_DELAY:
                # ====== CHUYEN PHASE 3 ======
                self._state.calibration_complete = True
                self._transition_to_phase3()
        
        elif not self._state.is_countdown_active and not self._state.is_calibrating_joint:
            # Bat dau countdown cho khop moi
            self._state.is_countdown_active = True
            self._state.calibration_countdown_start = timestamp
            output.status = "preparing"
            output.current_joint = current_joint.value if current_joint else None
            output.current_joint_name = get_joint_name_vi(current_joint.value) if current_joint else None
            
        elif self._state.is_countdown_active:
            # Dang countdown
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
                # Countdown xong -> Bat dau do
                self._state.is_countdown_active = False
                self._state.is_calibrating_joint = True
                self._start_calibration_for_joint(current_joint)
                output.status = "collecting"
        
        elif self._state.is_calibrating_joint and current_joint:
            # Dang do khop
            output.current_joint = current_joint.value
            output.current_joint_name = get_joint_name_vi(current_joint.value)
            output.status = "collecting"
            
            if self._calibrator and self._calibrator.state == CalibrationState.COLLECTING:
                output.progress = self._calibrator.progress
                output.message = f"Dang do {get_joint_name_vi(current_joint.value)}... {int(output.progress * 100)}%"
                
                # Them frame vao calibrator
                if result.has_pose():
                    try:
                        landmarks = result.pose_landmarks.to_numpy()
                        angle = calculate_joint_angle(landmarks, current_joint, use_3d=True)
                        self._state.user_angle = angle
                        output.current_angle = angle
                        self._calibrator.add_frame(result.pose_landmarks, timestamp_ms)
                        
                        # Kiem tra hoan thanh
                        if self._calibrator.state == CalibrationState.COMPLETED:
                            self._finish_calibration_for_joint(current_joint)
                    except ValueError:
                        pass
            
            elif self._calibrator and self._calibrator.state == CalibrationState.COMPLETED:
                output.status = "complete"
                output.user_max_angle = self._state.calibrated_joints.get(current_joint, 0)
        
        return EngineOutput(
            current_phase=2,
            phase_name="calibration",
            calibration=output.to_dict()
        )
    
    def _build_joints_status(self) -> List[Dict[str, Any]]:
        """Build danh sach trang thai cua cac khop."""
        joints_status = []
        for i, jt in enumerate(CALIBRATION_QUEUE):
            if jt in self._state.calibrated_joints:
                status = "complete"
            elif i == self._state.calibration_queue_index:
                status = "collecting"
            else:
                status = "pending"
            
            joint_status = JointCalibrationStatus(
                joint_name=get_joint_name_vi(jt.value),
                joint_type=jt.value,
                max_angle=self._state.calibrated_joints.get(jt),
                status=status
            )
            joints_status.append(joint_status.to_dict())
        
        return joints_status
    
    def _start_calibration_for_joint(self, joint_type: JointType) -> None:
        """Bat dau do mot khop."""
        if self._user_profile is None:
            self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        self._calibrator = SafeMaxCalibrator(
            duration_ms=self._config.calibration_duration_ms
        )
        self._calibrator.start_calibration(joint_type, self._user_profile)
    
    def _finish_calibration_for_joint(self, joint_type: JointType) -> None:
        """Hoan thanh do mot khop va chuyen sang khop tiep theo."""
        if self._user_profile and joint_type:
            max_angle = self._user_profile.get_max_angle(joint_type)
            if max_angle:
                self._state.calibrated_joints[joint_type] = max_angle
                
                if joint_type == self._default_joint:
                    self._state.user_max_angle = max_angle
        
        # Chuyen khop tiep
        self._state.calibration_queue_index += 1
        self._state.is_calibrating_joint = False
        self._state.user_angle = 0.0
        
        if self._state.calibration_queue_index >= len(CALIBRATION_QUEUE):
            self._state.all_joints_calibrated = True
    
    def _save_calibration_to_profile(self) -> None:
        """Luu calibration profile ra file."""
        if not self._user_profile:
            return
            
        profile_dir = Path("./data/user_profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        profile_path = profile_dir / f"user_{timestamp}.json"
        
        try:
            import json
            profile_data = {
                "user_id": self._user_profile.user_id,
                "instance_id": self._state.instance_id,
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
    
    def _run_phase3(self, result: Any, timestamp: float) -> EngineOutput:
        """
        Phase 3: Motion Sync voi multi-joint tracking.
        
        Logic tu dong chuyen Phase:
        1. Phat video mau va so sanh voi user
        2. Tinh diem real-time cho tung khop
        3. Khi video ket thuc HOAC SyncStatus.COMPLETE -> Tu dong chuyen Phase 4
        
        Args:
            result: Ket qua detect
            timestamp: Thoi gian hien tai (giay)
        
        Returns:
            EngineOutput voi phase=3 va sync data
        """
        output = SyncOutput()
        
        # Xu ly video engine
        if self._video_engine:
            # Kiem tra trang thai sync
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
                
                # ====== KIEM TRA HOAN THANH - TU DONG CHUYEN PHASE 4 ======
                if self._state.sync_state:
                    if self._state.sync_state.sync_status == SyncStatus.COMPLETE:
                        self._transition_to_phase4()
                        output.status = "complete"
                        return EngineOutput(
                            current_phase=3,
                            phase_name="sync",
                            sync=output.to_dict()
                        )
                
                if ref_status.state == PlaybackState.FINISHED:
                    # Video ket thuc -> Chuyen Phase 4
                    self._transition_to_phase4()
                    output.status = "complete"
                    return EngineOutput(
                        current_phase=3,
                        phase_name="sync",
                        sync=output.to_dict()
                    )
                
                # Kiem tra hoan thanh rep
                if self._state.sync_state:
                    current_mp = self._state.sync_state.current_phase
                    if (self._state.last_motion_phase == MotionPhase.CONCENTRIC and 
                        current_mp == MotionPhase.IDLE):
                        self._on_rep_complete()
                    self._state.last_motion_phase = current_mp
        
        output.video_paused = self._state.is_paused
        
        # Tinh goc multi-joint
        if result.has_pose():
            self._current_landmarks = result.pose_landmarks.to_numpy()
            
            # Tinh goc tat ca khop
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
        
        # Tinh target multi-joint
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
        
        # Tinh diem multi-joint (weighted)
        multi_joint_score = self._calculate_multi_joint_score()
        self._state.current_score = 0.7 * self._state.current_score + 0.3 * multi_joint_score
        
        if len(self._state.target_angles_dict) > 0:
            self._score_history.append(multi_joint_score)
            if len(self._score_history) > 0:
                self._state.average_score = sum(self._score_history) / len(self._score_history)
        
        # Update fatigue
        if self._scorer:
            scorer_status = self._scorer.get_current_status()
            self._state.fatigue_level = scorer_status.get("fatigue_level", "FRESH")
        
        # Build joint errors
        joint_errors = self._build_joint_errors()
        
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
        
        return EngineOutput(
            current_phase=3,
            phase_name="sync",
            sync=output.to_dict()
        )
    
    def _build_joint_errors(self) -> List[Dict[str, Any]]:
        """Build danh sach sai so cua cac khop."""
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
        
        return joint_errors
    
    def _calculate_all_joint_angles(self, landmarks: np.ndarray) -> Dict[JointType, float]:
        """Tinh goc tat ca khop dang tracking."""
        angles: Dict[JointType, float] = {}
        for joint_type in self._state.active_joints:
            try:
                angle = calculate_joint_angle(landmarks, joint_type, use_3d=True)
                angles[joint_type] = angle
            except (ValueError, IndexError):
                # Giu gia tri cu neu khong tinh duoc
                if joint_type in self._state.user_angles_dict:
                    angles[joint_type] = self._state.user_angles_dict[joint_type]
        return angles
    
    def _interpolate_target_angle(
        self, 
        current_frame: int, 
        total_frames: int, 
        joint_type: Optional[JointType] = None
    ) -> float:
        """Tinh target angle cho mot khop dua tren video progress."""
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
        
        # Interpolate between checkpoints
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
        
        # Scale theo user's calibrated max
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
        """Tinh target cho tat ca khop."""
        targets: Dict[JointType, float] = {}
        for joint_type in self._state.active_joints:
            if joint_type in self._state.calibrated_joints:
                target = self._interpolate_target_angle(current_frame, total_frames, joint_type)
                targets[joint_type] = target
        return targets
    
    def _calculate_realtime_score(self, user_angle: float, target_angle: float) -> float:
        """Tinh diem real-time cho mot khop."""
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
        """Tinh diem weighted multi-joint."""
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
        """Xu ly khi hoan thanh mot rep."""
        dtw_result = None
        if len(self._user_angles) > 20 and len(self._ref_angles) > 20:
            user_seq = self._user_angles[-50:]
            ref_seq = self._ref_angles[-50:]
            dtw_result = compute_single_joint_dtw(user_seq, ref_seq)
        
        target = self._state.target_angle or self._state.user_max_angle or 150
        
        if self._scorer:
            rep_score = self._scorer.complete_rep(target, dtw_result)
            
            # Track rep scores (JSON-serializable)
            self._rep_scores.append({
                "rep_number": rep_score.rep_number,
                "rom_score": float(rep_score.rom_score),
                "stability_score": float(rep_score.stability_score),
                "flow_score": float(rep_score.flow_score),
                "total_score": float(rep_score.total_score),
                "duration_ms": int(rep_score.duration_ms)
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
        """Xu ly pain detection tu face landmarks."""
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
    
    def _run_phase4(self) -> EngineOutput:
        """
        Phase 4: Final Report.
        
        Giai doan cuoi, tra ve bao cao tong ket.
        
        Returns:
            EngineOutput voi phase=4 va final_report data
        """
        report = self._generate_final_report()
        
        return EngineOutput(
            current_phase=4,
            phase_name="scoring",
            final_report=report.to_dict()
        )
    
    def _generate_final_report(self) -> FinalReportOutput:
        """Tao bao cao cuoi cung."""
        
        # Lay scores tu scorer
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
        
        # Calibrated joints (JSON-serializable)
        calibrated_joints: List[Dict[str, Any]] = []
        for jt, angle in self._state.calibrated_joints.items():
            calibrated_joints.append(
                JointCalibrationResult(
                    get_joint_name_vi(jt.value),
                    jt.value,
                    float(angle)
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
            total_score=float(total_score),
            rom_score=float(rom_score),
            stability_score=float(stability_score),
            flow_score=float(flow_score),
            grade=grade,
            grade_color=grade_color,
            total_reps=self._state.rep_count,
            fatigue_level=self._state.fatigue_level,
            calibrated_joints=calibrated_joints,
            primary_joint=primary_joint.value,
            primary_max_angle=float(self._state.user_max_angle),
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
        """Chuyen sang Phase 2: Calibration."""
        self._state.current_phase = AppPhase.PHASE2_CALIBRATION
        self._user_profile = UserProfile(user_id=f"user_{int(time.time())}")
        
        # Reset Phase 2 state
        self._state.calibration_queue_index = 0
        self._state.calibrated_joints = {}
        self._state.is_countdown_active = False
        self._state.is_calibrating_joint = False
        self._state.all_joints_calibrated = False
        self._state.phase2_complete_time = 0
    
    def _transition_to_phase3(self) -> None:
        """Chuyen sang Phase 3: Motion Sync."""
        # Kiem tra video path
        if not self._config.ref_video_path or not Path(self._config.ref_video_path).exists():
            # Khong co video -> Nhay thang sang Phase 4
            self._transition_to_phase4()
            return
        
        self._state.current_phase = AppPhase.PHASE3_SYNC
        
        # Setup video engine
        self._video_engine = VideoEngine(self._config.ref_video_path)
        total_frames = self._video_engine.total_frames
        fps = self._video_engine.fps
        
        # Setup multi-joint tracking
        primary_joint = self._state.selected_joint or self._default_joint
        
        # Xac dinh loai bai tap
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
        
        # Lay max angle
        if primary_joint in self._state.calibrated_joints:
            max_angle = self._state.calibrated_joints[primary_joint]
            self._state.user_max_angle = max_angle
        else:
            max_angle = self._state.user_max_angle or 150
        
        # Tao exercise
        if primary_joint in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
            exercise = create_elbow_flex_exercise(total_frames, fps, max_angle=max_angle)
        else:
            exercise = create_arm_raise_exercise(total_frames, fps, max_angle=max_angle)
        
        self._state.exercise_name = exercise.name
        
        # Tao sync controller
        self._sync_controller = MotionSyncController(
            exercise,
            user_max_angle=max_angle
        )
        
        # Init tracking dicts
        self._state.user_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.target_angles_dict = {jt: 0.0 for jt in self._state.active_joints}
        self._state.joint_scores_dict = {jt: 0.0 for jt in self._state.active_joints}
        
        # Setup video playback
        checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
        self._video_engine.set_checkpoints(checkpoint_frames)
        self._video_engine.set_speed(0.7)
        
        # Session ID
        session_id = f"session_{int(time.time())}"
        self._state.session_id = session_id
        
        # Start logging
        if self._logger:
            self._logger.start_session(session_id, exercise.name)
        if self._scorer:
            self._scorer.start_session(exercise.name, session_id)
        
        # Play video
        self._video_engine.play()
    
    def _transition_to_phase4(self) -> None:
        """Chuyen sang Phase 4: Scoring."""
        self._state.current_phase = AppPhase.PHASE4_SCORING
        
        if self._scorer:
            report = self._scorer.compute_session_report()
            self._state.average_score = report.average_scores.get('total', 0)
    
    # ==================== PUBLIC CONTROL METHODS ====================
    
    def pause(self) -> None:
        """Pause video (chi co tac dung o Phase 3)."""
        if self._state.current_phase == AppPhase.PHASE3_SYNC:
            self._state.is_paused = True
            if self._video_engine:
                self._video_engine.pause()
    
    def resume(self) -> None:
        """Resume video (chi co tac dung o Phase 3)."""
        if self._state.current_phase == AppPhase.PHASE3_SYNC:
            self._state.is_paused = False
            if self._video_engine:
                self._video_engine.play()
    
    def restart(self) -> None:
        """Restart tu dau - reset toan bo state."""
        # Tao state moi
        old_instance_id = self._state.instance_id
        self._state = EngineState()
        self._state.selected_joint = self._default_joint
        self._state.instance_id = old_instance_id
        self._state.session_start_time = time.time()
        
        # Reset tracking data
        self._user_angles = []
        self._ref_angles = []
        self._score_history = []
        self._rep_scores = []
        
        # Cleanup video
        if self._video_engine:
            self._video_engine.stop()
            self._video_engine = None
        
        # Reset controllers
        self._sync_controller = None
        self._calibrator = SafeMaxCalibrator(
            duration_ms=self._config.calibration_duration_ms
        )
    
    def skip_to_phase(self, phase: int) -> bool:
        """
        Nhay den phase chi dinh (debug/testing only).
        
        Args:
            phase: So phase (1-4)
        
        Returns:
            bool: True neu thanh cong
        """
        phase_map = {
            1: AppPhase.PHASE1_DETECTION,
            2: AppPhase.PHASE2_CALIBRATION,
            3: AppPhase.PHASE3_SYNC,
            4: AppPhase.PHASE4_SCORING,
        }
        
        if phase not in phase_map:
            return False
        
        target_phase = phase_map[phase]
        
        if target_phase == AppPhase.PHASE2_CALIBRATION:
            self._transition_to_phase2()
        elif target_phase == AppPhase.PHASE3_SYNC:
            # Phai co calibration data
            if not self._state.calibrated_joints:
                self._state.calibrated_joints = {self._default_joint: 150.0}
            self._transition_to_phase3()
        elif target_phase == AppPhase.PHASE4_SCORING:
            self._transition_to_phase4()
        else:
            self._state.current_phase = target_phase
        
        return True
    
    # ==================== GETTERS ====================
    
    def get_current_phase(self) -> int:
        """Lay phase hien tai (1-4)."""
        return self._get_phase_number()
    
    def get_current_phase_name(self) -> str:
        """Lay ten phase hien tai."""
        return PHASE_NAMES.get(self._state.current_phase, "unknown")
    
    def get_instance_id(self) -> str:
        """Lay instance ID."""
        return self._state.instance_id
    
    def is_complete(self) -> bool:
        """Kiem tra da hoan thanh chua."""
        return self._state.current_phase in (AppPhase.PHASE4_SCORING, AppPhase.COMPLETED)
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Lay snapshot cua state hien tai (debug/monitoring).
        
        Returns:
            Dict JSON-serializable chua thong tin state
        """
        return {
            "instance_id": self._state.instance_id,
            "session_id": self._state.session_id,
            "current_phase": self._get_phase_number(),
            "phase_name": self.get_current_phase_name(),
            "is_running": self._state.is_running,
            "is_paused": self._state.is_paused,
            "pose_detected": self._state.pose_detected,
            "calibration_progress": len(self._state.calibrated_joints) / len(CALIBRATION_QUEUE),
            "rep_count": self._state.rep_count,
            "average_score": float(self._state.average_score),
            "session_duration_seconds": int(time.time() - self._state.session_start_time),
        }
    
    # ==================== CLEANUP ====================
    
    def cleanup(self) -> None:
        """Don dep resources khi ket thuc."""
        if self._video_engine:
            self._video_engine.release()
            self._video_engine = None
        if self._detector:
            self._detector.close()
            self._detector = None
        if self._ref_detector:
            self._ref_detector.close()
            self._ref_detector = None
        
        self._initialized = False
    
    def __del__(self):
        """Destructor - tu dong cleanup."""
        try:
            self.cleanup()
        except:
            pass


# ==================== BACKWARD COMPATIBILITY ====================

# Alias cho ten cu
EngineService = MemotionEngine


# ==================== FACTORY FUNCTIONS ====================

def create_engine_for_user(
    user_id: str,
    config: Optional[EngineConfig] = None
) -> MemotionEngine:
    """
    Factory function: Tao engine cho mot user.
    
    Args:
        user_id: ID cua user
        config: Cau hinh tuy chinh
    
    Returns:
        MemotionEngine instance moi
    
    Example:
        engines = {}
        
        def handle_request(user_id, frame, timestamp):
            if user_id not in engines:
                engines[user_id] = create_engine_for_user(user_id)
            return engines[user_id].process_frame(frame, timestamp)
    """
    return MemotionEngine.create_instance(
        config=config,
        instance_id=f"engine_{user_id}"
    )
