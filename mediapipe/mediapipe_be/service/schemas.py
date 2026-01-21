"""
MEMOTION Backend Schemas - JSON-serializable data structures

Định nghĩa các class để chuẩn hóa dữ liệu trả về cho Backend.
Tất cả các class đều dễ dàng convert sang JSON (chỉ dùng kiểu dữ liệu cơ bản).

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Literal
from enum import Enum


# ==================== ENUMS ====================

class PhaseStatus(str, Enum):
    """Trạng thái của mỗi phase."""
    IDLE = "idle"
    PREPARING = "preparing"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class MotionPhaseType(str, Enum):
    """Loại phase trong bài tập."""
    IDLE = "idle"
    ECCENTRIC = "eccentric"
    HOLD = "hold"
    CONCENTRIC = "concentric"


class DirectionHint(str, Enum):
    """Hướng dẫn điều chỉnh góc."""
    RAISE = "raise"      # Nâng cao hơn
    LOWER = "lower"      # Hạ thấp hơn
    HOLD = "hold"        # Giữ nguyên
    OK = "ok"            # Đạt mục tiêu


# ==================== PHASE 1: DETECTION ====================

@dataclass
class DetectionOutput:
    """
    Output cho Phase 1: Pose Detection
    
    Attributes:
        pose_detected: True nếu đã phát hiện pose ổn định (30 frames)
        stable_count: Số frame ổn định hiện tại (0-30)
        progress: Tiến trình detection (0.0 - 1.0)
        countdown_remaining: Thời gian countdown còn lại (giây), None nếu chưa bắt đầu
        status: Trạng thái hiện tại
        message: Thông báo hiển thị cho user
    """
    pose_detected: bool = False
    stable_count: int = 0
    progress: float = 0.0
    countdown_remaining: Optional[float] = None
    status: str = "idle"  # idle, detecting, countdown, transitioning
    message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "pose_detected": self.pose_detected,
            "stable_count": self.stable_count,
            "progress": round(self.progress, 2),
            "countdown_remaining": round(self.countdown_remaining, 1) if self.countdown_remaining else None,
            "status": self.status,
            "message": self.message
        }


# ==================== PHASE 2: CALIBRATION ====================

@dataclass
class JointCalibrationStatus:
    """
    Trạng thái calibration của một khớp.
    
    Attributes:
        joint_name: Tên khớp (tiếng Việt không dấu)
        joint_type: Loại khớp (left_shoulder, right_elbow, etc.)
        max_angle: Góc max đã đo được (None nếu chưa đo)
        status: Trạng thái (pending, preparing, collecting, complete)
    """
    joint_name: str
    joint_type: str
    max_angle: Optional[float] = None
    status: str = "pending"  # pending, preparing, collecting, complete
    
    def to_dict(self) -> Dict:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "max_angle": round(self.max_angle, 1) if self.max_angle else None,
            "status": self.status
        }


@dataclass
class CalibrationOutput:
    """
    Output cho Phase 2: Safe-Max Calibration
    
    Attributes:
        current_joint: Khớp đang đo
        current_joint_name: Tên khớp đang đo (tiếng Việt)
        queue_index: Vị trí trong queue (0-5)
        total_joints: Tổng số khớp cần đo (6)
        progress: Tiến trình đo khớp hiện tại (0.0 - 1.0)
        overall_progress: Tiến trình tổng thể (0.0 - 1.0)
        current_angle: Góc hiện tại đang đo
        user_max_angle: Góc max đã ghi nhận
        countdown_remaining: Thời gian countdown còn lại (giây)
        status: Trạng thái (preparing, collecting, complete, all_complete)
        position_instruction: Hướng dẫn tư thế ("Moi ba dung NGANG/DOC")
        joints_status: Danh sách trạng thái của tất cả khớp
        message: Thông báo hiển thị
    """
    current_joint: Optional[str] = None
    current_joint_name: Optional[str] = None
    queue_index: int = 0
    total_joints: int = 6
    progress: float = 0.0
    overall_progress: float = 0.0
    current_angle: float = 0.0
    user_max_angle: float = 0.0
    countdown_remaining: Optional[float] = None
    status: str = "preparing"  # preparing, collecting, complete, all_complete
    position_instruction: str = ""
    joints_status: List[Dict] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "current_joint": self.current_joint,
            "current_joint_name": self.current_joint_name,
            "queue_index": self.queue_index,
            "total_joints": self.total_joints,
            "progress": round(self.progress, 2),
            "overall_progress": round(self.overall_progress, 2),
            "current_angle": round(self.current_angle, 1),
            "user_max_angle": round(self.user_max_angle, 1),
            "countdown_remaining": round(self.countdown_remaining, 1) if self.countdown_remaining else None,
            "status": self.status,
            "position_instruction": self.position_instruction,
            "joints_status": self.joints_status,
            "message": self.message
        }


# ==================== PHASE 3: SYNC ====================

@dataclass
class JointError:
    """
    Thông tin sai số của một khớp.
    
    Attributes:
        joint_name: Tên khớp (tiếng Việt không dấu)
        joint_type: Loại khớp
        user_angle: Góc hiện tại của user
        target_angle: Góc mục tiêu
        error: Sai số (độ)
        error_percent: Sai số phần trăm
        score: Điểm của khớp này (0-100)
        direction_hint: Hướng cần điều chỉnh (raise/lower/hold/ok)
        weight: Trọng số của khớp trong bài tập
    """
    joint_name: str
    joint_type: str
    user_angle: float
    target_angle: float
    error: float
    error_percent: float
    score: float
    direction_hint: str  # raise, lower, hold, ok
    weight: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "user_angle": round(self.user_angle, 1),
            "target_angle": round(self.target_angle, 1),
            "error": round(self.error, 1),
            "error_percent": round(self.error_percent, 1),
            "score": round(self.score, 1),
            "direction_hint": self.direction_hint,
            "weight": round(self.weight, 2)
        }


@dataclass
class SyncOutput:
    """
    Output cho Phase 3: Motion Sync (Multi-joint)
    
    Attributes:
        # Primary joint info (backward compatible)
        user_angle: Góc hiện tại của primary joint
        target_angle: Góc mục tiêu của primary joint
        error: Sai số của primary joint
        
        # Scoring
        current_score: Điểm hiện tại (weighted multi-joint)
        average_score: Điểm trung bình cả buổi
        
        # Motion state
        motion_phase: Phase động tác hiện tại (idle/eccentric/hold/concentric)
        rep_count: Số hiệp đã hoàn thành
        
        # Video progress
        video_progress: Tiến trình video (0.0 - 1.0)
        video_paused: Video đang pause
        
        # Health indicators
        pain_level: Mức độ đau (NONE/MILD/MODERATE/SEVERE)
        fatigue_level: Mức độ mệt mỏi (FRESH/MILD/MODERATE/FATIGUED)
        
        # Multi-joint details
        joint_errors: Danh sách chi tiết sai số từng khớp
        active_joints_count: Số khớp đang tracking
        
        # Feedback
        feedback_text: Text phản hồi (TUYET VOI/TOT/KHA/DIEU CHINH)
        direction_hint: Hướng cần điều chỉnh cho primary joint
        warning: Cảnh báo (nếu có)
        
        status: Trạng thái sync (syncing, paused, complete)
    """
    # Primary joint
    user_angle: float = 0.0
    target_angle: float = 0.0
    error: float = 0.0
    
    # Scoring
    current_score: float = 0.0
    average_score: float = 0.0
    
    # Motion state
    motion_phase: str = "idle"
    rep_count: int = 0
    
    # Video
    video_progress: float = 0.0
    video_paused: bool = False
    
    # Health
    pain_level: str = "NONE"
    fatigue_level: str = "FRESH"
    
    # Multi-joint
    joint_errors: List[Dict] = field(default_factory=list)
    active_joints_count: int = 0
    
    # Feedback
    feedback_text: str = ""
    direction_hint: str = "hold"
    warning: Optional[str] = None
    
    status: str = "syncing"  # syncing, paused, complete
    
    def to_dict(self) -> Dict:
        return {
            "user_angle": round(self.user_angle, 1),
            "target_angle": round(self.target_angle, 1),
            "error": round(self.error, 1),
            "current_score": round(self.current_score, 1),
            "average_score": round(self.average_score, 1),
            "motion_phase": self.motion_phase,
            "rep_count": self.rep_count,
            "video_progress": round(self.video_progress, 2),
            "video_paused": self.video_paused,
            "pain_level": self.pain_level,
            "fatigue_level": self.fatigue_level,
            "joint_errors": self.joint_errors,
            "active_joints_count": self.active_joints_count,
            "feedback_text": self.feedback_text,
            "direction_hint": self.direction_hint,
            "warning": self.warning,
            "status": self.status
        }


# ==================== PHASE 4: FINAL REPORT ====================

@dataclass
class RepScore:
    """
    Điểm số của một hiệp.
    
    Attributes:
        rep_number: Số thứ tự hiệp
        rom_score: Điểm ROM (Range of Motion)
        stability_score: Điểm ổn định
        flow_score: Điểm mượt mà
        total_score: Điểm tổng
        duration_ms: Thời gian thực hiện (ms)
    """
    rep_number: int
    rom_score: float
    stability_score: float
    flow_score: float
    total_score: float
    duration_ms: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "rep_number": self.rep_number,
            "rom_score": round(self.rom_score, 1),
            "stability_score": round(self.stability_score, 1),
            "flow_score": round(self.flow_score, 1),
            "total_score": round(self.total_score, 1),
            "duration_ms": self.duration_ms
        }


@dataclass
class JointCalibrationResult:
    """
    Kết quả calibration của một khớp.
    """
    joint_name: str
    joint_type: str
    max_angle: float
    
    def to_dict(self) -> Dict:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "max_angle": round(self.max_angle, 1)
        }


@dataclass
class FinalReportOutput:
    """
    Output cho Phase 4: Final Report
    
    Attributes:
        # Session info
        session_id: ID của buổi tập
        exercise_name: Tên bài tập
        duration_seconds: Tổng thời gian (giây)
        
        # Scores
        total_score: Điểm tổng kết
        rom_score: Điểm ROM trung bình
        stability_score: Điểm ổn định trung bình
        flow_score: Điểm mượt mà trung bình
        
        # Grade
        grade: Đánh giá (XUAT SAC / KHA / CAN CO GANG)
        grade_color: Màu grade (green/yellow/red)
        
        # Stats
        total_reps: Tổng số hiệp
        fatigue_level: Mức độ mệt mỏi cuối buổi
        
        # Calibration results
        calibrated_joints: Danh sách kết quả calibration
        primary_joint: Khớp chính đã sử dụng
        primary_max_angle: Góc max của khớp chính
        
        # Rep details
        rep_scores: Chi tiết điểm từng hiệp
        
        # Recommendations
        recommendations: Danh sách khuyến nghị
        
        # Timestamps
        start_time: Thời gian bắt đầu (ISO format)
        end_time: Thời gian kết thúc (ISO format)
    """
    # Session info
    session_id: str = ""
    exercise_name: str = ""
    duration_seconds: int = 0
    
    # Scores
    total_score: float = 0.0
    rom_score: float = 0.0
    stability_score: float = 0.0
    flow_score: float = 0.0
    
    # Grade
    grade: str = ""  # XUAT SAC, KHA, CAN CO GANG
    grade_color: str = "yellow"  # green, yellow, red
    
    # Stats
    total_reps: int = 0
    fatigue_level: str = "FRESH"
    
    # Calibration
    calibrated_joints: List[Dict] = field(default_factory=list)
    primary_joint: str = ""
    primary_max_angle: float = 0.0
    
    # Rep details
    rep_scores: List[Dict] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "exercise_name": self.exercise_name,
            "duration_seconds": self.duration_seconds,
            "total_score": round(self.total_score, 1),
            "rom_score": round(self.rom_score, 1),
            "stability_score": round(self.stability_score, 1),
            "flow_score": round(self.flow_score, 1),
            "grade": self.grade,
            "grade_color": self.grade_color,
            "total_reps": self.total_reps,
            "fatigue_level": self.fatigue_level,
            "calibrated_joints": self.calibrated_joints,
            "primary_joint": self.primary_joint,
            "primary_max_angle": round(self.primary_max_angle, 1),
            "rep_scores": self.rep_scores,
            "recommendations": self.recommendations,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


# ==================== COMPOSITE OUTPUT ====================

@dataclass
class EngineOutput:
    """
    Output tổng hợp từ Engine cho mỗi frame.
    
    Chỉ một trong các field phase sẽ có giá trị tùy theo phase hiện tại.
    
    Attributes:
        current_phase: Phase hiện tại (1-4)
        phase_name: Tên phase
        detection: Output Phase 1 (nếu phase 1)
        calibration: Output Phase 2 (nếu phase 2)
        sync: Output Phase 3 (nếu phase 3)
        final_report: Output Phase 4 (nếu phase 4)
        error: Thông báo lỗi (nếu có)
    """
    current_phase: int = 1
    phase_name: str = "detection"
    detection: Optional[Dict] = None
    calibration: Optional[Dict] = None
    sync: Optional[Dict] = None
    final_report: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = {
            "current_phase": self.current_phase,
            "phase_name": self.phase_name,
            "error": self.error
        }
        
        if self.detection:
            result["detection"] = self.detection
        if self.calibration:
            result["calibration"] = self.calibration
        if self.sync:
            result["sync"] = self.sync
        if self.final_report:
            result["final_report"] = self.final_report
        
        return result


# ==================== HELPER FUNCTIONS ====================

def get_direction_hint(user_angle: float, target_angle: float, tolerance: float = 10.0) -> str:
    """
    Xác định hướng cần điều chỉnh.
    
    Args:
        user_angle: Góc hiện tại
        target_angle: Góc mục tiêu
        tolerance: Ngưỡng chấp nhận (độ)
    
    Returns:
        str: "raise", "lower", "hold", hoặc "ok"
    """
    diff = user_angle - target_angle
    
    if abs(diff) <= tolerance:
        return "ok"
    elif diff < -tolerance:
        return "raise"  # Cần nâng cao hơn
    else:
        return "lower"  # Cần hạ thấp hơn


def get_feedback_text(error: float, target_angle: float) -> str:
    """
    Lấy text phản hồi dựa trên sai số.
    
    Args:
        error: Sai số (độ)
        target_angle: Góc mục tiêu
    
    Returns:
        str: Feedback text
    """
    if target_angle <= 0:
        return ""
    
    error_percent = (error / target_angle) * 100
    
    if error_percent < 10:
        return "TUYET VOI!"
    elif error_percent < 20:
        return "TOT!"
    elif error_percent < 30:
        return "KHA"
    else:
        return "DIEU CHINH!"


def get_grade(score: float) -> tuple:
    """
    Lấy grade dựa trên điểm số.
    
    Args:
        score: Điểm số (0-100)
    
    Returns:
        tuple: (grade_text, grade_color)
    """
    if score >= 80:
        return ("XUAT SAC", "green")
    elif score >= 60:
        return ("KHA", "yellow")
    else:
        return ("CAN CO GANG", "red")


# ==================== JOINT NAME MAPPING ====================

JOINT_NAMES_VI = {
    "left_shoulder": "Vai trai",
    "right_shoulder": "Vai phai",
    "left_elbow": "Khuyu tay trai",
    "right_elbow": "Khuyu tay phai",
    "left_knee": "Dau goi trai",
    "right_knee": "Dau goi phai",
    "left_hip": "Hong trai",
    "right_hip": "Hong phai",
    "left_wrist": "Co tay trai",
    "right_wrist": "Co tay phai",
    "left_ankle": "Mat ca trai",
    "right_ankle": "Mat ca phai",
}


def get_joint_name_vi(joint_type: str) -> str:
    """
    Lấy tên tiếng Việt của khớp.
    
    Args:
        joint_type: Loại khớp (e.g., "left_shoulder")
    
    Returns:
        str: Tên tiếng Việt
    """
    return JOINT_NAMES_VI.get(joint_type, joint_type)
