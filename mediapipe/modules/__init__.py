"""
Modules Package for MEMOTION.

Chứa các module chức năng:
- calibration: Safe-Max Calibration cho người già
- target_generator: Cá nhân hóa mục tiêu bài tập
- video_engine: Smart Video Player
- pain_detection: Nhận diện đau qua FACS
- scoring: Ma trận chấm điểm đa chiều

Author: MEMOTION Team
Version: 1.2.0
"""

from .calibration import (
    SafeMaxCalibrator,
    CalibrationState,
    JointCalibrationData,
    UserProfile,
)

from .target_generator import (
    rescale_reference_motion,
    rescale_multi_joint_motion,
    compute_scale_factor,
    compute_target_at_time,
    compare_with_target,
    print_comparison_report,
    RescaledMotion,
)

from .video_engine import (
    VideoEngine,
    VideoInfo,
    PlaybackState,
    PlaybackStatus,
    SyncedVideoPlayer,
)

from .pain_detection import (
    PainDetector,
    PainLevel,
    PainEvent,
    PainAnalysisResult,
)

from .scoring import (
    HealthScorer,
    FatigueLevel,
    RepScore,
    SessionReport,
    calculate_jerk,
    calculate_center_of_mass,
)

__all__ = [
    # Calibration
    "SafeMaxCalibrator",
    "CalibrationState",
    "JointCalibrationData",
    "UserProfile",
    # Target Generator
    "rescale_reference_motion",
    "rescale_multi_joint_motion",
    "compute_scale_factor",
    "compute_target_at_time",
    "compare_with_target",
    "print_comparison_report",
    "RescaledMotion",
    # Video Engine
    "VideoEngine",
    "VideoInfo",
    "PlaybackState",
    "PlaybackStatus",
    "SyncedVideoPlayer",
    # Pain Detection
    "PainDetector",
    "PainLevel",
    "PainEvent",
    "PainAnalysisResult",
    # Scoring
    "HealthScorer",
    "FatigueLevel",
    "RepScore",
    "SessionReport",
    "calculate_jerk",
    "calculate_center_of_mass",
]