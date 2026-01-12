"""
Modules Package for MEMOTION.

Chứa các module chức năng:
- calibration: Safe-Max Calibration cho người già
- target_generator: Cá nhân hóa mục tiêu bài tập

Author: MEMOTION Team
Version: 1.0.0
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
]