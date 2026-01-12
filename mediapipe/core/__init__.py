"""
Core Module for MEMOTION.

Chứa các thành phần cốt lõi:
- VisionDetector: Wrapper cho MediaPipe Tasks API
- Procrustes Analysis: Chuẩn hóa skeleton
- Kinematics: Tính toán góc khớp
- Data Types: Các data classes chuẩn hóa

Author: MEMOTION Team
Version: 1.1.0
"""

from .data_types import (
    Point3D,
    LandmarkType,
    LandmarkSet,
    DetectionResult,
    NormalizedSkeleton,
    ProcrustesResult,
    PoseLandmarkIndex,
)

from .detector import (
    VisionDetector,
    DetectorConfig,
)

from .procrustes import (
    normalize_skeleton,
    align_skeleton_to_reference,
    compute_procrustes_distance,
    compute_procrustes_similarity,
    extract_core_landmarks,
)

from .kinematics import (
    JointType,
    JointDefinition,
    JOINT_DEFINITIONS,
    calculate_angle,
    calculate_angle_safe,
    calculate_joint_angle,
    calculate_all_joint_angles,
    compute_angle_velocity,
    is_angle_in_normal_range,
)

__all__ = [
    # Data Types
    "Point3D",
    "LandmarkType", 
    "LandmarkSet",
    "DetectionResult",
    "NormalizedSkeleton",
    "ProcrustesResult",
    "PoseLandmarkIndex",
    # Detector
    "VisionDetector",
    "DetectorConfig",
    # Procrustes
    "normalize_skeleton",
    "align_skeleton_to_reference",
    "compute_procrustes_distance",
    "compute_procrustes_similarity",
    "extract_core_landmarks",
    # Kinematics
    "JointType",
    "JointDefinition",
    "JOINT_DEFINITIONS",
    "calculate_angle",
    "calculate_angle_safe",
    "calculate_joint_angle",
    "calculate_all_joint_angles",
    "compute_angle_velocity",
    "is_angle_in_normal_range",
]

__version__ = "1.1.0"