"""
Data Types Module for MEMOTION.

Chứa các Data Classes và Type Definitions chuẩn hóa
để dễ dàng chuyển đổi sang Flutter/Dart sau này.

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import numpy as np


class LandmarkType(Enum):
    """Enum định nghĩa loại landmark."""
    POSE = auto()
    FACE = auto()
    HAND_LEFT = auto()
    HAND_RIGHT = auto()


@dataclass(frozen=True)
class Point3D:
    """
    Đại diện một điểm trong không gian 3D.
    
    Attributes:
        x: Tọa độ X (normalized 0-1 hoặc world coordinates).
        y: Tọa độ Y (normalized 0-1 hoặc world coordinates).
        z: Tọa độ Z (depth, normalized hoặc world coordinates).
        visibility: Độ tin cậy của điểm (0-1), None nếu không áp dụng.
        presence: Xác suất điểm tồn tại trong frame (0-1).
    """
    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    presence: Optional[float] = None
    
    def to_array(self) -> np.ndarray:
        """Chuyển đổi sang numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    def to_2d(self) -> Tuple[float, float]:
        """Lấy tọa độ 2D (x, y)."""
        return (self.x, self.y)


@dataclass
class LandmarkSet:
    """
    Tập hợp các landmarks của một loại (pose/face/hand).
    
    Attributes:
        landmarks: Danh sách các điểm Point3D.
        landmark_type: Loại landmark (POSE, FACE, etc.).
        timestamp_ms: Timestamp của frame (milliseconds).
    """
    landmarks: List[Point3D]
    landmark_type: LandmarkType
    timestamp_ms: int = 0
    
    def __len__(self) -> int:
        return len(self.landmarks)
    
    def to_numpy(self) -> np.ndarray:
        """
        Chuyển đổi toàn bộ landmarks sang numpy array.
        
        Returns:
            np.ndarray: Ma trận shape (N, 3) với N là số landmarks.
        """
        if not self.landmarks:
            return np.array([], dtype=np.float32).reshape(0, 3)
        return np.array([lm.to_array() for lm in self.landmarks], dtype=np.float32)
    
    def get_visibility_mask(self, threshold: float = 0.5) -> np.ndarray:
        """
        Tạo mask cho các landmarks có visibility cao.
        
        Args:
            threshold: Ngưỡng visibility tối thiểu.
            
        Returns:
            np.ndarray: Boolean mask shape (N,).
        """
        mask = []
        for lm in self.landmarks:
            if lm.visibility is not None:
                mask.append(lm.visibility >= threshold)
            else:
                mask.append(True)
        return np.array(mask, dtype=bool)


@dataclass
class DetectionResult:
    """
    Kết quả detection từ một frame.
    
    Attributes:
        pose_landmarks: Landmarks của pose (33 điểm cho MediaPipe).
        face_landmarks: Landmarks của khuôn mặt (478 điểm).
        pose_world_landmarks: Pose landmarks trong world coordinates.
        frame_width: Chiều rộng frame gốc.
        frame_height: Chiều cao frame gốc.
        timestamp_ms: Timestamp của frame.
        is_valid: True nếu detection thành công.
        error_message: Thông báo lỗi nếu có.
    """
    pose_landmarks: Optional[LandmarkSet] = None
    face_landmarks: Optional[LandmarkSet] = None
    pose_world_landmarks: Optional[LandmarkSet] = None
    frame_width: int = 0
    frame_height: int = 0
    timestamp_ms: int = 0
    is_valid: bool = False
    error_message: Optional[str] = None
    
    def has_pose(self) -> bool:
        """Kiểm tra có pose landmarks không."""
        return self.pose_landmarks is not None and len(self.pose_landmarks) > 0
    
    def has_face(self) -> bool:
        """Kiểm tra có face landmarks không."""
        return self.face_landmarks is not None and len(self.face_landmarks) > 0


@dataclass
class NormalizedSkeleton:
    """
    Skeleton đã được chuẩn hóa qua Procrustes Analysis.
    
    Attributes:
        landmarks: Ma trận landmarks đã chuẩn hóa (N, 3).
        centroid: Tâm của skeleton trước chuẩn hóa.
        scale: Hệ số scale đã áp dụng.
        rotation_matrix: Ma trận rotation đã áp dụng (3, 3).
        original_landmarks: Landmarks gốc trước chuẩn hóa.
    """
    landmarks: np.ndarray
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    original_landmarks: Optional[np.ndarray] = None


@dataclass
class ProcrustesResult:
    """
    Kết quả của Procrustes Analysis.
    
    Attributes:
        aligned_skeleton: Skeleton đã căn chỉnh theo reference.
        disparity: Khoảng cách Procrustes (0 = hoàn toàn khớp).
        transformation: Dictionary chứa các tham số biến đổi.
    """
    aligned_skeleton: NormalizedSkeleton
    disparity: float
    transformation: dict = field(default_factory=dict)


# Định nghĩa các pose landmark indices quan trọng (MediaPipe Pose)
class PoseLandmarkIndex:
    """
    Chỉ số các landmarks quan trọng trong MediaPipe Pose.
    Tổng cộng 33 landmarks.
    """
    # Face
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    
    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    
    # Danh sách các landmarks chính cho so sánh tư thế
    # (Loại bỏ các điểm trên mặt và ngón tay để giảm noise)
    CORE_LANDMARKS = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    ]