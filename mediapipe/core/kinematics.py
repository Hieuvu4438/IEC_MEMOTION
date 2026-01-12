"""
Kinematics Module for MEMOTION.

Cung cấp các hàm tính toán động học cơ bản:
- Tính góc giữa 3 điểm trong không gian 3D
- Tính góc các khớp cơ thể từ pose landmarks

Công thức toán học:
    Góc giữa 3 điểm A, B, C (với B là đỉnh góc):
    
    Vector BA = A - B
    Vector BC = C - B
    
    cos(θ) = (BA · BC) / (|BA| × |BC|)
    θ = arccos(cos(θ))

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, List, Optional
from enum import Enum
import numpy as np

from .data_types import Point3D, LandmarkSet, PoseLandmarkIndex


class JointType(Enum):
    """
    Định nghĩa các khớp cần theo dõi trong phục hồi chức năng.
    
    Mỗi khớp được định nghĩa bởi 3 landmarks:
    - Điểm đầu (proximal)
    - Điểm giữa (vertex) - đỉnh góc
    - Điểm cuối (distal)
    """
    # Khớp chi trên
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    
    # Khớp chi dưới
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    
    # Khớp cột sống
    SPINE = "spine"
    NECK = "neck"


@dataclass
class JointDefinition:
    """
    Định nghĩa một khớp bằng 3 landmark indices.
    
    Attributes:
        proximal: Index của điểm đầu (gần thân).
        vertex: Index của điểm đỉnh góc (khớp cần đo).
        distal: Index của điểm cuối (xa thân).
        name: Tên khớp.
        normal_range: Tuple (min, max) góc bình thường (degrees).
    """
    proximal: int
    vertex: int
    distal: int
    name: str
    normal_range: Tuple[float, float] = (0.0, 180.0)


# Bảng định nghĩa các khớp theo MediaPipe Pose Landmarks
JOINT_DEFINITIONS: Dict[JointType, JointDefinition] = {
    # ===== KHỚP CHI TRÊN =====
    # Khuỷu tay: Vai → Khuỷu → Cổ tay
    JointType.LEFT_ELBOW: JointDefinition(
        proximal=PoseLandmarkIndex.LEFT_SHOULDER,
        vertex=PoseLandmarkIndex.LEFT_ELBOW,
        distal=PoseLandmarkIndex.LEFT_WRIST,
        name="Khuỷu tay trái",
        normal_range=(0.0, 145.0)  # Góc gập khuỷu tay
    ),
    JointType.RIGHT_ELBOW: JointDefinition(
        proximal=PoseLandmarkIndex.RIGHT_SHOULDER,
        vertex=PoseLandmarkIndex.RIGHT_ELBOW,
        distal=PoseLandmarkIndex.RIGHT_WRIST,
        name="Khuỷu tay phải",
        normal_range=(0.0, 145.0)
    ),
    
    # Vai: Hông → Vai → Khuỷu (đo góc dang tay)
    JointType.LEFT_SHOULDER: JointDefinition(
        proximal=PoseLandmarkIndex.LEFT_HIP,
        vertex=PoseLandmarkIndex.LEFT_SHOULDER,
        distal=PoseLandmarkIndex.LEFT_ELBOW,
        name="Vai trái",
        normal_range=(0.0, 180.0)  # Góc dang vai
    ),
    JointType.RIGHT_SHOULDER: JointDefinition(
        proximal=PoseLandmarkIndex.RIGHT_HIP,
        vertex=PoseLandmarkIndex.RIGHT_SHOULDER,
        distal=PoseLandmarkIndex.RIGHT_ELBOW,
        name="Vai phải",
        normal_range=(0.0, 180.0)
    ),
    
    # ===== KHỚP CHI DƯỚI =====
    # Đầu gối: Hông → Đầu gối → Mắt cá
    JointType.LEFT_KNEE: JointDefinition(
        proximal=PoseLandmarkIndex.LEFT_HIP,
        vertex=PoseLandmarkIndex.LEFT_KNEE,
        distal=PoseLandmarkIndex.LEFT_ANKLE,
        name="Đầu gối trái",
        normal_range=(0.0, 140.0)  # Góc gập gối
    ),
    JointType.RIGHT_KNEE: JointDefinition(
        proximal=PoseLandmarkIndex.RIGHT_HIP,
        vertex=PoseLandmarkIndex.RIGHT_KNEE,
        distal=PoseLandmarkIndex.RIGHT_ANKLE,
        name="Đầu gối phải",
        normal_range=(0.0, 140.0)
    ),
    
    # Hông: Vai → Hông → Đầu gối (đo góc gập hông)
    JointType.LEFT_HIP: JointDefinition(
        proximal=PoseLandmarkIndex.LEFT_SHOULDER,
        vertex=PoseLandmarkIndex.LEFT_HIP,
        distal=PoseLandmarkIndex.LEFT_KNEE,
        name="Hông trái",
        normal_range=(0.0, 125.0)  # Góc gập hông
    ),
    JointType.RIGHT_HIP: JointDefinition(
        proximal=PoseLandmarkIndex.RIGHT_SHOULDER,
        vertex=PoseLandmarkIndex.RIGHT_HIP,
        distal=PoseLandmarkIndex.RIGHT_KNEE,
        name="Hông phải",
        normal_range=(0.0, 125.0)
    ),
}


def calculate_angle(
    point_a: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    point_b: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    point_c: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    use_3d: bool = True
) -> float:
    """
    Tính góc giữa 3 điểm trong không gian, với B là đỉnh góc.
    
    Công thức toán học:
        1. Tạo vector BA = A - B và BC = C - B
        2. Tính dot product: BA · BC = |BA| × |BC| × cos(θ)
        3. Suy ra: θ = arccos((BA · BC) / (|BA| × |BC|))
    
    Ý nghĩa nhân văn:
        Góc khớp là thước đo quan trọng nhất trong phục hồi chức năng.
        Việc tính chính xác góc giúp:
        - Đánh giá tiến triển của bệnh nhân
        - Đặt mục tiêu tập luyện phù hợp
        - Phát hiện sớm các vấn đề về vận động
    
    Args:
        point_a: Điểm đầu (proximal).
        point_b: Điểm đỉnh góc (vertex) - khớp cần đo.
        point_c: Điểm cuối (distal).
        use_3d: Nếu True, sử dụng cả 3 tọa độ (x, y, z).
                Nếu False, chỉ dùng (x, y) - hữu ích khi z không đáng tin cậy.
    
    Returns:
        float: Góc tính bằng độ (degrees), trong khoảng [0, 180].
        
    Raises:
        ValueError: Nếu các điểm trùng nhau (không thể tính góc).
        
    Example:
        >>> # Góc vuông 90 độ
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([0, 0, 0])  # Đỉnh góc
        >>> c = np.array([0, 1, 0])
        >>> angle = calculate_angle(a, b, c)
        >>> print(f"{angle:.1f}")  # 90.0
    """
    # Chuyển đổi về numpy array
    a = _to_numpy(point_a, use_3d)
    b = _to_numpy(point_b, use_3d)
    c = _to_numpy(point_c, use_3d)
    
    # Tạo vector từ đỉnh góc B
    # Vector BA: từ B đến A (hướng về proximal)
    # Vector BC: từ B đến C (hướng về distal)
    vector_ba = a - b
    vector_bc = c - b
    
    # Tính độ dài (norm) của các vector
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    
    # Kiểm tra trường hợp đặc biệt: điểm trùng nhau
    if norm_ba < 1e-10 or norm_bc < 1e-10:
        raise ValueError(
            "Không thể tính góc: các điểm quá gần nhau hoặc trùng nhau. "
            "Điều này có thể xảy ra khi MediaPipe không detect chính xác."
        )
    
    # Tính dot product
    # BA · BC = |BA| × |BC| × cos(θ)
    dot_product = np.dot(vector_ba, vector_bc)
    
    # Tính cosine của góc
    # cos(θ) = (BA · BC) / (|BA| × |BC|)
    cos_angle = dot_product / (norm_ba * norm_bc)
    
    # Clamp giá trị về [-1, 1] để tránh lỗi số học
    # (do sai số floating point, cos có thể > 1 hoặc < -1 một chút)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Tính góc bằng arccos
    angle_radians = np.arccos(cos_angle)
    
    # Chuyển từ radians sang degrees
    angle_degrees = np.degrees(angle_radians)
    
    return float(angle_degrees)


def calculate_angle_safe(
    point_a: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    point_b: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    point_c: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    use_3d: bool = True,
    default_angle: float = 0.0
) -> float:
    """
    Phiên bản an toàn của calculate_angle, trả về default nếu lỗi.
    
    Hữu ích khi xử lý real-time và cần tránh crash do dữ liệu xấu.
    
    Args:
        point_a, point_b, point_c: 3 điểm.
        use_3d: Sử dụng 3D hay 2D.
        default_angle: Giá trị trả về nếu không tính được.
        
    Returns:
        float: Góc hoặc default_angle.
    """
    try:
        return calculate_angle(point_a, point_b, point_c, use_3d)
    except (ValueError, TypeError):
        return default_angle


def calculate_joint_angle(
    landmarks: Union[np.ndarray, LandmarkSet],
    joint_type: JointType,
    use_3d: bool = True
) -> float:
    """
    Tính góc của một khớp cụ thể từ pose landmarks.
    
    Args:
        landmarks: Ma trận landmarks (N, 3) hoặc LandmarkSet.
        joint_type: Loại khớp cần tính (từ JointType enum).
        use_3d: Sử dụng tọa độ 3D hay 2D.
        
    Returns:
        float: Góc của khớp (degrees).
        
    Example:
        >>> # Tính góc khuỷu tay trái
        >>> angle = calculate_joint_angle(landmarks, JointType.LEFT_ELBOW)
    """
    # Chuyển đổi LandmarkSet sang numpy nếu cần
    if isinstance(landmarks, LandmarkSet):
        landmarks = landmarks.to_numpy()
    
    # Lấy định nghĩa khớp
    joint_def = JOINT_DEFINITIONS.get(joint_type)
    if joint_def is None:
        raise ValueError(f"Joint type {joint_type} not defined")
    
    # Trích xuất 3 điểm
    point_a = landmarks[joint_def.proximal]
    point_b = landmarks[joint_def.vertex]
    point_c = landmarks[joint_def.distal]
    
    return calculate_angle(point_a, point_b, point_c, use_3d)


def calculate_all_joint_angles(
    landmarks: Union[np.ndarray, LandmarkSet],
    use_3d: bool = True,
    joints: Optional[List[JointType]] = None
) -> Dict[JointType, float]:
    """
    Tính góc của tất cả các khớp (hoặc một subset).
    
    Args:
        landmarks: Pose landmarks.
        use_3d: Sử dụng 3D hay 2D.
        joints: Danh sách khớp cần tính. Nếu None, tính tất cả.
        
    Returns:
        Dict[JointType, float]: Mapping từ loại khớp đến góc.
    """
    if joints is None:
        joints = list(JOINT_DEFINITIONS.keys())
    
    results = {}
    for joint_type in joints:
        try:
            angle = calculate_joint_angle(landmarks, joint_type, use_3d)
            results[joint_type] = angle
        except (ValueError, IndexError):
            # Skip joints that can't be calculated
            pass
    
    return results


def _to_numpy(
    point: Union[np.ndarray, Point3D, Tuple[float, float, float]],
    use_3d: bool = True
) -> np.ndarray:
    """
    Chuyển đổi điểm về numpy array.
    
    Args:
        point: Điểm có thể ở nhiều format.
        use_3d: Nếu True, lấy [x, y, z]. Nếu False, lấy [x, y].
        
    Returns:
        np.ndarray: Array 2D hoặc 3D.
    """
    if isinstance(point, Point3D):
        arr = point.to_array()
    elif isinstance(point, (tuple, list)):
        arr = np.array(point, dtype=np.float32)
    else:
        arr = np.asarray(point, dtype=np.float32)
    
    if use_3d:
        return arr[:3] if len(arr) >= 3 else np.pad(arr, (0, 3 - len(arr)))
    else:
        return arr[:2]


def compute_angle_velocity(
    angles: List[float],
    timestamps_ms: List[int]
) -> List[float]:
    """
    Tính vận tốc góc (angular velocity) từ chuỗi góc theo thời gian.
    
    Ý nghĩa:
        Vận tốc góc cho biết tốc độ thay đổi của khớp.
        - Vận tốc cao: Chuyển động nhanh, mạnh
        - Vận tốc thấp: Chuyển động chậm, nhẹ nhàng
        - Đột ngột thay đổi: Có thể là dấu hiệu mất kiểm soát
    
    Args:
        angles: Danh sách góc (degrees).
        timestamps_ms: Danh sách timestamps (milliseconds).
        
    Returns:
        List[float]: Vận tốc góc (degrees/second).
    """
    if len(angles) < 2:
        return []
    
    velocities = []
    for i in range(1, len(angles)):
        dt_seconds = (timestamps_ms[i] - timestamps_ms[i-1]) / 1000.0
        if dt_seconds > 0:
            velocity = (angles[i] - angles[i-1]) / dt_seconds
            velocities.append(velocity)
        else:
            velocities.append(0.0)
    
    return velocities


def is_angle_in_normal_range(
    angle: float,
    joint_type: JointType
) -> bool:
    """
    Kiểm tra xem góc có nằm trong phạm vi bình thường không.
    
    Args:
        angle: Góc cần kiểm tra (degrees).
        joint_type: Loại khớp.
        
    Returns:
        bool: True nếu góc trong phạm vi bình thường.
    """
    joint_def = JOINT_DEFINITIONS.get(joint_type)
    if joint_def is None:
        return True  # Không có định nghĩa thì không validate
    
    min_angle, max_angle = joint_def.normal_range
    return min_angle <= angle <= max_angle