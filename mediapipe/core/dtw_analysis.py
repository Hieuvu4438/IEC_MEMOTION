"""
DTW Analysis Module for MEMOTION.

Triển khai Weighted Dynamic Time Warping để so sánh nhịp điệu
chuyển động giữa người dùng và video mẫu.

Tại sao cần DTW thay vì so sánh trực tiếp?
    - Người già di chuyển với tốc độ khác nhau
    - Có thể dừng lại giữa chừng
    - DTW "kéo giãn" thời gian để tìm sự tương đồng tối ưu

Weighted DTW cho phép:
    - Tập trung vào khớp chính (trọng số cao)
    - Giảm ảnh hưởng của khớp nhiễu (trọng số thấp)
    
Ví dụ: Khi tập giơ tay
    - Vai: weight = 1.0 (quan trọng nhất)
    - Khuỷu: weight = 0.7
    - Đầu gối: weight = 0.1 (không liên quan)

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter1d

from .kinematics import JointType


@dataclass
class DTWResult:
    """
    Kết quả phân tích DTW.
    
    Attributes:
        distance: Khoảng cách DTW (càng nhỏ càng tốt).
        normalized_distance: Distance chuẩn hóa theo độ dài chuỗi.
        path: Đường đi tối ưu [(i1, j1), (i2, j2), ...].
        similarity_score: Điểm tương đồng (0-100%).
        rhythm_quality: Đánh giá nhịp điệu ("excellent", "good", "fair", "poor").
        details: Chi tiết phân tích.
    """
    distance: float
    normalized_distance: float
    path: List[Tuple[int, int]]
    similarity_score: float
    rhythm_quality: str
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


def preprocess_sequence(
    sequence: Union[List[float], np.ndarray],
    smooth_window: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    Tiền xử lý chuỗi trước khi tính DTW.
    
    Các bước:
        1. Làm mượt bằng moving average (giảm nhiễu)
        2. Chuẩn hóa về [0, 1] (để so sánh công bằng)
    
    Args:
        sequence: Chuỗi góc gốc.
        smooth_window: Kích thước cửa sổ làm mượt.
        normalize: Có chuẩn hóa không.
        
    Returns:
        np.ndarray: Chuỗi đã xử lý.
    """
    arr = np.array(sequence, dtype=np.float64)
    
    if len(arr) == 0:
        return arr
    
    # Làm mượt
    if smooth_window > 1 and len(arr) >= smooth_window:
        arr = uniform_filter1d(arr, size=smooth_window, mode='nearest')
    
    # Chuẩn hóa về [0, 1]
    if normalize:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val)
    
    return arr


def compute_dtw_distance(
    seq1: Union[List[float], np.ndarray],
    seq2: Union[List[float], np.ndarray],
    radius: int = 1
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Tính khoảng cách DTW giữa 2 chuỗi 1D.
    
    Sử dụng FastDTW nếu có, fallback về implementation đơn giản.
    
    Args:
        seq1: Chuỗi thứ nhất.
        seq2: Chuỗi thứ hai.
        radius: Bán kính tìm kiếm cho FastDTW.
        
    Returns:
        Tuple[distance, path]: Khoảng cách và đường đi.
    """
    arr1 = np.array(seq1).reshape(-1, 1)
    arr2 = np.array(seq2).reshape(-1, 1)
    
    if len(arr1) == 0 or len(arr2) == 0:
        return 0.0, []
    
    if FASTDTW_AVAILABLE:
        distance, path = fastdtw(arr1, arr2, radius=radius, dist=euclidean)
    else:
        # Fallback: Simple DTW implementation
        distance, path = _simple_dtw(arr1.flatten(), arr2.flatten())
    
    return float(distance), list(path)


def _simple_dtw(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Implementation DTW đơn giản khi không có fastdtw.
    
    Độ phức tạp: O(n*m) với n, m là độ dài 2 chuỗi.
    """
    n, m = len(seq1), len(seq2)
    
    # Ma trận chi phí tích lũy
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Điền ma trận
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # Insertion
                dtw_matrix[i, j-1],     # Deletion
                dtw_matrix[i-1, j-1]    # Match
            )
    
    # Backtrack để tìm đường đi
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (dtw_matrix[i-1, j-1], i-1, j-1),
                (dtw_matrix[i-1, j], i-1, j),
                (dtw_matrix[i, j-1], i, j-1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    return dtw_matrix[n, m], path


def compute_weighted_dtw(
    user_sequences: Dict[JointType, List[float]],
    ref_sequences: Dict[JointType, List[float]],
    weights: Dict[JointType, float],
    preprocess: bool = True
) -> DTWResult:
    """
    Tính Weighted DTW cho nhiều khớp.
    
    Công thức:
        Total Distance = Σ (weight_i × dtw_distance_i) / Σ weight_i
    
    Ý nghĩa của weights:
        - Weight cao = khớp quan trọng, ảnh hưởng lớn đến điểm
        - Weight thấp = khớp phụ, có thể "tha thứ" sai lệch
        
    Ví dụ cho bài tập giơ tay:
        weights = {
            JointType.LEFT_SHOULDER: 1.0,   # Quan trọng nhất
            JointType.LEFT_ELBOW: 0.5,      # Quan trọng vừa
            JointType.LEFT_KNEE: 0.1,       # Không liên quan
        }
    
    Args:
        user_sequences: Dict mapping JointType → chuỗi góc của user.
        ref_sequences: Dict mapping JointType → chuỗi góc mẫu.
        weights: Dict mapping JointType → trọng số (0-1).
        preprocess: Có tiền xử lý chuỗi không.
        
    Returns:
        DTWResult: Kết quả phân tích.
    """
    if not user_sequences or not ref_sequences:
        return DTWResult(
            distance=0.0,
            normalized_distance=0.0,
            path=[],
            similarity_score=100.0,
            rhythm_quality="unknown"
        )
    
    total_weighted_distance = 0.0
    total_weight = 0.0
    joint_details = {}
    combined_path = []
    
    for joint_type, user_seq in user_sequences.items():
        if joint_type not in ref_sequences:
            continue
        
        ref_seq = ref_sequences[joint_type]
        weight = weights.get(joint_type, 0.5)  # Default weight
        
        if weight < 1e-6:
            continue  # Skip joints with zero weight
        
        # Tiền xử lý
        if preprocess:
            user_processed = preprocess_sequence(user_seq)
            ref_processed = preprocess_sequence(ref_seq)
        else:
            user_processed = np.array(user_seq)
            ref_processed = np.array(ref_seq)
        
        # Tính DTW cho khớp này
        distance, path = compute_dtw_distance(user_processed, ref_processed)
        
        # Chuẩn hóa theo độ dài
        seq_len = max(len(user_seq), len(ref_seq))
        normalized = distance / seq_len if seq_len > 0 else 0
        
        # Tích lũy
        total_weighted_distance += weight * normalized
        total_weight += weight
        
        joint_details[joint_type.value] = {
            "distance": distance,
            "normalized_distance": normalized,
            "weight": weight,
            "weighted_contribution": weight * normalized,
            "path_length": len(path),
        }
        
        if not combined_path:
            combined_path = path
    
    # Tính tổng hợp
    if total_weight > 0:
        final_distance = total_weighted_distance / total_weight
    else:
        final_distance = 0.0
    
    # Chuyển đổi distance sang similarity score
    # Sử dụng hàm exponential decay
    # distance = 0 → similarity = 100%
    # distance = 1 → similarity ≈ 37%
    similarity_score = 100.0 * np.exp(-final_distance * 3)
    similarity_score = float(np.clip(similarity_score, 0, 100))
    
    # Đánh giá nhịp điệu
    rhythm_quality = _evaluate_rhythm_quality(similarity_score)
    
    return DTWResult(
        distance=total_weighted_distance,
        normalized_distance=final_distance,
        path=combined_path,
        similarity_score=similarity_score,
        rhythm_quality=rhythm_quality,
        details={"joints": joint_details, "total_weight": total_weight}
    )


def compute_single_joint_dtw(
    user_sequence: List[float],
    ref_sequence: List[float],
    preprocess: bool = True
) -> DTWResult:
    """
    Tính DTW cho một khớp duy nhất.
    
    Phiên bản đơn giản của compute_weighted_dtw.
    
    Args:
        user_sequence: Chuỗi góc của user.
        ref_sequence: Chuỗi góc mẫu.
        preprocess: Có tiền xử lý không.
        
    Returns:
        DTWResult: Kết quả phân tích.
    """
    if not user_sequence or not ref_sequence:
        return DTWResult(
            distance=0.0,
            normalized_distance=0.0,
            path=[],
            similarity_score=100.0,
            rhythm_quality="unknown"
        )
    
    if preprocess:
        user_processed = preprocess_sequence(user_sequence)
        ref_processed = preprocess_sequence(ref_sequence)
    else:
        user_processed = np.array(user_sequence)
        ref_processed = np.array(ref_sequence)
    
    distance, path = compute_dtw_distance(user_processed, ref_processed)
    
    seq_len = max(len(user_sequence), len(ref_sequence))
    normalized = distance / seq_len if seq_len > 0 else 0
    
    similarity_score = 100.0 * np.exp(-normalized * 3)
    similarity_score = float(np.clip(similarity_score, 0, 100))
    
    rhythm_quality = _evaluate_rhythm_quality(similarity_score)
    
    return DTWResult(
        distance=distance,
        normalized_distance=normalized,
        path=path,
        similarity_score=similarity_score,
        rhythm_quality=rhythm_quality
    )


def _evaluate_rhythm_quality(similarity_score: float) -> str:
    """
    Đánh giá chất lượng nhịp điệu từ điểm tương đồng.
    
    Args:
        similarity_score: Điểm tương đồng (0-100).
        
    Returns:
        str: "excellent", "good", "fair", hoặc "poor".
    """
    if similarity_score >= 85:
        return "excellent"
    elif similarity_score >= 70:
        return "good"
    elif similarity_score >= 50:
        return "fair"
    else:
        return "poor"


def get_rhythm_feedback(dtw_result: DTWResult) -> str:
    """
    Tạo phản hồi về nhịp điệu cho người dùng.
    
    Args:
        dtw_result: Kết quả DTW.
        
    Returns:
        str: Thông điệp phản hồi thân thiện.
    """
    quality = dtw_result.rhythm_quality
    score = dtw_result.similarity_score
    
    feedback_map = {
        "excellent": f"Tuyệt vời! Nhịp điệu rất mượt mà ({score:.0f}%)!",
        "good": f"Tốt lắm! Nhịp điệu khá ổn ({score:.0f}%).",
        "fair": f"Được rồi! Cố gắng đều tay hơn nhé ({score:.0f}%).",
        "poor": f"Không sao! Từ từ luyện tập sẽ tốt hơn ({score:.0f}%).",
        "unknown": "Đang phân tích..."
    }
    
    return feedback_map.get(quality, feedback_map["unknown"])


def analyze_speed_variation(
    timestamps: List[float],
    angles: List[float]
) -> Dict[str, float]:
    """
    Phân tích độ biến thiên tốc độ của chuyển động.
    
    Người già lý tưởng nên di chuyển đều đặn, không giật cục.
    
    Args:
        timestamps: Chuỗi timestamps (seconds).
        angles: Chuỗi góc tương ứng.
        
    Returns:
        Dict chứa các metrics về tốc độ.
    """
    if len(timestamps) < 2 or len(angles) < 2:
        return {"mean_velocity": 0, "velocity_std": 0, "smoothness": 1.0}
    
    ts = np.array(timestamps)
    ang = np.array(angles)
    
    # Tính vận tốc góc (degrees/second)
    dt = np.diff(ts)
    da = np.diff(ang)
    
    # Tránh chia cho 0
    dt = np.where(dt < 1e-6, 1e-6, dt)
    velocity = da / dt
    
    mean_vel = float(np.mean(np.abs(velocity)))
    std_vel = float(np.std(velocity))
    
    # Smoothness: 1 = rất mượt, 0 = rất giật
    # Dựa trên tỷ lệ std/mean
    if mean_vel > 1e-6:
        smoothness = 1.0 / (1.0 + std_vel / mean_vel)
    else:
        smoothness = 1.0
    
    return {
        "mean_velocity": mean_vel,
        "velocity_std": std_vel,
        "smoothness": float(smoothness),
        "max_velocity": float(np.max(np.abs(velocity))),
        "min_velocity": float(np.min(np.abs(velocity))),
    }


def create_exercise_weights(
    exercise_type: str
) -> Dict[JointType, float]:
    """
    Tạo bảng trọng số phù hợp cho từng loại bài tập.
    
    Args:
        exercise_type: "arm_raise", "squat", "bicep_curl", etc.
        
    Returns:
        Dict[JointType, float]: Bảng trọng số.
    """
    # Default: tất cả đều quan trọng như nhau
    default_weights = {jt: 0.5 for jt in JointType}
    
    exercise_weights = {
        "arm_raise": {
            JointType.LEFT_SHOULDER: 1.0,
            JointType.RIGHT_SHOULDER: 1.0,
            JointType.LEFT_ELBOW: 0.6,
            JointType.RIGHT_ELBOW: 0.6,
            JointType.LEFT_KNEE: 0.1,
            JointType.RIGHT_KNEE: 0.1,
            JointType.LEFT_HIP: 0.2,
            JointType.RIGHT_HIP: 0.2,
        },
        "squat": {
            JointType.LEFT_KNEE: 1.0,
            JointType.RIGHT_KNEE: 1.0,
            JointType.LEFT_HIP: 0.8,
            JointType.RIGHT_HIP: 0.8,
            JointType.LEFT_SHOULDER: 0.2,
            JointType.RIGHT_SHOULDER: 0.2,
            JointType.LEFT_ELBOW: 0.1,
            JointType.RIGHT_ELBOW: 0.1,
        },
        "bicep_curl": {
            JointType.LEFT_ELBOW: 1.0,
            JointType.RIGHT_ELBOW: 1.0,
            JointType.LEFT_SHOULDER: 0.5,
            JointType.RIGHT_SHOULDER: 0.5,
            JointType.LEFT_KNEE: 0.1,
            JointType.RIGHT_KNEE: 0.1,
            JointType.LEFT_HIP: 0.2,
            JointType.RIGHT_HIP: 0.2,
        },
    }
    
    weights = exercise_weights.get(exercise_type.lower(), {})
    
    # Merge với default
    result = default_weights.copy()
    result.update(weights)
    
    return result