"""
Procrustes Analysis Module for MEMOTION.

Triển khai thuật toán Procrustes để chuẩn hóa skeleton,
loại bỏ sự khác biệt về vị trí, kích thước, và hướng quay.

Thuật toán gồm 3 bước:
1. Translation: Dịch chuyển centroid về gốc tọa độ
2. Scaling: Chuẩn hóa kích thước về unit norm
3. Rotation: Xoay để minimize khoảng cách với reference

Author: MEMOTION Team
Version: 1.0.0
"""

from typing import Optional, Tuple, List
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes as scipy_procrustes

from .data_types import (
    NormalizedSkeleton,
    ProcrustesResult,
    LandmarkSet,
    PoseLandmarkIndex,
)


def extract_core_landmarks(
    landmarks: np.ndarray,
    indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Trích xuất các landmarks chính cho so sánh.
    
    Loại bỏ các điểm nhiễu như ngón tay, mặt để tập trung
    vào các khớp chính của cơ thể.
    
    Args:
        landmarks: Ma trận landmarks đầy đủ, shape (N, 3).
        indices: Danh sách chỉ số landmarks cần trích xuất.
                 Mặc định sử dụng PoseLandmarkIndex.CORE_LANDMARKS.
    
    Returns:
        np.ndarray: Ma trận landmarks đã lọc, shape (M, 3) với M <= N.
    """
    if indices is None:
        indices = PoseLandmarkIndex.CORE_LANDMARKS
    
    if landmarks.shape[0] == 0:
        return landmarks
    
    valid_indices = [i for i in indices if i < landmarks.shape[0]]
    
    return landmarks[valid_indices]


def compute_centroid(landmarks: np.ndarray) -> np.ndarray:
    """
    Tính centroid (tâm hình học) của tập landmarks.
    
    Args:
        landmarks: Ma trận landmarks, shape (N, 3).
        
    Returns:
        np.ndarray: Centroid, shape (3,).
    """
    if landmarks.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    return np.mean(landmarks, axis=0)


def compute_scale(landmarks: np.ndarray) -> float:
    """
    Tính hệ số scale (Frobenius norm) của landmarks.
    
    Args:
        landmarks: Ma trận landmarks đã center, shape (N, 3).
        
    Returns:
        float: Frobenius norm của ma trận.
    """
    if landmarks.shape[0] == 0:
        return 1.0
    return np.linalg.norm(landmarks)


def translate_to_origin(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dịch chuyển landmarks để centroid ở gốc tọa độ.
    
    Args:
        landmarks: Ma trận landmarks, shape (N, 3).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Landmarks đã dịch chuyển
            - Centroid gốc
    """
    centroid = compute_centroid(landmarks)
    translated = landmarks - centroid
    return translated, centroid


def normalize_scale(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Chuẩn hóa kích thước landmarks về unit norm.
    
    Args:
        landmarks: Ma trận landmarks đã center, shape (N, 3).
        
    Returns:
        Tuple[np.ndarray, float]:
            - Landmarks đã chuẩn hóa scale
            - Hệ số scale ban đầu
    """
    scale = compute_scale(landmarks)
    if scale < 1e-10:
        return landmarks, 1.0
    normalized = landmarks / scale
    return normalized, scale


def compute_optimal_rotation(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Tính ma trận rotation tối ưu để căn chỉnh source với target.
    
    Sử dụng Orthogonal Procrustes để tìm rotation matrix R
    sao cho minimize ||source @ R - target||_F.
    
    Args:
        source: Ma trận source landmarks (đã normalize), shape (N, 3).
        target: Ma trận target landmarks (đã normalize), shape (N, 3).
        
    Returns:
        np.ndarray: Ma trận rotation, shape (3, 3).
    """
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.eye(3, dtype=np.float32)
    
    R, _ = orthogonal_procrustes(source, target)
    return R.astype(np.float32)


def normalize_skeleton(
    skeleton: np.ndarray,
    use_core_landmarks: bool = True
) -> NormalizedSkeleton:
    """
    Chuẩn hóa skeleton về dạng canonical (centered, unit scale).
    
    Thực hiện Translation và Scaling mà KHÔNG xoay.
    Dùng để tạo representation chuẩn trước khi so sánh.
    
    Args:
        skeleton: Ma trận landmarks, shape (N, 3).
        use_core_landmarks: Nếu True, chỉ sử dụng core landmarks.
        
    Returns:
        NormalizedSkeleton: Skeleton đã chuẩn hóa.
    """
    if skeleton.shape[0] == 0:
        return NormalizedSkeleton(
            landmarks=skeleton,
            centroid=np.zeros(3),
            scale=1.0,
            rotation_matrix=np.eye(3),
            original_landmarks=skeleton.copy()
        )
    
    working_skeleton = skeleton.copy()
    if use_core_landmarks and skeleton.shape[0] >= 33:
        working_skeleton = extract_core_landmarks(skeleton)
    
    translated, centroid = translate_to_origin(working_skeleton)
    normalized, scale = normalize_scale(translated)
    
    return NormalizedSkeleton(
        landmarks=normalized,
        centroid=centroid,
        scale=scale,
        rotation_matrix=np.eye(3),
        original_landmarks=skeleton.copy()
    )


def align_skeleton_to_reference(
    target_skeleton: np.ndarray,
    reference_skeleton: np.ndarray,
    use_core_landmarks: bool = True
) -> ProcrustesResult:
    """
    Căn chỉnh target skeleton theo reference skeleton.
    
    Thực hiện đầy đủ Procrustes Analysis:
    1. Center cả hai về gốc
    2. Normalize scale
    3. Tìm optimal rotation
    4. Tính disparity
    
    Args:
        target_skeleton: Skeleton cần căn chỉnh, shape (N, 3).
        reference_skeleton: Skeleton tham chiếu, shape (N, 3).
        use_core_landmarks: Nếu True, chỉ sử dụng core landmarks.
        
    Returns:
        ProcrustesResult: Kết quả alignment bao gồm disparity.
        
    Raises:
        ValueError: Nếu hai skeleton có số landmarks khác nhau.
    """
    target_work = target_skeleton.copy()
    ref_work = reference_skeleton.copy()
    
    if use_core_landmarks:
        if target_skeleton.shape[0] >= 33:
            target_work = extract_core_landmarks(target_skeleton)
        if reference_skeleton.shape[0] >= 33:
            ref_work = extract_core_landmarks(reference_skeleton)
    
    if target_work.shape != ref_work.shape:
        raise ValueError(
            f"Skeleton shapes must match. "
            f"Got target {target_work.shape} vs reference {ref_work.shape}"
        )
    
    if target_work.shape[0] == 0:
        return ProcrustesResult(
            aligned_skeleton=NormalizedSkeleton(
                landmarks=np.array([]),
                centroid=np.zeros(3),
                scale=1.0,
                rotation_matrix=np.eye(3)
            ),
            disparity=0.0,
            transformation={}
        )
    
    # Step 1 & 2: Normalize cả hai
    target_centered, target_centroid = translate_to_origin(target_work)
    target_normalized, target_scale = normalize_scale(target_centered)
    
    ref_centered, ref_centroid = translate_to_origin(ref_work)
    ref_normalized, ref_scale = normalize_scale(ref_centered)
    
    # Step 3: Tìm optimal rotation
    rotation_matrix = compute_optimal_rotation(target_normalized, ref_normalized)
    
    # Áp dụng rotation
    aligned = target_normalized @ rotation_matrix
    
    # Step 4: Tính disparity (sum of squared differences)
    disparity = np.sum((aligned - ref_normalized) ** 2)
    
    result = ProcrustesResult(
        aligned_skeleton=NormalizedSkeleton(
            landmarks=aligned,
            centroid=target_centroid,
            scale=target_scale,
            rotation_matrix=rotation_matrix,
            original_landmarks=target_skeleton.copy()
        ),
        disparity=float(disparity),
        transformation={
            "target_centroid": target_centroid.tolist(),
            "target_scale": float(target_scale),
            "ref_centroid": ref_centroid.tolist(),
            "ref_scale": float(ref_scale),
            "rotation_matrix": rotation_matrix.tolist(),
        }
    )
    
    return result


def compute_procrustes_distance(
    skeleton_a: np.ndarray,
    skeleton_b: np.ndarray,
    use_core_landmarks: bool = True
) -> float:
    """
    Tính khoảng cách Procrustes giữa hai skeleton.
    
    Đây là metric để so sánh độ giống nhau của hai tư thế,
    đã loại bỏ ảnh hưởng của vị trí, kích thước, hướng quay.
    
    Args:
        skeleton_a: Skeleton thứ nhất, shape (N, 3).
        skeleton_b: Skeleton thứ hai, shape (N, 3).
        use_core_landmarks: Nếu True, chỉ sử dụng core landmarks.
        
    Returns:
        float: Khoảng cách Procrustes (0 = giống hệt, càng lớn = càng khác).
    """
    result = align_skeleton_to_reference(
        skeleton_a, skeleton_b, use_core_landmarks
    )
    return result.disparity


def compute_procrustes_similarity(
    skeleton_a: np.ndarray,
    skeleton_b: np.ndarray,
    use_core_landmarks: bool = True
) -> float:
    """
    Tính độ tương đồng Procrustes giữa hai skeleton.
    
    Trả về giá trị từ 0 đến 1, với 1 là giống hệt hoàn toàn.
    
    Args:
        skeleton_a: Skeleton thứ nhất.
        skeleton_b: Skeleton thứ hai.
        use_core_landmarks: Nếu True, chỉ sử dụng core landmarks.
        
    Returns:
        float: Độ tương đồng (0-1).
    """
    distance = compute_procrustes_distance(
        skeleton_a, skeleton_b, use_core_landmarks
    )
    # Chuyển đổi distance sang similarity score
    # Sử dụng exponential decay
    similarity = np.exp(-distance * 10)  # Scale factor có thể điều chỉnh
    return float(np.clip(similarity, 0.0, 1.0))


def scipy_procrustes_wrapper(
    skeleton_a: np.ndarray,
    skeleton_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Wrapper cho scipy.spatial.procrustes.
    
    Tiện lợi để so sánh kết quả với implementation tự viết.
    
    Args:
        skeleton_a: Ma trận thứ nhất, shape (N, 3).
        skeleton_b: Ma trận thứ hai, shape (N, 3).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - skeleton_a đã standardized
            - skeleton_b đã standardized  
            - disparity
    """
    mtx1, mtx2, disparity = scipy_procrustes(skeleton_a, skeleton_b)
    return mtx1, mtx2, disparity


def apply_transformation(
    skeleton: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    rotation: np.ndarray
) -> np.ndarray:
    """
    Áp dụng biến đổi Procrustes lên skeleton.
    
    Hữu ích khi muốn biến đổi skeleton mới sử dụng
    tham số từ lần alignment trước.
    
    Args:
        skeleton: Skeleton cần biến đổi, shape (N, 3).
        centroid: Centroid để dịch chuyển.
        scale: Hệ số scale.
        rotation: Ma trận rotation, shape (3, 3).
        
    Returns:
        np.ndarray: Skeleton đã biến đổi.
    """
    centered = skeleton - centroid
    scaled = centered / scale if scale > 1e-10 else centered
    rotated = scaled @ rotation
    return rotated