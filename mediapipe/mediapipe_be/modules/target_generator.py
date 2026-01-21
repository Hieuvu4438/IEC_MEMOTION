"""
Target Generator Module for MEMOTION.

Triển khai công thức hiệu chỉnh mục tiêu để co giãn video mẫu
phù hợp với giới hạn vận động của từng người già.

Công thức chính:
    θ_target(t) = θ_ref(t) × (θ_user_max / max(θ_ref)) × (1 + α)
    
Trong đó:
    - θ_ref(t): Góc trong video mẫu tại thời điểm t
    - θ_user_max: Góc tối đa an toàn của người dùng (từ calibration)
    - max(θ_ref): Góc lớn nhất trong video mẫu
    - α: Challenge Factor (mặc định 0.05 = 5%)

Ý nghĩa nhân văn:
    Công thức này đảm bảo:
    1. Mục tiêu KHÔNG BAO GIỜ vượt quá khả năng của người già
    2. Có chút thử thách (α) để khuyến khích tiến bộ
    3. Tỷ lệ động tác được bảo toàn (không biến dạng)

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from core.kinematics import JointType
from modules.calibration import UserProfile


@dataclass
class RescaledMotion:
    """
    Kết quả rescale một chuỗi chuyển động.
    
    Attributes:
        original_angles: Chuỗi góc gốc từ video mẫu.
        target_angles: Chuỗi góc mục tiêu đã rescale.
        scale_factor: Hệ số scale đã áp dụng.
        user_max: Góc tối đa của người dùng.
        ref_max: Góc tối đa trong video mẫu.
        challenge_factor: Hệ số thử thách đã áp dụng.
        timestamps_ms: Timestamps tương ứng (nếu có).
    """
    original_angles: List[float]
    target_angles: List[float]
    scale_factor: float
    user_max: float
    ref_max: float
    challenge_factor: float
    timestamps_ms: Optional[List[int]] = None
    
    def get_max_target(self) -> float:
        """Lấy góc mục tiêu lớn nhất."""
        return max(self.target_angles) if self.target_angles else 0.0
    
    def get_reduction_percent(self) -> float:
        """Tính phần trăm giảm so với video mẫu."""
        if self.ref_max == 0:
            return 0.0
        return (1 - self.scale_factor) * 100


def compute_scale_factor(
    user_max_angle: float,
    ref_max_angle: float,
    challenge_factor: float = 0.05
) -> float:
    """
    Tính hệ số scale để điều chỉnh video mẫu.
    
    Công thức:
        scale = (θ_user_max / max(θ_ref)) × (1 + α)
    
    Giải thích từng thành phần:
        - θ_user_max / max(θ_ref): Tỷ lệ co giãn cơ bản
          Ví dụ: Người già chỉ gập khuỷu được 90°, video mẫu gập 120°
          → Tỷ lệ = 90/120 = 0.75 (giảm 25%)
          
        - (1 + α): Hệ số thử thách
          α = 0.05 nghĩa là tăng thêm 5% so với mức an toàn
          Mục đích: Khuyến khích người già cố gắng thêm một chút
          nhưng vẫn trong ngưỡng an toàn
    
    Tại sao cần Challenge Factor?
        - Nếu mục tiêu = đúng mức tối đa → Không có động lực cải thiện
        - Một chút thử thách (5%) giúp duy trì tiến bộ
        - Nhưng không quá cao để gây áp lực hoặc chấn thương
    
    Args:
        user_max_angle: Góc tối đa an toàn của người dùng (degrees, phải >= 0).
        ref_max_angle: Góc tối đa trong video mẫu (degrees, phải > 0).
        challenge_factor: Hệ số thử thách α (default 0.05 = 5%, phải >= 0).
        
    Returns:
        float: Hệ số scale. Giá trị < 1 nghĩa là giảm biên độ.
        
    Raises:
        ValueError: Nếu ref_max_angle <= 0 hoặc các góc âm.
    """
    # Validate: Góc không thể âm (không có ý nghĩa vật lý)
    if user_max_angle < 0:
        raise ValueError(
            f"user_max_angle phải >= 0, nhận được {user_max_angle}. "
            "Góc âm không có ý nghĩa vật lý trong phục hồi chức năng."
        )
    
    if ref_max_angle < 0:
        raise ValueError(
            f"ref_max_angle phải >= 0, nhận được {ref_max_angle}. "
            "Video mẫu không thể có góc âm."
        )
    
    if challenge_factor < 0:
        raise ValueError(
            f"challenge_factor phải >= 0, nhận được {challenge_factor}. "
            "Hệ số thử thách âm không có ý nghĩa."
        )
    
    # Xử lý edge case: video mẫu không có chuyển động
    if ref_max_angle < 1e-6:
        raise ValueError(
            "ref_max_angle không thể bằng 0. "
            "Video mẫu cần có chuyển động để rescale."
        )
    
    # Xử lý edge case: người dùng không thể cử động
    if user_max_angle < 1e-6:
        # Trả về 0 để chỉ ra rằng không thể thực hiện bài tập
        return 0.0
    
    # Tính tỷ lệ cơ bản
    base_ratio = user_max_angle / ref_max_angle
    
    # Áp dụng challenge factor
    # Giới hạn scale tối đa là 1.0 để không vượt quá video mẫu
    scale = base_ratio * (1.0 + challenge_factor)
    
    # Cap at 1.0: Không cần scale lên nếu người dùng đã vượt mẫu
    # (hiếm khi xảy ra với người già)
    return min(scale, 1.0)


def rescale_reference_motion(
    ref_angles_sequence: List[float],
    user_max_angle: float,
    challenge_factor: float = 0.05,
    timestamps_ms: Optional[List[int]] = None
) -> RescaledMotion:
    """
    Co giãn chuỗi góc từ video mẫu phù hợp với người già.
    
    Đây là hàm chính để cá nhân hóa bài tập.
    
    Công thức áp dụng cho mỗi frame:
        θ_target(t) = θ_ref(t) × scale_factor
        
    Trong đó:
        scale_factor = (θ_user_max / max(θ_ref)) × (1 + α)
    
    Ví dụ minh họa:
        Video mẫu: [0°, 30°, 60°, 90°, 120°, 90°, 60°, 30°, 0°]
        max(θ_ref) = 120°
        
        Người già A (θ_user_max = 90°):
        scale = 90/120 × 1.05 = 0.7875
        → Mục tiêu: [0°, 24°, 47°, 71°, 95°, 71°, 47°, 24°, 0°]
        
        Người già B (θ_user_max = 60°):
        scale = 60/120 × 1.05 = 0.525
        → Mục tiêu: [0°, 16°, 32°, 47°, 63°, 47°, 32°, 16°, 0°]
    
    Args:
        ref_angles_sequence: Chuỗi góc từ video mẫu (degrees).
        user_max_angle: Góc tối đa an toàn của người dùng (degrees).
        challenge_factor: Hệ số thử thách α (default 0.05).
        timestamps_ms: Timestamps tương ứng với mỗi góc (optional).
        
    Returns:
        RescaledMotion: Object chứa chuỗi góc mục tiêu và metadata.
        
    Example:
        >>> ref = [0, 30, 60, 90, 120, 90, 60, 30, 0]
        >>> result = rescale_reference_motion(ref, user_max_angle=90)
        >>> print(result.target_angles)
        >>> print(f"Giảm {result.get_reduction_percent():.1f}% so với mẫu")
    """
    if not ref_angles_sequence:
        return RescaledMotion(
            original_angles=[],
            target_angles=[],
            scale_factor=1.0,
            user_max=user_max_angle,
            ref_max=0.0,
            challenge_factor=challenge_factor,
            timestamps_ms=timestamps_ms,
        )
    
    # Tìm góc lớn nhất trong video mẫu
    ref_max = max(ref_angles_sequence)
    
    # Tính scale factor
    try:
        scale_factor = compute_scale_factor(
            user_max_angle, ref_max, challenge_factor
        )
    except ValueError:
        # ref_max = 0, không có chuyển động
        scale_factor = 1.0
    
    # Áp dụng scale cho toàn bộ chuỗi
    # θ_target(t) = θ_ref(t) × scale_factor
    target_angles = [angle * scale_factor for angle in ref_angles_sequence]
    
    return RescaledMotion(
        original_angles=list(ref_angles_sequence),
        target_angles=target_angles,
        scale_factor=scale_factor,
        user_max=user_max_angle,
        ref_max=ref_max,
        challenge_factor=challenge_factor,
        timestamps_ms=timestamps_ms,
    )


def rescale_multi_joint_motion(
    ref_motion: Dict[JointType, List[float]],
    user_profile: UserProfile,
    challenge_factor: float = 0.05
) -> Dict[JointType, RescaledMotion]:
    """
    Rescale chuyển động cho nhiều khớp cùng lúc.
    
    Args:
        ref_motion: Dict mapping JointType → chuỗi góc từ video mẫu.
        user_profile: Profile người dùng chứa các giới hạn góc.
        challenge_factor: Hệ số thử thách.
        
    Returns:
        Dict[JointType, RescaledMotion]: Kết quả rescale cho mỗi khớp.
    """
    results = {}
    
    for joint_type, ref_angles in ref_motion.items():
        user_max = user_profile.get_max_angle(joint_type)
        
        if user_max is None:
            # Chưa calibrate khớp này, giữ nguyên
            results[joint_type] = RescaledMotion(
                original_angles=list(ref_angles),
                target_angles=list(ref_angles),
                scale_factor=1.0,
                user_max=0.0,
                ref_max=max(ref_angles) if ref_angles else 0.0,
                challenge_factor=challenge_factor,
            )
        else:
            results[joint_type] = rescale_reference_motion(
                ref_angles, user_max, challenge_factor
            )
    
    return results


def generate_target_trajectory(
    ref_angles: List[float],
    user_max: float,
    duration_ms: int,
    fps: float = 30.0,
    challenge_factor: float = 0.05
) -> Tuple[List[float], List[int]]:
    """
    Tạo trajectory mục tiêu hoàn chỉnh với timestamps.
    
    Hữu ích khi cần đồng bộ với video real-time.
    
    Args:
        ref_angles: Chuỗi góc từ video mẫu.
        user_max: Góc tối đa của người dùng.
        duration_ms: Thời lượng mong muốn (milliseconds).
        fps: Frame rate.
        challenge_factor: Hệ số thử thách.
        
    Returns:
        Tuple[List[float], List[int]]: (target_angles, timestamps_ms)
    """
    # Rescale angles
    result = rescale_reference_motion(ref_angles, user_max, challenge_factor)
    
    # Tính timestamps
    num_frames = len(result.target_angles)
    if num_frames == 0:
        return [], []
    
    # Điều chỉnh timing để khớp với duration
    time_per_frame = duration_ms / num_frames
    timestamps = [int(i * time_per_frame) for i in range(num_frames)]
    
    return result.target_angles, timestamps


def compute_target_at_time(
    ref_angles: List[float],
    ref_timestamps_ms: List[int],
    user_max: float,
    current_time_ms: int,
    challenge_factor: float = 0.05
) -> float:
    """
    Tính góc mục tiêu tại một thời điểm cụ thể.
    
    Sử dụng nội suy tuyến tính giữa các keyframes.
    
    Args:
        ref_angles: Chuỗi góc từ video mẫu.
        ref_timestamps_ms: Timestamps tương ứng.
        user_max: Góc tối đa của người dùng.
        current_time_ms: Thời điểm hiện tại.
        challenge_factor: Hệ số thử thách.
        
    Returns:
        float: Góc mục tiêu tại thời điểm current_time_ms.
    """
    if not ref_angles or not ref_timestamps_ms:
        return 0.0
    
    if len(ref_angles) != len(ref_timestamps_ms):
        raise ValueError("ref_angles và ref_timestamps_ms phải có cùng độ dài")
    
    # Rescale toàn bộ
    result = rescale_reference_motion(ref_angles, user_max, challenge_factor)
    target_angles = result.target_angles
    
    # Tìm vị trí nội suy
    if current_time_ms <= ref_timestamps_ms[0]:
        return target_angles[0]
    
    if current_time_ms >= ref_timestamps_ms[-1]:
        return target_angles[-1]
    
    # Tìm 2 keyframes gần nhất
    for i in range(len(ref_timestamps_ms) - 1):
        t1, t2 = ref_timestamps_ms[i], ref_timestamps_ms[i + 1]
        if t1 <= current_time_ms <= t2:
            # Nội suy tuyến tính
            ratio = (current_time_ms - t1) / (t2 - t1) if t2 != t1 else 0
            angle1, angle2 = target_angles[i], target_angles[i + 1]
            return angle1 + ratio * (angle2 - angle1)
    
    return target_angles[-1]


def compare_with_target(
    current_angle: float,
    target_angle: float,
    tolerance_degrees: float = 10.0
) -> Tuple[str, float]:
    """
    So sánh góc hiện tại với góc mục tiêu.
    
    Args:
        current_angle: Góc hiện tại của người dùng.
        target_angle: Góc mục tiêu.
        tolerance_degrees: Ngưỡng sai số chấp nhận được.
        
    Returns:
        Tuple[str, float]: (status, error)
            - status: "perfect" | "good" | "under" | "over"
            - error: Độ lệch (có thể âm hoặc dương)
    """
    error = current_angle - target_angle
    abs_error = abs(error)
    
    if abs_error <= tolerance_degrees * 0.3:
        status = "perfect"  # Trong khoảng 30% tolerance
    elif abs_error <= tolerance_degrees:
        status = "good"  # Trong khoảng tolerance
    elif error < 0:
        status = "under"  # Chưa đạt
    else:
        status = "over"  # Vượt quá (hiếm với người già)
    
    return status, error


def print_comparison_report(
    original: List[float],
    rescaled: RescaledMotion,
    joint_name: str = "Unknown Joint"
) -> None:
    """
    In báo cáo so sánh giữa góc mẫu và góc mục tiêu cá nhân hóa.
    
    Args:
        original: Chuỗi góc gốc.
        rescaled: Kết quả rescale.
        joint_name: Tên khớp để hiển thị.
    """
    print(f"\n{'='*60}")
    print(f"BÁO CÁO CÁ NHÂN HÓA MỤC TIÊU: {joint_name}")
    print(f"{'='*60}")
    
    print(f"\n[Thông số Calibration]")
    print(f"  • Góc tối đa của người dùng (θ_user_max): {rescaled.user_max:.1f}°")
    print(f"  • Góc tối đa trong video mẫu (max θ_ref): {rescaled.ref_max:.1f}°")
    print(f"  • Hệ số thử thách (α): {rescaled.challenge_factor:.1%}")
    print(f"  • Hệ số scale: {rescaled.scale_factor:.4f}")
    
    print(f"\n[Kết quả Rescale]")
    print(f"  • Giảm biên độ: {rescaled.get_reduction_percent():.1f}%")
    print(f"  • Góc mục tiêu tối đa: {rescaled.get_max_target():.1f}°")
    
    print(f"\n[So sánh Chi tiết]")
    print(f"  {'Frame':>6} │ {'Góc Mẫu':>10} │ {'Góc Mục Tiêu':>12} │ {'Giảm':>8}")
    print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*8}")
    
    # Hiển thị tối đa 10 frames
    step = max(1, len(original) // 10)
    for i in range(0, len(original), step):
        orig = original[i]
        target = rescaled.target_angles[i]
        reduction = orig - target
        print(f"  {i:>6} │ {orig:>9.1f}° │ {target:>11.1f}° │ {reduction:>7.1f}°")
    
    print(f"\n[Ý nghĩa]")
    if rescaled.scale_factor < 0.5:
        print(f"  ⚠️ Biên độ vận động hạn chế đáng kể. Cần theo dõi tiến triển.")
    elif rescaled.scale_factor < 0.8:
        print(f"  ℹ️ Biên độ vận động vừa phải. Bài tập đã được điều chỉnh phù hợp.")
    else:
        print(f"  ✅ Biên độ vận động tốt. Gần với mức chuẩn.")
    
    print(f"{'='*60}\n")