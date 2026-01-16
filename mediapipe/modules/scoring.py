"""
Scoring Module for MEMOTION.

Ma trận chấm điểm đa chiều đánh giá chất lượng tập luyện:
1. ROM Score: Mức độ đạt góc mục tiêu
2. Stability Score: Độ ổn định trong pha HOLD
3. Flow Score: Độ mượt mà (từ DTW)
4. Symmetry Score: Cân bằng trái-phải

Phân tích Jerk để phát hiện mệt mỏi:
    Jerk = d³x/dt³ (đạo hàm bậc 3 của vị trí)
    
    Ý nghĩa:
    - Jerk thấp = chuyển động mượt mà
    - Jerk cao = chuyển động giật, không kiểm soát
    - Jerk tăng dần qua các rep = dấu hiệu mệt mỏi

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import numpy as np

from core.kinematics import JointType
from core.synchronizer import MotionPhase
from core.dtw_analysis import DTWResult


class FatigueLevel(Enum):
    """Mức độ mệt mỏi."""
    FRESH = 0       # Khỏe, có thể tiếp tục
    LIGHT = 1       # Hơi mệt
    MODERATE = 2    # Mệt vừa, cần chú ý
    HEAVY = 3       # Rất mệt, nên nghỉ


@dataclass
class RepScore:
    """
    Điểm của một lần lặp (rep).
    
    Attributes:
        rep_number: Số thứ tự rep.
        rom_score: Điểm ROM (0-100).
        stability_score: Điểm ổn định (0-100).
        flow_score: Điểm mượt mà (0-100).
        symmetry_score: Điểm cân bằng (0-100).
        compensation_score: Điểm bù trừ - cao = ít bù trừ (0-100).
        total_score: Điểm tổng hợp (0-100).
        jerk_value: Giá trị Jerk.
        duration_ms: Thời gian thực hiện (ms).
        notes: Ghi chú.
        compensation_detected: Các loại bù trừ phát hiện được.
    """
    rep_number: int
    rom_score: float = 0.0
    stability_score: float = 0.0
    flow_score: float = 0.0
    symmetry_score: float = 0.0
    compensation_score: float = 100.0  # Mới: điểm cho việc không bù trừ
    total_score: float = 0.0
    jerk_value: float = 0.0
    duration_ms: int = 0
    notes: str = ""
    compensation_detected: List[str] = field(default_factory=list)  # Mới
    
    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "rom_score": round(self.rom_score, 1),
            "stability_score": round(self.stability_score, 1),
            "flow_score": round(self.flow_score, 1),
            "symmetry_score": round(self.symmetry_score, 1),
            "compensation_score": round(self.compensation_score, 1),
            "total_score": round(self.total_score, 1),
            "jerk_value": round(self.jerk_value, 4),
            "duration_ms": self.duration_ms,
            "notes": self.notes,
            "compensation_detected": self.compensation_detected,
        }


@dataclass
class SessionReport:
    """
    Báo cáo tổng hợp buổi tập.
    
    Attributes:
        session_id: ID buổi tập.
        start_time: Thời gian bắt đầu.
        end_time: Thời gian kết thúc.
        exercise_name: Tên bài tập.
        total_reps: Tổng số rep.
        rep_scores: Điểm từng rep.
        average_scores: Điểm trung bình.
        fatigue_analysis: Phân tích mệt mỏi.
        pain_events: Các sự kiện đau.
        recommendations: Khuyến nghị.
    """
    session_id: str
    start_time: float
    end_time: float = 0.0
    exercise_name: str = ""
    total_reps: int = 0
    rep_scores: List[RepScore] = field(default_factory=list)
    average_scores: Dict[str, float] = field(default_factory=dict)
    fatigue_analysis: Dict = field(default_factory=dict)
    pain_events: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.end_time - self.start_time if self.end_time else 0,
            "exercise_name": self.exercise_name,
            "total_reps": self.total_reps,
            "rep_scores": [r.to_dict() for r in self.rep_scores],
            "average_scores": self.average_scores,
            "fatigue_analysis": self.fatigue_analysis,
            "pain_events": self.pain_events,
            "recommendations": self.recommendations,
        }


class HealthScorer:
    """
    Bộ chấm điểm sức khỏe đa chiều.
    
    Đánh giá chất lượng tập luyện dựa trên 4 chỉ số:
    
    1. ROM Score (Range of Motion):
       - So sánh góc đạt được với góc mục tiêu cá nhân hóa
       - 100% nếu đạt hoặc vượt mục tiêu
       
    2. Stability Score:
       - Đo độ rung lắc trong pha HOLD
       - Dựa trên std deviation của góc khớp
       
    3. Flow Score:
       - Từ kết quả DTW (Giai đoạn 3)
       - Đánh giá sự mượt mà của chuyển động
       
    4. Symmetry Score:
       - So sánh bên trái và bên phải
       - Quan trọng cho các bài tập đối xứng
    
    Example:
        >>> scorer = HealthScorer()
        >>> scorer.start_session("arm_raise")
        >>> 
        >>> # Mỗi khi hoàn thành 1 rep
        >>> scorer.record_rep(angles, target, dtw_result)
        >>> 
        >>> # Cuối buổi tập
        >>> report = scorer.compute_session_report()
    """
    
    # Trọng số của từng thành phần - cập nhật có compensation
    SCORE_WEIGHTS = {
        "rom": 0.30,           # Giảm để có chỗ cho compensation
        "stability": 0.20,
        "flow": 0.20,
        "symmetry": 0.15,
        "compensation": 0.15,  # Mới: trừ điểm nếu bù trừ
    }
    
    # Ngưỡng Jerk để phát hiện mệt mỏi
    JERK_THRESHOLDS = {
        FatigueLevel.LIGHT: 1.5,      # Tăng 50%
        FatigueLevel.MODERATE: 2.0,   # Tăng 100%
        FatigueLevel.HEAVY: 3.0,      # Tăng 200%
    }
    
    def __init__(self):
        """Khởi tạo HealthScorer."""
        self._session_id: Optional[str] = None
        self._start_time: float = 0.0
        self._exercise_name: str = ""
        
        self._rep_scores: List[RepScore] = []
        self._current_rep: int = 0
        
        # Data collection cho rep hiện tại
        self._current_rep_angles: List[float] = []
        self._current_rep_timestamps: List[float] = []
        self._current_rep_phases: List[MotionPhase] = []
        
        # Jerk tracking
        self._jerk_values: List[float] = []
        self._baseline_jerk: Optional[float] = None
        
        # Symmetry tracking
        self._left_angles: List[float] = []
        self._right_angles: List[float] = []
        
        # Pain events
        self._pain_events: List[Dict] = []
        
        # Compensation tracking - mới
        self._shoulder_heights: List[Tuple[float, float]] = []  # (left_y, right_y)
        self._hip_positions: List[Tuple[float, float]] = []  # (left_y, right_y)
        self._torso_tilts: List[float] = []  # Góc nghiêng thân
    
    def start_session(
        self,
        exercise_name: str,
        session_id: Optional[str] = None
    ) -> str:
        """
        Bắt đầu một buổi tập mới.
        
        Args:
            exercise_name: Tên bài tập.
            session_id: ID tùy chỉnh (tự tạo nếu None).
            
        Returns:
            str: Session ID.
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self._session_id = session_id
        self._start_time = time.time()
        self._exercise_name = exercise_name
        
        self._rep_scores = []
        self._current_rep = 0
        self._jerk_values = []
        self._baseline_jerk = None
        self._pain_events = []
        
        self._reset_current_rep()
        
        print(f"[SCORER] Session started: {session_id}")
        print(f"[SCORER] Exercise: {exercise_name}")
        
        return session_id
    
    def _reset_current_rep(self) -> None:
        """Reset data cho rep mới."""
        self._current_rep_angles = []
        self._current_rep_timestamps = []
        self._current_rep_phases = []
        self._left_angles = []
        self._right_angles = []
        # Reset compensation tracking
        self._shoulder_heights = []
        self._hip_positions = []
        self._torso_tilts = []
    
    def add_frame(
        self,
        angle: float,
        timestamp: float,
        phase: MotionPhase,
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None,
        pose_landmarks: Optional[np.ndarray] = None  # Mới: thêm landmarks để detect compensation
    ) -> None:
        """
        Thêm dữ liệu một frame.
        
        Args:
            angle: Góc khớp chính.
            timestamp: Timestamp (seconds).
            phase: Pha hiện tại.
            left_angle: Góc bên trái (cho symmetry).
            right_angle: Góc bên phải (cho symmetry).
            pose_landmarks: Full pose landmarks (33, 3) để detect compensation.
        """
        self._current_rep_angles.append(angle)
        self._current_rep_timestamps.append(timestamp)
        self._current_rep_phases.append(phase)
        
        if left_angle is not None:
            self._left_angles.append(left_angle)
        if right_angle is not None:
            self._right_angles.append(right_angle)
        
        # Track compensation data nếu có landmarks
        if pose_landmarks is not None and len(pose_landmarks) >= 25:
            self._track_compensation_data(pose_landmarks)
    
    def _track_compensation_data(self, landmarks: np.ndarray) -> None:
        """
        Thu thập dữ liệu để phát hiện bù trừ.
        
        Các loại bù trừ cần phát hiện:
        1. Vai không đều (shoulder hiking)
        2. Nghiêng thân (trunk lean)
        3. Xoay hông (hip rotation)
        """
        try:
            # Lấy tọa độ vai (indices 11, 12)
            left_shoulder_y = landmarks[11][1]
            right_shoulder_y = landmarks[12][1]
            self._shoulder_heights.append((left_shoulder_y, right_shoulder_y))
            
            # Lấy tọa độ hông (indices 23, 24)
            left_hip_y = landmarks[23][1]
            right_hip_y = landmarks[24][1]
            self._hip_positions.append((left_hip_y, right_hip_y))
            
            # Tính góc nghiêng thân (từ mid-shoulder đến mid-hip)
            mid_shoulder = (landmarks[11] + landmarks[12]) / 2
            mid_hip = (landmarks[23] + landmarks[24]) / 2
            
            # Góc với vertical (trục y)
            dx = mid_hip[0] - mid_shoulder[0]
            dy = mid_hip[1] - mid_shoulder[1]
            if abs(dy) > 1e-6:
                tilt_angle = np.degrees(np.arctan2(dx, dy))
                self._torso_tilts.append(tilt_angle)
        except (IndexError, ValueError):
            pass  # Skip nếu landmarks không hợp lệ
    
    def complete_rep(
        self,
        target_angle: float,
        dtw_result: Optional[DTWResult] = None
    ) -> RepScore:
        """
        Hoàn thành một rep và tính điểm.
        
        Args:
            target_angle: Góc mục tiêu (đã cá nhân hóa).
            dtw_result: Kết quả DTW (nếu có).
            
        Returns:
            RepScore: Điểm của rep này.
        """
        self._current_rep += 1
        
        angles = np.array(self._current_rep_angles)
        timestamps = np.array(self._current_rep_timestamps)
        phases = self._current_rep_phases
        
        if len(angles) < 10:
            score = RepScore(rep_number=self._current_rep, notes="Không đủ data")
            self._rep_scores.append(score)
            self._reset_current_rep()
            return score
        
        # 1. ROM Score
        rom_score = self._calculate_rom_score(angles, target_angle)
        
        # 2. Stability Score
        stability_score = self._calculate_stability_score(angles, phases)
        
        # 3. Flow Score
        if dtw_result is not None:
            flow_score = dtw_result.similarity_score
        else:
            flow_score = self._estimate_flow_score(angles, timestamps)
        
        # 4. Symmetry Score
        symmetry_score = self._calculate_symmetry_score()
        
        # 5. Compensation Score - MỚI
        compensation_score, compensation_issues = self._calculate_compensation_score()
        
        # 6. Jerk
        jerk = self._calculate_jerk(angles, timestamps)
        self._jerk_values.append(jerk)
        
        # Set baseline jerk từ rep đầu tiên
        if self._baseline_jerk is None and jerk > 0:
            self._baseline_jerk = jerk
        
        # Total Score - cập nhật có compensation
        total = (
            self.SCORE_WEIGHTS["rom"] * rom_score +
            self.SCORE_WEIGHTS["stability"] * stability_score +
            self.SCORE_WEIGHTS["flow"] * flow_score +
            self.SCORE_WEIGHTS["symmetry"] * symmetry_score +
            self.SCORE_WEIGHTS["compensation"] * compensation_score
        )
        
        # Duration
        duration_ms = int((timestamps[-1] - timestamps[0]) * 1000) if len(timestamps) > 1 else 0
        
        # Create score - thêm compensation
        score = RepScore(
            rep_number=self._current_rep,
            rom_score=rom_score,
            stability_score=stability_score,
            flow_score=flow_score,
            symmetry_score=symmetry_score,
            compensation_score=compensation_score,
            total_score=total,
            jerk_value=jerk,
            duration_ms=duration_ms,
            compensation_detected=compensation_issues,
        )
        
        # Check fatigue
        fatigue = self._check_fatigue()
        notes_list = []
        if fatigue != FatigueLevel.FRESH:
            notes_list.append(f"Mệt mỏi: {fatigue.name}")
        if compensation_issues:
            notes_list.append(f"Bù trừ: {', '.join(compensation_issues)}")
        score.notes = "; ".join(notes_list)
        
        self._rep_scores.append(score)
        self._reset_current_rep()
        
        return score
    
    def _calculate_rom_score(self, angles: np.ndarray, target: float) -> float:
        """
        Tính ROM Score - cải tiến để phát hiện chính xác hơn.
        
        Đánh giá dựa trên nhiều yếu tố:
        1. Max angle đạt được (40%)
        2. Thời gian giữ gần target (30%)
        3. Chất lượng đỉnh - không giật lên rồi xuống ngay (30%)
        
        Args:
            angles: Chuỗi góc trong rep.
            target: Góc mục tiêu.
            
        Returns:
            float: Điểm ROM (0-100).
        """
        if target <= 0:
            return 100.0
        
        if len(angles) < 5:
            return 0.0
        
        # 1. Max angle score (40%)
        max_achieved = np.max(angles)
        max_score = min(100.0, (max_achieved / target) * 100)
        
        # 2. Hold time score - thời gian giữ >= 80% target (30%)
        threshold = target * 0.8
        frames_above_threshold = np.sum(angles >= threshold)
        # Yêu cầu tối thiểu 10% số frame phải ở trên threshold
        min_frames_required = max(3, len(angles) * 0.1)
        hold_ratio = min(1.0, frames_above_threshold / min_frames_required)
        hold_score = hold_ratio * 100
        
        # 3. Peak quality score - kiểm tra đạt góc có ổn định không (30%)
        # Tìm vùng xung quanh peak
        peak_idx = np.argmax(angles)
        window = max(3, len(angles) // 10)  # 10% số frame hoặc tối thiểu 3
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(angles), peak_idx + window + 1)
        peak_region = angles[start_idx:end_idx]
        
        if len(peak_region) >= 3:
            # Độ ổn định của vùng peak (std thấp = tốt)
            peak_std = np.std(peak_region)
            # Chuẩn hóa: std < 5° = tốt (100), std > 20° = kém (0)
            peak_quality_score = max(0, 100 - peak_std * 5)
        else:
            peak_quality_score = 50.0  # Default nếu không đủ data
        
        # Tổng hợp với trọng số
        final_score = (
            0.40 * max_score +
            0.30 * hold_score +
            0.30 * peak_quality_score
        )
        
        return min(100.0, max(0.0, final_score))
    
    def _calculate_stability_score(
        self,
        angles: np.ndarray,
        phases: List[MotionPhase]
    ) -> float:
        """
        Tính Stability Score từ pha HOLD - cải tiến.
        
        Đánh giá dựa trên:
        1. Standard deviation trong pha HOLD (50%)
        2. Số lần vượt ngưỡng dao động (30%)
        3. Xu hướng giảm góc trong HOLD - dấu hiệu mệt (20%)
        
        Args:
            angles: Chuỗi góc.
            phases: Chuỗi pha tương ứng.
            
        Returns:
            float: Điểm stability (0-100).
        """
        # Lọc ra các góc trong pha HOLD
        hold_angles = [
            angles[i] for i in range(len(phases))
            if phases[i] == MotionPhase.HOLD
        ]
        
        if len(hold_angles) < 5:
            # Không đủ data trong HOLD, đánh giá cả eccentric/concentric
            if len(angles) < 5:
                return 80.0  # Default
            # Đánh giá độ mượt của toàn bộ chuyển động
            overall_std = np.std(np.diff(angles))  # Std của velocity
            return min(100.0, max(0.0, 100 - overall_std * 5))
        
        hold_arr = np.array(hold_angles)
        
        # 1. Standard deviation score (50%)
        std = np.std(hold_arr)
        # Chuẩn hóa: std < 2° = tuyệt vời (100), std > 10° = kém (0)
        std_score = max(0, 100 - std * 10)
        
        # 2. Oscillation count - số lần dao động vượt ngưỡng (30%)
        mean_angle = np.mean(hold_arr)
        oscillation_threshold = 3.0  # 3 độ
        # Đếm số lần cross threshold
        deviations = np.abs(hold_arr - mean_angle)
        crossings = np.sum(deviations > oscillation_threshold)
        # Cho phép tối đa 20% số frame vượt ngưỡng
        max_allowed_crossings = max(1, len(hold_arr) * 0.2)
        oscillation_ratio = min(1.0, crossings / max_allowed_crossings)
        oscillation_score = (1 - oscillation_ratio) * 100
        
        # 3. Drift score - góc có giảm dần không (dấu hiệu mệt) (20%)
        if len(hold_arr) >= 3:
            # So sánh nửa đầu và nửa sau
            first_half = np.mean(hold_arr[:len(hold_arr)//2])
            second_half = np.mean(hold_arr[len(hold_arr)//2:])
            drift = first_half - second_half  # Dương = góc giảm
            # Cho phép giảm tối đa 5 độ
            drift_penalty = min(1.0, max(0, drift) / 5.0)
            drift_score = (1 - drift_penalty) * 100
        else:
            drift_score = 100.0
        
        # Tổng hợp
        final_score = (
            0.50 * std_score +
            0.30 * oscillation_score +
            0.20 * drift_score
        )
        
        return min(100.0, max(0.0, final_score))
    
    def _estimate_flow_score(
        self,
        angles: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Ước tính Flow Score khi không có DTW - cải tiến.
        
        Đánh giá dựa trên nhiều yếu tố:
        1. Độ mượt của velocity (không giật) (40%)
        2. Tính liên tục - không có jump đột ngột (30%)
        3. Tỷ lệ velocity âm/dương hợp lý (30%)
        
        Returns:
            float: Flow score (0-100)
        """
        if len(angles) < 5:
            return 70.0
        
        # Tính velocity
        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        velocity = np.diff(angles) / dt
        
        # 1. Velocity smoothness (40%)
        # Std của acceleration (đạo hàm velocity) - thấp = mượt
        if len(velocity) >= 3:
            acceleration = np.diff(velocity) / dt[:-1]
            accel_std = np.std(acceleration)
            # Chuẩn hóa: accel_std < 50 = tốt, > 500 = kém
            smoothness_score = max(0, 100 - accel_std * 0.2)
        else:
            smoothness_score = 70.0
        
        # 2. Continuity - không có jump đột ngột (30%)
        angle_diffs = np.abs(np.diff(angles))
        max_allowed_jump = 15.0  # Tối đa 15 độ/frame
        jumps = np.sum(angle_diffs > max_allowed_jump)
        jump_ratio = jumps / len(angle_diffs) if len(angle_diffs) > 0 else 0
        continuity_score = (1 - min(1.0, jump_ratio * 5)) * 100
        
        # 3. Direction consistency (30%)
        # Trong một pha, velocity nên chủ yếu cùng chiều
        if len(velocity) >= 5:
            # Đếm số lần đổi chiều
            sign_changes = np.sum(np.abs(np.diff(np.sign(velocity))) > 0)
            # Cho phép tối đa 30% số frame có đổi chiều
            max_changes = len(velocity) * 0.3
            direction_ratio = min(1.0, sign_changes / max(1, max_changes))
            direction_score = (1 - direction_ratio) * 100
        else:
            direction_score = 70.0
        
        # Tổng hợp
        final_score = (
            0.40 * smoothness_score +
            0.30 * continuity_score +
            0.30 * direction_score
        )
        
        return min(100.0, max(0.0, final_score))
    
    def _calculate_symmetry_score(self) -> float:
        """
        Tính Symmetry Score.
        
        So sánh góc bên trái và phải.
        """
        if len(self._left_angles) < 5 or len(self._right_angles) < 5:
            return 85.0  # Default nếu không có data
        
        left = np.array(self._left_angles)
        right = np.array(self._right_angles)
        
        # Cắt về cùng độ dài
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]
        
        # Tính mean absolute difference
        diff = np.mean(np.abs(left - right))
        
        # Chuẩn hóa: diff < 5° = tuyệt vời, diff > 20° = kém
        score = 100 - (diff * 4)
        
        return min(100.0, max(0.0, score))
    
    def _calculate_compensation_score(self) -> Tuple[float, List[str]]:
        """
        Tính Compensation Score - phát hiện động tác bù trừ.
        
        Các loại bù trừ phổ biến trong phục hồi chức năng:
        1. Shoulder hiking: Nhún vai lên để tăng góc giơ tay
        2. Trunk lean: Nghiêng thân để "gian lận" góc
        3. Hip shift: Xoay hông khi tập chi dưới
        
        Returns:
            Tuple[float, List[str]]: (score, list of detected compensations)
        """
        issues = []
        penalties = []
        
        # 1. Kiểm tra shoulder hiking (vai không đều)
        if len(self._shoulder_heights) >= 5:
            shoulder_diffs = [abs(left - right) for left, right in self._shoulder_heights]
            avg_diff = np.mean(shoulder_diffs)
            max_diff = np.max(shoulder_diffs)
            
            # Ngưỡng: chênh lệch vai > 0.05 (5% chiều cao frame) là đáng kể
            if max_diff > 0.08:  # Bù trừ nặng
                issues.append("Vai không đều (nặng)")
                penalties.append(40)
            elif max_diff > 0.05:  # Bù trừ nhẹ
                issues.append("Vai không đều (nhẹ)")
                penalties.append(20)
            elif avg_diff > 0.03:  # Có xu hướng
                penalties.append(10)
        
        # 2. Kiểm tra trunk lean (nghiêng thân)
        if len(self._torso_tilts) >= 5:
            torso_arr = np.array(self._torso_tilts)
            avg_tilt = np.mean(np.abs(torso_arr))
            max_tilt = np.max(np.abs(torso_arr))
            tilt_change = np.max(torso_arr) - np.min(torso_arr)
            
            # Ngưỡng: nghiêng > 15 độ hoặc thay đổi > 20 độ trong rep
            if max_tilt > 20 or tilt_change > 25:
                issues.append("Nghiêng thân nhiều")
                penalties.append(35)
            elif max_tilt > 15 or tilt_change > 15:
                issues.append("Nghiêng thân")
                penalties.append(20)
            elif avg_tilt > 10:
                penalties.append(10)
        
        # 3. Kiểm tra hip asymmetry (xoay hông)
        if len(self._hip_positions) >= 5:
            hip_diffs = [abs(left - right) for left, right in self._hip_positions]
            max_hip_diff = np.max(hip_diffs)
            
            # Ngưỡng: chênh lệch hông > 0.06 là đáng kể
            if max_hip_diff > 0.08:
                issues.append("Hông không cân bằng")
                penalties.append(25)
            elif max_hip_diff > 0.05:
                penalties.append(15)
        
        # Tính điểm: bắt đầu từ 100, trừ penalties
        total_penalty = sum(penalties)
        score = max(0, 100 - total_penalty)
        
        return score, issues
    
    def _calculate_jerk(
        self,
        angles: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Tính Squared Jerk metric.
        
        Jerk = d³θ/dt³ (đạo hàm bậc 3)
        
        Công thức:
            1. Tính velocity: v = dθ/dt
            2. Tính acceleration: a = dv/dt
            3. Tính jerk: j = da/dt
            4. Squared Jerk = Σ j²
        
        Args:
            angles: Chuỗi góc (degrees).
            timestamps: Chuỗi thời gian (seconds).
            
        Returns:
            float: Giá trị Squared Jerk.
        """
        if len(angles) < 4:
            return 0.0
        
        # Đảm bảo timestamps tăng dần
        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        
        # Velocity (đạo hàm bậc 1)
        velocity = np.diff(angles) / dt
        
        # Acceleration (đạo hàm bậc 2)
        dt2 = dt[:-1]
        acceleration = np.diff(velocity) / dt2
        
        # Jerk (đạo hàm bậc 3)
        dt3 = dt2[:-1]
        if len(dt3) == 0:
            return 0.0
        
        jerk = np.diff(acceleration) / dt3
        
        # Squared Jerk (chuẩn hóa theo thời gian)
        total_time = timestamps[-1] - timestamps[0]
        if total_time < 1e-6:
            return 0.0
        
        squared_jerk = np.sum(jerk ** 2) / total_time
        
        return float(squared_jerk)
    
    def _check_fatigue(self) -> FatigueLevel:
        """
        Kiểm tra mức độ mệt mỏi dựa trên Jerk.
        
        Logic:
            - So sánh Jerk hiện tại với baseline
            - Nếu tăng dần qua các rep → mệt mỏi
        """
        if self._baseline_jerk is None or self._baseline_jerk < 1e-6:
            return FatigueLevel.FRESH
        
        if len(self._jerk_values) < 2:
            return FatigueLevel.FRESH
        
        current_jerk = self._jerk_values[-1]
        jerk_ratio = current_jerk / self._baseline_jerk
        
        if jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.HEAVY]:
            return FatigueLevel.HEAVY
        elif jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.MODERATE]:
            return FatigueLevel.MODERATE
        elif jerk_ratio >= self.JERK_THRESHOLDS[FatigueLevel.LIGHT]:
            return FatigueLevel.LIGHT
        else:
            return FatigueLevel.FRESH
    
    def add_pain_event(self, event: Dict) -> None:
        """Ghi nhận một pain event."""
        self._pain_events.append(event)
    
    def compute_session_report(self) -> SessionReport:
        """
        Tổng hợp báo cáo buổi tập.
        
        Returns:
            SessionReport: Báo cáo đầy đủ.
        """
        end_time = time.time()
        
        # Average scores - cập nhật có compensation
        if self._rep_scores:
            average_scores = {
                "rom": np.mean([r.rom_score for r in self._rep_scores]),
                "stability": np.mean([r.stability_score for r in self._rep_scores]),
                "flow": np.mean([r.flow_score for r in self._rep_scores]),
                "symmetry": np.mean([r.symmetry_score for r in self._rep_scores]),
                "compensation": np.mean([r.compensation_score for r in self._rep_scores]),
                "total": np.mean([r.total_score for r in self._rep_scores]),
            }
        else:
            average_scores = {}
        
        # Fatigue analysis
        fatigue_analysis = self._analyze_fatigue_trend()
        
        # Recommendations
        recommendations = self._generate_recommendations(
            average_scores, fatigue_analysis
        )
        
        report = SessionReport(
            session_id=self._session_id or "",
            start_time=self._start_time,
            end_time=end_time,
            exercise_name=self._exercise_name,
            total_reps=len(self._rep_scores),
            rep_scores=self._rep_scores,
            average_scores=average_scores,
            fatigue_analysis=fatigue_analysis,
            pain_events=self._pain_events,
            recommendations=recommendations,
        )
        
        return report
    
    def _analyze_fatigue_trend(self) -> Dict:
        """Phân tích xu hướng mệt mỏi."""
        if len(self._jerk_values) < 2:
            return {
                "trend": "stable",
                "fatigue_level": FatigueLevel.FRESH.name,
                "jerk_increase_percent": 0,
            }
        
        # Tính xu hướng tăng
        first_half = np.mean(self._jerk_values[:len(self._jerk_values)//2])
        second_half = np.mean(self._jerk_values[len(self._jerk_values)//2:])
        
        if first_half > 1e-6:
            increase_percent = ((second_half - first_half) / first_half) * 100
        else:
            increase_percent = 0
        
        if increase_percent > 100:
            trend = "increasing_fast"
        elif increase_percent > 30:
            trend = "increasing"
        elif increase_percent < -20:
            trend = "improving"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "fatigue_level": self._check_fatigue().name,
            "jerk_increase_percent": round(increase_percent, 1),
            "jerk_values": [round(j, 4) for j in self._jerk_values],
        }
    
    def _generate_recommendations(
        self,
        avg_scores: Dict,
        fatigue: Dict
    ) -> List[str]:
        """Tạo khuyến nghị cho người dùng."""
        recommendations = []
        
        # ROM recommendations
        rom = avg_scores.get("rom", 100)
        if rom < 70:
            recommendations.append(
                "Bác chưa đạt được góc mục tiêu. Hãy cố gắng thêm một chút, "
                "nhưng đừng ép nếu thấy đau nhé!"
            )
        elif rom >= 95:
            recommendations.append(
                "Tuyệt vời! Bác đã đạt góc mục tiêu rất tốt!"
            )
        
        # Stability recommendations
        stability = avg_scores.get("stability", 100)
        if stability < 60:
            recommendations.append(
                "Khi giữ tư thế, bác hãy cố giữ yên hơn nhé. "
                "Thở đều và tập trung."
            )
        
        # Compensation recommendations - MỚI
        compensation = avg_scores.get("compensation", 100)
        if compensation < 70:
            # Phân tích loại compensation phổ biến
            all_compensations = []
            for rep_score in self._rep_scores:
                all_compensations.extend(rep_score.compensation_detected)
            
            if "Vai không đều" in " ".join(all_compensations) or "Vai không đều (nặng)" in all_compensations:
                recommendations.append(
                    "⚠️ Bác có xu hướng nhún vai khi tập. "
                    "Hãy giữ vai thả lỏng, không nâng vai lên nhé!"
                )
            if "Nghiêng thân" in " ".join(all_compensations):
                recommendations.append(
                    "⚠️ Bác có nghiêng người khi tập. "
                    "Hãy giữ lưng thẳng, không nghiêng sang bên nhé!"
                )
            if "Hông không cân bằng" in all_compensations:
                recommendations.append(
                    "⚠️ Hông bác bị lệch khi tập. "
                    "Hãy đứng vững trên hai chân, phân bổ trọng lượng đều nhé!"
                )
        elif compensation < 85:
            recommendations.append(
                "Bác có chút bù trừ khi tập. Cố gắng giữ tư thế chuẩn hơn nhé!"
            )
        
        # Fatigue recommendations
        fatigue_level = fatigue.get("fatigue_level", "FRESH")
        if fatigue_level == "HEAVY":
            recommendations.append(
                "⚠️ Bác đã mệt nhiều rồi. Nên nghỉ ngơi và uống nước!"
            )
        elif fatigue_level == "MODERATE":
            recommendations.append(
                "Bác có vẻ hơi mệt. Có thể nghỉ một chút rồi tiếp tục."
            )
        
        # Pain recommendations
        if self._pain_events:
            recommendations.append(
                "⚠️ Có dấu hiệu không thoải mái trong buổi tập. "
                "Hãy báo bác sĩ nếu còn đau sau khi nghỉ."
            )
        
        # Default positive
        if not recommendations:
            recommendations.append(
                "Buổi tập tốt! Hẹn gặp lại bác vào buổi tập sau nhé!"
            )
        
        return recommendations
    
    def get_current_status(self) -> Dict:
        """Lấy trạng thái hiện tại (cho real-time display)."""
        if not self._rep_scores:
            return {
                "rep_count": self._current_rep,
                "last_score": 0,
                "average_score": 0,
                "fatigue_level": FatigueLevel.FRESH.name,
            }
        
        last_score = self._rep_scores[-1].total_score
        avg_score = np.mean([r.total_score for r in self._rep_scores])
        
        return {
            "rep_count": len(self._rep_scores),
            "last_score": round(last_score, 1),
            "average_score": round(avg_score, 1),
            "fatigue_level": self._check_fatigue().name,
        }


def calculate_jerk(
    positions: np.ndarray,
    timestamps: np.ndarray
) -> float:
    """
    Tính Squared Jerk từ chuỗi vị trí 3D.
    
    Standalone function cho các use case khác.
    
    Args:
        positions: Array (N, 3) chứa vị trí x, y, z.
        timestamps: Array (N,) chứa timestamps.
        
    Returns:
        float: Squared Jerk value.
    """
    if len(positions) < 4 or len(timestamps) < 4:
        return 0.0
    
    # Tính velocity
    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1e-6, dt).reshape(-1, 1)
    
    velocity = np.diff(positions, axis=0) / dt
    
    # Tính acceleration
    dt2 = dt[:-1]
    acceleration = np.diff(velocity, axis=0) / dt2
    
    # Tính jerk
    dt3 = dt2[:-1]
    jerk = np.diff(acceleration, axis=0) / dt3
    
    # Squared jerk (sum of squared norms)
    jerk_norms_squared = np.sum(jerk ** 2, axis=1)
    
    total_time = timestamps[-1] - timestamps[0]
    if total_time < 1e-6:
        return 0.0
    
    return float(np.sum(jerk_norms_squared) / total_time)


def calculate_center_of_mass(landmarks: np.ndarray) -> np.ndarray:
    """
    Tính Center of Mass từ pose landmarks.
    
    Sử dụng weighted average với trọng số ước lượng.
    
    Args:
        landmarks: Pose landmarks array (33, 3).
        
    Returns:
        np.ndarray: Tọa độ CoM [x, y, z].
    """
    # Trọng số ước lượng dựa trên khối lượng cơ thể
    # Thân > Đầu > Tay/Chân
    weights = np.ones(len(landmarks))
    
    # Torso (cao hơn)
    torso_indices = [11, 12, 23, 24]  # Vai và hông
    for idx in torso_indices:
        if idx < len(weights):
            weights[idx] = 2.0
    
    # Head (vừa)
    head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for idx in head_indices:
        if idx < len(weights):
            weights[idx] = 1.5
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Weighted average
    com = np.average(landmarks, axis=0, weights=weights)
    
    return com