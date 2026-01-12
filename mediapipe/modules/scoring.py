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
        total_score: Điểm tổng hợp (0-100).
        jerk_value: Giá trị Jerk.
        duration_ms: Thời gian thực hiện (ms).
        notes: Ghi chú.
    """
    rep_number: int
    rom_score: float = 0.0
    stability_score: float = 0.0
    flow_score: float = 0.0
    symmetry_score: float = 0.0
    total_score: float = 0.0
    jerk_value: float = 0.0
    duration_ms: int = 0
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "rep_number": self.rep_number,
            "rom_score": round(self.rom_score, 1),
            "stability_score": round(self.stability_score, 1),
            "flow_score": round(self.flow_score, 1),
            "symmetry_score": round(self.symmetry_score, 1),
            "total_score": round(self.total_score, 1),
            "jerk_value": round(self.jerk_value, 4),
            "duration_ms": self.duration_ms,
            "notes": self.notes,
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
    
    # Trọng số của từng thành phần
    SCORE_WEIGHTS = {
        "rom": 0.35,
        "stability": 0.25,
        "flow": 0.25,
        "symmetry": 0.15,
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
    
    def add_frame(
        self,
        angle: float,
        timestamp: float,
        phase: MotionPhase,
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None
    ) -> None:
        """
        Thêm dữ liệu một frame.
        
        Args:
            angle: Góc khớp chính.
            timestamp: Timestamp (seconds).
            phase: Pha hiện tại.
            left_angle: Góc bên trái (cho symmetry).
            right_angle: Góc bên phải (cho symmetry).
        """
        self._current_rep_angles.append(angle)
        self._current_rep_timestamps.append(timestamp)
        self._current_rep_phases.append(phase)
        
        if left_angle is not None:
            self._left_angles.append(left_angle)
        if right_angle is not None:
            self._right_angles.append(right_angle)
    
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
        
        # 5. Jerk
        jerk = self._calculate_jerk(angles, timestamps)
        self._jerk_values.append(jerk)
        
        # Set baseline jerk từ rep đầu tiên
        if self._baseline_jerk is None and jerk > 0:
            self._baseline_jerk = jerk
        
        # Total Score
        total = (
            self.SCORE_WEIGHTS["rom"] * rom_score +
            self.SCORE_WEIGHTS["stability"] * stability_score +
            self.SCORE_WEIGHTS["flow"] * flow_score +
            self.SCORE_WEIGHTS["symmetry"] * symmetry_score
        )
        
        # Duration
        duration_ms = int((timestamps[-1] - timestamps[0]) * 1000) if len(timestamps) > 1 else 0
        
        # Create score
        score = RepScore(
            rep_number=self._current_rep,
            rom_score=rom_score,
            stability_score=stability_score,
            flow_score=flow_score,
            symmetry_score=symmetry_score,
            total_score=total,
            jerk_value=jerk,
            duration_ms=duration_ms,
        )
        
        # Check fatigue
        fatigue = self._check_fatigue()
        if fatigue != FatigueLevel.FRESH:
            score.notes = f"Mệt mỏi: {fatigue.name}"
        
        self._rep_scores.append(score)
        self._reset_current_rep()
        
        return score
    
    def _calculate_rom_score(self, angles: np.ndarray, target: float) -> float:
        """
        Tính ROM Score.
        
        Công thức:
            score = min(100, (max_achieved / target) × 100)
        
        Args:
            angles: Chuỗi góc trong rep.
            target: Góc mục tiêu.
            
        Returns:
            float: Điểm ROM (0-100).
        """
        if target <= 0:
            return 100.0
        
        max_achieved = np.max(angles)
        score = (max_achieved / target) * 100
        
        return min(100.0, max(0.0, score))
    
    def _calculate_stability_score(
        self,
        angles: np.ndarray,
        phases: List[MotionPhase]
    ) -> float:
        """
        Tính Stability Score từ pha HOLD.
        
        Công thức:
            score = 100 - (std_deviation × 10)
            
        Std thấp = ổn định = điểm cao.
        
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
            return 80.0  # Default nếu không đủ data
        
        std = np.std(hold_angles)
        
        # Chuẩn hóa: std < 2° = tuyệt vời, std > 10° = kém
        score = 100 - (std * 10)
        
        return min(100.0, max(0.0, score))
    
    def _estimate_flow_score(
        self,
        angles: np.ndarray,
        timestamps: np.ndarray
    ) -> float:
        """
        Ước tính Flow Score khi không có DTW.
        
        Dựa trên độ mượt của velocity.
        """
        if len(angles) < 3:
            return 70.0
        
        # Tính velocity
        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        velocity = np.diff(angles) / dt
        
        # Flow cao = velocity ít biến thiên
        velocity_std = np.std(velocity)
        
        # Chuẩn hóa
        score = 100 - (velocity_std * 2)
        
        return min(100.0, max(0.0, score))
    
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
        
        # Average scores
        if self._rep_scores:
            average_scores = {
                "rom": np.mean([r.rom_score for r in self._rep_scores]),
                "stability": np.mean([r.stability_score for r in self._rep_scores]),
                "flow": np.mean([r.flow_score for r in self._rep_scores]),
                "symmetry": np.mean([r.symmetry_score for r in self._rep_scores]),
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