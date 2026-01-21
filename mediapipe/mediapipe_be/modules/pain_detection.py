"""
Pain Detection Module for MEMOTION.

Nhận diện đau đớn qua phân tích biểu cảm khuôn mặt sử dụng
Facial Action Coding System (FACS).

Cơ sở khoa học:
    FACS được phát triển bởi Paul Ekman, mô tả các cử động
    cơ mặt như "Action Units" (AU). Khi đau, người ta thường:
    - AU4: Cau mày (Brow Lowerer)
    - AU6: Nheo má (Cheek Raiser)
    - AU7: Căng mí mắt (Lid Tightener)
    - AU9: Nhăn mũi (Nose Wrinkler)
    - AU10: Nâng môi trên (Upper Lip Raiser)
    - AU43: Nhắm mắt (Eye Closure)

Ý nghĩa nhân văn:
    Người già Việt Nam thường "chịu đựng" và không nói khi đau.
    Hệ thống này giúp phát hiện sớm để:
    - Dừng bài tập trước khi gây hại
    - Điều chỉnh cường độ phù hợp
    - Tạo môi trường tập luyện AN TOÀN

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from collections import deque
import time
import numpy as np

from core.data_types import LandmarkSet


class PainLevel(Enum):
    """Mức độ đau được phát hiện."""
    NONE = 0        # Không có dấu hiệu đau
    MILD = 1        # Nhẹ - có thể tiếp tục
    MODERATE = 2    # Trung bình - cần chú ý
    SEVERE = 3      # Nặng - nên dừng lại


@dataclass
class PainEvent:
    """
    Ghi nhận một sự kiện đau.
    
    Attributes:
        timestamp: Thời điểm xảy ra.
        level: Mức độ đau.
        duration_ms: Thời gian kéo dài (ms).
        au_scores: Điểm của từng Action Unit.
        message: Thông báo cho người dùng.
    """
    timestamp: float
    level: PainLevel
    duration_ms: int = 0
    au_scores: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level.name,
            "duration_ms": self.duration_ms,
            "au_scores": self.au_scores,
            "message": self.message,
        }


@dataclass
class PainAnalysisResult:
    """
    Kết quả phân tích đau từ một frame.
    
    Attributes:
        pain_level: Mức độ đau.
        pain_score: Điểm đau tổng hợp (0-100).
        au_activations: Dict mapping AU → mức độ kích hoạt (0-1).
        is_pain_detected: True nếu phát hiện đau.
        confidence: Độ tin cậy của phân tích.
    """
    pain_level: PainLevel
    pain_score: float
    au_activations: Dict[str, float]
    is_pain_detected: bool
    confidence: float
    message: str = ""


# MediaPipe Face Mesh landmark indices
# Tham khảo: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
class FaceLandmarkIndex:
    """Indices của các landmarks quan trọng cho FACS."""
    
    # Lông mày trái
    LEFT_EYEBROW_INNER = 107
    LEFT_EYEBROW_MIDDLE = 66
    LEFT_EYEBROW_OUTER = 105
    
    # Lông mày phải
    RIGHT_EYEBROW_INNER = 336
    RIGHT_EYEBROW_MIDDLE = 296
    RIGHT_EYEBROW_OUTER = 334
    
    # Mắt trái
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    
    # Mắt phải
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    
    # Mũi
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_LEFT = 129
    NOSE_RIGHT = 358
    
    # Môi
    UPPER_LIP_TOP = 0
    UPPER_LIP_BOTTOM = 13
    LOWER_LIP_TOP = 14
    LOWER_LIP_BOTTOM = 17
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    
    # Mặt tổng thể
    FACE_TOP = 10      # Trán
    FACE_BOTTOM = 152  # Cằm
    FACE_LEFT = 234
    FACE_RIGHT = 454


class PainDetector:
    """
    Bộ phát hiện đau qua biểu cảm khuôn mặt.
    
    Sử dụng FACS (Facial Action Coding System) để phân tích
    các Action Units liên quan đến đau:
    - AU4: Cau mày
    - AU6/7: Nheo mắt
    - AU9/10: Nhăn mũi/môi
    - AU43: Nhắm mắt
    
    Thuật toán:
        1. Tính các tỷ lệ khoảng cách giữa landmarks
        2. So sánh với baseline (trạng thái bình thường)
        3. Nếu nhiều AU kích hoạt đồng thời → đau
        4. Theo dõi thời gian để lọc nhiễu (>500ms mới tính)
    
    Example:
        >>> detector = PainDetector()
        >>> detector.set_baseline(neutral_landmarks)
        >>> 
        >>> result = detector.analyze(face_landmarks)
        >>> if result.is_pain_detected:
        ...     print(f"Phát hiện đau: {result.message}")
    """
    
    # Ngưỡng phát hiện AU
    AU_THRESHOLDS = {
        "AU4": 0.15,   # Cau mày - giảm khoảng cách lông mày-mắt
        "AU6": 0.12,   # Nheo má - giảm độ mở mắt
        "AU7": 0.15,   # Căng mí - giảm độ mở mắt
        "AU9": 0.10,   # Nhăn mũi
        "AU10": 0.12,  # Nâng môi trên
        "AU43": 0.40,  # Nhắm mắt - giảm độ mở mắt >40%
    }
    
    # Trọng số của từng AU trong tính điểm đau
    AU_WEIGHTS = {
        "AU4": 0.25,   # Cau mày - quan trọng
        "AU6": 0.15,
        "AU7": 0.20,   # Căng mí - quan trọng
        "AU9": 0.10,
        "AU10": 0.10,
        "AU43": 0.20,  # Nhắm mắt - quan trọng
    }
    
    # Ngưỡng mức độ đau
    PAIN_THRESHOLDS = {
        PainLevel.MILD: 20,
        PainLevel.MODERATE: 45,
        PainLevel.SEVERE: 70,
    }
    
    # Thời gian tối thiểu để xác nhận đau (ms)
    MIN_PAIN_DURATION_MS = 500
    
    def __init__(self, history_size: int = 30):
        """
        Khởi tạo PainDetector.
        
        Args:
            history_size: Số frame lưu trong history (để lọc nhiễu).
        """
        self._history_size = history_size
        
        # Baseline measurements (từ trạng thái bình thường)
        self._baseline: Optional[Dict[str, float]] = None
        self._baseline_set = False
        
        # History để làm mượt và lọc nhiễu
        self._pain_score_history: deque = deque(maxlen=history_size)
        self._au_history: Dict[str, deque] = {
            au: deque(maxlen=history_size) for au in self.AU_THRESHOLDS.keys()
        }
        
        # Pain event tracking
        self._pain_start_time: Optional[float] = None
        self._current_pain_level = PainLevel.NONE
        self._pain_events: List[PainEvent] = []
        
        # Calibration
        self._calibration_frames: List[Dict[str, float]] = []
        self._is_calibrating = False
    
    def start_calibration(self) -> None:
        """Bắt đầu calibration để lấy baseline."""
        self._calibration_frames = []
        self._is_calibrating = True
        print("[PAIN] Bắt đầu calibration biểu cảm trung tính...")
        print("[PAIN] Hãy giữ khuôn mặt thư giãn trong 3 giây...")
    
    def add_calibration_frame(self, landmarks: np.ndarray) -> bool:
        """
        Thêm frame vào quá trình calibration.
        
        Args:
            landmarks: Face landmarks array.
            
        Returns:
            bool: True nếu calibration hoàn thành.
        """
        if not self._is_calibrating:
            return False
        
        measurements = self._compute_measurements(landmarks)
        self._calibration_frames.append(measurements)
        
        # Cần 90 frames (~3 giây @ 30fps)
        if len(self._calibration_frames) >= 90:
            self._finalize_calibration()
            return True
        
        return False
    
    def _finalize_calibration(self) -> None:
        """Hoàn thành calibration và set baseline."""
        if not self._calibration_frames:
            return
        
        # Lấy trung bình các measurements
        self._baseline = {}
        keys = self._calibration_frames[0].keys()
        
        for key in keys:
            values = [f[key] for f in self._calibration_frames if key in f]
            self._baseline[key] = np.median(values)
        
        self._baseline_set = True
        self._is_calibrating = False
        
        print("[PAIN] ✓ Calibration hoàn thành!")
        print(f"[PAIN] Baseline: eye_ratio={self._baseline.get('eye_aspect_ratio', 0):.3f}")
    
    def set_baseline(self, landmarks: np.ndarray) -> None:
        """
        Đặt baseline từ một frame đơn.
        
        Dùng khi không có thời gian calibration đầy đủ.
        
        Args:
            landmarks: Face landmarks từ trạng thái bình thường.
        """
        self._baseline = self._compute_measurements(landmarks)
        self._baseline_set = True
    
    def analyze(
        self,
        landmarks: np.ndarray,
        timestamp: Optional[float] = None
    ) -> PainAnalysisResult:
        """
        Phân tích biểu cảm để phát hiện đau.
        
        Args:
            landmarks: Face landmarks array (478, 3) từ MediaPipe.
            timestamp: Timestamp hiện tại (seconds).
            
        Returns:
            PainAnalysisResult: Kết quả phân tích.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Nếu đang calibration
        if self._is_calibrating:
            self.add_calibration_frame(landmarks)
            return PainAnalysisResult(
                pain_level=PainLevel.NONE,
                pain_score=0.0,
                au_activations={},
                is_pain_detected=False,
                confidence=0.0,
                message="Đang calibration..."
            )
        
        # Tính measurements hiện tại
        current = self._compute_measurements(landmarks)
        
        # Nếu chưa có baseline, sử dụng default
        if not self._baseline_set:
            self._baseline = self._get_default_baseline()
            self._baseline_set = True
        
        # Tính AU activations
        au_activations = self._compute_au_activations(current)
        
        # Cập nhật history
        for au, value in au_activations.items():
            self._au_history[au].append(value)
        
        # Tính điểm đau
        pain_score = self._compute_pain_score(au_activations)
        self._pain_score_history.append(pain_score)
        
        # Làm mượt với moving average
        smoothed_score = np.mean(list(self._pain_score_history))
        
        # Xác định mức độ đau
        pain_level = self._classify_pain_level(smoothed_score)
        
        # Xử lý pain events
        is_pain_detected = self._process_pain_event(
            pain_level, smoothed_score, au_activations, timestamp
        )
        
        # Tính confidence
        confidence = self._compute_confidence(au_activations)
        
        # Tạo message
        message = self._generate_message(pain_level, is_pain_detected)
        
        return PainAnalysisResult(
            pain_level=pain_level,
            pain_score=smoothed_score,
            au_activations=au_activations,
            is_pain_detected=is_pain_detected,
            confidence=confidence,
            message=message
        )
    
    def _compute_measurements(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Tính các measurements từ landmarks."""
        if len(landmarks) < 468:
            return {}
        
        measurements = {}
        
        try:
            # Eye Aspect Ratio (EAR) - cho AU6, AU7, AU43
            left_ear = self._eye_aspect_ratio(
                landmarks, "left"
            )
            right_ear = self._eye_aspect_ratio(
                landmarks, "right"
            )
            measurements["eye_aspect_ratio"] = (left_ear + right_ear) / 2
            measurements["left_ear"] = left_ear
            measurements["right_ear"] = right_ear
            
            # Eyebrow position - cho AU4
            left_brow = self._eyebrow_position(landmarks, "left")
            right_brow = self._eyebrow_position(landmarks, "right")
            measurements["eyebrow_position"] = (left_brow + right_brow) / 2
            measurements["left_brow"] = left_brow
            measurements["right_brow"] = right_brow
            
            # Nose wrinkle - cho AU9
            measurements["nose_wrinkle"] = self._nose_wrinkle_ratio(landmarks)
            
            # Upper lip raise - cho AU10
            measurements["upper_lip_raise"] = self._upper_lip_position(landmarks)
            
            # Mouth aspect ratio
            measurements["mouth_aspect_ratio"] = self._mouth_aspect_ratio(landmarks)
            
        except (IndexError, ValueError):
            pass
        
        return measurements
    
    def _eye_aspect_ratio(self, landmarks: np.ndarray, side: str) -> float:
        """
        Tính Eye Aspect Ratio (EAR).
        
        EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
        
        Giá trị nhỏ = mắt nhắm/nheo
        """
        if side == "left":
            top = landmarks[FaceLandmarkIndex.LEFT_EYE_TOP]
            bottom = landmarks[FaceLandmarkIndex.LEFT_EYE_BOTTOM]
            inner = landmarks[FaceLandmarkIndex.LEFT_EYE_INNER]
            outer = landmarks[FaceLandmarkIndex.LEFT_EYE_OUTER]
        else:
            top = landmarks[FaceLandmarkIndex.RIGHT_EYE_TOP]
            bottom = landmarks[FaceLandmarkIndex.RIGHT_EYE_BOTTOM]
            inner = landmarks[FaceLandmarkIndex.RIGHT_EYE_INNER]
            outer = landmarks[FaceLandmarkIndex.RIGHT_EYE_OUTER]
        
        vertical = np.linalg.norm(top[:2] - bottom[:2])
        horizontal = np.linalg.norm(inner[:2] - outer[:2])
        
        if horizontal < 1e-6:
            return 0.3  # Default
        
        return vertical / horizontal
    
    def _eyebrow_position(self, landmarks: np.ndarray, side: str) -> float:
        """
        Tính vị trí lông mày so với mắt.
        
        Giá trị nhỏ = lông mày hạ thấp (cau mày)
        """
        if side == "left":
            brow = landmarks[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE]
            eye = landmarks[FaceLandmarkIndex.LEFT_EYE_TOP]
        else:
            brow = landmarks[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE]
            eye = landmarks[FaceLandmarkIndex.RIGHT_EYE_TOP]
        
        # Chuẩn hóa theo chiều cao mặt
        face_height = np.linalg.norm(
            landmarks[FaceLandmarkIndex.FACE_TOP][:2] - 
            landmarks[FaceLandmarkIndex.FACE_BOTTOM][:2]
        )
        
        if face_height < 1e-6:
            return 0.1
        
        brow_eye_dist = np.linalg.norm(brow[:2] - eye[:2])
        return brow_eye_dist / face_height
    
    def _nose_wrinkle_ratio(self, landmarks: np.ndarray) -> float:
        """Tính độ nhăn mũi (AU9)."""
        nose_tip = landmarks[FaceLandmarkIndex.NOSE_TIP]
        nose_bridge = landmarks[FaceLandmarkIndex.NOSE_BRIDGE]
        
        face_height = np.linalg.norm(
            landmarks[FaceLandmarkIndex.FACE_TOP][:2] - 
            landmarks[FaceLandmarkIndex.FACE_BOTTOM][:2]
        )
        
        if face_height < 1e-6:
            return 0.1
        
        nose_length = np.linalg.norm(nose_tip[:2] - nose_bridge[:2])
        return nose_length / face_height
    
    def _upper_lip_position(self, landmarks: np.ndarray) -> float:
        """Tính vị trí môi trên (AU10)."""
        upper_lip = landmarks[FaceLandmarkIndex.UPPER_LIP_TOP]
        nose_tip = landmarks[FaceLandmarkIndex.NOSE_TIP]
        
        face_height = np.linalg.norm(
            landmarks[FaceLandmarkIndex.FACE_TOP][:2] - 
            landmarks[FaceLandmarkIndex.FACE_BOTTOM][:2]
        )
        
        if face_height < 1e-6:
            return 0.1
        
        lip_nose_dist = np.linalg.norm(upper_lip[:2] - nose_tip[:2])
        return lip_nose_dist / face_height
    
    def _mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Tính tỷ lệ miệng."""
        top = landmarks[FaceLandmarkIndex.UPPER_LIP_BOTTOM]
        bottom = landmarks[FaceLandmarkIndex.LOWER_LIP_TOP]
        left = landmarks[FaceLandmarkIndex.MOUTH_LEFT]
        right = landmarks[FaceLandmarkIndex.MOUTH_RIGHT]
        
        vertical = np.linalg.norm(top[:2] - bottom[:2])
        horizontal = np.linalg.norm(left[:2] - right[:2])
        
        if horizontal < 1e-6:
            return 0.2
        
        return vertical / horizontal
    
    def _compute_au_activations(self, current: Dict[str, float]) -> Dict[str, float]:
        """Tính mức độ kích hoạt của từng AU."""
        if not self._baseline:
            return {au: 0.0 for au in self.AU_THRESHOLDS}
        
        activations = {}
        
        # AU4: Cau mày - giảm khoảng cách lông mày-mắt
        baseline_brow = self._baseline.get("eyebrow_position", 0.1)
        current_brow = current.get("eyebrow_position", 0.1)
        if baseline_brow > 1e-6:
            au4 = max(0, (baseline_brow - current_brow) / baseline_brow)
            activations["AU4"] = min(1.0, au4)
        else:
            activations["AU4"] = 0.0
        
        # AU6/AU7: Nheo mắt - giảm EAR
        baseline_ear = self._baseline.get("eye_aspect_ratio", 0.3)
        current_ear = current.get("eye_aspect_ratio", 0.3)
        if baseline_ear > 1e-6:
            eye_change = max(0, (baseline_ear - current_ear) / baseline_ear)
            activations["AU6"] = min(1.0, eye_change * 0.8)
            activations["AU7"] = min(1.0, eye_change)
        else:
            activations["AU6"] = 0.0
            activations["AU7"] = 0.0
        
        # AU43: Nhắm mắt - EAR rất thấp
        if baseline_ear > 1e-6:
            activations["AU43"] = min(1.0, max(0, (baseline_ear - current_ear) / baseline_ear))
        else:
            activations["AU43"] = 0.0
        
        # AU9: Nhăn mũi
        baseline_nose = self._baseline.get("nose_wrinkle", 0.1)
        current_nose = current.get("nose_wrinkle", 0.1)
        if baseline_nose > 1e-6:
            activations["AU9"] = min(1.0, max(0, abs(baseline_nose - current_nose) / baseline_nose))
        else:
            activations["AU9"] = 0.0
        
        # AU10: Nâng môi trên
        baseline_lip = self._baseline.get("upper_lip_raise", 0.1)
        current_lip = current.get("upper_lip_raise", 0.1)
        if baseline_lip > 1e-6:
            activations["AU10"] = min(1.0, max(0, (baseline_lip - current_lip) / baseline_lip))
        else:
            activations["AU10"] = 0.0
        
        return activations
    
    def _compute_pain_score(self, au_activations: Dict[str, float]) -> float:
        """Tính điểm đau tổng hợp (0-100)."""
        score = 0.0
        total_weight = 0.0
        
        for au, activation in au_activations.items():
            threshold = self.AU_THRESHOLDS.get(au, 0.15)
            weight = self.AU_WEIGHTS.get(au, 0.1)
            
            # Chỉ tính khi vượt ngưỡng
            if activation > threshold:
                normalized = (activation - threshold) / (1 - threshold)
                score += normalized * weight * 100
            
            total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        
        return min(100, max(0, score))
    
    def _classify_pain_level(self, score: float) -> PainLevel:
        """Phân loại mức độ đau."""
        if score >= self.PAIN_THRESHOLDS[PainLevel.SEVERE]:
            return PainLevel.SEVERE
        elif score >= self.PAIN_THRESHOLDS[PainLevel.MODERATE]:
            return PainLevel.MODERATE
        elif score >= self.PAIN_THRESHOLDS[PainLevel.MILD]:
            return PainLevel.MILD
        else:
            return PainLevel.NONE
    
    def _process_pain_event(
        self,
        level: PainLevel,
        score: float,
        au_activations: Dict[str, float],
        timestamp: float
    ) -> bool:
        """Xử lý và ghi nhận pain events."""
        is_painful = level != PainLevel.NONE
        
        if is_painful:
            if self._pain_start_time is None:
                # Bắt đầu pain event mới
                self._pain_start_time = timestamp
                self._current_pain_level = level
            
            # Cập nhật level nếu cao hơn
            if level.value > self._current_pain_level.value:
                self._current_pain_level = level
            
            # Kiểm tra duration
            duration_ms = int((timestamp - self._pain_start_time) * 1000)
            
            if duration_ms >= self.MIN_PAIN_DURATION_MS:
                return True
        else:
            # Kết thúc pain event
            if self._pain_start_time is not None:
                duration_ms = int((timestamp - self._pain_start_time) * 1000)
                
                if duration_ms >= self.MIN_PAIN_DURATION_MS:
                    # Ghi nhận event
                    event = PainEvent(
                        timestamp=self._pain_start_time,
                        level=self._current_pain_level,
                        duration_ms=duration_ms,
                        au_scores=dict(au_activations),
                        message=self._generate_message(self._current_pain_level, True)
                    )
                    self._pain_events.append(event)
                
                self._pain_start_time = None
                self._current_pain_level = PainLevel.NONE
        
        return False
    
    def _compute_confidence(self, au_activations: Dict[str, float]) -> float:
        """Tính độ tin cậy của phân tích."""
        # Confidence cao khi nhiều AU đồng nhất
        active_aus = sum(1 for v in au_activations.values() if v > 0.1)
        max_activation = max(au_activations.values()) if au_activations else 0
        
        confidence = min(1.0, (active_aus / 3) * max_activation)
        return confidence
    
    def _generate_message(self, level: PainLevel, is_detected: bool) -> str:
        """Tạo thông báo thân thiện cho người già Việt Nam."""
        if not is_detected:
            return ""
        
        messages = {
            PainLevel.MILD: "Bác ơi, có vẻ hơi mỏi rồi. Bác nghỉ một chút nhé!",
            PainLevel.MODERATE: "Bác ơi, để cháu giảm nhẹ bài tập cho bác nhé. Đừng cố quá!",
            PainLevel.SEVERE: "⚠️ Bác ơi, mình dừng lại nghỉ ngơi nhé. Sức khỏe là quan trọng nhất!",
        }
        
        return messages.get(level, "")
    
    def _get_default_baseline(self) -> Dict[str, float]:
        """Trả về baseline mặc định."""
        return {
            "eye_aspect_ratio": 0.28,
            "left_ear": 0.28,
            "right_ear": 0.28,
            "eyebrow_position": 0.08,
            "left_brow": 0.08,
            "right_brow": 0.08,
            "nose_wrinkle": 0.12,
            "upper_lip_raise": 0.10,
            "mouth_aspect_ratio": 0.15,
        }
    
    def get_pain_events(self) -> List[PainEvent]:
        """Lấy danh sách các pain events đã ghi nhận."""
        return self._pain_events.copy()
    
    def get_pain_summary(self) -> Dict:
        """Tổng kết pain analysis."""
        if not self._pain_events:
            return {
                "total_events": 0,
                "max_level": PainLevel.NONE.name,
                "total_duration_ms": 0,
            }
        
        max_level = max(e.level.value for e in self._pain_events)
        total_duration = sum(e.duration_ms for e in self._pain_events)
        
        return {
            "total_events": len(self._pain_events),
            "max_level": PainLevel(max_level).name,
            "total_duration_ms": total_duration,
            "events": [e.to_dict() for e in self._pain_events],
        }
    
    def reset(self) -> None:
        """Reset về trạng thái ban đầu."""
        self._pain_score_history.clear()
        for history in self._au_history.values():
            history.clear()
        self._pain_start_time = None
        self._current_pain_level = PainLevel.NONE
        self._pain_events = []