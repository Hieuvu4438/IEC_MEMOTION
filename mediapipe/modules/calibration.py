"""
Calibration Module for MEMOTION.

Triển khai cơ chế Safe-Max Calibration để xác định giới hạn vận động
(Range of Motion - ROM) thực tế của người già.

Ý nghĩa nhân văn:
    Mỗi người già có giới hạn vận động khác nhau do:
    - Tuổi tác và tình trạng sức khỏe
    - Tiền sử chấn thương
    - Bệnh lý mãn tính (viêm khớp, thoái hóa...)
    
    Việc calibration giúp:
    - Đặt mục tiêu tập luyện AN TOÀN, không gây đau đớn
    - Giảm áp lực tâm lý khi không đạt được tư thế "chuẩn"
    - Theo dõi tiến triển theo thời gian một cách khách quan

Author: MEMOTION Team
Version: 1.0.0
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from core.kinematics import (
    JointType,
    calculate_joint_angle,
    JOINT_DEFINITIONS,
)
from core.data_types import LandmarkSet


class CalibrationState(Enum):
    """Trạng thái của quá trình calibration."""
    IDLE = "idle"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class JointCalibrationData:
    """
    Dữ liệu calibration cho một khớp.
    
    Attributes:
        joint_type: Loại khớp.
        max_angle: Góc tối đa an toàn (đã lọc nhiễu).
        min_angle: Góc tối thiểu ghi nhận.
        raw_angles: Danh sách góc thô từ các frame.
        timestamps_ms: Timestamps tương ứng.
        confidence: Độ tin cậy của calibration (0-1).
        calibration_date: Ngày thực hiện calibration.
    """
    joint_type: str
    max_angle: float
    min_angle: float
    raw_angles: List[float] = field(default_factory=list)
    timestamps_ms: List[int] = field(default_factory=list)
    confidence: float = 0.0
    calibration_date: str = ""
    
    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary để lưu JSON."""
        return {
            "joint_type": self.joint_type,
            "max_angle": round(self.max_angle, 2),
            "min_angle": round(self.min_angle, 2),
            "confidence": round(self.confidence, 3),
            "calibration_date": self.calibration_date,
            "num_samples": len(self.raw_angles),
        }


@dataclass
class UserProfile:
    """
    Profile người dùng chứa thông số calibration.
    
    Attributes:
        user_id: ID định danh người dùng.
        name: Tên hiển thị.
        age: Tuổi (quan trọng cho việc đánh giá ROM).
        joint_limits: Dict mapping JointType → JointCalibrationData.
        created_at: Ngày tạo profile.
        last_calibration: Ngày calibration gần nhất.
        notes: Ghi chú về tình trạng sức khỏe.
    """
    user_id: str
    name: str = ""
    age: int = 0
    joint_limits: Dict[str, JointCalibrationData] = field(default_factory=dict)
    created_at: str = ""
    last_calibration: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary để lưu JSON."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "age": self.age,
            "created_at": self.created_at,
            "last_calibration": self.last_calibration,
            "notes": self.notes,
            "joint_limits": {
                k: v.to_dict() for k, v in self.joint_limits.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Tạo UserProfile từ dictionary."""
        joint_limits = {}
        for k, v in data.get("joint_limits", {}).items():
            joint_limits[k] = JointCalibrationData(
                joint_type=v["joint_type"],
                max_angle=v["max_angle"],
                min_angle=v["min_angle"],
                confidence=v.get("confidence", 0.0),
                calibration_date=v.get("calibration_date", ""),
            )
        
        return cls(
            user_id=data["user_id"],
            name=data.get("name", ""),
            age=data.get("age", 0),
            joint_limits=joint_limits,
            created_at=data.get("created_at", ""),
            last_calibration=data.get("last_calibration", ""),
            notes=data.get("notes", ""),
        )
    
    def get_max_angle(self, joint_type: JointType) -> Optional[float]:
        """Lấy góc tối đa của một khớp."""
        key = joint_type.value
        if key in self.joint_limits:
            return self.joint_limits[key].max_angle
        return None


class SafeMaxCalibrator:
    """
    Bộ calibration xác định giới hạn vận động an toàn.
    
    Quy trình calibration:
        1. Người dùng được hướng dẫn thực hiện động tác "hết khả năng"
           (nhưng KHÔNG GÂY ĐAU)
        2. Hệ thống thu thập góc khớp trong 5-10 giây
        3. Áp dụng bộ lọc nhiễu (Median Filter) để loại bỏ outliers
        4. Trích xuất giá trị max ổn định làm θ_user_max
        
    Tại sao cần lọc nhiễu?
        - MediaPipe có thể "nhảy" tọa độ do occlusion hoặc blur
        - Người già có thể run tay, gây dao động nhỏ
        - Cần lấy giá trị ĐẠI DIỆN, không phải giá trị cực đoan nhất
    
    Example:
        >>> calibrator = SafeMaxCalibrator()
        >>> calibrator.start_calibration(JointType.LEFT_ELBOW)
        >>> 
        >>> # Trong vòng lặp video...
        >>> calibrator.add_frame(landmarks, timestamp_ms)
        >>> 
        >>> # Sau khi thu thập đủ
        >>> result = calibrator.finish_calibration()
        >>> print(f"Max angle: {result.max_angle}°")
    """
    
    # Cấu hình mặc định
    DEFAULT_DURATION_MS = 5000  # 5 giây
    MIN_SAMPLES = 30  # Ít nhất 30 frames
    MEDIAN_WINDOW_SIZE = 5  # Kích thước cửa sổ median filter
    STABILITY_THRESHOLD = 5.0  # Ngưỡng ổn định (degrees)
    
    def __init__(
        self,
        duration_ms: int = DEFAULT_DURATION_MS,
        min_samples: int = MIN_SAMPLES,
    ):
        """
        Khởi tạo SafeMaxCalibrator.
        
        Args:
            duration_ms: Thời gian thu thập (milliseconds).
            min_samples: Số mẫu tối thiểu cần thu thập.
        """
        self._duration_ms = duration_ms
        self._min_samples = min_samples
        
        self._state = CalibrationState.IDLE
        self._current_joint: Optional[JointType] = None
        self._collected_angles: List[float] = []
        self._collected_timestamps: List[int] = []
        self._start_timestamp: Optional[int] = None
        
        self._user_profile: Optional[UserProfile] = None
    
    @property
    def state(self) -> CalibrationState:
        """Trạng thái hiện tại."""
        return self._state
    
    @property
    def current_joint(self) -> Optional[JointType]:
        """Khớp đang calibrate."""
        return self._current_joint
    
    @property
    def progress(self) -> float:
        """Tiến độ calibration (0-1)."""
        if self._state != CalibrationState.COLLECTING:
            return 0.0 if self._state == CalibrationState.IDLE else 1.0
        
        if self._start_timestamp is None or not self._collected_timestamps:
            return 0.0
        
        elapsed = self._collected_timestamps[-1] - self._start_timestamp
        return min(1.0, elapsed / self._duration_ms)
    
    @property
    def elapsed_ms(self) -> int:
        """Thời gian đã thu thập (ms)."""
        if self._start_timestamp is None or not self._collected_timestamps:
            return 0
        return self._collected_timestamps[-1] - self._start_timestamp
    
    def start_calibration(
        self,
        joint_type: JointType,
        user_profile: Optional[UserProfile] = None
    ) -> None:
        """
        Bắt đầu quá trình calibration cho một khớp.
        
        Args:
            joint_type: Loại khớp cần calibrate.
            user_profile: Profile người dùng (tạo mới nếu None).
        """
        self._current_joint = joint_type
        self._collected_angles = []
        self._collected_timestamps = []
        self._start_timestamp = None
        self._state = CalibrationState.COLLECTING
        
        if user_profile is not None:
            self._user_profile = user_profile
        elif self._user_profile is None:
            self._user_profile = UserProfile(
                user_id=f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        joint_name = JOINT_DEFINITIONS[joint_type].name
        print(f"[CALIBRATION] Bắt đầu calibration cho: {joint_name}")
        print(f"[CALIBRATION] Hãy thực hiện động tác HẾT KHẢ NĂNG (không gây đau)")
        print(f"[CALIBRATION] Thu thập trong {self._duration_ms/1000:.1f} giây...")
    
    def add_frame(
        self,
        landmarks: LandmarkSet,
        timestamp_ms: int
    ) -> Tuple[bool, float]:
        """
        Thêm một frame vào quá trình calibration.
        
        Args:
            landmarks: Pose landmarks từ detector.
            timestamp_ms: Timestamp của frame.
            
        Returns:
            Tuple[bool, float]: (is_valid, current_angle)
                - is_valid: True nếu frame được chấp nhận
                - current_angle: Góc hiện tại (hoặc 0 nếu lỗi)
        """
        if self._state != CalibrationState.COLLECTING:
            return False, 0.0
        
        if self._current_joint is None:
            return False, 0.0
        
        # Tính góc
        try:
            angle = calculate_joint_angle(
                landmarks.to_numpy(),
                self._current_joint,
                use_3d=True
            )
        except (ValueError, IndexError) as e:
            return False, 0.0
        
        # Ghi nhận timestamp bắt đầu
        if self._start_timestamp is None:
            self._start_timestamp = timestamp_ms
        
        # Thêm vào danh sách
        self._collected_angles.append(angle)
        self._collected_timestamps.append(timestamp_ms)
        
        # Kiểm tra đã đủ thời gian chưa
        if self.elapsed_ms >= self._duration_ms:
            self._auto_finish()
        
        return True, angle
    
    def _auto_finish(self) -> None:
        """Tự động kết thúc khi đủ thời gian."""
        if self._state == CalibrationState.COLLECTING:
            print(f"\n[CALIBRATION] Đã thu thập {len(self._collected_angles)} samples")
            self.finish_calibration()
    
    def finish_calibration(self) -> Optional[JointCalibrationData]:
        """
        Kết thúc calibration và tính toán kết quả.
        
        Thuật toán lọc nhiễu:
            1. Áp dụng Median Filter để làm mượt chuỗi góc
            2. Loại bỏ outliers (góc nằm ngoài 2 độ lệch chuẩn)
            3. Lấy percentile 95 thay vì max tuyệt đối
               (tránh các spike do lỗi tracking)
        
        Returns:
            JointCalibrationData hoặc None nếu không đủ dữ liệu.
        """
        if self._state != CalibrationState.COLLECTING:
            return None
        
        self._state = CalibrationState.PROCESSING
        
        if len(self._collected_angles) < self._min_samples:
            self._state = CalibrationState.ERROR
            print(f"[CALIBRATION] Lỗi: Chỉ thu được {len(self._collected_angles)}/{self._min_samples} samples")
            return None
        
        angles = np.array(self._collected_angles)
        
        # Step 1: Median Filter để làm mượt
        smoothed_angles = self._median_filter(angles)
        
        # Step 2: Loại bỏ outliers
        filtered_angles = self._remove_outliers(smoothed_angles)
        
        if len(filtered_angles) < 10:
            self._state = CalibrationState.ERROR
            print("[CALIBRATION] Lỗi: Quá nhiều outliers, dữ liệu không ổn định")
            return None
        
        # Step 3: Tính max ổn định (percentile 95)
        # Tại sao percentile 95 thay vì max?
        # → Tránh các giá trị cực đoan do lỗi tracking
        # → Lấy giá trị mà người dùng ĐẠT ĐƯỢC ỔN ĐỊNH
        max_angle = float(np.percentile(filtered_angles, 95))
        min_angle = float(np.percentile(filtered_angles, 5))
        
        # Tính độ tin cậy dựa trên độ ổn định
        std_dev = float(np.std(filtered_angles))
        # Độ tin cậy cao nếu std thấp
        confidence = max(0.0, 1.0 - (std_dev / 30.0))
        
        # Tạo kết quả
        result = JointCalibrationData(
            joint_type=self._current_joint.value,
            max_angle=max_angle,
            min_angle=min_angle,
            raw_angles=self._collected_angles,
            timestamps_ms=self._collected_timestamps,
            confidence=confidence,
            calibration_date=datetime.now().isoformat(),
        )
        
        # Lưu vào profile
        if self._user_profile is not None:
            self._user_profile.joint_limits[self._current_joint.value] = result
            self._user_profile.last_calibration = datetime.now().isoformat()
        
        self._state = CalibrationState.COMPLETED
        
        # In kết quả
        joint_name = JOINT_DEFINITIONS[self._current_joint].name
        print(f"\n[CALIBRATION] ✓ Hoàn thành calibration: {joint_name}")
        print(f"  → Góc tối đa (θ_user_max): {max_angle:.1f}°")
        print(f"  → Góc tối thiểu: {min_angle:.1f}°")
        print(f"  → Biên độ vận động: {max_angle - min_angle:.1f}°")
        print(f"  → Độ tin cậy: {confidence:.1%}")
        
        return result
    
    def _median_filter(self, angles: np.ndarray) -> np.ndarray:
        """
        Áp dụng Median Filter để làm mượt chuỗi góc.
        
        Tại sao Median Filter thay vì Mean?
        → Median không bị ảnh hưởng bởi outliers
        → Bảo toàn các giá trị cực đại thực sự
        
        Args:
            angles: Chuỗi góc gốc.
            
        Returns:
            np.ndarray: Chuỗi góc đã làm mượt.
        """
        if len(angles) < self.MEDIAN_WINDOW_SIZE:
            return angles
        
        result = np.zeros_like(angles)
        half_window = self.MEDIAN_WINDOW_SIZE // 2
        
        for i in range(len(angles)):
            start = max(0, i - half_window)
            end = min(len(angles), i + half_window + 1)
            result[i] = np.median(angles[start:end])
        
        return result
    
    def _remove_outliers(
        self,
        angles: np.ndarray,
        num_std: float = 2.0
    ) -> np.ndarray:
        """
        Loại bỏ outliers dựa trên độ lệch chuẩn.
        
        Args:
            angles: Chuỗi góc.
            num_std: Số độ lệch chuẩn để xác định outlier.
            
        Returns:
            np.ndarray: Chuỗi góc đã loại bỏ outliers.
        """
        mean = np.mean(angles)
        std = np.std(angles)
        
        if std < 1e-10:
            return angles
        
        lower_bound = mean - num_std * std
        upper_bound = mean + num_std * std
        
        mask = (angles >= lower_bound) & (angles <= upper_bound)
        return angles[mask]
    
    def get_user_limit(self, joint_type: JointType) -> Optional[float]:
        """
        Lấy giới hạn góc của một khớp từ profile.
        
        Args:
            joint_type: Loại khớp.
            
        Returns:
            float hoặc None nếu chưa calibrate.
        """
        if self._user_profile is None:
            return None
        return self._user_profile.get_max_angle(joint_type)
    
    def get_profile(self) -> Optional[UserProfile]:
        """Lấy user profile hiện tại."""
        return self._user_profile
    
    def save_profile(
        self,
        path: str,
        create_dirs: bool = True
    ) -> bool:
        """
        Lưu user profile vào file JSON.
        
        Args:
            path: Đường dẫn file hoặc thư mục.
            create_dirs: Tự động tạo thư mục nếu chưa có.
            
        Returns:
            bool: True nếu lưu thành công.
        """
        if self._user_profile is None:
            print("[ERROR] Không có profile để lưu")
            return False
        
        file_path = Path(path)
        
        # Nếu path là thư mục, tạo tên file từ user_id
        if file_path.is_dir() or not file_path.suffix:
            if create_dirs:
                file_path.mkdir(parents=True, exist_ok=True)
            file_path = file_path / f"{self._user_profile.user_id}.json"
        else:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self._user_profile.to_dict(),
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            print(f"[CALIBRATION] Đã lưu profile: {file_path}")
            return True
        except IOError as e:
            print(f"[ERROR] Không thể lưu profile: {e}")
            return False
    
    def load_profile(self, path: str) -> Optional[UserProfile]:
        """
        Tải user profile từ file JSON.
        
        Args:
            path: Đường dẫn file JSON.
            
        Returns:
            UserProfile hoặc None nếu lỗi.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._user_profile = UserProfile.from_dict(data)
            print(f"[CALIBRATION] Đã tải profile: {self._user_profile.user_id}")
            return self._user_profile
        except (IOError, json.JSONDecodeError) as e:
            print(f"[ERROR] Không thể tải profile: {e}")
            return None
    
    def reset(self) -> None:
        """Reset về trạng thái ban đầu."""
        self._state = CalibrationState.IDLE
        self._current_joint = None
        self._collected_angles = []
        self._collected_timestamps = []
        self._start_timestamp = None