"""
Motion Synchronizer Module for MEMOTION.

Triển khai Finite State Machine (FSM) để điều khiển video mẫu
đồng bộ với chuyển động của người già.

Ý nghĩa nhân văn:
    Người già thường di chuyển chậm hơn video mẫu. Thay vì ép họ
    theo kịp tốc độ (gây stress và nguy hiểm), hệ thống sẽ:
    - CHỜ người dùng hoàn thành từng pha
    - KHÔNG PHÁN XÉT về tốc độ
    - KHUYẾN KHÍCH bằng phản hồi tích cực

Mô hình FSM cho một động tác:
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   IDLE ──► ECCENTRIC ──► HOLD ──► CONCENTRIC ──┐    │
    │     ▲                                          │    │
    │     └──────────────────────────────────────────┘    │
    │                                                      │
    └──────────────────────────────────────────────────────┘

Giải thích các pha:
    - IDLE: Tư thế nghỉ, chuẩn bị bắt đầu
    - ECCENTRIC: Pha "đi ra" - cơ duỗi ra (vd: hạ người xuống squat)
    - HOLD: Giữ tại điểm cao trào (vd: đáy squat)
    - CONCENTRIC: Pha "đi về" - cơ co lại (vd: đứng lên từ squat)

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Callable, Dict
import time
import numpy as np

from .kinematics import JointType


class MotionPhase(Enum):
    """
    Các pha của một động tác tập luyện.
    
    Một rep (repetition) hoàn chỉnh gồm 4 pha theo thứ tự:
    IDLE → ECCENTRIC → HOLD → CONCENTRIC → IDLE
    """
    IDLE = "idle"                  # Tư thế nghỉ/bắt đầu
    ECCENTRIC = "eccentric"        # Pha duỗi cơ (đi xa tâm)
    HOLD = "hold"                  # Giữ tại điểm cao trào
    CONCENTRIC = "concentric"      # Pha co cơ (đi về tâm)


class SyncStatus(Enum):
    """
    Trạng thái đồng bộ giữa người dùng và video mẫu.
    
    Được sử dụng để điều khiển video player.
    """
    PLAY = "play"          # Video chạy bình thường
    PAUSE = "pause"        # Video tạm dừng chờ user
    LOOP = "loop"          # Video lặp đoạn hiện tại
    SKIP = "skip"          # Bỏ qua (user đã vượt qua)
    COMPLETE = "complete"  # Hoàn thành bài tập


@dataclass
class PhaseCheckpoint:
    """
    Điểm mốc (checkpoint) trong video mẫu.
    
    Mỗi checkpoint đánh dấu ranh giới giữa các pha.
    Video sẽ chờ tại checkpoint nếu user chưa đạt ngưỡng.
    
    Attributes:
        frame_index: Số frame trong video mẫu.
        phase_start: Pha bắt đầu từ checkpoint này.
        target_angle: Góc mục tiêu cần đạt để qua checkpoint.
        tolerance: Sai số cho phép (degrees).
        message: Thông báo hiển thị cho người dùng.
    """
    frame_index: int
    phase_start: MotionPhase
    target_angle: float
    tolerance: float = 10.0
    message: str = ""
    
    def is_reached(self, user_angle: float) -> bool:
        """Kiểm tra user đã đạt checkpoint chưa."""
        return abs(user_angle - self.target_angle) <= self.tolerance


@dataclass
class ExerciseDefinition:
    """
    Định nghĩa một bài tập với các checkpoints.
    
    Attributes:
        name: Tên bài tập.
        joint_type: Khớp chính cần theo dõi.
        checkpoints: Danh sách các checkpoint.
        total_frames: Tổng số frame của video mẫu.
        angle_increasing: True nếu góc tăng trong pha ECCENTRIC.
    """
    name: str
    joint_type: JointType
    checkpoints: List[PhaseCheckpoint]
    total_frames: int
    angle_increasing: bool = True  # True: góc tăng khi ECCENTRIC (vd: giơ tay)
    
    def get_current_phase(self, frame_index: int) -> MotionPhase:
        """Xác định pha hiện tại dựa trên frame index."""
        current_phase = MotionPhase.IDLE
        for cp in self.checkpoints:
            if frame_index >= cp.frame_index:
                current_phase = cp.phase_start
        return current_phase
    
    def get_next_checkpoint(self, frame_index: int) -> Optional[PhaseCheckpoint]:
        """Lấy checkpoint tiếp theo."""
        for cp in self.checkpoints:
            if cp.frame_index > frame_index:
                return cp
        return None


@dataclass
class SyncState:
    """
    Trạng thái đồng bộ hiện tại.
    
    Được cập nhật liên tục trong quá trình tập luyện.
    """
    current_phase: MotionPhase = MotionPhase.IDLE
    sync_status: SyncStatus = SyncStatus.PLAY
    
    # Frame tracking
    ref_frame: int = 0          # Frame hiện tại của video mẫu
    user_frame: int = 0         # "Frame tương đương" của user
    
    # Angle tracking
    user_angle: float = 0.0
    target_angle: float = 0.0
    angle_error: float = 0.0
    
    # Timing
    phase_start_time: float = 0.0
    wait_start_time: Optional[float] = None
    total_wait_time: float = 0.0
    
    # Rep counting
    rep_count: int = 0
    
    # Messages
    status_message: str = ""
    encouragement: str = ""


class MotionSyncController:
    """
    Bộ điều khiển đồng bộ chuyển động.
    
    Quản lý FSM và quyết định khi nào video mẫu nên:
    - Chạy tiếp (PLAY)
    - Tạm dừng (PAUSE)
    - Lặp lại (LOOP)
    
    Nguyên tắc "Wait-for-User":
        1. Video chạy bình thường cho đến checkpoint
        2. Tại checkpoint, kiểm tra user đã đạt ngưỡng chưa
        3. Nếu chưa → PAUSE/LOOP cho đến khi user đạt
        4. Khi user đạt → tiếp tục đến checkpoint tiếp theo
    
    Example:
        >>> exercise = create_arm_raise_exercise(total_frames=300)
        >>> controller = MotionSyncController(exercise)
        >>> 
        >>> # Trong vòng lặp video
        >>> state = controller.update(user_angle=45.0, ref_frame=100)
        >>> if state.sync_status == SyncStatus.PAUSE:
        ...     # Không tăng ref_frame
        ...     print(state.status_message)
    """
    
    # Thời gian chờ tối đa trước khi bỏ qua (giây)
    MAX_WAIT_TIME = 10.0
    
    # Ngưỡng để phát hiện user bắt đầu di chuyển
    MOTION_THRESHOLD = 5.0  # degrees
    
    # Thông điệp khuyến khích
    ENCOURAGEMENT_MESSAGES = [
        "Tuyệt vời! Bà đang làm rất tốt!",
        "Cố lên! Gần đạt rồi!",
        "Chậm thôi, không vội đâu!",
        "Bà làm đúng rồi, tiếp tục nhé!",
        "Rất tốt! Giữ nhịp như vậy!",
    ]
    
    def __init__(
        self,
        exercise: ExerciseDefinition,
        user_max_angle: Optional[float] = None,
        challenge_factor: float = 0.05
    ):
        """
        Khởi tạo MotionSyncController.
        
        Args:
            exercise: Định nghĩa bài tập.
            user_max_angle: Góc tối đa của user (từ calibration).
            challenge_factor: Hệ số thử thách.
        """
        self._exercise = exercise
        self._user_max_angle = user_max_angle
        self._challenge_factor = challenge_factor
        
        self._state = SyncState()
        self._angle_history: List[float] = []
        self._timestamp_history: List[float] = []
        
        # Callbacks
        self._on_phase_change: Optional[Callable[[MotionPhase, MotionPhase], None]] = None
        self._on_rep_complete: Optional[Callable[[int], None]] = None
        
        # Rescale checkpoints nếu có user_max_angle
        if user_max_angle is not None:
            self._rescale_checkpoints()
    
    @property
    def state(self) -> SyncState:
        """Trạng thái hiện tại."""
        return self._state
    
    @property
    def exercise(self) -> ExerciseDefinition:
        """Định nghĩa bài tập."""
        return self._exercise
    
    @property
    def angle_history(self) -> List[float]:
        """Lịch sử góc khớp của user."""
        return self._angle_history.copy()
    
    def _rescale_checkpoints(self) -> None:
        """
        Điều chỉnh target angles theo khả năng của user.
        
        Công thức (từ GĐ2):
            θ_target = θ_ref × (θ_user_max / max(θ_ref)) × (1 + α)
        """
        if self._user_max_angle is None:
            return
        
        # Tìm góc max trong các checkpoints
        ref_max = max(cp.target_angle for cp in self._exercise.checkpoints)
        if ref_max < 1e-6:
            return
        
        scale = (self._user_max_angle / ref_max) * (1 + self._challenge_factor)
        scale = min(scale, 1.0)  # Không vượt quá mẫu
        
        for cp in self._exercise.checkpoints:
            cp.target_angle *= scale
    
    def update(
        self,
        user_angle: float,
        ref_frame: int,
        timestamp: Optional[float] = None
    ) -> SyncState:
        """
        Cập nhật trạng thái đồng bộ.
        
        Đây là hàm chính được gọi mỗi frame.
        
        Args:
            user_angle: Góc khớp hiện tại của user (degrees).
            ref_frame: Frame hiện tại của video mẫu.
            timestamp: Timestamp (seconds), None để tự động.
            
        Returns:
            SyncState: Trạng thái mới.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Lưu lịch sử
        self._angle_history.append(user_angle)
        self._timestamp_history.append(timestamp)
        
        # Giới hạn history
        max_history = 1000
        if len(self._angle_history) > max_history:
            self._angle_history = self._angle_history[-max_history:]
            self._timestamp_history = self._timestamp_history[-max_history:]
        
        # Cập nhật state cơ bản
        self._state.user_angle = user_angle
        self._state.ref_frame = ref_frame
        
        # Xác định pha hiện tại
        current_phase = self._exercise.get_current_phase(ref_frame)
        old_phase = self._state.current_phase
        
        if current_phase != old_phase:
            self._on_phase_changed(old_phase, current_phase, timestamp)
        
        self._state.current_phase = current_phase
        
        # Kiểm tra checkpoint
        sync_status = self._check_sync_status(user_angle, ref_frame, timestamp)
        self._state.sync_status = sync_status
        
        # Cập nhật thông điệp
        self._update_messages()
        
        return self._state
    
    def _check_sync_status(
        self,
        user_angle: float,
        ref_frame: int,
        timestamp: float
    ) -> SyncStatus:
        """
        Quyết định video mẫu nên chạy hay dừng.
        
        Logic Wait-for-User:
            1. Tìm checkpoint tiếp theo
            2. Nếu ref_frame đến checkpoint nhưng user chưa đạt → PAUSE
            3. Nếu chờ quá lâu → SKIP (để không block mãi)
            4. Nếu user đạt → PLAY
        """
        next_cp = self._exercise.get_next_checkpoint(ref_frame - 1)
        
        # Không có checkpoint tiếp theo → check hoàn thành
        if next_cp is None:
            if ref_frame >= self._exercise.total_frames - 1:
                return SyncStatus.COMPLETE
            return SyncStatus.PLAY
        
        # Chưa đến checkpoint → chạy bình thường
        if ref_frame < next_cp.frame_index:
            self._state.wait_start_time = None
            return SyncStatus.PLAY
        
        # Đã đến checkpoint - kiểm tra user
        self._state.target_angle = next_cp.target_angle
        self._state.angle_error = user_angle - next_cp.target_angle
        
        if next_cp.is_reached(user_angle):
            # User đạt checkpoint → tiếp tục
            self._state.wait_start_time = None
            return SyncStatus.PLAY
        
        # User chưa đạt → cần chờ
        if self._state.wait_start_time is None:
            self._state.wait_start_time = timestamp
        
        wait_duration = timestamp - self._state.wait_start_time
        self._state.total_wait_time += wait_duration
        
        # Chờ quá lâu → bỏ qua
        if wait_duration > self.MAX_WAIT_TIME:
            self._state.wait_start_time = None
            self._state.status_message = "Không sao, ta tiếp tục nhé!"
            return SyncStatus.SKIP
        
        # Còn trong thời gian chờ
        return SyncStatus.PAUSE
    
    def _on_phase_changed(
        self,
        old_phase: MotionPhase,
        new_phase: MotionPhase,
        timestamp: float
    ) -> None:
        """Xử lý khi chuyển pha."""
        self._state.phase_start_time = timestamp
        
        # Đếm rep khi hoàn thành CONCENTRIC
        if old_phase == MotionPhase.CONCENTRIC and new_phase == MotionPhase.IDLE:
            self._state.rep_count += 1
            if self._on_rep_complete:
                self._on_rep_complete(self._state.rep_count)
        
        if self._on_phase_change:
            self._on_phase_change(old_phase, new_phase)
    
    def _update_messages(self) -> None:
        """Cập nhật thông điệp trạng thái và khuyến khích."""
        phase = self._state.current_phase
        status = self._state.sync_status
        
        # Thông điệp theo pha
        phase_messages = {
            MotionPhase.IDLE: "Chuẩn bị sẵn sàng...",
            MotionPhase.ECCENTRIC: "Từ từ thực hiện động tác...",
            MotionPhase.HOLD: "Giữ tư thế này...",
            MotionPhase.CONCENTRIC: "Từ từ trở về...",
        }
        
        if status == SyncStatus.PAUSE:
            # Đang chờ user
            angle_diff = self._state.target_angle - self._state.user_angle
            if angle_diff > 0:
                self._state.status_message = f"Cố thêm {abs(angle_diff):.0f}° nữa!"
            else:
                self._state.status_message = f"Giảm {abs(angle_diff):.0f}° đi!"
        elif status == SyncStatus.COMPLETE:
            self._state.status_message = f"Hoàn thành! {self._state.rep_count} lần tập."
        else:
            self._state.status_message = phase_messages.get(phase, "")
        
        # Chọn ngẫu nhiên lời khuyến khích
        if len(self._angle_history) % 30 == 0:  # Mỗi ~1 giây
            idx = len(self._angle_history) % len(self.ENCOURAGEMENT_MESSAGES)
            self._state.encouragement = self.ENCOURAGEMENT_MESSAGES[idx]
    
    def check_sync_status(
        self,
        user_angle: float,
        ref_angle: float,
        current_phase: MotionPhase
    ) -> SyncStatus:
        """
        API đơn giản để kiểm tra trạng thái đồng bộ.
        
        Dùng khi không cần full state machine.
        
        Args:
            user_angle: Góc của user.
            ref_angle: Góc trong video mẫu.
            current_phase: Pha hiện tại.
            
        Returns:
            SyncStatus: PLAY, PAUSE, hoặc LOOP.
        """
        tolerance = 15.0  # degrees
        
        if current_phase == MotionPhase.HOLD:
            # Trong pha HOLD, cần khớp chặt hơn
            tolerance = 10.0
        
        angle_diff = abs(user_angle - ref_angle)
        
        if angle_diff <= tolerance:
            return SyncStatus.PLAY
        elif angle_diff <= tolerance * 2:
            return SyncStatus.LOOP
        else:
            return SyncStatus.PAUSE
    
    def get_sequence_for_dtw(self) -> Tuple[List[float], List[float]]:
        """
        Lấy chuỗi góc để tính DTW.
        
        Returns:
            Tuple[angles, timestamps]: Chuỗi góc và timestamps.
        """
        return self._angle_history.copy(), self._timestamp_history.copy()
    
    def reset(self) -> None:
        """Reset về trạng thái ban đầu."""
        self._state = SyncState()
        self._angle_history = []
        self._timestamp_history = []
    
    def set_on_phase_change(
        self,
        callback: Callable[[MotionPhase, MotionPhase], None]
    ) -> None:
        """Đặt callback khi chuyển pha."""
        self._on_phase_change = callback
    
    def set_on_rep_complete(
        self,
        callback: Callable[[int], None]
    ) -> None:
        """Đặt callback khi hoàn thành 1 rep."""
        self._on_rep_complete = callback


def create_arm_raise_exercise(
    total_frames: int,
    fps: float = 30.0,
    max_angle: float = 150.0
) -> ExerciseDefinition:
    """
    Tạo định nghĩa bài tập Giơ tay (Arm Raise).
    
    Động tác: Đứng thẳng → Giơ tay lên cao → Giữ → Hạ xuống
    Khớp theo dõi: Vai (Shoulder)
    
    Args:
        total_frames: Tổng số frame của video mẫu.
        fps: Frame rate.
        max_angle: Góc tối đa khi giơ tay.
        
    Returns:
        ExerciseDefinition: Định nghĩa bài tập.
    """
    # Chia video thành 4 phần
    # 0-25%: IDLE (chuẩn bị)
    # 25-50%: ECCENTRIC (giơ tay lên)
    # 50-60%: HOLD (giữ)
    # 60-100%: CONCENTRIC (hạ tay xuống)
    
    checkpoints = [
        PhaseCheckpoint(
            frame_index=0,
            phase_start=MotionPhase.IDLE,
            target_angle=30.0,  # Tay buông thõng
            message="Đứng thẳng, tay buông tự nhiên"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.25),
            phase_start=MotionPhase.ECCENTRIC,
            target_angle=30.0,  # Bắt đầu giơ
            message="Từ từ giơ tay lên"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.50),
            phase_start=MotionPhase.HOLD,
            target_angle=max_angle,  # Đỉnh
            message="Giữ tay ở vị trí cao nhất"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.60),
            phase_start=MotionPhase.CONCENTRIC,
            target_angle=max_angle * 0.9,  # Bắt đầu hạ
            message="Từ từ hạ tay xuống"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.95),
            phase_start=MotionPhase.IDLE,
            target_angle=30.0,  # Về vị trí ban đầu
            message="Nghỉ ngơi"
        ),
    ]
    
    return ExerciseDefinition(
        name="Giơ tay (Arm Raise)",
        joint_type=JointType.LEFT_SHOULDER,
        checkpoints=checkpoints,
        total_frames=total_frames,
        angle_increasing=True
    )


def create_squat_exercise(
    total_frames: int,
    fps: float = 30.0,
    max_angle: float = 90.0
) -> ExerciseDefinition:
    """
    Tạo định nghĩa bài tập Squat.
    
    Động tác: Đứng thẳng → Ngồi xuống → Giữ → Đứng lên
    Khớp theo dõi: Đầu gối (Knee)
    
    Lưu ý: Với Squat, góc đầu gối GIẢM khi ngồi xuống,
    nên angle_increasing=False.
    """
    checkpoints = [
        PhaseCheckpoint(
            frame_index=0,
            phase_start=MotionPhase.IDLE,
            target_angle=170.0,  # Chân thẳng
            message="Đứng thẳng, chân rộng bằng vai"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.20),
            phase_start=MotionPhase.ECCENTRIC,
            target_angle=170.0,
            message="Từ từ ngồi xuống"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.45),
            phase_start=MotionPhase.HOLD,
            target_angle=max_angle,  # Đáy squat
            message="Giữ tư thế ngồi"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.55),
            phase_start=MotionPhase.CONCENTRIC,
            target_angle=max_angle + 10,
            message="Từ từ đứng lên"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.95),
            phase_start=MotionPhase.IDLE,
            target_angle=170.0,
            message="Nghỉ ngơi"
        ),
    ]
    
    return ExerciseDefinition(
        name="Squat (Ngồi xổm)",
        joint_type=JointType.LEFT_KNEE,
        checkpoints=checkpoints,
        total_frames=total_frames,
        angle_increasing=False  # Góc giảm khi ngồi xuống
    )


def create_elbow_flex_exercise(
    total_frames: int,
    fps: float = 30.0,
    max_angle: float = 145.0
) -> ExerciseDefinition:
    """
    Tạo định nghĩa bài tập Gập khuỷu tay (Bicep Curl).
    
    Động tác: Tay thẳng → Gập lên → Giữ → Duỗi ra
    Khớp theo dõi: Khuỷu tay (Elbow)
    """
    checkpoints = [
        PhaseCheckpoint(
            frame_index=0,
            phase_start=MotionPhase.IDLE,
            target_angle=160.0,  # Tay gần thẳng
            message="Tay buông thẳng tự nhiên"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.20),
            phase_start=MotionPhase.ECCENTRIC,
            target_angle=160.0,
            message="Từ từ gập tay lên"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.50),
            phase_start=MotionPhase.HOLD,
            target_angle=max_angle,  # Gập tối đa
            tolerance=15.0,
            message="Giữ tay gập"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.60),
            phase_start=MotionPhase.CONCENTRIC,
            target_angle=max_angle - 10,
            message="Từ từ duỗi tay ra"
        ),
        PhaseCheckpoint(
            frame_index=int(total_frames * 0.95),
            phase_start=MotionPhase.IDLE,
            target_angle=160.0,
            message="Nghỉ ngơi"
        ),
    ]
    
    return ExerciseDefinition(
        name="Gập khuỷu tay (Bicep Curl)",
        joint_type=JointType.LEFT_ELBOW,
        checkpoints=checkpoints,
        total_frames=total_frames,
        angle_increasing=False  # Góc giảm khi gập
    )