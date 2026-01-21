"""
Video Engine Module for MEMOTION.

Cung cấp Smart Video Player với khả năng:
- Tạm dừng tại checkpoint chờ người dùng
- Lặp lại đoạn video khi cần
- Nhảy đến frame cụ thể
- Điều khiển tốc độ phát

Thiết kế tách biệt logic và UI:
    VideoEngine chỉ quản lý frame data và trạng thái.
    Việc hiển thị do caller quyết định.

Author: MEMOTION Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class PlaybackState(Enum):
    """Trạng thái của video player."""
    STOPPED = auto()    # Đã dừng hoàn toàn
    PLAYING = auto()    # Đang phát
    PAUSED = auto()     # Tạm dừng
    LOOPING = auto()    # Đang lặp đoạn
    SEEKING = auto()    # Đang seek
    FINISHED = auto()   # Phát xong


@dataclass
class VideoInfo:
    """Thông tin về video."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str = ""


@dataclass
class PlaybackStatus:
    """
    Trạng thái phát hiện tại.
    
    Được trả về mỗi lần gọi get_frame().
    """
    state: PlaybackState
    current_frame: int
    current_time_ms: int
    progress: float  # 0.0 - 1.0
    is_at_checkpoint: bool = False
    checkpoint_message: str = ""
    loop_count: int = 0


class VideoEngine:
    """
    Smart Video Player cho MEMOTION.
    
    Hỗ trợ các chế độ đặc biệt:
    - Wait-at-checkpoint: Dừng tại các điểm mốc
    - Loop-segment: Lặp lại một đoạn
    - Speed control: Điều chỉnh tốc độ
    
    Example:
        >>> engine = VideoEngine("exercise.mp4")
        >>> engine.set_checkpoints([100, 200, 300])
        >>> engine.play()
        >>> 
        >>> while True:
        ...     frame, status = engine.get_frame()
        ...     if status.state == PlaybackState.FINISHED:
        ...         break
        ...     if status.is_at_checkpoint:
        ...         # Chờ user
        ...         engine.pause()
        ...     cv2.imshow("Video", frame)
    """
    
    def __init__(self, video_path: str):
        """
        Khởi tạo VideoEngine.
        
        Args:
            video_path: Đường dẫn đến file video.
            
        Raises:
            FileNotFoundError: Nếu video không tồn tại.
            RuntimeError: Nếu không thể mở video.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        
        self._path = Path(video_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._state = PlaybackState.STOPPED
        
        # Frame tracking
        self._current_frame = 0
        self._target_frame = 0
        
        # Checkpoints
        self._checkpoints: List[int] = []
        self._checkpoint_messages: dict = {}
        self._current_checkpoint_idx = 0
        
        # Loop control
        self._loop_start = 0
        self._loop_end = 0
        self._loop_count = 0
        self._max_loops = 3
        
        # Speed control
        self._speed_factor = 1.0
        
        # Timing
        self._last_frame_time = 0.0
        self._frame_interval = 0.0
        
        # Callbacks
        self._on_checkpoint: Optional[Callable[[int, str], None]] = None
        self._on_loop: Optional[Callable[[int], None]] = None
        self._on_finish: Optional[Callable[[], None]] = None
        
        # Initialize
        self._open_video()
    
    def _open_video(self) -> None:
        """Mở video và đọc thông tin."""
        self._cap = cv2.VideoCapture(str(self._path))
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")
        
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default
        
        self._info = VideoInfo(
            path=str(self._path),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps,
            codec=self._get_codec()
        )
        
        self._frame_interval = 1.0 / fps
    
    def _get_codec(self) -> str:
        """Lấy codec của video."""
        if self._cap is None:
            return ""
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    @property
    def info(self) -> Optional[VideoInfo]:
        """Thông tin video."""
        return self._info
    
    @property
    def state(self) -> PlaybackState:
        """Trạng thái hiện tại."""
        return self._state
    
    @property
    def current_frame(self) -> int:
        """Frame hiện tại."""
        return self._current_frame
    
    @property
    def fps(self) -> float:
        """Frame rate."""
        return self._info.fps if self._info else 30.0
    
    @property
    def total_frames(self) -> int:
        """Tổng số frames."""
        return self._info.total_frames if self._info else 0
    
    def set_checkpoints(
        self,
        frames: List[int],
        messages: Optional[dict] = None
    ) -> None:
        """
        Đặt các checkpoint trong video.
        
        Args:
            frames: Danh sách frame index của các checkpoint.
            messages: Dict mapping frame → message (optional).
        """
        self._checkpoints = sorted(frames)
        self._checkpoint_messages = messages or {}
        self._current_checkpoint_idx = 0
    
    def set_speed(self, factor: float) -> None:
        """
        Đặt tốc độ phát.
        
        Args:
            factor: 1.0 = bình thường, 0.5 = chậm 2x, 2.0 = nhanh 2x.
        """
        self._speed_factor = max(0.1, min(3.0, factor))
    
    def play(self) -> None:
        """Bắt đầu/tiếp tục phát video."""
        if self._state == PlaybackState.FINISHED:
            self.seek(0)
        
        self._state = PlaybackState.PLAYING
        self._last_frame_time = time.time()
    
    def pause(self) -> None:
        """Tạm dừng video."""
        self._state = PlaybackState.PAUSED
    
    def stop(self) -> None:
        """Dừng và reset về đầu."""
        self._state = PlaybackState.STOPPED
        self._current_frame = 0
        self._loop_count = 0
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def seek(self, frame_index: int) -> bool:
        """
        Nhảy đến frame cụ thể.
        
        Args:
            frame_index: Số frame cần nhảy đến.
            
        Returns:
            bool: True nếu seek thành công.
        """
        if self._cap is None or self._info is None:
            return False
        
        frame_index = max(0, min(frame_index, self._info.total_frames - 1))
        
        self._state = PlaybackState.SEEKING
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        if success:
            self._current_frame = frame_index
            self._target_frame = frame_index
            self._update_checkpoint_index()
        
        # Restore previous state
        if self._state == PlaybackState.SEEKING:
            self._state = PlaybackState.PAUSED
        
        return success
    
    def seek_time(self, time_seconds: float) -> bool:
        """
        Nhảy đến thời điểm cụ thể.
        
        Args:
            time_seconds: Thời gian (giây).
            
        Returns:
            bool: True nếu seek thành công.
        """
        if self._info is None:
            return False
        frame = int(time_seconds * self._info.fps)
        return self.seek(frame)
    
    def start_loop(self, start_frame: int, end_frame: int, max_loops: int = 3) -> None:
        """
        Bắt đầu lặp một đoạn video.
        
        Args:
            start_frame: Frame bắt đầu đoạn lặp.
            end_frame: Frame kết thúc đoạn lặp.
            max_loops: Số lần lặp tối đa.
        """
        self._loop_start = max(0, start_frame)
        self._loop_end = min(end_frame, self.total_frames - 1)
        self._loop_count = 0
        self._max_loops = max_loops
        self._state = PlaybackState.LOOPING
        
        # Seek đến điểm bắt đầu
        self.seek(self._loop_start)
    
    def stop_loop(self) -> None:
        """Dừng lặp và tiếp tục phát bình thường."""
        self._state = PlaybackState.PLAYING
        self._loop_count = 0
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], PlaybackStatus]:
        """
        Lấy frame tiếp theo.
        
        Đây là hàm chính để lấy frame trong vòng lặp render.
        Tự động xử lý timing, checkpoints, và loops.
        
        Returns:
            Tuple[frame, status]:
                - frame: Numpy array (BGR) hoặc None nếu lỗi
                - status: PlaybackStatus
        """
        if self._cap is None or self._info is None:
            return None, PlaybackStatus(
                state=PlaybackState.STOPPED,
                current_frame=0,
                current_time_ms=0,
                progress=0.0
            )
        
        # Tính timing
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        adjusted_interval = self._frame_interval / self._speed_factor
        
        # Tạo status cơ bản
        status = PlaybackStatus(
            state=self._state,
            current_frame=self._current_frame,
            current_time_ms=int((self._current_frame / self._info.fps) * 1000),
            progress=self._current_frame / self._info.total_frames if self._info.total_frames > 0 else 0,
            loop_count=self._loop_count
        )
        
        # Xử lý theo state
        if self._state == PlaybackState.STOPPED:
            return self._read_current_frame(), status
        
        if self._state == PlaybackState.PAUSED:
            return self._read_current_frame(), status
        
        if self._state == PlaybackState.FINISHED:
            return self._read_current_frame(), status
        
        # PLAYING hoặc LOOPING
        if elapsed < adjusted_interval:
            # Chưa đến lúc đọc frame mới
            return self._read_current_frame(), status
        
        self._last_frame_time = current_time
        
        # Đọc frame tiếp theo
        ret, frame = self._cap.read()
        
        if not ret:
            # Hết video
            if self._state == PlaybackState.LOOPING:
                # Quay lại đầu đoạn lặp
                self._loop_count += 1
                if self._loop_count >= self._max_loops:
                    self._state = PlaybackState.PLAYING
                    if self._on_loop:
                        self._on_loop(self._loop_count)
                else:
                    self.seek(self._loop_start)
                    return self.get_frame()
            else:
                self._state = PlaybackState.FINISHED
                if self._on_finish:
                    self._on_finish()
            
            status.state = self._state
            return frame, status
        
        self._current_frame += 1
        
        # Kiểm tra loop boundary
        if self._state == PlaybackState.LOOPING:
            if self._current_frame >= self._loop_end:
                self._loop_count += 1
                if self._loop_count >= self._max_loops:
                    self._state = PlaybackState.PLAYING
                    if self._on_loop:
                        self._on_loop(self._loop_count)
                else:
                    self.seek(self._loop_start)
                    if self._on_loop:
                        self._on_loop(self._loop_count)
        
        # Kiểm tra checkpoint
        status.is_at_checkpoint = self._check_checkpoint()
        if status.is_at_checkpoint:
            status.checkpoint_message = self._checkpoint_messages.get(
                self._checkpoints[self._current_checkpoint_idx - 1],
                "Checkpoint reached"
            )
        
        # Cập nhật status
        status.state = self._state
        status.current_frame = self._current_frame
        status.current_time_ms = int((self._current_frame / self._info.fps) * 1000)
        status.progress = self._current_frame / self._info.total_frames
        status.loop_count = self._loop_count
        
        return frame, status
    
    def _read_current_frame(self) -> Optional[np.ndarray]:
        """Đọc frame hiện tại (không advance)."""
        if self._cap is None:
            return None
        
        # Lưu vị trí
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Seek đến frame hiện tại
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self._current_frame))
        ret, frame = self._cap.read()
        
        # Restore vị trí
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        
        return frame if ret else None
    
    def _check_checkpoint(self) -> bool:
        """Kiểm tra và xử lý checkpoint."""
        if not self._checkpoints:
            return False
        
        if self._current_checkpoint_idx >= len(self._checkpoints):
            return False
        
        next_cp = self._checkpoints[self._current_checkpoint_idx]
        
        if self._current_frame >= next_cp:
            self._current_checkpoint_idx += 1
            
            if self._on_checkpoint:
                msg = self._checkpoint_messages.get(next_cp, "")
                self._on_checkpoint(next_cp, msg)
            
            return True
        
        return False
    
    def _update_checkpoint_index(self) -> None:
        """Cập nhật checkpoint index sau khi seek."""
        self._current_checkpoint_idx = 0
        for i, cp in enumerate(self._checkpoints):
            if self._current_frame >= cp:
                self._current_checkpoint_idx = i + 1
    
    def advance_frame(self) -> bool:
        """
        Tiến 1 frame (manual control).
        
        Dùng khi muốn điều khiển từng frame một.
        
        Returns:
            bool: True nếu còn frame.
        """
        if self._cap is None:
            return False
        
        ret, _ = self._cap.read()
        if ret:
            self._current_frame += 1
            self._check_checkpoint()
            return True
        return False
    
    def get_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Lấy frame tại vị trí cụ thể (không thay đổi state).
        
        Args:
            frame_index: Số frame cần lấy.
            
        Returns:
            Frame hoặc None.
        """
        if self._cap is None or self._info is None:
            return None
        
        frame_index = max(0, min(frame_index, self._info.total_frames - 1))
        
        # Lưu vị trí hiện tại
        current_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Seek và đọc
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        
        # Restore vị trí
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frame if ret else None
    
    def set_on_checkpoint(self, callback: Callable[[int, str], None]) -> None:
        """Đặt callback khi đến checkpoint."""
        self._on_checkpoint = callback
    
    def set_on_loop(self, callback: Callable[[int], None]) -> None:
        """Đặt callback khi lặp xong 1 vòng."""
        self._on_loop = callback
    
    def set_on_finish(self, callback: Callable[[], None]) -> None:
        """Đặt callback khi phát xong."""
        self._on_finish = callback
    
    def release(self) -> None:
        """Giải phóng tài nguyên."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._state = PlaybackState.STOPPED
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class SyncedVideoPlayer:
    """
    Video Player đồng bộ với MotionSyncController.
    
    Tự động pause/play dựa trên sync status.
    
    Example:
        >>> from core.synchronizer import MotionSyncController
        >>> controller = MotionSyncController(exercise)
        >>> player = SyncedVideoPlayer("exercise.mp4", controller)
        >>> 
        >>> while True:
        ...     # Update controller với góc của user
        ...     sync_state = controller.update(user_angle, player.current_frame)
        ...     
        ...     # Player tự động điều chỉnh
        ...     frame, status = player.get_frame(sync_state.sync_status)
    """
    
    def __init__(
        self,
        video_path: str,
        checkpoints: Optional[List[int]] = None
    ):
        """
        Khởi tạo SyncedVideoPlayer.
        
        Args:
            video_path: Đường dẫn video.
            checkpoints: Danh sách checkpoint frames.
        """
        self._engine = VideoEngine(video_path)
        
        if checkpoints:
            self._engine.set_checkpoints(checkpoints)
        
        self._engine.set_speed(0.8)  # Chậm hơn cho người già
    
    @property
    def current_frame(self) -> int:
        """Frame hiện tại."""
        return self._engine.current_frame
    
    @property
    def total_frames(self) -> int:
        """Tổng số frames."""
        return self._engine.total_frames
    
    @property
    def info(self) -> Optional[VideoInfo]:
        """Thông tin video."""
        return self._engine.info
    
    def get_frame(
        self,
        sync_status=None
    ) -> Tuple[Optional[np.ndarray], PlaybackStatus]:
        """
        Lấy frame với điều khiển đồng bộ.
        
        Args:
            sync_status: SyncStatus từ controller (optional).
            
        Returns:
            Tuple[frame, status].
        """
        # Import here để tránh circular
        from core.synchronizer import SyncStatus
        
        if sync_status is not None:
            if sync_status == SyncStatus.PAUSE:
                self._engine.pause()
            elif sync_status == SyncStatus.PLAY:
                if self._engine.state != PlaybackState.PLAYING:
                    self._engine.play()
            elif sync_status == SyncStatus.LOOP:
                if self._engine.state != PlaybackState.LOOPING:
                    # Loop 5 frames xung quanh vị trí hiện tại
                    current = self._engine.current_frame
                    self._engine.start_loop(
                        max(0, current - 5),
                        min(self.total_frames - 1, current + 5),
                        max_loops=10
                    )
            elif sync_status == SyncStatus.SKIP:
                self._engine.play()
        
        return self._engine.get_frame()
    
    def start(self) -> None:
        """Bắt đầu phát."""
        self._engine.play()
    
    def release(self) -> None:
        """Giải phóng tài nguyên."""
        self._engine.release()