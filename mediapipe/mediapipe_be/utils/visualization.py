"""
Visualization Module for MEMOTION.

Cung cấp các hàm tiện ích để:
- Vẽ skeleton và keypoints lên frame
- Hiển thị text tiếng Việt (sử dụng PIL để tránh lỗi font)
- Vẽ giao diện người dùng (UI panels, progress bars, etc.)

Author: MEMOTION Team
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from core.data_types import PoseLandmarkIndex


# MediaPipe Pose Connections (định nghĩa các đường nối skeleton)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    # Body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # Right leg
    # Hands
    (15, 17), (15, 19), (15, 21), (17, 19),  # Left hand
    (16, 18), (16, 20), (16, 22), (18, 20),  # Right hand
]

# Core body connections (không bao gồm mặt và tay)
CORE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]

# Color schemes
COLORS = {
    'skeleton': (0, 255, 0),          # Green
    'skeleton_ref': (0, 200, 255),    # Orange
    'keypoint': (0, 255, 255),        # Yellow
    'keypoint_ref': (255, 165, 0),    # Blue-ish
    'highlight': (0, 0, 255),         # Red
    'text': (255, 255, 255),          # White
    'panel_bg': (40, 40, 40),         # Dark gray
    'success': (0, 255, 0),           # Green
    'warning': (0, 165, 255),         # Orange
    'error': (0, 0, 255),             # Red
    'info': (255, 255, 0),            # Cyan
    'phase_idle': (128, 128, 128),    # Gray
    'phase_eccentric': (0, 255, 255), # Yellow
    'phase_hold': (0, 255, 0),        # Green
    'phase_concentric': (255, 255, 0),# Cyan
}

# Font path cho tiếng Việt
VIETNAMESE_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


class VietnameseTextRenderer:
    """Render text tiếng Việt sử dụng PIL."""
    
    def __init__(self, font_size: int = 20):
        self.font_size = font_size
        self.font = None
        self._init_font()
    
    def _init_font(self):
        """Tìm và load font hỗ trợ tiếng Việt."""
        if not PIL_AVAILABLE:
            return
        
        for font_path in VIETNAMESE_FONT_PATHS:
            if Path(font_path).exists():
                try:
                    self.font = ImageFont.truetype(font_path, self.font_size)
                    return
                except Exception:
                    continue
        
        # Fallback to default
        try:
            self.font = ImageFont.load_default()
        except:
            pass
    
    def put_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        font_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Vẽ text tiếng Việt lên frame.
        
        Args:
            frame: OpenCV frame (BGR).
            text: Text cần vẽ.
            position: Vị trí (x, y).
            color: Màu BGR.
            font_size: Kích thước font (None để dùng mặc định).
            
        Returns:
            Frame với text đã vẽ.
        """
        if not PIL_AVAILABLE or self.font is None:
            # Fallback to OpenCV (không hỗ trợ tiếng Việt tốt)
            cv2.putText(
                frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
            return frame
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Adjust font size if needed
        font = self.font
        if font_size and font_size != self.font_size:
            try:
                for font_path in VIETNAMESE_FONT_PATHS:
                    if Path(font_path).exists():
                        font = ImageFont.truetype(font_path, font_size)
                        break
            except:
                pass
        
        # Convert BGR color to RGB
        color_rgb = (color[2], color[1], color[0])
        
        # Draw text
        draw.text(position, text, font=font, fill=color_rgb)
        
        # Convert back to BGR
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Global text renderer
_text_renderer = None


def get_text_renderer(font_size: int = 20) -> VietnameseTextRenderer:
    """Lấy text renderer singleton."""
    global _text_renderer
    if _text_renderer is None or _text_renderer.font_size != font_size:
        _text_renderer = VietnameseTextRenderer(font_size)
    return _text_renderer


def put_vietnamese_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 20
) -> np.ndarray:
    """
    Vẽ text tiếng Việt lên frame.
    
    Args:
        frame: OpenCV frame.
        text: Text cần vẽ.
        position: Vị trí (x, y).
        color: Màu BGR.
        font_size: Kích thước font.
        
    Returns:
        Frame với text.
    """
    renderer = get_text_renderer(font_size)
    return renderer.put_text(frame, text, position, color, font_size)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = COLORS['skeleton'],
    keypoint_color: Tuple[int, int, int] = COLORS['keypoint'],
    highlight_indices: Optional[List[int]] = None,
    highlight_color: Tuple[int, int, int] = COLORS['highlight'],
    line_thickness: int = 2,
    keypoint_radius: int = 4,
    use_core_only: bool = False,
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """
    Vẽ skeleton lên frame.
    
    Args:
        frame: OpenCV frame.
        landmarks: Numpy array shape (N, 3) hoặc (N, 4) với visibility.
        color: Màu đường skeleton.
        keypoint_color: Màu keypoints.
        highlight_indices: Danh sách indices cần highlight (vd: khớp đang đo).
        highlight_color: Màu highlight.
        line_thickness: Độ dày đường.
        keypoint_radius: Bán kính keypoint.
        use_core_only: Chỉ vẽ body chính (bỏ mặt, tay).
        visibility_threshold: Ngưỡng visibility để vẽ.
        
    Returns:
        Frame với skeleton.
    """
    output = frame.copy()
    h, w = frame.shape[:2]
    
    if landmarks is None or len(landmarks) == 0:
        return output
    
    connections = CORE_CONNECTIONS if use_core_only else POSE_CONNECTIONS
    
    # Vẽ các đường nối
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        
        # Kiểm tra visibility nếu có
        if len(p1) > 3 and p1[3] < visibility_threshold:
            continue
        if len(p2) > 3 and p2[3] < visibility_threshold:
            continue
        
        # Convert normalized coords to pixel coords
        x1, y1 = int(p1[0] * w), int(p1[1] * h)
        x2, y2 = int(p2[0] * w), int(p2[1] * h)
        
        # Kiểm tra bounds
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue
        
        cv2.line(output, (x1, y1), (x2, y2), color, line_thickness)
    
    # Vẽ keypoints
    for idx, point in enumerate(landmarks):
        if len(point) > 3 and point[3] < visibility_threshold:
            continue
        
        x, y = int(point[0] * w), int(point[1] * h)
        
        if not (0 <= x < w and 0 <= y < h):
            continue
        
        # Chọn màu
        pt_color = keypoint_color
        pt_radius = keypoint_radius
        
        if highlight_indices and idx in highlight_indices:
            pt_color = highlight_color
            pt_radius = keypoint_radius + 4
        
        cv2.circle(output, (x, y), pt_radius, pt_color, -1)
        cv2.circle(output, (x, y), pt_radius + 2, color, 1)
    
    return output


def draw_angle_arc(
    frame: np.ndarray,
    p1: Tuple[int, int],
    vertex: Tuple[int, int],
    p2: Tuple[int, int],
    angle: float,
    color: Tuple[int, int, int] = COLORS['info'],
    radius: int = 40,
    thickness: int = 2,
    show_value: bool = True
) -> np.ndarray:
    """
    Vẽ cung thể hiện góc giữa 3 điểm.
    
    Args:
        frame: OpenCV frame.
        p1, vertex, p2: Ba điểm tạo góc (vertex là đỉnh).
        angle: Giá trị góc (degrees).
        color: Màu cung.
        radius: Bán kính cung.
        thickness: Độ dày.
        show_value: Hiển thị giá trị góc.
        
    Returns:
        Frame với cung góc.
    """
    output = frame.copy()
    
    # Tính góc start và end
    angle1 = np.degrees(np.arctan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    angle2 = np.degrees(np.arctan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    
    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)
    
    # Vẽ cung
    cv2.ellipse(
        output, vertex, (radius, radius),
        0, start_angle, end_angle,
        color, thickness
    )
    
    # Hiển thị giá trị góc
    if show_value:
        mid_angle = np.radians((start_angle + end_angle) / 2)
        text_x = int(vertex[0] + (radius + 20) * np.cos(mid_angle))
        text_y = int(vertex[1] + (radius + 20) * np.sin(mid_angle))
        cv2.putText(
            output, f"{angle:.0f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    
    return output


def draw_panel(
    frame: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int],
    title: str = "",
    bg_color: Tuple[int, int, int] = COLORS['panel_bg'],
    title_color: Tuple[int, int, int] = COLORS['text'],
    alpha: float = 0.7
) -> np.ndarray:
    """
    Vẽ panel nền mờ với tiêu đề.
    
    Args:
        frame: OpenCV frame.
        position: Vị trí góc trên trái (x, y).
        size: Kích thước (width, height).
        title: Tiêu đề panel.
        bg_color: Màu nền.
        title_color: Màu tiêu đề.
        alpha: Độ trong suốt (0-1).
        
    Returns:
        Frame với panel.
    """
    output = frame.copy()
    x, y = position
    w, h = size
    
    # Vẽ nền mờ
    overlay = output.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # Vẽ viền
    cv2.rectangle(output, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
    # Vẽ tiêu đề
    if title:
        output = put_vietnamese_text(
            output, title, (x + 10, y + 25),
            title_color, font_size=18
        )
    
    return output


def draw_progress_bar(
    frame: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int],
    progress: float,
    color: Tuple[int, int, int] = COLORS['success'],
    bg_color: Tuple[int, int, int] = (100, 100, 100),
    show_percentage: bool = True
) -> np.ndarray:
    """
    Vẽ thanh progress.
    
    Args:
        frame: OpenCV frame.
        position: Vị trí (x, y).
        size: Kích thước (width, height).
        progress: Tiến độ 0-1.
        color: Màu thanh progress.
        bg_color: Màu nền.
        show_percentage: Hiển thị phần trăm.
        
    Returns:
        Frame với progress bar.
    """
    output = frame.copy()
    x, y = position
    w, h = size
    
    progress = max(0, min(1, progress))
    
    # Vẽ nền
    cv2.rectangle(output, (x, y), (x + w, y + h), bg_color, -1)
    
    # Vẽ progress
    progress_w = int(w * progress)
    if progress_w > 0:
        cv2.rectangle(output, (x, y), (x + progress_w, y + h), color, -1)
    
    # Vẽ viền
    cv2.rectangle(output, (x, y), (x + w, y + h), (150, 150, 150), 1)
    
    # Hiển thị phần trăm
    if show_percentage:
        text = f"{int(progress * 100)}%"
        text_x = x + w // 2 - 15
        text_y = y + h // 2 + 5
        cv2.putText(
            output, text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1
        )
    
    return output


def draw_button(
    frame: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int],
    text: str,
    key: str = "",
    active: bool = False,
    color: Tuple[int, int, int] = (80, 80, 80),
    active_color: Tuple[int, int, int] = COLORS['success']
) -> np.ndarray:
    """
    Vẽ button với phím tắt.
    
    Args:
        frame: OpenCV frame.
        position: Vị trí (x, y).
        size: Kích thước (width, height).
        text: Nhãn button.
        key: Phím tắt (vd: "1", "SPACE").
        active: Trạng thái active.
        color: Màu nền.
        active_color: Màu khi active.
        
    Returns:
        Frame với button.
    """
    output = frame.copy()
    x, y = position
    w, h = size
    
    bg = active_color if active else color
    
    # Vẽ nền
    cv2.rectangle(output, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(output, (x, y), (x + w, y + h), (150, 150, 150), 1)
    
    # Vẽ text
    display_text = f"[{key}] {text}" if key else text
    output = put_vietnamese_text(
        output, display_text, (x + 5, y + h // 2 + 5),
        COLORS['text'], font_size=14
    )
    
    return output


def draw_phase_indicator(
    frame: np.ndarray,
    phase: str,
    position: Tuple[int, int] = (20, 50),
    size: int = 120
) -> np.ndarray:
    """
    Vẽ indicator cho phase hiện tại.
    
    Args:
        frame: OpenCV frame.
        phase: Tên phase ("idle", "eccentric", "hold", "concentric").
        position: Vị trí.
        size: Kích thước.
        
    Returns:
        Frame với phase indicator.
    """
    output = frame.copy()
    x, y = position
    
    phase_colors = {
        "idle": COLORS['phase_idle'],
        "eccentric": COLORS['phase_eccentric'],
        "hold": COLORS['phase_hold'],
        "concentric": COLORS['phase_concentric'],
    }
    
    phase_names = {
        "idle": "Nghỉ",
        "eccentric": "Duỗi",
        "hold": "Giữ",
        "concentric": "Co",
    }
    
    color = phase_colors.get(phase.lower(), COLORS['phase_idle'])
    name = phase_names.get(phase.lower(), phase.upper())
    
    # Vẽ vòng tròn indicator
    cv2.circle(output, (x + 30, y + 30), 25, color, -1)
    cv2.circle(output, (x + 30, y + 30), 25, (200, 200, 200), 2)
    
    # Vẽ tên phase
    output = put_vietnamese_text(
        output, name.upper(), (x + 65, y + 35),
        color, font_size=18
    )
    
    return output


def draw_score_display(
    frame: np.ndarray,
    score: float,
    position: Tuple[int, int],
    label: str = "Score",
    max_score: float = 100
) -> np.ndarray:
    """
    Vẽ hiển thị điểm số.
    
    Args:
        frame: OpenCV frame.
        score: Điểm số.
        position: Vị trí.
        label: Nhãn.
        max_score: Điểm tối đa.
        
    Returns:
        Frame với score display.
    """
    output = frame.copy()
    x, y = position
    
    # Chọn màu theo điểm
    if score >= 80:
        color = COLORS['success']
    elif score >= 60:
        color = COLORS['warning']
    else:
        color = COLORS['error']
    
    # Vẽ label
    output = put_vietnamese_text(output, label, (x, y), COLORS['text'], 14)
    
    # Vẽ điểm
    score_text = f"{score:.0f}/{max_score:.0f}"
    cv2.putText(
        output, score_text, (x, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    
    return output


def create_dashboard(
    width: int,
    height: int,
    data: Dict,
    bg_color: Tuple[int, int, int] = COLORS['panel_bg']
) -> np.ndarray:
    """
    Tạo dashboard hiển thị thông tin.
    
    Args:
        width: Chiều rộng.
        height: Chiều cao.
        data: Dict chứa thông tin cần hiển thị.
        bg_color: Màu nền.
        
    Returns:
        Dashboard image.
    """
    dashboard = np.zeros((height, width, 3), dtype=np.uint8)
    dashboard[:] = bg_color
    
    y_offset = 30
    line_height = 30
    
    # Tiêu đề
    dashboard = put_vietnamese_text(
        dashboard, "DASHBOARD", (20, y_offset),
        COLORS['text'], font_size=22
    )
    y_offset += 45
    
    # Hiển thị từng item
    for key, value in data.items():
        # Format value
        if isinstance(value, float):
            display_value = f"{value:.1f}"
        else:
            display_value = str(value)
        
        # Vẽ key: value
        dashboard = put_vietnamese_text(
            dashboard, f"{key}:", (20, y_offset),
            (180, 180, 180), font_size=14
        )
        dashboard = put_vietnamese_text(
            dashboard, display_value, (120, y_offset),
            COLORS['text'], font_size=14
        )
        
        y_offset += line_height
    
    return dashboard


def draw_instructions(
    frame: np.ndarray,
    instructions: List[str],
    position: Tuple[int, int] = (20, 400),
    title: str = "Huong dan:"
) -> np.ndarray:
    """
    Vẽ panel hướng dẫn.
    
    Args:
        frame: OpenCV frame.
        instructions: Danh sách hướng dẫn.
        position: Vị trí.
        title: Tiêu đề.
        
    Returns:
        Frame với instructions.
    """
    output = frame.copy()
    x, y = position
    
    # Tính kích thước panel
    panel_height = 30 + len(instructions) * 25 + 10
    panel_width = max(len(inst) for inst in instructions) * 8 + 40
    
    # Vẽ panel
    output = draw_panel(output, (x, y), (panel_width, panel_height), title)
    
    # Vẽ từng instruction
    for i, inst in enumerate(instructions):
        output = put_vietnamese_text(
            output, f"• {inst}",
            (x + 15, y + 45 + i * 25),
            COLORS['text'], font_size=14
        )
    
    return output


def combine_frames_horizontal(
    frames: List[np.ndarray],
    target_height: int = 480,
    gap: int = 2,
    gap_color: Tuple[int, int, int] = (50, 50, 50)
) -> np.ndarray:
    """
    Ghép nhiều frame theo chiều ngang.
    
    Args:
        frames: Danh sách frames.
        target_height: Chiều cao đích.
        gap: Khoảng cách giữa các frame.
        gap_color: Màu khoảng cách.
        
    Returns:
        Frame đã ghép.
    """
    if not frames:
        return np.zeros((target_height, 640, 3), dtype=np.uint8)
    
    resized = []
    for frame in frames:
        if frame is None:
            continue
        h, w = frame.shape[:2]
        new_w = int(w * target_height / h)
        resized.append(cv2.resize(frame, (new_w, target_height)))
    
    if not resized:
        return np.zeros((target_height, 640, 3), dtype=np.uint8)
    
    # Tạo gaps
    if gap > 0:
        gap_img = np.zeros((target_height, gap, 3), dtype=np.uint8)
        gap_img[:] = gap_color
        
        result = []
        for i, frame in enumerate(resized):
            result.append(frame)
            if i < len(resized) - 1:
                result.append(gap_img)
        return np.hstack(result)
    
    return np.hstack(resized)


# Export all
__all__ = [
    'VietnameseTextRenderer',
    'get_text_renderer',
    'put_vietnamese_text',
    'draw_skeleton',
    'draw_angle_arc',
    'draw_panel',
    'draw_progress_bar',
    'draw_button',
    'draw_phase_indicator',
    'draw_score_display',
    'create_dashboard',
    'draw_instructions',
    'combine_frames_horizontal',
    'COLORS',
    'POSE_CONNECTIONS',
    'CORE_CONNECTIONS',
]
