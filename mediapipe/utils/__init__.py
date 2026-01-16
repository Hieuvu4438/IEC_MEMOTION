"""
Utils Package for MEMOTION.

Chứa các tiện ích:
- logger: Hệ thống ghi nhật ký
- visualization: Các hàm vẽ và hiển thị

Author: MEMOTION Team
Version: 1.1.0
"""

from .logger import (
    SessionLogger,
    LogLevel,
    LogCategory,
    LogEntry,
    create_session_logger,
)

from .visualization import (
    VietnameseTextRenderer,
    get_text_renderer,
    put_vietnamese_text,
    draw_skeleton,
    draw_angle_arc,
    draw_panel,
    draw_progress_bar,
    draw_button,
    draw_phase_indicator,
    draw_score_display,
    create_dashboard,
    draw_instructions,
    combine_frames_horizontal,
    COLORS,
    POSE_CONNECTIONS,
    CORE_CONNECTIONS,
)

__all__ = [
    # Logger
    "SessionLogger",
    "LogLevel",
    "LogCategory",
    "LogEntry",
    "create_session_logger",
    # Visualization
    "VietnameseTextRenderer",
    "get_text_renderer",
    "put_vietnamese_text",
    "draw_skeleton",
    "draw_angle_arc",
    "draw_panel",
    "draw_progress_bar",
    "draw_button",
    "draw_phase_indicator",
    "draw_score_display",
    "create_dashboard",
    "draw_instructions",
    "combine_frames_horizontal",
    "COLORS",
    "POSE_CONNECTIONS",
    "CORE_CONNECTIONS",
]