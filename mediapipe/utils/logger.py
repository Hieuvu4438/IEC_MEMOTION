"""
Logger Module for MEMOTION.

Hệ thống ghi nhật ký chi tiết cho:
- Kết quả từng hiệp tập (rep)
- Cảnh báo đau/mệt mỏi
- Báo cáo buổi tập

Định dạng output:
- JSON: Cấu trúc đầy đủ cho phân tích
- CSV: Dễ mở bằng Excel cho bác sĩ
- Console: Real-time monitoring

Author: MEMOTION Team
Version: 1.0.0
"""

import json
import csv
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue


class LogLevel(Enum):
    """Mức độ log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Loại log."""
    SESSION = "session"       # Thông tin buổi tập
    REP = "rep"              # Kết quả từng rep
    PAIN = "pain"            # Cảnh báo đau
    FATIGUE = "fatigue"      # Cảnh báo mệt mỏi
    SAFETY = "safety"        # Cảnh báo an toàn
    SYNC = "sync"            # Thông tin đồng bộ
    SYSTEM = "system"        # Thông tin hệ thống


@dataclass
class LogEntry:
    """
    Một entry trong log.
    
    Attributes:
        timestamp: Thời điểm ghi log.
        level: Mức độ log.
        category: Loại log.
        message: Nội dung.
        data: Dữ liệu bổ sung (dict).
        session_id: ID buổi tập (nếu có).
    """
    timestamp: str
    level: str
    category: str
    message: str
    data: Dict = None
    session_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "data": self.data or {},
            "session_id": self.session_id,
        }
    
    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp,
            self.level,
            self.category,
            self.message,
            json.dumps(self.data or {}),
            self.session_id,
        ]


class SessionLogger:
    """
    Logger cho một buổi tập.
    
    Tự động ghi log vào:
    - Console (real-time)
    - JSON file (cấu trúc đầy đủ)
    - CSV file (dễ đọc)
    
    Thread-safe để không block main thread.
    
    Example:
        >>> logger = SessionLogger("./logs")
        >>> logger.start_session("session_001", "arm_raise")
        >>> 
        >>> logger.log_rep(1, {"score": 85, "rom": 90})
        >>> logger.log_pain("mild", {"au4": 0.3})
        >>> 
        >>> logger.end_session(report)
    """
    
    CSV_HEADERS = [
        "timestamp", "level", "category", "message", "data", "session_id"
    ]
    
    def __init__(
        self,
        log_dir: str = "./logs",
        console_output: bool = True,
        async_write: bool = True
    ):
        """
        Khởi tạo SessionLogger.
        
        Args:
            log_dir: Thư mục lưu log.
            console_output: Có in ra console không.
            async_write: Ghi async để không block.
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        self._console_output = console_output
        self._async_write = async_write
        
        self._session_id: Optional[str] = None
        self._session_start: Optional[datetime] = None
        self._entries: List[LogEntry] = []
        
        # File handles
        self._json_file: Optional[Path] = None
        self._csv_file: Optional[Path] = None
        self._csv_writer = None
        self._file_handle = None
        
        # Async queue
        self._write_queue: Queue = Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = False
        
        # Setup console logger
        self._setup_console_logger()
    
    def _setup_console_logger(self) -> None:
        """Setup Python logging cho console."""
        self._console_logger = logging.getLogger("MEMOTION")
        self._console_logger.setLevel(logging.DEBUG)
        
        if not self._console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._console_logger.addHandler(handler)
    
    def start_session(
        self,
        session_id: str,
        exercise_name: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Bắt đầu logging cho một buổi tập.
        
        Args:
            session_id: ID buổi tập.
            exercise_name: Tên bài tập.
            user_id: ID người dùng (optional).
        """
        self._session_id = session_id
        self._session_start = datetime.now()
        self._entries = []
        
        # Tạo file paths
        date_str = self._session_start.strftime("%Y%m%d")
        time_str = self._session_start.strftime("%H%M%S")
        
        session_dir = self._log_dir / date_str
        session_dir.mkdir(exist_ok=True)
        
        self._json_file = session_dir / f"{session_id}_{time_str}.json"
        self._csv_file = session_dir / f"{session_id}_{time_str}.csv"
        
        # Khởi tạo CSV file
        self._init_csv_file()
        
        # Start async writer
        if self._async_write:
            self._start_async_writer()
        
        # Log session start
        self.log(
            LogLevel.INFO,
            LogCategory.SESSION,
            f"Session started: {exercise_name}",
            {
                "session_id": session_id,
                "exercise_name": exercise_name,
                "user_id": user_id or "anonymous",
                "start_time": self._session_start.isoformat(),
            }
        )
    
    def _init_csv_file(self) -> None:
        """Khởi tạo CSV file với headers."""
        if self._csv_file is None:
            return
        
        self._file_handle = open(self._csv_file, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._file_handle)
        self._csv_writer.writerow(self.CSV_HEADERS)
        self._file_handle.flush()
    
    def _start_async_writer(self) -> None:
        """Bắt đầu thread ghi async."""
        self._stop_writer = False
        self._writer_thread = threading.Thread(target=self._async_write_loop)
        self._writer_thread.daemon = True
        self._writer_thread.start()
    
    def _async_write_loop(self) -> None:
        """Loop ghi async."""
        while not self._stop_writer:
            try:
                entry = self._write_queue.get(timeout=0.5)
                self._write_entry(entry)
            except:
                continue
    
    def _write_entry(self, entry: LogEntry) -> None:
        """Ghi một entry vào file."""
        # CSV
        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow(entry.to_csv_row())
                self._file_handle.flush()
            except:
                pass
    
    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Ghi một log entry.
        
        Args:
            level: Mức độ log.
            category: Loại log.
            message: Nội dung.
            data: Dữ liệu bổ sung.
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            message=message,
            data=data,
            session_id=self._session_id or "",
        )
        
        self._entries.append(entry)
        
        # Console output
        if self._console_output:
            log_method = getattr(self._console_logger, level.value.lower(), self._console_logger.info)
            log_method(f"[{category.value}] {message}")
        
        # Write to file
        if self._async_write:
            self._write_queue.put(entry)
        else:
            self._write_entry(entry)
    
    def log_rep(
        self,
        rep_number: int,
        scores: Dict,
        jerk: float = 0.0,
        duration_ms: int = 0
    ) -> None:
        """
        Ghi log kết quả một rep.
        
        Args:
            rep_number: Số thứ tự rep.
            scores: Dict chứa các điểm.
            jerk: Giá trị Jerk.
            duration_ms: Thời gian thực hiện.
        """
        total = scores.get("total", 0)
        message = f"Rep {rep_number}: {total:.0f}/100"
        
        self.log(
            LogLevel.INFO,
            LogCategory.REP,
            message,
            {
                "rep_number": rep_number,
                "scores": scores,
                "jerk": jerk,
                "duration_ms": duration_ms,
            }
        )
    
    def log_pain(
        self,
        pain_level: str,
        pain_score: float,
        au_scores: Optional[Dict] = None,
        message: str = ""
    ) -> None:
        """
        Ghi log cảnh báo đau.
        
        Args:
            pain_level: Mức độ đau.
            pain_score: Điểm đau.
            au_scores: Điểm các Action Units.
            message: Thông báo.
        """
        level = LogLevel.WARNING if pain_level in ("MILD", "MODERATE") else LogLevel.CRITICAL
        
        self.log(
            level,
            LogCategory.PAIN,
            f"Pain detected: {pain_level} ({pain_score:.0f}%)",
            {
                "pain_level": pain_level,
                "pain_score": pain_score,
                "au_scores": au_scores or {},
                "user_message": message,
            }
        )
    
    def log_fatigue(
        self,
        fatigue_level: str,
        jerk_increase: float,
        message: str = ""
    ) -> None:
        """
        Ghi log cảnh báo mệt mỏi.
        
        Args:
            fatigue_level: Mức độ mệt mỏi.
            jerk_increase: Phần trăm tăng Jerk.
            message: Thông báo.
        """
        level = LogLevel.WARNING if fatigue_level in ("LIGHT", "MODERATE") else LogLevel.CRITICAL
        
        self.log(
            level,
            LogCategory.FATIGUE,
            f"Fatigue detected: {fatigue_level} (Jerk +{jerk_increase:.0f}%)",
            {
                "fatigue_level": fatigue_level,
                "jerk_increase_percent": jerk_increase,
                "user_message": message,
            }
        )
    
    def log_safety(
        self,
        warning_type: str,
        message: str,
        details: Optional[Dict] = None
    ) -> None:
        """
        Ghi log cảnh báo an toàn.
        
        Args:
            warning_type: Loại cảnh báo.
            message: Thông báo.
            details: Chi tiết.
        """
        self.log(
            LogLevel.WARNING,
            LogCategory.SAFETY,
            f"Safety warning: {message}",
            {
                "warning_type": warning_type,
                "details": details or {},
            }
        )
    
    def log_sync_status(
        self,
        status: str,
        phase: str,
        user_angle: float,
        target_angle: float
    ) -> None:
        """
        Ghi log trạng thái đồng bộ.
        
        Args:
            status: Trạng thái sync.
            phase: Pha hiện tại.
            user_angle: Góc của user.
            target_angle: Góc mục tiêu.
        """
        self.log(
            LogLevel.DEBUG,
            LogCategory.SYNC,
            f"Sync {status}: phase={phase}, angle={user_angle:.1f}°/{target_angle:.1f}°",
            {
                "status": status,
                "phase": phase,
                "user_angle": user_angle,
                "target_angle": target_angle,
            }
        )
    
    def end_session(self, report: Optional[Dict] = None) -> str:
        """
        Kết thúc session và lưu báo cáo.
        
        Args:
            report: Báo cáo buổi tập (SessionReport.to_dict()).
            
        Returns:
            str: Đường dẫn đến JSON report.
        """
        # Log session end
        self.log(
            LogLevel.INFO,
            LogCategory.SESSION,
            "Session ended",
            {
                "end_time": datetime.now().isoformat(),
                "total_entries": len(self._entries),
            }
        )
        
        # Stop async writer
        if self._async_write and self._writer_thread:
            self._stop_writer = True
            self._writer_thread.join(timeout=2.0)
        
        # Close CSV
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        
        # Save JSON report
        if self._json_file:
            full_report = {
                "session_id": self._session_id,
                "session_start": self._session_start.isoformat() if self._session_start else "",
                "session_end": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self._entries],
                "report": report or {},
            }
            
            with open(self._json_file, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, ensure_ascii=False, indent=2)
            
            if self._console_output:
                self._console_logger.info(f"Report saved: {self._json_file}")
            
            return str(self._json_file)
        
        return ""
    
    def get_entries(
        self,
        category: Optional[LogCategory] = None,
        level: Optional[LogLevel] = None
    ) -> List[LogEntry]:
        """
        Lấy các entries đã log.
        
        Args:
            category: Lọc theo category.
            level: Lọc theo level.
            
        Returns:
            List[LogEntry]: Danh sách entries.
        """
        entries = self._entries
        
        if category:
            entries = [e for e in entries if e.category == category.value]
        
        if level:
            entries = [e for e in entries if e.level == level.value]
        
        return entries
    
    def get_summary(self) -> Dict:
        """
        Lấy tóm tắt session.
        
        Returns:
            Dict chứa thống kê.
        """
        rep_entries = [e for e in self._entries if e.category == LogCategory.REP.value]
        pain_entries = [e for e in self._entries if e.category == LogCategory.PAIN.value]
        fatigue_entries = [e for e in self._entries if e.category == LogCategory.FATIGUE.value]
        
        return {
            "session_id": self._session_id,
            "total_entries": len(self._entries),
            "total_reps": len(rep_entries),
            "pain_events": len(pain_entries),
            "fatigue_warnings": len(fatigue_entries),
            "files": {
                "json": str(self._json_file) if self._json_file else "",
                "csv": str(self._csv_file) if self._csv_file else "",
            }
        }


def create_session_logger(
    log_dir: str = "./data/logs",
    console: bool = True
) -> SessionLogger:
    """
    Factory function để tạo SessionLogger.
    
    Args:
        log_dir: Thư mục lưu log.
        console: Có output console không.
        
    Returns:
        SessionLogger instance.
    """
    return SessionLogger(log_dir, console_output=console, async_write=True)