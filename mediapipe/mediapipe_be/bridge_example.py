#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MEMOTION BACKEND INTEGRATION GUIDE                        ║
║                                                                              ║
║  File này mô phỏng cách một kỹ sư Backend sử dụng EngineService              ║
║  để tích hợp vào hệ thống của họ (FastAPI, Flask, WebSocket, etc.)           ║
║                                                                              ║
║  Author: MEMOTION Team                                                       ║
║  Version: 1.0.0                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CÁCH SỬ DỤNG:
    1. Khởi tạo EngineService MỘT LẦN khi server start
    2. Gọi process_frame() mỗi khi nhận được frame từ client
    3. Trả kết quả JSON về cho Frontend qua WebSocket/HTTP
    4. Lấy báo cáo cuối cùng khi engine.is_complete() == True

LUỒNG DỮ LIỆU:
    Camera/Video → Frame → process_frame() → JSON → Frontend
                                ↓
                          EngineState (nội bộ)
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Thêm path để import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import từ service layer
from service import (
    EngineService,
    EngineConfig,
    EngineOutput,
)

# Optional: Import OpenCV nếu có
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[WARNING] OpenCV không được cài đặt. Chỉ có thể chạy simulation mode.")
    print("          Cài đặt: pip install opencv-python")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: KHỞI TẠO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def create_engine() -> EngineService:
    """
    Khởi tạo EngineService với cấu hình.
    
    Backend Engineer chỉ cần gọi hàm này MỘT LẦN khi server khởi động.
    
    Returns:
        EngineService: Instance đã được cấu hình
    """
    # Lấy đường dẫn tuyệt đối dựa vào vị trí script
    script_dir = Path(__file__).parent  # mediapipe_be/
    project_root = script_dir.parent    # mediapipe/
    
    # Cấu hình engine với đường dẫn tuyệt đối
    config = EngineConfig(
        # Đường dẫn tới thư mục chứa model files
        # Có thể dùng models trong mediapipe_be/ hoặc trong mediapipe/
        models_dir=str(script_dir / "models"),  # mediapipe_be/models/
        
        # Đường dẫn lưu logs
        log_dir=str(project_root / "data" / "logs"),  # mediapipe/data/logs/
        
        # Video mẫu để sync (Phase 3)
        # Nếu không có, sẽ skip Phase 3 và chuyển thẳng sang Phase 4
        ref_video_path=str(project_root / "videos" / "arm_raise.mp4"),
        
        # Khớp mặc định để tracking
        # Có thể là: left_shoulder, right_shoulder, left_elbow, 
        #            right_elbow, left_knee, right_knee
        default_joint="left_shoulder",
        
        # Số frame cần stable để xác nhận pose (Phase 1)
        detection_stable_threshold=30,
        
        # Thời gian đo mỗi khớp (ms) - Phase 2
        calibration_duration_ms=5000,
    )
    
    # Debug: In ra đường dẫn đang sử dụng
    print(f"   📁 Models dir: {config.models_dir}")
    print(f"   📁 Log dir: {config.log_dir}")
    print(f"   📁 Video path: {config.ref_video_path}")
    
    # Kiểm tra models tồn tại
    models_path = Path(config.models_dir)
    pose_model = models_path / "pose_landmarker_lite.task"
    if pose_model.exists():
        print(f"   ✅ Pose model found: {pose_model}")
    else:
        print(f"   ❌ Pose model NOT found: {pose_model}")
    
    # Tạo engine instance
    engine = EngineService(config)
    
    # Khởi tạo các components (detector, scorer, etc.)
    # Nếu không gọi initialize(), engine sẽ tự gọi trong process_frame()
    success = engine.initialize()
    
    if success:
        print("✅ Engine đã khởi tạo thành công!")
    else:
        print("❌ Lỗi khởi tạo engine. Kiểm tra model files.")
    
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: XỬ LÝ FRAME VÀ LẤY KẾT QUẢ JSON
# ═══════════════════════════════════════════════════════════════════════════════

def process_and_print_result(engine: EngineService, frame: np.ndarray, timestamp_ms: int):
    """
    Xử lý frame và in ra kết quả theo từng phase.
    
    Đây là hàm mẫu cho thấy cách xử lý output từ engine.
    Trong thực tế, Backend sẽ convert to JSON và gửi qua WebSocket.
    
    Args:
        engine: EngineService instance
        frame: Frame ảnh (numpy array BGR)
        timestamp_ms: Timestamp tính bằng milliseconds
    
    Returns:
        Dict: Kết quả JSON-serializable
    """
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 1: GỌI process_frame() - Đây là hàm CHÍNH của Engine
    # ═══════════════════════════════════════════════════════════════════════
    result: EngineOutput = engine.process_frame(frame, timestamp_ms)
    
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 2: CONVERT SANG JSON
    # ═══════════════════════════════════════════════════════════════════════
    json_data = result.to_dict()
    
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 3: XỬ LÝ THEO PHASE
    # ═══════════════════════════════════════════════════════════════════════
    
    phase = result.current_phase
    phase_name = result.phase_name
    
    print(f"\n{'═' * 60}")
    print(f"PHASE {phase}: {phase_name.upper()}")
    print(f"{'═' * 60}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: DETECTION - Nhận diện tư thế
    # ─────────────────────────────────────────────────────────────────────────
    if phase == 1 and result.detection:
        det = result.detection
        print(f"""
┌─ DETECTION STATUS ─────────────────────────────────────────┐
│ Pose Detected: {det.get('pose_detected', False)}
│ Stable Count:  {det.get('stable_count', 0)}/30 frames
│ Progress:      {det.get('progress', 0):.0%}
│ Status:        {det.get('status', 'idle')}
│ Message:       {det.get('message', '')}
│ Countdown:     {det.get('countdown_remaining', '-')} giây
└────────────────────────────────────────────────────────────┘
        """)
        
        # Thông số quan trọng cho Frontend
        print("📤 DỮ LIỆU GỬI FRONTEND:")
        print(f"   - pose_detected: {det.get('pose_detected')} → Hiển thị skeleton")
        print(f"   - progress: {det.get('progress', 0):.0%} → Progress bar")
        print(f"   - countdown_remaining: {det.get('countdown_remaining')} → Đếm ngược")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: CALIBRATION - Đo giới hạn vận động
    # ─────────────────────────────────────────────────────────────────────────
    elif phase == 2 and result.calibration:
        cal = result.calibration
        print(f"""
┌─ CALIBRATION STATUS ───────────────────────────────────────┐
│ Current Joint:     {cal.get('current_joint_name', 'N/A')}
│ Queue Position:    {cal.get('queue_index', 0) + 1}/6 khớp
│ Overall Progress:  {cal.get('overall_progress', 0):.0%}
│ Joint Progress:    {cal.get('progress', 0):.0%}
│ Current Angle:     {cal.get('current_angle', 0):.1f}°
│ Max Angle:         {cal.get('user_max_angle', 0):.1f}°
│ Status:            {cal.get('status', 'preparing')}
│ Instruction:       {cal.get('position_instruction', '')}
│ Countdown:         {cal.get('countdown_remaining', '-')} giây
└────────────────────────────────────────────────────────────┘
        """)
        
        # Danh sách khớp đã/đang/chưa đo
        print("📊 TRẠNG THÁI CÁC KHỚP:")
        for joint in cal.get('joints_status', []):
            status_icon = "✅" if joint['status'] == 'complete' else "🔄" if joint['status'] == 'collecting' else "⏳"
            angle = f"{joint['max_angle']:.1f}°" if joint['max_angle'] else "---"
            print(f"   {status_icon} {joint['joint_name']}: {angle}")
        
        # Thông số quan trọng cho Frontend
        print("\n📤 DỮ LIỆU GỬI FRONTEND:")
        print(f"   - current_joint: {cal.get('current_joint')} → Highlight khớp")
        print(f"   - position_instruction: '{cal.get('position_instruction')}' → Hướng dẫn")
        print(f"   - overall_progress: {cal.get('overall_progress', 0):.0%} → Progress bar tổng")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: SYNC - Đồng bộ chuyển động
    # ─────────────────────────────────────────────────────────────────────────
    elif phase == 3 and result.sync:
        sync = result.sync
        print(f"""
┌─ SYNC STATUS ──────────────────────────────────────────────┐
│ User Angle:      {sync.get('user_angle', 0):.1f}° → Target: {sync.get('target_angle', 0):.1f}°
│ Error:           {sync.get('error', 0):.1f}° ({sync.get('direction_hint', 'hold')})
│ Current Score:   {sync.get('current_score', 0):.1f}/100
│ Average Score:   {sync.get('average_score', 0):.1f}/100
│ Motion Phase:    {sync.get('motion_phase', 'idle').upper()}
│ Rep Count:       {sync.get('rep_count', 0)}
│ Video Progress:  {sync.get('video_progress', 0):.0%}
│ Pain Level:      {sync.get('pain_level', 'NONE')}
│ Fatigue Level:   {sync.get('fatigue_level', 'FRESH')}
│ Feedback:        {sync.get('feedback_text', '')}
└────────────────────────────────────────────────────────────┘
        """)
        
        # Chi tiết sai số từng khớp (MULTI-JOINT)
        joint_errors = sync.get('joint_errors', [])
        if joint_errors:
            print(f"📊 CHI TIẾT {len(joint_errors)} KHỚP:")
            print("   ┌─────────────────────┬────────┬────────┬───────┬──────────────┐")
            print("   │ Khớp                │ User   │ Target │ Score │ Hướng        │")
            print("   ├─────────────────────┼────────┼────────┼───────┼──────────────┤")
            for je in joint_errors:
                direction_vi = {
                    'raise': '↑ Nâng cao',
                    'lower': '↓ Hạ thấp', 
                    'hold': '= Giữ',
                    'ok': '✓ Đạt'
                }.get(je.get('direction_hint', 'hold'), je.get('direction_hint', ''))
                print(f"   │ {je.get('joint_name', ''):<19} │ {je.get('user_angle', 0):>5.1f}° │ {je.get('target_angle', 0):>5.1f}° │ {je.get('score', 0):>5.1f} │ {direction_vi:<12} │")
            print("   └─────────────────────┴────────┴────────┴───────┴──────────────┘")
        
        # Cảnh báo
        if sync.get('warning'):
            print(f"\n⚠️  CẢNH BÁO: {sync.get('warning')}")
        
        # Thông số quan trọng cho Frontend
        print("\n📤 DỮ LIỆU GỬI FRONTEND:")
        print(f"   - user_angle/target_angle → Hiển thị góc và target")
        print(f"   - direction_hint: '{sync.get('direction_hint')}' → Mũi tên hướng dẫn")
        print(f"   - joint_errors[] → Danh sách chi tiết từng khớp")
        print(f"   - feedback_text: '{sync.get('feedback_text')}' → Banner feedback")
        print(f"   - motion_phase: '{sync.get('motion_phase')}' → Trạng thái động tác")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: SCORING - Báo cáo kết quả
    # ─────────────────────────────────────────────────────────────────────────
    elif phase == 4 and result.final_report:
        report = result.final_report
        print(f"""
╔════════════════════════════════════════════════════════════╗
║                    BÁO CÁO CUỐI BUỔI TẬP                   ║
╠════════════════════════════════════════════════════════════╣
║ Session ID:    {report.get('session_id', 'N/A'):<42} ║
║ Exercise:      {report.get('exercise_name', 'N/A'):<42} ║
║ Duration:      {report.get('duration_seconds', 0)} giây{' ' * 35}║
╠════════════════════════════════════════════════════════════╣
║                      ĐIỂM SỐ                               ║
║────────────────────────────────────────────────────────────║
║ ★ TOTAL SCORE:  {report.get('total_score', 0):.1f}/100{' ' * 32}║
║   ROM Score:    {report.get('rom_score', 0):.1f}{' ' * 40}║
║   Stability:    {report.get('stability_score', 0):.1f}{' ' * 40}║
║   Flow:         {report.get('flow_score', 0):.1f}{' ' * 40}║
║────────────────────────────────────────────────────────────║
║ Grade:          {report.get('grade', 'N/A'):<42} ║
║ Total Reps:     {report.get('total_reps', 0):<42} ║
║ Fatigue Level:  {report.get('fatigue_level', 'FRESH'):<42} ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        # Kết quả calibration
        calibrated = report.get('calibrated_joints', [])
        if calibrated:
            print("📐 GÓC TỐI ĐA ĐÃ CALIBRATE:")
            for joint in calibrated:
                print(f"   - {joint['joint_name']}: {joint['max_angle']:.1f}°")
        
        # Chi tiết từng rep
        rep_scores = report.get('rep_scores', [])
        if rep_scores:
            print(f"\n📈 CHI TIẾT {len(rep_scores)} REP:")
            for rep in rep_scores[:5]:  # Hiển thị 5 rep đầu
                print(f"   Rep {rep['rep_number']}: {rep['total_score']:.1f} pts "
                      f"(ROM:{rep['rom_score']:.0f} | Stab:{rep['stability_score']:.0f} | Flow:{rep['flow_score']:.0f})")
        
        # Khuyến nghị
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\n💡 KHUYẾN NGHỊ:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ERROR HANDLING
    # ─────────────────────────────────────────────────────────────────────────
    if result.error:
        print(f"\n❌ LỖI: {result.error}")
    
    return json_data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LẤY BÁO CÁO CUỐI CÙNG
# ═══════════════════════════════════════════════════════════════════════════════

def get_final_report(engine: EngineService) -> Optional[Dict]:
    """
    Lấy báo cáo cuối cùng sau khi Phase 3 kết thúc.
    
    ⚠️  QUAN TRỌNG: Hàm này chỉ nên gọi khi engine.is_complete() == True
    
    Args:
        engine: EngineService instance
    
    Returns:
        Dict: Báo cáo cuối cùng hoặc None nếu chưa hoàn thành
    """
    if not engine.is_complete():
        print("⚠️  Session chưa hoàn thành. Không có báo cáo.")
        return None
    
    # Cách 1: Lấy từ kết quả process_frame cuối cùng
    # Khi phase == 4, result.final_report sẽ có dữ liệu
    
    # Cách 2: Tạo dummy frame để lấy báo cáo
    if HAS_OPENCV:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp_ms = int(time.time() * 1000)
        result = engine.process_frame(dummy_frame, timestamp_ms)
        
        if result.final_report:
            return result.final_report
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SIMULATION - MÔ PHỎNG VÒNG LẶP NHẬN FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_video_processing():
    """
    Mô phỏng việc nhận frame từ nguồn video và xử lý.
    
    Trong thực tế, frame sẽ đến từ:
    - WebSocket (client gửi base64 encoded frame)
    - RTSP stream
    - USB camera
    - Video file
    """
    if not HAS_OPENCV:
        print("❌ Cần OpenCV để chạy simulation.")
        print("   Cài đặt: pip install opencv-python numpy")
        return
    
    print("\n" + "═" * 70)
    print("              MÔ PHỎNG VÒNG LẶP BACKEND NHẬN FRAME")
    print("═" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 1: KHỞI TẠO ENGINE (chỉ làm 1 lần khi server start)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/4] Khởi tạo Engine...")
    engine = create_engine()
    
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 2: MỞ NGUỒN VIDEO (webcam, file, hoặc stream)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/4] Mở nguồn video...")
    
    # Option 1: Webcam
    # cap = cv2.VideoCapture(0)
    
    # Option 2: Video file
    video_path = "../videos/arm_raise.mp4"  # Hoặc video test bất kỳ
    
    if Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        print(f"   Đang sử dụng video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print("   Đang sử dụng webcam (video file không tồn tại)")
    
    if not cap.isOpened():
        print("❌ Không thể mở nguồn video!")
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # BƯỚC 3: VÒNG LẶP CHÍNH - XỬ LÝ TỪNG FRAME
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/4] Bắt đầu xử lý frames...")
    print("      (Nhấn Ctrl+C để dừng)\n")
    
    frame_count = 0
    last_print_time = 0
    
    try:
        while True:
            # Đọc frame từ nguồn
            ret, frame = cap.read()
            if not ret:
                # Nếu là video file, có thể loop lại
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Mirror frame nếu là webcam
            # frame = cv2.flip(frame, 1)
            
            # Timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # ═══════════════════════════════════════════════════════════════
            # GỌI process_frame() - NƠI XỬ LÝ CHÍNH
            # ═══════════════════════════════════════════════════════════════
            result = engine.process_frame(frame, timestamp_ms)
            
            # Convert to JSON để gửi qua WebSocket
            json_data = result.to_dict()
            
            # In kết quả mỗi 2 giây để không spam console
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                process_and_print_result(engine, frame, timestamp_ms)
                last_print_time = current_time
            
            frame_count += 1
            
            # ═══════════════════════════════════════════════════════════════
            # KIỂM TRA HOÀN THÀNH
            # ═══════════════════════════════════════════════════════════════
            if engine.is_complete():
                print("\n" + "🎉" * 20)
                print("         SESSION HOÀN THÀNH!")
                print("🎉" * 20)
                
                # Lấy báo cáo cuối cùng
                final_report = get_final_report(engine)
                if final_report:
                    print("\n📋 BÁO CÁO CUỐI CÙNG (JSON):")
                    print(json.dumps(final_report, indent=2, ensure_ascii=False))
                
                break
            
            # Delay nhỏ để giảm CPU (trong thực tế không cần)
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Dừng bởi người dùng")
    
    finally:
        # ═══════════════════════════════════════════════════════════════════
        # BƯỚC 4: CLEANUP
        # ═══════════════════════════════════════════════════════════════════
        print("\n[4/4] Cleanup...")
        cap.release()
        engine.cleanup()
        print(f"   Đã xử lý {frame_count} frames")
        print("   ✅ Done!")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VÍ DỤ TÍCH HỢP FASTAPI + WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

FASTAPI_EXAMPLE = '''
# ═══════════════════════════════════════════════════════════════════════════════
# FILE: main.py - FastAPI + WebSocket Integration
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import asyncio

from service import EngineService, EngineConfig

app = FastAPI(title="MEMOTION Backend")

# CORS cho Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: EngineService = None

@app.on_event("startup")
async def startup():
    """Khởi tạo engine khi server start."""
    global engine
    
    config = EngineConfig(
        models_dir="./models",
        ref_video_path="./videos/exercise.mp4",
        default_joint="left_shoulder"
    )
    
    engine = EngineService(config)
    success = engine.initialize()
    
    if success:
        print("✅ Engine initialized")
    else:
        print("❌ Engine initialization failed")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup khi server shutdown."""
    global engine
    if engine:
        engine.cleanup()

@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    """
    WebSocket endpoint cho session tập luyện.
    
    Client gửi:
    {
        "type": "frame",
        "frame": "<base64 encoded image>",
        "timestamp": 1234567890
    }
    
    hoặc:
    {
        "type": "control",
        "command": "pause" | "resume" | "restart"
    }
    
    Server trả về:
    {
        "current_phase": 1-4,
        "phase_name": "detection" | "calibration" | "sync" | "scoring",
        "detection": {...} | null,
        "calibration": {...} | null,
        "sync": {...} | null,
        "final_report": {...} | null
    }
    """
    await websocket.accept()
    print(f"Client connected")
    
    try:
        while True:
            # Nhận message từ client
            data = await websocket.receive_json()
            
            if data.get("type") == "frame":
                # ═══════════════════════════════════════════════════════
                # DECODE FRAME TỪ BASE64
                # ═══════════════════════════════════════════════════════
                try:
                    img_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({"error": "Invalid frame data"})
                        continue
                    
                    timestamp_ms = data.get("timestamp", int(time.time() * 1000))
                    
                    # ═══════════════════════════════════════════════════
                    # XỬ LÝ FRAME VÀ GỬI KẾT QUẢ
                    # ═══════════════════════════════════════════════════
                    result = engine.process_frame(frame, timestamp_ms)
                    
                    # Gửi JSON về client
                    await websocket.send_json(result.to_dict())
                    
                    # Check if complete
                    if engine.is_complete():
                        # Gửi final report
                        final = result.final_report
                        await websocket.send_json({
                            "type": "session_complete",
                            "final_report": final
                        })
                
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
            
            elif data.get("type") == "control":
                # ═══════════════════════════════════════════════════════
                # XỬ LÝ LỆNH ĐIỀU KHIỂN
                # ═══════════════════════════════════════════════════════
                command = data.get("command")
                
                if command == "pause":
                    engine.pause()
                    await websocket.send_json({"status": "paused"})
                
                elif command == "resume":
                    engine.resume()
                    await websocket.send_json({"status": "resumed"})
                
                elif command == "restart":
                    engine.restart()
                    await websocket.send_json({"status": "restarted"})
                
                else:
                    await websocket.send_json({"error": f"Unknown command: {command}"})
    
    except WebSocketDisconnect:
        print(f"Client disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/api/status")
async def get_status():
    """Lấy trạng thái hiện tại của engine."""
    return {
        "initialized": engine is not None,
        "current_phase": engine.get_current_phase() if engine else 0,
        "is_complete": engine.is_complete() if engine else False
    }

# Chạy với: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
'''


def print_fastapi_example():
    """In ví dụ FastAPI integration."""
    print("\n" + "═" * 70)
    print("         VÍ DỤ TÍCH HỢP FASTAPI + WEBSOCKET")
    print("═" * 70)
    print(FASTAPI_EXAMPLE)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: VÍ DỤ JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_json_examples():
    """In ví dụ JSON output cho từng phase."""
    
    print("\n" + "═" * 70)
    print("              CẤU TRÚC JSON OUTPUT TỪNG PHASE")
    print("═" * 70)
    
    # Phase 1
    phase1 = {
        "current_phase": 1,
        "phase_name": "detection",
        "detection": {
            "pose_detected": True,
            "stable_count": 28,
            "progress": 0.93,
            "countdown_remaining": 1.5,
            "status": "countdown",
            "message": "Chuan bi... 2 giay"
        }
    }
    print("\n📌 PHASE 1 - DETECTION:")
    print(json.dumps(phase1, indent=2, ensure_ascii=False))
    
    # Phase 2
    phase2 = {
        "current_phase": 2,
        "phase_name": "calibration",
        "calibration": {
            "current_joint": "left_elbow",
            "current_joint_name": "Khuyu tay trai",
            "queue_index": 2,
            "total_joints": 6,
            "progress": 0.75,
            "overall_progress": 0.33,
            "current_angle": 128.5,
            "status": "collecting",
            "position_instruction": "Moi ba dung NGANG",
            "joints_status": [
                {"joint_name": "Vai trai", "status": "complete", "max_angle": 145.3},
                {"joint_name": "Vai phai", "status": "complete", "max_angle": 142.1},
                {"joint_name": "Khuyu tay trai", "status": "collecting", "max_angle": None}
            ]
        }
    }
    print("\n📌 PHASE 2 - CALIBRATION:")
    print(json.dumps(phase2, indent=2, ensure_ascii=False))
    
    # Phase 3
    phase3 = {
        "current_phase": 3,
        "phase_name": "sync",
        "sync": {
            "user_angle": 85.3,
            "target_angle": 90.0,
            "error": 4.7,
            "current_score": 87.5,
            "average_score": 82.3,
            "motion_phase": "eccentric",
            "rep_count": 3,
            "video_progress": 0.45,
            "pain_level": "NONE",
            "fatigue_level": "MILD",
            "feedback_text": "TOT!",
            "direction_hint": "raise",
            "joint_errors": [
                {
                    "joint_name": "Vai trai",
                    "user_angle": 85.3,
                    "target_angle": 90.0,
                    "score": 92.5,
                    "direction_hint": "raise"
                }
            ]
        }
    }
    print("\n📌 PHASE 3 - SYNC (Multi-joint):")
    print(json.dumps(phase3, indent=2, ensure_ascii=False))
    
    # Phase 4
    phase4 = {
        "current_phase": 4,
        "phase_name": "scoring",
        "final_report": {
            "session_id": "session_1737458123",
            "total_score": 82.5,
            "rom_score": 85.3,
            "stability_score": 78.2,
            "flow_score": 83.8,
            "grade": "XUAT SAC",
            "total_reps": 5,
            "fatigue_level": "MILD",
            "calibrated_joints": [
                {"joint_name": "Vai trai", "max_angle": 145.3}
            ],
            "recommendations": [
                "Tiep tuc tap luyen deu dan moi ngay"
            ]
        }
    }
    print("\n📌 PHASE 4 - FINAL REPORT:")
    print(json.dumps(phase4, indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗███████╗███╗   ███╗ ██████╗ ████████╗██╗ ██████╗ ███╗   ██╗    ║
║   ████╗ ████║██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║    ║
║   ██╔████╔██║█████╗  ██╔████╔██║██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║    ║
║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██║██║   ██║██║╚██╗██║    ║
║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║    ║
║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ║
║                                                                              ║
║                    BACKEND INTEGRATION EXAMPLES                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("Chọn ví dụ để chạy:\n")
    print("  1. In cấu trúc JSON output của từng Phase")
    print("  2. In ví dụ FastAPI + WebSocket integration")
    print("  3. Chạy simulation xử lý video (cần OpenCV)")
    print("  4. Thoát")
    
    while True:
        try:
            choice = input("\nNhập lựa chọn (1-4): ").strip()
            
            if choice == "1":
                print_json_examples()
            elif choice == "2":
                print_fastapi_example()
            elif choice == "3":
                simulate_video_processing()
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ. Vui lòng chọn 1-4.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
