"""
MEMOTION Backend Integration Example

Vi du su dung MemotionEngine trong Backend da nguoi dung.
File nay minh hoa cach tich hop Engine vao FastAPI/Flask.

Author: MEMOTION Team
Version: 2.0.0
"""

import numpy as np
import time
from typing import Dict, Optional
from pathlib import Path

# Import tu service layer
from service import (
    MemotionEngine,
    EngineConfig,
    create_engine_for_user,
    EngineOutput,
)


# ==================== MULTI-USER ENGINE MANAGER ====================

class EngineManager:
    """
    Quan ly nhieu Engine instance cho nhieu user.
    
    Moi user co 1 engine instance rieng, dam bao state doc lap.
    
    Usage:
        manager = EngineManager()
        
        # Xu ly frame cho user
        result = manager.process_frame("user_123", frame, timestamp_ms)
        
        # Lay trang thai
        state = manager.get_user_state("user_123")
        
        # Cleanup khi user disconnect
        manager.remove_user("user_123")
    """
    
    def __init__(self, default_config: Optional[EngineConfig] = None):
        """
        Khoi tao manager.
        
        Args:
            default_config: Config mac dinh cho tat ca engine
        """
        self._engines: Dict[str, MemotionEngine] = {}
        self._default_config = default_config or EngineConfig(
            models_dir="./models",
            log_dir="./data/logs",
            ref_video_path="./videos/exercise.mp4"
        )
        self._last_activity: Dict[str, float] = {}
    
    def get_or_create_engine(self, user_id: str) -> MemotionEngine:
        """
        Lay engine cua user, tao moi neu chua co.
        
        Args:
            user_id: ID cua user
        
        Returns:
            MemotionEngine instance
        """
        if user_id not in self._engines:
            self._engines[user_id] = create_engine_for_user(
                user_id=user_id,
                config=self._default_config
            )
            print(f"[EngineManager] Created engine for user: {user_id}")
        
        self._last_activity[user_id] = time.time()
        return self._engines[user_id]
    
    def process_frame(
        self, 
        user_id: str, 
        frame: np.ndarray, 
        timestamp_ms: int
    ) -> Dict:
        """
        Xu ly frame cho mot user.
        
        Args:
            user_id: ID cua user
            frame: Frame anh (BGR numpy array)
            timestamp_ms: Timestamp (ms)
        
        Returns:
            Dict JSON-serializable voi key "phase" bat buoc
        """
        engine = self.get_or_create_engine(user_id)
        result = engine.process_frame(frame, timestamp_ms)
        
        # Tra ve JSON-serializable dict
        return result.to_dict()
    
    def get_user_state(self, user_id: str) -> Optional[Dict]:
        """Lay trang thai hien tai cua user."""
        if user_id not in self._engines:
            return None
        return self._engines[user_id].get_state_snapshot()
    
    def get_user_phase(self, user_id: str) -> int:
        """Lay phase hien tai cua user."""
        if user_id not in self._engines:
            return 0
        return self._engines[user_id].get_current_phase()
    
    def pause_user(self, user_id: str) -> bool:
        """Pause video cho user (Phase 3)."""
        if user_id not in self._engines:
            return False
        self._engines[user_id].pause()
        return True
    
    def resume_user(self, user_id: str) -> bool:
        """Resume video cho user (Phase 3)."""
        if user_id not in self._engines:
            return False
        self._engines[user_id].resume()
        return True
    
    def restart_user(self, user_id: str) -> bool:
        """Restart session cho user."""
        if user_id not in self._engines:
            return False
        self._engines[user_id].restart()
        return True
    
    def remove_user(self, user_id: str) -> bool:
        """
        Xoa engine cua user va don dep resources.
        
        Goi khi user disconnect hoac timeout.
        """
        if user_id not in self._engines:
            return False
        
        self._engines[user_id].cleanup()
        del self._engines[user_id]
        del self._last_activity[user_id]
        print(f"[EngineManager] Removed engine for user: {user_id}")
        return True
    
    def cleanup_inactive(self, timeout_seconds: int = 3600) -> int:
        """
        Don dep cac engine khong hoat dong.
        
        Args:
            timeout_seconds: Thoi gian timeout (default 1 gio)
        
        Returns:
            So engine da don dep
        """
        now = time.time()
        inactive_users = [
            user_id for user_id, last_time in self._last_activity.items()
            if now - last_time > timeout_seconds
        ]
        
        for user_id in inactive_users:
            self.remove_user(user_id)
        
        return len(inactive_users)
    
    def get_stats(self) -> Dict:
        """Lay thong ke cua manager."""
        return {
            "total_users": len(self._engines),
            "users": list(self._engines.keys()),
            "user_phases": {
                user_id: engine.get_current_phase()
                for user_id, engine in self._engines.items()
            }
        }


# ==================== FASTAPI EXAMPLE ====================

def create_fastapi_app():
    """
    Tao FastAPI app voi MEMOTION integration.
    
    Yeu cau: pip install fastapi uvicorn python-multipart
    """
    try:
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.responses import JSONResponse
        import cv2
    except ImportError:
        print("Chua cai dat FastAPI. Chay: pip install fastapi uvicorn python-multipart")
        return None
    
    app = FastAPI(
        title="MEMOTION Backend API",
        description="API xu ly frame cho ung dung phuc hoi chuc nang",
        version="2.0.0"
    )
    
    # Global engine manager
    manager = EngineManager()
    
    @app.post("/api/process_frame")
    async def process_frame(
        user_id: str = Form(...),
        timestamp_ms: int = Form(...),
        frame_file: UploadFile = File(...)
    ):
        """
        Xu ly mot frame cho user.
        
        Returns JSON voi key "phase" bat buoc (1, 2, 3, hoac 4).
        """
        try:
            # Doc frame tu file upload
            contents = await frame_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid frame data")
            
            # Xu ly frame
            result = manager.process_frame(user_id, frame, timestamp_ms)
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/user/{user_id}/state")
    async def get_user_state(user_id: str):
        """Lay trang thai hien tai cua user."""
        state = manager.get_user_state(user_id)
        if state is None:
            raise HTTPException(status_code=404, detail="User not found")
        return JSONResponse(content=state)
    
    @app.post("/api/user/{user_id}/pause")
    async def pause_user(user_id: str):
        """Pause video cho user (Phase 3)."""
        if not manager.pause_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "paused"}
    
    @app.post("/api/user/{user_id}/resume")
    async def resume_user(user_id: str):
        """Resume video cho user (Phase 3)."""
        if not manager.resume_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "resumed"}
    
    @app.post("/api/user/{user_id}/restart")
    async def restart_user(user_id: str):
        """Restart session cho user."""
        if not manager.restart_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "restarted"}
    
    @app.delete("/api/user/{user_id}")
    async def remove_user(user_id: str):
        """Xoa user va don dep resources."""
        if not manager.remove_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "removed"}
    
    @app.get("/api/stats")
    async def get_stats():
        """Lay thong ke cua he thong."""
        return JSONResponse(content=manager.get_stats())
    
    return app


# ==================== DEMO: SINGLE USER ====================

def demo_single_user():
    """Demo xu ly frame cho 1 user."""
    print("=" * 60)
    print("DEMO: Single User Processing")
    print("=" * 60)
    
    # Tao engine
    config = EngineConfig(
        models_dir="./models",
        ref_video_path="./videos/exercise.mp4"
    )
    engine = MemotionEngine(config)
    
    print(f"Instance ID: {engine.get_instance_id()}")
    print(f"Current Phase: {engine.get_current_phase()}")
    
    # Simulate frames
    print("\nSimulating frames...")
    for i in range(5):
        # Tao frame gia
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp_ms = i * 33  # ~30fps
        
        result = engine.process_frame(fake_frame, timestamp_ms)
        output = result.to_dict()
        
        print(f"Frame {i}: phase={output['phase']}, phase_name={output['phase_name']}")
    
    # Cleanup
    engine.cleanup()
    print("\nDemo completed!")


# ==================== DEMO: MULTI USER ====================

def demo_multi_user():
    """Demo xu ly frame cho nhieu user dong thoi."""
    print("=" * 60)
    print("DEMO: Multi-User Processing")
    print("=" * 60)
    
    manager = EngineManager()
    users = ["user_001", "user_002", "user_003"]
    
    # Moi user gui 3 frames
    for frame_idx in range(3):
        print(f"\n--- Frame batch {frame_idx + 1} ---")
        
        for user_id in users:
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            timestamp_ms = frame_idx * 33
            
            result = manager.process_frame(user_id, fake_frame, timestamp_ms)
            
            print(f"{user_id}: phase={result['phase']}, "
                  f"phase_name={result['phase_name']}")
    
    # Stats
    print("\n--- Stats ---")
    stats = manager.get_stats()
    print(f"Total users: {stats['total_users']}")
    print(f"User phases: {stats['user_phases']}")
    
    # Cleanup
    for user_id in users:
        manager.remove_user(user_id)
    
    print("\nDemo completed!")


# ==================== OUTPUT FORMAT DEMO ====================

def demo_output_format():
    """Demo dinh dang output JSON."""
    print("=" * 60)
    print("DEMO: Output JSON Format")
    print("=" * 60)
    
    # Tao output mau cho moi phase
    from service import (
        EngineOutput, DetectionOutput, CalibrationOutput,
        SyncOutput, FinalReportOutput
    )
    
    # Phase 1 output
    phase1_output = EngineOutput(
        current_phase=1,
        phase_name="detection",
        detection=DetectionOutput(
            pose_detected=False,
            stable_count=15,
            progress=0.5,
            status="detecting",
            message="Dang xac nhan... 50%"
        ).to_dict()
    )
    
    print("\n--- Phase 1 Output ---")
    import json
    print(json.dumps(phase1_output.to_dict(), indent=2, ensure_ascii=False))
    
    # Phase 2 output
    phase2_output = EngineOutput(
        current_phase=2,
        phase_name="calibration",
        calibration=CalibrationOutput(
            current_joint="left_shoulder",
            current_joint_name="Vai trai",
            queue_index=0,
            total_joints=6,
            progress=0.3,
            status="collecting",
            message="Dang do Vai trai... 30%"
        ).to_dict()
    )
    
    print("\n--- Phase 2 Output ---")
    print(json.dumps(phase2_output.to_dict(), indent=2, ensure_ascii=False))
    
    # Phase 3 output
    phase3_output = EngineOutput(
        current_phase=3,
        phase_name="sync",
        sync=SyncOutput(
            user_angle=85.5,
            target_angle=90.0,
            error=4.5,
            current_score=92.5,
            average_score=88.0,
            motion_phase="concentric",
            rep_count=3,
            video_progress=0.45,
            pain_level="NONE",
            fatigue_level="MILD",
            feedback_text="TUYET VOI!",
            direction_hint="raise",
            status="syncing"
        ).to_dict()
    )
    
    print("\n--- Phase 3 Output ---")
    print(json.dumps(phase3_output.to_dict(), indent=2, ensure_ascii=False))
    
    # Phase 4 output
    phase4_output = EngineOutput(
        current_phase=4,
        phase_name="scoring",
        final_report=FinalReportOutput(
            session_id="session_12345",
            exercise_name="Arm Raise",
            duration_seconds=300,
            total_score=85.5,
            rom_score=88.0,
            stability_score=82.0,
            flow_score=86.0,
            grade="XUAT SAC",
            grade_color="green",
            total_reps=10,
            fatigue_level="MILD"
        ).to_dict()
    )
    
    print("\n--- Phase 4 Output ---")
    print(json.dumps(phase4_output.to_dict(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("KEY BAT BUOC trong moi output: 'phase' (int: 1-4)")
    print("=" * 60)


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "single":
            demo_single_user()
        elif mode == "multi":
            demo_multi_user()
        elif mode == "format":
            demo_output_format()
        elif mode == "server":
            app = create_fastapi_app()
            if app:
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print(f"Unknown mode: {mode}")
            print("Available: single, multi, format, server")
    else:
        print("MEMOTION Backend Example")
        print("-" * 40)
        print("Usage:")
        print("  python backend_example.py single  - Demo single user")
        print("  python backend_example.py multi   - Demo multi user")
        print("  python backend_example.py format  - Demo output format")
        print("  python backend_example.py server  - Run FastAPI server")
        print()
        
        # Default: show output format
        demo_output_format()
