"""
MEMOTION Backend Service Layer

Cung cap:
- MemotionEngine: Class chinh xu ly frame (stateful, multi-instance)
- EngineConfig: Cau hinh cho engine
- Schemas: Cac class JSON-serializable cho output

Usage:
    from service import MemotionEngine, EngineConfig, create_engine_for_user
    
    # Tao engine cho user
    engine = MemotionEngine.create_instance(
        config=EngineConfig(ref_video_path="./videos/exercise.mp4")
    )
    
    # Xu ly frame
    result = engine.process_frame(frame, timestamp_ms)
    json_output = result.to_dict()
    # json_output["phase"] = 1, 2, 3, or 4
"""

from .schemas import (
    # Enums
    PhaseStatus,
    MotionPhaseType,
    DirectionHint,
    # Phase 1
    DetectionOutput,
    # Phase 2
    JointCalibrationStatus,
    CalibrationOutput,
    # Phase 3
    JointError,
    SyncOutput,
    # Phase 4
    RepScore,
    JointCalibrationResult,
    FinalReportOutput,
    # Composite
    EngineOutput,
    # Helpers
    get_direction_hint,
    get_feedback_text,
    get_grade,
    get_joint_name_vi,
    JOINT_NAMES_VI,
)

from .engine_service import (
    # Main class (new name)
    MemotionEngine,
    # Backward compatible alias
    EngineService,
    # Config and State
    EngineConfig,
    EngineState,
    AppPhase,
    # Constants
    CALIBRATION_QUEUE,
    JOINT_POSITION_INSTRUCTIONS,
    PHASE_NAMES,
    # Factory function
    create_engine_for_user,
)

__all__ = [
    # ===== MAIN ENGINE (Recommended) =====
    'MemotionEngine',
    'EngineConfig',
    'create_engine_for_user',
    
    # ===== BACKWARD COMPATIBLE =====
    'EngineService',  # Alias cua MemotionEngine
    'EngineState',
    'AppPhase',
    
    # ===== CONSTANTS =====
    'CALIBRATION_QUEUE',
    'JOINT_POSITION_INSTRUCTIONS',
    'PHASE_NAMES',
    
    # ===== SCHEMAS =====
    # Enums
    'PhaseStatus',
    'MotionPhaseType',
    'DirectionHint',
    # Phase 1
    'DetectionOutput',
    # Phase 2
    'JointCalibrationStatus',
    'CalibrationOutput',
    # Phase 3
    'JointError',
    'SyncOutput',
    # Phase 4
    'RepScore',
    'JointCalibrationResult',
    'FinalReportOutput',
    # Composite Output
    'EngineOutput',
    
    # ===== HELPERS =====
    'get_direction_hint',
    'get_feedback_text',
    'get_grade',
    'get_joint_name_vi',
    'JOINT_NAMES_VI',
]
