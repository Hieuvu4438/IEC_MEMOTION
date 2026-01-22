# PROJECT_CONTEXT.md - MEMOTION

> **Document for Claude Code context across sessions**
> Version: 2.0.0
> Last Updated: 2026-01-22

---

## 1. PROJECT OVERVIEW

### 1.1 Purpose
**MEMOTION** is a **rehabilitation support system for elderly people** using Computer Vision. The system focuses on:

- **Safety**: Never forces users beyond their range of motion limits
- **Personalization**: Adjusts targets based on individual capabilities
- **Pain Monitoring**: Automatically detects when users are in pain (via FACS - Facial Action Coding System)
- **Encouragement**: Provides positive feedback without judgment

### 1.2 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Main language |
| **MediaPipe Tasks API** | Pose detection (33 landmarks), Face detection (478 landmarks) |
| **OpenCV** | Video processing & display |
| **NumPy / SciPy** | Scientific computing, Procrustes analysis |
| **FastDTW** | Dynamic Time Warping for motion rhythm comparison |
| **Pandas** | Data handling |
| **Matplotlib** | Visualization (optional) |

### 1.3 Four Main Phases

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMOTION WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PHASE 1          PHASE 2           PHASE 3          PHASE 4      │
│   ────────         ────────          ────────         ────────     │
│   Pose             Safe-Max          Motion           Scoring &    │
│   Detection        Calibration       Sync             Analysis     │
│                                                                     │
│   ↓ Detect         ↓ Measure         ↓ Sync with      ↓ Multi-     │
│     pose             safe ROM          reference        dimensional │
│     skeleton         limits            video            scoring     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. DIRECTORY STRUCTURE

```
mediapipe/
│
├── 📁 core/                      # CORE COMPONENTS
│   ├── __init__.py               # Module exports
│   ├── data_types.py             # Data classes (Point3D, LandmarkSet, DetectionResult)
│   ├── detector.py               # VisionDetector - MediaPipe wrapper
│   ├── kinematics.py             # Joint angle calculations
│   ├── procrustes.py             # Skeleton normalization (Procrustes Analysis)
│   ├── synchronizer.py           # Motion sync FSM (Finite State Machine)
│   └── dtw_analysis.py           # Dynamic Time Warping analysis
│
├── 📁 modules/                   # BUSINESS LOGIC
│   ├── __init__.py               # Module exports
│   ├── calibration.py            # Safe-Max Calibration for elderly
│   ├── pain_detection.py         # Pain detection via FACS
│   ├── scoring.py                # Multi-dimensional scoring matrix
│   ├── target_generator.py       # Personalized target generation
│   └── video_engine.py           # Smart Video Player
│
├── 📁 utils/                     # UTILITIES
│   ├── __init__.py               # Module exports
│   ├── logger.py                 # Session logging (JSON)
│   └── visualization.py          # Drawing utilities (skeleton, panels, Vietnamese text)
│
├── 📁 models/                    # MEDIAPIPE MODEL FILES
│   ├── pose_landmarker_lite.task
│   ├── pose_landmarker_full.task
│   └── face_landmarker.task
│
├── 📁 data/                      # DATA STORAGE
│   ├── logs/                     # Session logs (JSON)
│   └── user_profiles/            # User calibration profiles
│
├── 📁 mediapipe_be/              # BACKEND-READY VERSION (No UI)
│   ├── service/                  # Backend integration layer
│   │   ├── engine_service.py     # Main class - replaces main_v2.py for backend
│   │   └── schemas.py            # JSON-serializable output schemas
│   ├── core/                     # Same as mediapipe/core/
│   ├── modules/                  # Same as mediapipe/modules/
│   ├── utils/                    # Same as mediapipe/utils/
│   ├── models/                   # Model files
│   ├── bridge_example.py         # Example integration file
│   └── README.md                 # Backend documentation
│
├── 📁 scripts/                   # COMMAND SCRIPTS (txt format)
│   ├── main.txt                  # Run main_final.py
│   ├── main_test.txt             # Run main_test.py
│   └── compare.txt               # Run comparison
│
├── 📁 videos/                    # Reference videos for exercises
├── 📁 test_logs/                 # Test session logs
│
├── 📄 main_v2.py                 # MAIN ENTRY POINT (with UI) - Version 2.0
├── 📄 main_final.py              # Previous main entry point
├── 📄 main_test.py               # Test script for Phase 1
├── 📄 main_sync_test.py          # Test script for motion sync
├── 📄 test_calibration.py        # Test script for calibration
├── 📄 comprehensive_audit.py     # Full system audit
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 .gitignore                 # Git ignore rules
├── 📄 .cursorrules               # Cursor IDE rules
├── 📄 README1.md                 # Basic README
├── 📄 README_STRUCTURE.md        # Detailed structure documentation
├── 📄 main_context.md            # Detailed system documentation
└── 📄 main_v2_docs.md            # main_v2.py documentation
```

---

## 3. MAIN WORKFLOW

### 3.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              MEMOTION APP                                │
│                            (main_v2.py)                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐           │
│  │   INPUT LAYER  │   │  PROCESS LAYER │   │  OUTPUT LAYER  │           │
│  │                │   │                │   │                │           │
│  │  • Webcam      │──▶│  • Detector    │──▶│  • Display     │           │
│  │  • Video file  │   │  • Calibrator  │   │  • Dashboard   │           │
│  │  • Ref video   │   │  • Synchronizer│   │  • Logger      │           │
│  │                │   │  • Scorer      │   │  • Reports     │           │
│  └────────────────┘   │  • PainDetector│   └────────────────┘           │
│                       └────────────────┘                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase Flow

| Phase | Component | Description |
|-------|-----------|-------------|
| **Phase 1** | `VisionDetector` | Detect pose, draw skeleton, wait for stable detection (30 frames) |
| **Phase 2** | `SafeMaxCalibrator` | Measure safe ROM for 6 joints (shoulders, elbows, knees) |
| **Phase 3** | `MotionSyncController` + `VideoEngine` | Sync user movement with reference video |
| **Phase 4** | `HealthScorer` | Multi-dimensional scoring (ROM, stability, flow) + final report |

### 3.3 Data Flow (Backend)

```
Camera → WebSocket/API → EngineService → process_frame() → EngineOutput.to_dict() → JSON → Frontend
```

---

## 4. KEY COMPONENTS

### 4.1 Core Module (`core/`)

| File | Key Classes/Functions | Purpose |
|------|----------------------|---------|
| `data_types.py` | `Point3D`, `LandmarkSet`, `DetectionResult`, `PoseLandmarkIndex` | Data structures for landmarks |
| `detector.py` | `VisionDetector`, `DetectorConfig` | MediaPipe wrapper for pose/face detection |
| `kinematics.py` | `JointType`, `calculate_joint_angle()`, `JOINT_DEFINITIONS` | Joint angle calculations |
| `procrustes.py` | `normalize_skeleton()`, `compute_procrustes_distance()` | Skeleton normalization |
| `synchronizer.py` | `MotionSyncController`, `MotionPhase`, `SyncState` | FSM for motion phases (idle, eccentric, hold, concentric) |
| `dtw_analysis.py` | `compute_weighted_dtw()`, `compute_single_joint_dtw()` | Rhythm comparison using DTW |

### 4.2 Modules (`modules/`)

| File | Key Classes | Purpose |
|------|-------------|---------|
| `calibration.py` | `SafeMaxCalibrator`, `CalibrationState`, `UserProfile` | Safe-Max calibration for elderly |
| `pain_detection.py` | `PainDetector`, `PainLevel`, `PainEvent` | Pain detection via facial expressions (FACS) |
| `scoring.py` | `HealthScorer`, `FatigueLevel`, `RepScore`, `SessionReport` | Multi-dimensional scoring |
| `target_generator.py` | `rescale_reference_motion()`, `compute_target_at_time()` | Personalized target generation |
| `video_engine.py` | `VideoEngine`, `SyncedVideoPlayer`, `PlaybackState` | Smart video playback |

### 4.3 Utils (`utils/`)

| File | Functions | Purpose |
|------|-----------|---------|
| `logger.py` | `SessionLogger` | Session logging to JSON |
| `visualization.py` | `draw_skeleton()`, `draw_panel()`, `put_vietnamese_text()`, `COLORS` | Drawing utilities |

---

## 5. INSTALLATION & RUNNING

### 5.1 Installation

```bash
# Create virtual environment
python -m venv med_venv

# Activate (Windows)
med_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Download MediaPipe Models

Download and place in `./models/` directory:

| Model | URL |
|-------|-----|
| Pose Landmarker Lite | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task |
| Pose Landmarker Full | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task |
| Face Landmarker | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |

### 5.3 Run Commands

```bash
# Main application with UI (webcam)
python main_v2.py --source webcam

# Main application with reference video
python main_v2.py --source webcam --ref-video videos/exercise.mp4

# Test mode (unit tests)
python main_v2.py --mode test

# Phase 1 test (video processing)
python main_test.py --source video.mp4

# Backend example (no UI)
cd mediapipe_be
python bridge_example.py
```

### 5.4 Keyboard Controls (main_v2.py)

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume or Start calibration |
| `R` | Restart |
| `Q` | Quit |
| `1-6` | Select joint for calibration (Phase 2) |
| `ENTER` | Confirm / Next phase |
| `ESC` | Exit |

---

## 6. IMPORTANT NOTES

### 6.1 Two Versions

| Version | Location | Purpose | Has UI |
|---------|----------|---------|--------|
| **Demo/Test** | `mediapipe/` | Local testing with OpenCV window | Yes |
| **Backend** | `mediapipe_be/` | Production integration (FastAPI, WebSocket) | No |

### 6.2 Calibration Queue (6 Joints)

```python
CALIBRATION_QUEUE = [
    JointType.LEFT_SHOULDER,   # 1
    JointType.RIGHT_SHOULDER,  # 2
    JointType.LEFT_ELBOW,      # 3
    JointType.RIGHT_ELBOW,     # 4
    JointType.LEFT_KNEE,       # 5
    JointType.RIGHT_KNEE,      # 6
]
```

### 6.3 Motion Phases

| Phase | Vietnamese | Description |
|-------|------------|-------------|
| `idle` | Nghi | Rest position |
| `eccentric` | Duoi co | Stretching/lowering |
| `hold` | Giu | Hold position |
| `concentric` | Co co | Contracting/raising |

### 6.4 Scoring Dimensions

- **ROM Score**: Range of motion accuracy
- **Stability Score**: Movement smoothness (jerk-based)
- **Flow Score**: Rhythm matching with reference (DTW)

---

## 7. FOR FUTURE SESSIONS

### Key Entry Points to Read First

1. `main_v2.py` - Main application with UI
2. `mediapipe_be/service/engine_service.py` - Backend service class
3. `core/__init__.py` - Core module exports
4. `modules/__init__.py` - Business logic exports

### When Working on Specific Features

| Feature | Files to Read |
|---------|---------------|
| Pose Detection | `core/detector.py`, `core/data_types.py` |
| Joint Angles | `core/kinematics.py` |
| Calibration | `modules/calibration.py` |
| Pain Detection | `modules/pain_detection.py` |
| Scoring | `modules/scoring.py` |
| Motion Sync | `core/synchronizer.py`, `core/dtw_analysis.py` |
| Backend Integration | `mediapipe_be/service/engine_service.py`, `mediapipe_be/service/schemas.py` |

---

*This document is intended as context for Claude Code sessions. Update as the project evolves.*
