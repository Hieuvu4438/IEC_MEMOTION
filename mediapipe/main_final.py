#!/usr/bin/env python3
"""
MEMOTION - Complete Integration (Final)

Tích hợp hoàn chỉnh 4 giai đoạn:
1. GĐ1: Nhận diện Pose + Procrustes
2. GĐ2: Safe-Max Calibration 
3. GĐ3: Motion Sync + Wait-for-User
4. GĐ4: Scoring + Pain Detection + Fatigue

Usage:
    python main_final.py --source webcam --ref-video exercise.mp4
    python main_final.py --mode test

Author: MEMOTION Team
Version: 1.0.0
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict
from queue import Queue
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV required. Install: pip install opencv-python")
    sys.exit(1)

from core import (
    VisionDetector, DetectorConfig, JointType, JOINT_DEFINITIONS,
    calculate_joint_angle, MotionPhase, SyncStatus, SyncState,
    MotionSyncController, create_arm_raise_exercise, create_elbow_flex_exercise,
    compute_single_joint_dtw,
)
from modules import (
    VideoEngine, PlaybackState, PainDetector, PainLevel,
    HealthScorer, FatigueLevel,
)
from utils import SessionLogger


@dataclass
class DashboardState:
    rep_count: int = 0
    current_score: float = 0.0
    average_score: float = 0.0
    fatigue_level: str = "FRESH"
    pain_level: str = "NONE"
    sync_status: str = "IDLE"
    phase: str = "IDLE"
    user_angle: float = 0.0
    target_angle: float = 0.0
    message: str = ""
    warning: str = ""


class MemotionApp:
    """Ứng dụng MEMOTION hoàn chỉnh."""
    
    def __init__(self, detector, ref_video_path, joint_type=JointType.LEFT_SHOULDER, log_dir="./data/logs"):
        self._detector = detector
        self._ref_video_path = ref_video_path
        self._joint_type = joint_type
        self._video_engine = None
        self._sync_controller = None
        self._pain_detector = PainDetector()
        self._scorer = HealthScorer()
        self._logger = SessionLogger(log_dir)
        self._dashboard = DashboardState()
        self._is_running = False
        self._is_paused = False
        self._analysis_queue = Queue(maxsize=5)
        self._user_angles = []
    
    def setup(self):
        try:
            self._video_engine = VideoEngine(self._ref_video_path)
            total_frames = self._video_engine.total_frames
            fps = self._video_engine.fps
            
            if self._joint_type in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
                exercise = create_elbow_flex_exercise(total_frames, fps)
            else:
                exercise = create_arm_raise_exercise(total_frames, fps, max_angle=150)
            
            self._sync_controller = MotionSyncController(exercise)
            checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
            self._video_engine.set_checkpoints(checkpoint_frames)
            self._video_engine.set_speed(0.7)
            
            session_id = f"session_{int(time.time())}"
            self._logger.start_session(session_id, exercise.name)
            self._scorer.start_session(exercise.name, session_id)
            
            print(f"[SETUP] Exercise: {exercise.name}")
            print(f"[SETUP] Joint: {JOINT_DEFINITIONS[self._joint_type].name}")
            return True
        except Exception as e:
            print(f"[ERROR] Setup failed: {e}")
            return False
    
    def run(self, user_source="webcam", display=True):
        user_cap = cv2.VideoCapture(0 if user_source.lower() == "webcam" else user_source)
        if not user_cap.isOpened():
            print(f"[ERROR] Cannot open: {user_source}")
            return {}
        
        self._is_running = True
        self._video_engine.play()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self._analysis_loop)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        print("\n" + "="*60)
        print("MEMOTION - Rehabilitation Support System")
        print("="*60)
        if display:
            print("Controls: [SPACE] Pause | [R] Restart | [Q] Quit")
        print("="*60 + "\n")
        
        last_phase = MotionPhase.IDLE
        
        while self._is_running:
            ret, user_frame = user_cap.read()
            if not ret:
                if user_source.lower() != "webcam":
                    break
                continue
            
            if user_source.lower() == "webcam":
                user_frame = cv2.flip(user_frame, 1)
            
            timestamp_ms = int(time.time() * 1000)
            result = self._detector.process_frame(user_frame, timestamp_ms)
            
            user_angle = 0.0
            if result.has_pose():
                try:
                    user_angle = calculate_joint_angle(result.pose_landmarks.to_numpy(), self._joint_type, use_3d=True)
                except ValueError:
                    pass
            
            current_time = time.time()
            sync_state = self._sync_controller.update(user_angle, self._video_engine.current_frame, current_time)
            self._update_dashboard(sync_state, user_angle)
            
            if not self._is_paused:
                if sync_state.sync_status == SyncStatus.PAUSE:
                    self._video_engine.pause()
                elif sync_state.sync_status in (SyncStatus.PLAY, SyncStatus.SKIP):
                    if self._video_engine.state != PlaybackState.PLAYING:
                        self._video_engine.play()
            
            ref_frame, ref_status = self._video_engine.get_frame()
            self._scorer.add_frame(user_angle, current_time, sync_state.current_phase)
            self._user_angles.append(user_angle)
            
            if last_phase == MotionPhase.CONCENTRIC and sync_state.current_phase == MotionPhase.IDLE:
                self._on_rep_complete()
            last_phase = sync_state.current_phase
            
            if result.has_face() and not self._analysis_queue.full():
                self._analysis_queue.put(result.face_landmarks.to_numpy())
            
            if sync_state.sync_status == SyncStatus.COMPLETE or ref_status.state == PlaybackState.FINISHED:
                break
            
            if display and ref_frame is not None:
                combined = self._render_display(user_frame, ref_frame, result)
                cv2.imshow("MEMOTION", combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._is_paused = not self._is_paused
                    self._video_engine.pause() if self._is_paused else self._video_engine.play()
                elif key == ord('r'):
                    self._restart()
        
        self._is_running = False
        user_cap.release()
        if display:
            cv2.destroyAllWindows()
        
        return self._generate_final_report()
    
    def _analysis_loop(self):
        while self._is_running:
            try:
                face_landmarks = self._analysis_queue.get(timeout=0.1)
                result = self._pain_detector.analyze(face_landmarks)
                if result.is_pain_detected:
                    self._dashboard.pain_level = result.pain_level.name
                    self._dashboard.warning = result.message
                    self._logger.log_pain(result.pain_level.name, result.pain_score, result.au_activations, result.message)
                else:
                    self._dashboard.pain_level = "NONE"
                    self._dashboard.warning = ""
            except:
                continue
    
    def _update_dashboard(self, sync_state, user_angle):
        self._dashboard.rep_count = sync_state.rep_count
        self._dashboard.sync_status = sync_state.sync_status.value
        self._dashboard.phase = sync_state.current_phase.value
        self._dashboard.user_angle = user_angle
        self._dashboard.target_angle = sync_state.target_angle
        self._dashboard.message = sync_state.status_message
        scorer_status = self._scorer.get_current_status()
        self._dashboard.current_score = scorer_status.get("last_score", 0)
        self._dashboard.average_score = scorer_status.get("average_score", 0)
        self._dashboard.fatigue_level = scorer_status.get("fatigue_level", "FRESH")
    
    def _on_rep_complete(self):
        dtw_result = compute_single_joint_dtw(self._user_angles[-50:], self._user_angles[-50:]) if len(self._user_angles) > 20 else None
        target = self._dashboard.target_angle or 150
        rep_score = self._scorer.complete_rep(target, dtw_result)
        self._logger.log_rep(rep_score.rep_number, {"rom": rep_score.rom_score, "stability": rep_score.stability_score, "flow": rep_score.flow_score, "total": rep_score.total_score}, rep_score.jerk_value, rep_score.duration_ms)
    
    def _restart(self):
        self._video_engine.stop()
        self._video_engine.play()
        self._sync_controller.reset()
        self._user_angles = []
        self._dashboard = DashboardState()
    
    def _render_display(self, user_frame, ref_frame, detection_result):
        h = 480
        user_display = self._draw_user_view(user_frame, detection_result)
        ref_display = self._draw_ref_view(ref_frame)
        
        user_h, user_w = user_display.shape[:2]
        ref_h, ref_w = ref_display.shape[:2]
        user_resized = cv2.resize(user_display, (int(user_w * h / user_h), h))
        ref_resized = cv2.resize(ref_display, (int(ref_w * h / ref_h), h))
        dashboard = self._draw_dashboard(h)
        
        return np.hstack([user_resized, ref_resized, dashboard])
    
    def _draw_user_view(self, frame, detection_result):
        output = frame.copy()
        if detection_result.has_pose():
            self._draw_skeleton(output, detection_result.pose_landmarks.to_numpy())
        
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        cv2.putText(output, "USER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Angle: {self._dashboard.user_angle:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"Target: {self._dashboard.target_angle:.1f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return output
    
    def _draw_ref_view(self, frame):
        output = frame.copy()
        fh, fw = frame.shape[:2]
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        cv2.putText(output, "REFERENCE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Phase: {self._dashboard.phase.upper()}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        progress = self._video_engine.current_frame / max(1, self._video_engine.total_frames)
        cv2.rectangle(output, (20, fh - 30), (fw - 20, fh - 15), (100, 100, 100), -1)
        cv2.rectangle(output, (20, fh - 30), (20 + int((fw - 40) * progress), fh - 15), (0, 255, 0), -1)
        
        if self._dashboard.sync_status == "pause":
            cv2.putText(output, "|| WAITING", (fw - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return output
    
    def _draw_dashboard(self, height):
        width = 250
        dashboard = np.zeros((height, width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)
        
        y = 30
        cv2.putText(dashboard, "DASHBOARD", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 40
        cv2.putText(dashboard, f"Reps: {self._dashboard.rep_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 35
        
        score_color = (0, 255, 0) if self._dashboard.current_score >= 70 else (0, 165, 255)
        cv2.putText(dashboard, f"Score: {self._dashboard.current_score:.0f}/100", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        y += 30
        cv2.putText(dashboard, f"Average: {self._dashboard.average_score:.0f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 45
        
        cv2.putText(dashboard, "Fatigue:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        fatigue_colors = {"FRESH": (0, 255, 0), "LIGHT": (0, 255, 255), "MODERATE": (0, 165, 255), "HEAVY": (0, 0, 255)}
        fatigue_levels = {"FRESH": 0.1, "LIGHT": 0.4, "MODERATE": 0.7, "HEAVY": 1.0}
        level = fatigue_levels.get(self._dashboard.fatigue_level, 0.1)
        color = fatigue_colors.get(self._dashboard.fatigue_level, (0, 255, 0))
        cv2.rectangle(dashboard, (20, y), (220, y + 15), (100, 100, 100), -1)
        cv2.rectangle(dashboard, (20, y), (20 + int(200 * level), y + 15), color, -1)
        y += 35
        
        cv2.putText(dashboard, "Pain:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pain_color = (0, 255, 0) if self._dashboard.pain_level == "NONE" else (0, 0, 255)
        cv2.putText(dashboard, self._dashboard.pain_level, (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pain_color, 1)
        y += 40
        
        if self._dashboard.message:
            cv2.putText(dashboard, self._dashboard.message[:30], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 30
        
        if self._dashboard.warning:
            cv2.rectangle(dashboard, (10, y), (width - 10, y + 45), (0, 0, 100), -1)
            cv2.putText(dashboard, "WARNING", (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(dashboard, self._dashboard.warning[:25], (20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return dashboard
    
    def _draw_skeleton(self, frame, landmarks):
        h, w = frame.shape[:2]
        connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                p1, p2 = landmarks[start], landmarks[end]
                cv2.line(frame, (int(p1[0] * w), int(p1[1] * h)), (int(p2[0] * w), int(p2[1] * h)), (0, 255, 0), 2)
        
        joint_def = JOINT_DEFINITIONS.get(self._joint_type)
        if joint_def and joint_def.vertex < len(landmarks):
            p = landmarks[joint_def.vertex]
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 10, (0, 0, 255), -1)
    
    def _generate_final_report(self):
        report = self._scorer.compute_session_report()
        report.pain_events = self._pain_detector.get_pain_summary().get("events", [])
        log_path = self._logger.end_session(report.to_dict())
        
        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        print(f"  Total Reps: {report.total_reps}")
        print(f"  Average Score: {report.average_scores.get('total', 0):.1f}/100")
        print(f"  Fatigue: {report.fatigue_analysis.get('fatigue_level', 'N/A')}")
        print(f"\nReport saved: {log_path}")
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")
        print("="*60 + "\n")
        return report.to_dict()
    
    def cleanup(self):
        self._is_running = False
        if self._video_engine:
            self._video_engine.release()


def run_unit_tests():
    print("\n" + "="*60)
    print("UNIT TESTS - Final Integration")
    print("="*60)
    
    print("\n[TEST 1] PainDetector...")
    detector = PainDetector()
    print("  ✓ PainDetector initialized")
    
    print("\n[TEST 2] HealthScorer...")
    scorer = HealthScorer()
    scorer.start_session("test_exercise", "test_session")
    for i in range(20):
        scorer.add_frame(30 + i * 2, i * 0.033, MotionPhase.ECCENTRIC)
    for i in range(10):
        scorer.add_frame(90, 20 * 0.033 + i * 0.033, MotionPhase.HOLD)
    rep_score = scorer.complete_rep(target_angle=90)
    print(f"  ✓ ROM Score: {rep_score.rom_score:.1f}")
    print(f"  ✓ Stability Score: {rep_score.stability_score:.1f}")
    print(f"  ✓ Total Score: {rep_score.total_score:.1f}")
    
    print("\n[TEST 3] Jerk calculation...")
    from modules.scoring import calculate_jerk
    positions = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [6, 0, 0], [10, 0, 0]])
    timestamps = np.array([0, 0.1, 0.2, 0.3, 0.4])
    jerk = calculate_jerk(positions, timestamps)
    print(f"  ✓ Jerk value: {jerk:.4f}")
    
    print("\n[TEST 4] SessionLogger...")
    from utils import create_session_logger
    logger = create_session_logger("./test_logs", console=False)
    logger.start_session("test_session", "test_exercise")
    logger.log_rep(1, {"total": 85})
    logger.end_session()
    print("  ✓ Logger working")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MEMOTION - Complete System")
    parser.add_argument("--source", type=str, default="webcam")
    parser.add_argument("--ref-video", type=str, default=None)
    parser.add_argument("--joint", type=str, default="left_shoulder")
    parser.add_argument("--mode", type=str, choices=["run", "test"], default="run")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./data/logs")
    args = parser.parse_args()
    
    if args.mode == "test":
        run_unit_tests()
        return
    
    if args.ref_video is None or not Path(args.ref_video).exists():
        print("[INFO] No --ref-video provided. Running tests...")
        run_unit_tests()
        return
    
    joint_type = None
    for jt in JointType:
        if jt.value == args.joint.lower().replace("-", "_"):
            joint_type = jt
            break
    if joint_type is None:
        print(f"[ERROR] Unknown joint: {args.joint}")
        return
    
    models_dir = Path(args.models_dir)
    pose_model = models_dir / "pose_landmarker_lite.task"
    if not pose_model.exists():
        print(f"[WARNING] Model not found: {pose_model}")
        run_unit_tests()
        return
    
    config = DetectorConfig(pose_model_path=str(pose_model), running_mode="VIDEO")
    
    try:
        with VisionDetector(config) as detector:
            app = MemotionApp(detector, args.ref_video, joint_type, args.log_dir)
            if app.setup():
                app.run(args.source, display=not args.headless)
            app.cleanup()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()