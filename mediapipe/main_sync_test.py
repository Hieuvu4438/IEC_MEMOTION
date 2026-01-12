#!/usr/bin/env python3
"""
Motion Synchronization Test Script - MEMOTION Phase 3.

Script tích hợp:
1. Mở webcam/video của người dùng
2. Mở video mẫu
3. Đồng bộ 2 video với cơ chế Wait-for-User
4. Tính DTW score

Usage:
    python main_sync_test.py --user-source webcam --ref-video exercise.mp4
    python main_sync_test.py --mode test

Author: MEMOTION Team
Version: 1.0.0
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

from core import (
    VisionDetector,
    DetectorConfig,
    JointType,
    JOINT_DEFINITIONS,
    calculate_joint_angle,
    MotionPhase,
    SyncStatus,
    MotionSyncController,
    create_arm_raise_exercise,
    create_elbow_flex_exercise,
    compute_single_joint_dtw,
    get_rhythm_feedback,
    create_exercise_weights,
)
from modules import VideoEngine, PlaybackState


def check_display_available() -> bool:
    if sys.platform in ('win32', 'darwin'):
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


PHASE_COLORS = {
    MotionPhase.IDLE: (128, 128, 128),
    MotionPhase.ECCENTRIC: (0, 255, 255),
    MotionPhase.HOLD: (0, 255, 0),
    MotionPhase.CONCENTRIC: (255, 255, 0),
}

STATUS_COLORS = {
    SyncStatus.PLAY: (0, 255, 0),
    SyncStatus.PAUSE: (0, 165, 255),
    SyncStatus.LOOP: (255, 255, 0),
    SyncStatus.SKIP: (0, 0, 255),
    SyncStatus.COMPLETE: (0, 255, 0),
}


def draw_sync_info(frame, sync_state, user_angle, joint_name):
    """Vẽ thông tin đồng bộ lên frame user."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    cv2.putText(output, "USER VIEW", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Angle ({joint_name}): {user_angle:.1f} deg", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    target_str = f"Target: {sync_state.target_angle:.1f}" if sync_state.target_angle > 0 else "Target: --"
    cv2.putText(output, target_str, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    phase_color = PHASE_COLORS.get(sync_state.current_phase, (255, 255, 255))
    cv2.putText(output, f"Phase: {sync_state.current_phase.value.upper()}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
    
    status_color = STATUS_COLORS.get(sync_state.sync_status, (255, 255, 255))
    cv2.putText(output, f"Sync: {sync_state.sync_status.value.upper()}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    cv2.putText(output, f"Reps: {sync_state.rep_count}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if sync_state.status_message:
        cv2.putText(output, sync_state.status_message, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return output


def draw_video_info(frame, current_frame, total_frames, sync_status, phase):
    """Vẽ thông tin lên frame video mẫu."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    cv2.putText(output, "REFERENCE VIDEO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    progress = current_frame / total_frames if total_frames > 0 else 0
    cv2.putText(output, f"Frame: {current_frame}/{total_frames} ({progress:.0%})", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    phase_color = PHASE_COLORS.get(phase, (255, 255, 255))
    cv2.putText(output, f"Phase: {phase.value.upper()}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
    
    bar_y, bar_width = h - 30, w - 40
    cv2.rectangle(output, (20, bar_y), (20 + bar_width, bar_y + 15), (100, 100, 100), -1)
    cv2.rectangle(output, (20, bar_y), (20 + int(bar_width * progress), bar_y + 15), (0, 255, 0), -1)
    
    if sync_status == SyncStatus.PAUSE:
        cv2.putText(output, "|| WAITING", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return output


def draw_pose_overlay(frame, landmarks, joint_type, angle):
    """Vẽ pose skeleton với highlight khớp chính."""
    if landmarks is None:
        return frame
    
    output = frame.copy()
    h, w = frame.shape[:2]
    joint_def = JOINT_DEFINITIONS.get(joint_type)
    if joint_def is None:
        return output
    
    try:
        lm_array = landmarks.to_numpy()
        connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
        
        for start_idx, end_idx in connections:
            if start_idx < len(lm_array) and end_idx < len(lm_array):
                p1, p2 = lm_array[start_idx], lm_array[end_idx]
                pt1, pt2 = (int(p1[0] * w), int(p1[1] * h)), (int(p2[0] * w), int(p2[1] * h))
                cv2.line(output, pt1, pt2, (100, 100, 100), 2)
        
        p1, p2, p3 = lm_array[joint_def.proximal], lm_array[joint_def.vertex], lm_array[joint_def.distal]
        pt1, pt2, pt3 = (int(p1[0] * w), int(p1[1] * h)), (int(p2[0] * w), int(p2[1] * h)), (int(p3[0] * w), int(p3[1] * h))
        
        cv2.line(output, pt1, pt2, (0, 255, 0), 3)
        cv2.line(output, pt2, pt3, (0, 255, 0), 3)
        cv2.circle(output, pt2, 10, (0, 0, 255), -1)
        cv2.putText(output, f"{angle:.0f}", (pt2[0] + 15, pt2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except (IndexError, ValueError):
        pass
    
    return output


def run_sync_session(detector, ref_video_path, user_source="webcam", joint_type=JointType.LEFT_SHOULDER, display=True):
    """Chạy session đồng bộ đầy đủ."""
    try:
        ref_engine = VideoEngine(ref_video_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] Cannot open reference video: {e}")
        return {}
    
    total_frames = ref_engine.total_frames
    exercise = create_arm_raise_exercise(total_frames, ref_engine.fps, max_angle=150)
    if joint_type in (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW):
        exercise = create_elbow_flex_exercise(total_frames, ref_engine.fps)
    
    checkpoint_frames = [cp.frame_index for cp in exercise.checkpoints]
    checkpoint_messages = {cp.frame_index: cp.message for cp in exercise.checkpoints}
    ref_engine.set_checkpoints(checkpoint_frames, checkpoint_messages)
    
    user_cap = cv2.VideoCapture(0 if user_source.lower() == "webcam" else user_source)
    if not user_cap.isOpened():
        print(f"[ERROR] Cannot open user source: {user_source}")
        ref_engine.release()
        return {}
    
    controller = MotionSyncController(exercise)
    joint_name = JOINT_DEFINITIONS[joint_type].name
    
    print(f"\n{'='*60}")
    print(f"MOTION SYNC SESSION: {exercise.name}")
    print(f"Joint: {joint_name} | Frames: {total_frames}")
    print(f"{'='*60}")
    if display:
        print("Controls: [SPACE] Pause | [R] Restart | [Q] Quit\n")
    
    ref_engine.play()
    ref_engine.set_speed(0.7)
    
    user_angles, ref_angles = [], []
    results = {"exercise": exercise.name, "total_reps": 0, "dtw_score": 0, "rhythm_feedback": ""}
    paused = False
    
    while True:
        ret_user, user_frame = user_cap.read()
        if not ret_user:
            if user_source.lower() != "webcam":
                break
            continue
        
        if user_source.lower() == "webcam":
            user_frame = cv2.flip(user_frame, 1)
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.process_frame(user_frame, timestamp_ms)
        
        user_angle = 0.0
        if result.has_pose():
            try:
                user_angle = calculate_joint_angle(result.pose_landmarks.to_numpy(), joint_type, use_3d=True)
            except ValueError:
                pass
        
        sync_state = controller.update(user_angle, ref_engine.current_frame, time.time())
        
        if not paused:
            if sync_state.sync_status == SyncStatus.PAUSE:
                ref_engine.pause()
            elif sync_state.sync_status in (SyncStatus.PLAY, SyncStatus.SKIP):
                if ref_engine.state != PlaybackState.PLAYING:
                    ref_engine.play()
        
        ref_frame, ref_status = ref_engine.get_frame()
        
        if result.has_pose():
            user_angles.append(user_angle)
            ref_angles.append(sync_state.target_angle if sync_state.target_angle > 0 else user_angle)
        
        if sync_state.sync_status == SyncStatus.COMPLETE or ref_status.state == PlaybackState.FINISHED:
            results["total_reps"] = sync_state.rep_count
            break
        
        if display and ref_frame is not None:
            user_display = draw_pose_overlay(user_frame, result.pose_landmarks if result.has_pose() else None, joint_type, user_angle)
            user_display = draw_sync_info(user_display, sync_state, user_angle, joint_name)
            ref_display = draw_video_info(ref_frame, ref_engine.current_frame, ref_engine.total_frames, sync_state.sync_status, sync_state.current_phase)
            
            target_height = 480
            user_resized = cv2.resize(user_display, (int(user_display.shape[1] * target_height / user_display.shape[0]), target_height))
            ref_resized = cv2.resize(ref_display, (int(ref_display.shape[1] * target_height / ref_display.shape[0]), target_height))
            combined = np.hstack([user_resized, ref_resized])
            
            cv2.imshow("MEMOTION - Motion Sync", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                ref_engine.pause() if paused else ref_engine.play()
            elif key == ord('r'):
                ref_engine.stop()
                ref_engine.play()
                controller.reset()
                user_angles, ref_angles = [], []
    
    if len(user_angles) > 10:
        dtw_result = compute_single_joint_dtw(user_angles, ref_angles)
        results["dtw_score"] = dtw_result.similarity_score
        results["rhythm_feedback"] = get_rhythm_feedback(dtw_result)
    
    user_cap.release()
    ref_engine.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"SESSION COMPLETE | Reps: {results['total_reps']} | Score: {results.get('dtw_score', 0):.1f}%")
    print(f"Feedback: {results.get('rhythm_feedback', 'N/A')}")
    print(f"{'='*60}\n")
    
    return results


def run_unit_tests():
    """Chạy unit tests cho module đồng bộ."""
    print("\n" + "="*60)
    print("UNIT TESTS - Motion Synchronization")
    print("="*60)
    
    print("\n[TEST 1] MotionSyncController FSM...")
    exercise = create_arm_raise_exercise(total_frames=100, fps=30)
    controller = MotionSyncController(exercise)
    assert controller.state.current_phase == MotionPhase.IDLE
    for i in range(30):
        controller.update(user_angle=30 + i, ref_frame=i, timestamp=i/30)
    print(f"  ✓ FSM working, phase={controller.state.current_phase.value}")
    
    print("\n[TEST 2] DTW identical sequences...")
    seq1 = [0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0]
    result = compute_single_joint_dtw(seq1, seq1)
    assert result.similarity_score > 95
    print(f"  ✓ Identical: {result.similarity_score:.1f}%")
    
    print("\n[TEST 3] DTW different sequences...")
    seq2 = [0, 5, 10, 15, 20, 25, 20, 15, 10, 5, 0]
    result = compute_single_joint_dtw(seq1, seq2)
    print(f"  ✓ Different: {result.similarity_score:.1f}%")
    
    print("\n[TEST 4] Exercise weights...")
    weights = create_exercise_weights("arm_raise")
    assert weights[JointType.LEFT_SHOULDER] > weights[JointType.LEFT_KNEE]
    print(f"  ✓ Shoulder={weights[JointType.LEFT_SHOULDER]}, Knee={weights[JointType.LEFT_KNEE]}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MEMOTION Motion Sync Test")
    parser.add_argument("--user-source", type=str, default="webcam")
    parser.add_argument("--ref-video", type=str, default=None)
    parser.add_argument("--joint", type=str, default="left_shoulder")
    parser.add_argument("--mode", type=str, choices=["sync", "test"], default="sync")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--models-dir", type=str, default="./models")
    args = parser.parse_args()
    
    if args.mode == "test":
        run_unit_tests()
        return
    
    if args.ref_video is None or not Path(args.ref_video).exists():
        print("[INFO] No valid --ref-video provided. Running unit tests...")
        run_unit_tests()
        return
    
    use_display = args.display or (not args.headless and check_display_available())
    
    joint_type = None
    for jt in JointType:
        if jt.value == args.joint.lower().replace("-", "_"):
            joint_type = jt
            break
    if joint_type is None:
        print(f"[ERROR] Unknown joint: {args.joint}")
        return
    
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pose_model = models_dir / "pose_landmarker_lite.task"
    
    if not pose_model.exists():
        print(f"[WARNING] Model not found: {pose_model}")
        run_unit_tests()
        return
    
    config = DetectorConfig(pose_model_path=str(pose_model), running_mode="VIDEO")
    
    try:
        with VisionDetector(config) as detector:
            run_sync_session(detector, args.ref_video, args.user_source, joint_type, use_display)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()