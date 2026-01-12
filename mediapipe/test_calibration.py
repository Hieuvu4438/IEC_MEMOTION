#!/usr/bin/env python3
"""
Test Script for Calibration System - MEMOTION Phase 2.

Script này cho phép:
1. Calibrate góc khớp tối đa của người dùng
2. So sánh góc mẫu vs góc mục tiêu cá nhân hóa
3. Lưu và tải user profile

Usage:
    python test_calibration.py --source webcam --joint left_elbow
    python test_calibration.py --source video.mp4 --joint left_shoulder --headless
    python test_calibration.py --mode test

Author: MEMOTION Team
Version: 1.0.0
"""

import argparse
import sys
import time
import os
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
    calculate_angle,
)
from modules import (
    SafeMaxCalibrator,
    CalibrationState,
    UserProfile,
    rescale_reference_motion,
    print_comparison_report,
)


def check_display_available() -> bool:
    """Kiểm tra có display không."""
    if sys.platform in ('win32', 'darwin'):
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def get_joint_type_from_string(joint_name: str) -> Optional[JointType]:
    """Chuyển đổi tên khớp từ string sang JointType enum."""
    name_lower = joint_name.lower().replace("-", "_")
    for jt in JointType:
        if jt.value == name_lower or name_lower in jt.value:
            return jt
    return None


def draw_angle_arc(frame, pt1, pt2, pt3, radius=40):
    """Vẽ cung thể hiện góc."""
    angle1 = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    angle2 = np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0])
    start_angle = np.degrees(min(angle1, angle2))
    end_angle = np.degrees(max(angle1, angle2))
    cv2.ellipse(frame, pt2, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)


def draw_pose_landmarks(frame, landmarks, joint_type, current_angle):
    """Vẽ pose và highlight khớp đang calibrate."""
    output = frame.copy()
    h, w = frame.shape[:2]
    
    if landmarks is None:
        return output
    
    joint_def = JOINT_DEFINITIONS.get(joint_type)
    if joint_def is None:
        return output
    
    try:
        lm_array = landmarks.to_numpy()
        p1, p2, p3 = lm_array[joint_def.proximal], lm_array[joint_def.vertex], lm_array[joint_def.distal]
        
        pt1 = (int(p1[0] * w), int(p1[1] * h))
        pt2 = (int(p2[0] * w), int(p2[1] * h))
        pt3 = (int(p3[0] * w), int(p3[1] * h))
        
        cv2.line(output, pt1, pt2, (0, 255, 0), 3)
        cv2.line(output, pt2, pt3, (0, 255, 0), 3)
        cv2.circle(output, pt1, 8, (255, 0, 0), -1)
        cv2.circle(output, pt2, 12, (0, 0, 255), -1)
        cv2.circle(output, pt3, 8, (255, 0, 0), -1)
        cv2.putText(output, f"{current_angle:.1f} deg", (pt2[0] + 15, pt2[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_angle_arc(output, pt1, pt2, pt3)
    except (IndexError, ValueError):
        pass
    
    return output


def draw_calibration_ui(frame, calibrator, current_angle, joint_name):
    """Vẽ giao diện calibration lên frame."""
    output = frame.copy()
    
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    cv2.putText(output, f"CALIBRATION: {joint_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    state_colors = {
        CalibrationState.IDLE: (128, 128, 128),
        CalibrationState.COLLECTING: (0, 255, 0),
        CalibrationState.PROCESSING: (255, 255, 0),
        CalibrationState.COMPLETED: (0, 255, 0),
        CalibrationState.ERROR: (0, 0, 255),
    }
    cv2.putText(output, f"Status: {calibrator.state.value}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_colors.get(calibrator.state, (255, 255, 255)), 2)
    
    cv2.putText(output, f"Current Angle: {current_angle:.1f} deg", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if calibrator.state == CalibrationState.COLLECTING:
        progress = calibrator.progress
        filled = int(300 * progress)
        cv2.rectangle(output, (20, 120), (320, 140), (100, 100, 100), -1)
        cv2.rectangle(output, (20, 120), (20 + filled, 140), (0, 255, 0), -1)
        cv2.putText(output, f"{progress:.0%}", (330, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"Elapsed: {calibrator.elapsed_ms/1000:.1f}s", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(output, "Move joint to MAXIMUM safe position!", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif calibrator.state == CalibrationState.IDLE:
        cv2.putText(output, "Press [SPACE] to start calibration", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    elif calibrator.state == CalibrationState.COMPLETED:
        user_max = calibrator.get_user_limit(calibrator.current_joint)
        if user_max:
            cv2.putText(output, f"Max Angle: {user_max:.1f} deg", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, "Press [S] save, [R] recalibrate, [C] compare", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return output


def run_calibration_session(detector, joint_type, source="webcam", display=True, output_dir="./data/user_profiles"):
    """Chạy session calibration đầy đủ."""
    joint_name = JOINT_DEFINITIONS[joint_type].name
    
    cap = cv2.VideoCapture(0 if source.lower() == "webcam" else source)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở source: {source}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"\n{'='*60}")
    print(f"CALIBRATION SESSION: {joint_name}")
    print(f"{'='*60}")
    if display:
        print("Controls: [SPACE] Start | [S] Save | [R] Recalibrate | [C] Compare | [Q] Quit\n")
    
    calibrator = SafeMaxCalibrator(duration_ms=5000)
    current_angle = 0.0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if source.lower() != "webcam":
                break
            continue
        
        if source.lower() == "webcam":
            frame = cv2.flip(frame, 1)
        
        timestamp_ms = int(frame_idx * (1000 / fps))
        result = detector.process_frame(frame, timestamp_ms)
        
        if result.has_pose():
            try:
                current_angle = calculate_joint_angle(result.pose_landmarks.to_numpy(), joint_type, use_3d=True)
            except ValueError:
                current_angle = 0.0
            
            if calibrator.state == CalibrationState.COLLECTING:
                calibrator.add_frame(result.pose_landmarks, timestamp_ms)
        
        if display:
            output = draw_pose_landmarks(frame, result.pose_landmarks if result.has_pose() else None, joint_type, current_angle)
            output = draw_calibration_ui(output, calibrator, current_angle, joint_name)
            
            cv2.imshow("MEMOTION - Calibration", output)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                calibrator.reset()
                calibrator.start_calibration(joint_type)
            elif key == ord('s') and calibrator.state == CalibrationState.COMPLETED:
                calibrator.save_profile(output_dir)
            elif key == ord('r'):
                calibrator.reset()
                calibrator.start_calibration(joint_type)
            elif key == ord('c') and calibrator.state == CalibrationState.COMPLETED:
                run_comparison_demo(calibrator.get_profile(), joint_type)
        else:
            if calibrator.state == CalibrationState.IDLE:
                calibrator.start_calibration(joint_type)
            if calibrator.state == CalibrationState.COMPLETED:
                break
        
        frame_idx += 1
    
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    return calibrator.get_profile()


def run_comparison_demo(user_profile, joint_type):
    """Demo so sánh góc mẫu vs góc mục tiêu cá nhân hóa."""
    ref_angles = [90 + 60 * np.sin(2 * np.pi * t / 60) for t in range(60)]
    
    user_max = user_profile.get_max_angle(joint_type)
    if user_max is None:
        print(f"[ERROR] Chưa calibrate khớp {joint_type.value}")
        return
    
    result = rescale_reference_motion(ref_angles, user_max, challenge_factor=0.05)
    joint_name = JOINT_DEFINITIONS[joint_type].name
    print_comparison_report(ref_angles, result, joint_name)


def run_unit_tests():
    """Chạy unit tests cho module calibration và kinematics."""
    print("\n" + "="*60)
    print("UNIT TESTS - Calibration & Kinematics")
    print("="*60)
    
    # Test 1: calculate_angle - Góc vuông 90°
    print("\n[TEST 1] calculate_angle - Góc vuông 90°...")
    a, b, c = np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 0])
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 0.1, f"Expected 90°, got {angle}°"
    print(f"  ✓ Góc tính được: {angle:.2f}°")
    
    # Test 2: calculate_angle - Góc thẳng 180°
    print("\n[TEST 2] calculate_angle - Góc thẳng 180°...")
    a, b, c = np.array([-1, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0])
    angle = calculate_angle(a, b, c)
    assert abs(angle - 180.0) < 0.1, f"Expected 180°, got {angle}°"
    print(f"  ✓ Góc tính được: {angle:.2f}°")
    
    # Test 3: calculate_angle - Góc 45°
    print("\n[TEST 3] calculate_angle - Góc 45°...")
    a, b, c = np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 0])
    angle = calculate_angle(a, b, c)
    assert abs(angle - 45.0) < 0.1, f"Expected 45°, got {angle}°"
    print(f"  ✓ Góc tính được: {angle:.2f}°")
    
    # Test 4: rescale_reference_motion
    print("\n[TEST 4] rescale_reference_motion...")
    ref = [0, 30, 60, 90, 120, 90, 60, 30, 0]
    result = rescale_reference_motion(ref, user_max_angle=90, challenge_factor=0.05)
    expected_scale = (90 / 120) * 1.05
    assert abs(result.scale_factor - expected_scale) < 0.001
    print(f"  ✓ Scale factor: {result.scale_factor:.4f}")
    print(f"  ✓ Max target: {result.get_max_target():.1f}°")
    print(f"  ✓ Reduction: {result.get_reduction_percent():.1f}%")
    
    # Test 5: Median filter
    print("\n[TEST 5] SafeMaxCalibrator - Median filter...")
    calibrator = SafeMaxCalibrator()
    angles = np.array([90, 95, 100, 150, 95, 100, 98, 102, 97, 99])
    filtered = calibrator._median_filter(angles)
    assert max(filtered) < 150, "Median filter should reduce spikes"
    print(f"  ✓ Original max: {max(angles)}, Filtered max: {max(filtered):.1f}")
    
    # Test 6: Remove outliers
    print("\n[TEST 6] Remove outliers...")
    angles_clean = calibrator._remove_outliers(angles)
    assert 150 not in angles_clean
    print(f"  ✓ Removed outliers, remaining: {len(angles_clean)} samples")
    
    # Test 7: UserProfile serialization
    print("\n[TEST 7] UserProfile JSON serialization...")
    from modules.calibration import JointCalibrationData
    profile = UserProfile(user_id="test_user", name="Test", age=70)
    profile.joint_limits["left_elbow"] = JointCalibrationData(
        joint_type="left_elbow", max_angle=120.5, min_angle=30.0, confidence=0.95,
        calibration_date="2024-01-01"
    )
    profile_dict = profile.to_dict()
    restored = UserProfile.from_dict(profile_dict)
    assert restored.user_id == profile.user_id
    assert restored.get_max_angle(JointType.LEFT_ELBOW) == 120.5
    print(f"  ✓ Serialization/deserialization OK")
    
    # Test 8: Full comparison demo
    print("\n[TEST 8] Full comparison demo...")
    ref_angles = [0, 30, 60, 90, 120, 90, 60, 30, 0]
    result = rescale_reference_motion(ref_angles, user_max_angle=90, challenge_factor=0.05)
    print_comparison_report(ref_angles, result, "Test Joint")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")


def download_models(models_dir):
    """Hướng dẫn download model files."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pose_model = models_dir / "pose_landmarker_lite.task"
    
    if not pose_model.exists():
        print(f"\n[!] Pose model not found: {pose_model}")
        print("    Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    
    return str(pose_model)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MEMOTION Calibration Test")
    parser.add_argument("--source", type=str, default="webcam", help="webcam or video path")
    parser.add_argument("--joint", type=str, default="left_elbow", 
                        help="Joint to calibrate: left_elbow, right_elbow, left_shoulder, etc.")
    parser.add_argument("--mode", type=str, choices=["calibrate", "test"], default="calibrate")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--display", action="store_true", help="Force display mode")
    parser.add_argument("--output-dir", type=str, default="./data/user_profiles")
    parser.add_argument("--models-dir", type=str, default="./models")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        run_unit_tests()
        return
    
    # Determine display mode
    if args.display:
        use_display = True
    elif args.headless:
        use_display = False
    else:
        use_display = check_display_available()
    
    print(f"[INFO] Display mode: {'ON' if use_display else 'OFF'}")
    
    # Parse joint type
    joint_type = get_joint_type_from_string(args.joint)
    if joint_type is None:
        print(f"[ERROR] Unknown joint: {args.joint}")
        print(f"Available joints: {[jt.value for jt in JointType]}")
        return
    
    # Setup model
    pose_model = download_models(args.models_dir)
    if not Path(pose_model).exists():
        print("[WARNING] Model not found. Running unit tests only.")
        run_unit_tests()
        return
    
    # Initialize detector
    config = DetectorConfig(pose_model_path=pose_model, running_mode="VIDEO")
    
    try:
        with VisionDetector(config) as detector:
            profile = run_calibration_session(
                detector, joint_type, args.source, use_display, args.output_dir
            )
            
            if profile and profile.get_max_angle(joint_type):
                print("\n[INFO] Calibration completed successfully!")
                run_comparison_demo(profile, joint_type)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        run_unit_tests()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()