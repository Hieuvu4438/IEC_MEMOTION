#!/usr/bin/env python3
"""
MEMOTION Comprehensive Code Audit Test Suite.

Kiểm tra:
1. Algorithm Integrity - Procrustes, Safe-Max, DTW
2. Domain-Specific Logic - Jerk, Wait-for-User, Pain Detection
3. Edge Cases - Out of bounds, Deadlocks, False Positives

Author: Code Audit Team
"""

import sys
import numpy as np
from typing import Dict, List, Tuple
import traceback

# =============================================
# TEST UTILITIES
# =============================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
    
    def add_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def add_fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗ {name}: {reason}")
    
    def add_warning(self, name: str, reason: str):
        self.warnings.append((name, reason))
        print(f"  ⚠ WARNING: {name}: {reason}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {len(self.warnings)} warnings")
        if self.errors:
            print("\nERRORS:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        if self.warnings:
            print("\nWARNINGS:")
            for name, reason in self.warnings:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")

results = TestResult()

# =============================================
# 1. PROCRUSTES ANALYSIS TESTS
# =============================================

def test_procrustes_analysis():
    """Kiểm tra Procrustes Analysis"""
    print("\n[1] PROCRUSTES ANALYSIS TESTS")
    print("-" * 40)
    
    from core.procrustes import (
        normalize_skeleton, 
        align_skeleton_to_reference,
        compute_procrustes_distance,
        compute_procrustes_similarity,
        extract_core_landmarks,
        compute_centroid,
        compute_scale,
    )
    
    # Test 1.1: Basic normalization
    skeleton = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    
    normalized = normalize_skeleton(skeleton, use_core_landmarks=False)
    centroid = compute_centroid(normalized.landmarks)
    
    # Centroid should be near origin after normalization
    if np.allclose(centroid, [0, 0, 0], atol=1e-5):
        results.add_pass("Procrustes: Centroid at origin")
    else:
        results.add_fail("Procrustes: Centroid at origin", f"Got {centroid}")
    
    # Test 1.2: Người đứng chéo góc với camera (rotation test)
    # Skeleton 1: Người đứng thẳng
    standing = np.array([
        [0, 0, 0],   # Hip center
        [0, 1, 0],   # Torso
        [0, 2, 0],   # Neck
        [-0.5, 1, 0], # Left shoulder
        [0.5, 1, 0],  # Right shoulder
        [-0.5, 0, 0], # Left hip
        [0.5, 0, 0],  # Right hip
    ], dtype=np.float32)
    
    # Skeleton 2: Người xoay 45° so với camera
    theta = np.pi / 4  # 45 degrees
    rot_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    rotated = standing @ rot_matrix.T
    
    # Align rotated to standing
    try:
        proc_result = align_skeleton_to_reference(rotated, standing, use_core_landmarks=False)
        disparity = proc_result.disparity
        
        # Disparity should be very small since they're same pose just rotated
        if disparity < 0.1:
            results.add_pass("Procrustes: Handles 45° rotation correctly")
        else:
            results.add_warning("Procrustes: 45° rotation", f"Disparity={disparity:.4f}, expected < 0.1")
    except Exception as e:
        results.add_fail("Procrustes: Rotation handling", str(e))
    
    # Test 1.3: Scale distortion test
    # Large person vs small person
    large_person = standing * 2.0
    
    try:
        proc_result = align_skeleton_to_reference(large_person, standing, use_core_landmarks=False)
        similarity = compute_procrustes_similarity(large_person, standing, use_core_landmarks=False)
        
        if similarity > 0.9:
            results.add_pass("Procrustes: Scale invariance maintained")
        else:
            results.add_warning("Procrustes: Scale invariance", f"Similarity={similarity:.4f}")
    except Exception as e:
        results.add_fail("Procrustes: Scale invariance", str(e))
    
    # Test 1.4: Empty skeleton handling
    empty_skeleton = np.array([], dtype=np.float32).reshape(0, 3)
    try:
        normalized = normalize_skeleton(empty_skeleton, use_core_landmarks=False)
        if normalized.landmarks.shape[0] == 0:
            results.add_pass("Procrustes: Empty skeleton handled")
        else:
            results.add_fail("Procrustes: Empty skeleton", "Should return empty")
    except Exception as e:
        results.add_fail("Procrustes: Empty skeleton", str(e))
    
    # Test 1.5: Tỷ lệ cơ thể có bị biến dạng không?
    # Kiểm tra shoulder-to-hip ratio trước và sau alignment
    shoulder_width_before = np.linalg.norm(standing[3] - standing[4])
    hip_width_before = np.linalg.norm(standing[5] - standing[6])
    ratio_before = shoulder_width_before / hip_width_before if hip_width_before > 0 else 0
    
    # Apply transformation
    aligned = proc_result.aligned_skeleton.landmarks
    if len(aligned) >= 7:
        shoulder_width_after = np.linalg.norm(aligned[3] - aligned[4])
        hip_width_after = np.linalg.norm(aligned[5] - aligned[6])
        ratio_after = shoulder_width_after / hip_width_after if hip_width_after > 0 else 0
        
        ratio_change = abs(ratio_before - ratio_after) / ratio_before if ratio_before > 0 else 0
        
        if ratio_change < 0.05:  # < 5% change
            results.add_pass("Procrustes: Body proportions preserved")
        else:
            results.add_warning("Procrustes: Body proportions", f"Ratio changed by {ratio_change*100:.1f}%")
    else:
        results.add_warning("Procrustes: Body proportions", "Not enough landmarks to check")


# =============================================
# 2. SAFE-MAX & θ_TARGET TESTS  
# =============================================

def test_safe_max_and_target():
    """Kiểm tra công thức Safe-Max và θ_target"""
    print("\n[2] SAFE-MAX & θ_TARGET TESTS")
    print("-" * 40)
    
    from modules.target_generator import (
        compute_scale_factor,
        rescale_reference_motion,
        compare_with_target,
    )
    
    # Test 2.1: Normal case
    user_max = 90.0   # Người già chỉ gập được 90°
    ref_max = 120.0   # Video mẫu gập 120°
    alpha = 0.05
    
    scale = compute_scale_factor(user_max, ref_max, alpha)
    expected_scale = (90 / 120) * 1.05  # = 0.7875
    
    if abs(scale - expected_scale) < 0.001:
        results.add_pass("Safe-Max: Normal scale calculation")
    else:
        results.add_fail("Safe-Max: Normal scale", f"Got {scale}, expected {expected_scale}")
    
    # Test 2.2: Edge case - user_max > ref_max (hiếm với người già)
    user_max_high = 150.0
    ref_max = 120.0
    scale_high = compute_scale_factor(user_max_high, ref_max, alpha)
    
    if scale_high == 1.0:  # Should be capped at 1.0
        results.add_pass("Safe-Max: Scale capped at 1.0")
    else:
        results.add_fail("Safe-Max: Scale cap", f"Got {scale_high}, expected 1.0")
    
    # Test 2.3: Edge case - user_max = 0 (không thể cử động)
    user_max_zero = 0.0
    scale_zero = compute_scale_factor(user_max_zero, ref_max, alpha)
    
    if scale_zero == 0.0:
        results.add_pass("Safe-Max: Zero user_max handled")
    else:
        results.add_fail("Safe-Max: Zero user_max", f"Got {scale_zero}, expected 0.0")
    
    # Test 2.4: Edge case - ref_max = 0 (không có chuyển động)
    try:
        scale_error = compute_scale_factor(90, 0.0, alpha)
        results.add_fail("Safe-Max: Zero ref_max", "Should raise ValueError")
    except ValueError:
        results.add_pass("Safe-Max: Zero ref_max raises error")
    
    # Test 2.5: α = 0 (không có challenge)
    scale_no_challenge = compute_scale_factor(90, 120, 0.0)
    expected_no_challenge = 90 / 120  # = 0.75
    
    if abs(scale_no_challenge - expected_no_challenge) < 0.001:
        results.add_pass("Safe-Max: Zero challenge factor")
    else:
        results.add_fail("Safe-Max: Zero challenge", f"Got {scale_no_challenge}")
    
    # Test 2.6: Negative angle (invalid input) - NOW SHOULD RAISE ERROR
    try:
        scale_neg = compute_scale_factor(-90, 120, alpha)
        results.add_fail("Safe-Max: Negative angle", f"Should raise error, got scale={scale_neg}")
    except ValueError as e:
        results.add_pass("Safe-Max: Negative angle validated correctly")
    
    # Test 2.7: Full rescale sequence
    ref_sequence = [0, 30, 60, 90, 120, 90, 60, 30, 0]
    user_max = 90
    
    rescaled = rescale_reference_motion(ref_sequence, user_max, alpha)
    
    # Check max target <= user_max * (1 + alpha)
    max_target = max(rescaled.target_angles)
    allowed_max = user_max * (1 + alpha)
    
    if max_target <= allowed_max + 0.1:  # Small tolerance
        results.add_pass("Safe-Max: Target never exceeds user limit")
    else:
        results.add_fail("Safe-Max: Target exceeds limit", 
                        f"Max target={max_target:.1f}, allowed={allowed_max:.1f}")
    
    # Test 2.8: Rất quan trọng - α quá cao
    # Nếu α = 1.0 (100%), có thể gây chấn thương
    try:
        scale_high_alpha = compute_scale_factor(90, 120, 1.0)  # 100% challenge
        # 90/120 * 2.0 = 1.5, but should be capped at 1.0
        if scale_high_alpha <= 1.0:
            results.add_pass("Safe-Max: High α capped correctly")
        else:
            results.add_fail("Safe-Max: High α", f"Scale={scale_high_alpha}, exceeds video!")
    except Exception as e:
        results.add_fail("Safe-Max: High α handling", str(e))


# =============================================
# 3. WEIGHTED DTW TESTS
# =============================================

def test_weighted_dtw():
    """Kiểm tra Weighted DTW"""
    print("\n[3] WEIGHTED DTW TESTS")
    print("-" * 40)
    
    from core.dtw_analysis import (
        compute_weighted_dtw,
        compute_single_joint_dtw,
        create_exercise_weights,
        preprocess_sequence,
    )
    from core.kinematics import JointType
    
    # Test 3.1: Identical sequences
    seq1 = [0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0]
    result = compute_single_joint_dtw(seq1, seq1)
    
    if result.similarity_score > 95:
        results.add_pass("DTW: Identical sequences score > 95%")
    else:
        results.add_fail("DTW: Identical sequences", f"Score={result.similarity_score}")
    
    # Test 3.2: Completely different sequences
    seq_diff = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    result_diff = compute_single_joint_dtw(seq1, seq_diff)
    
    if result_diff.similarity_score < 50:
        results.add_pass("DTW: Different sequences score < 50%")
    else:
        results.add_warning("DTW: Different sequences", f"Score={result_diff.similarity_score}")
    
    # Test 3.3: Time-shifted sequences (DTW should handle)
    seq_shifted = [0, 0, 0, 10, 20, 30, 40, 50, 40, 30, 20]
    result_shift = compute_single_joint_dtw(seq1, seq_shifted)
    
    if result_shift.similarity_score > 70:
        results.add_pass("DTW: Time-shifted sequences handled")
    else:
        results.add_warning("DTW: Time-shifted", f"Score={result_shift.similarity_score}")
    
    # Test 3.4: Exercise-specific weights
    arm_weights = create_exercise_weights("arm_raise")
    squat_weights = create_exercise_weights("squat")
    
    # Arm raise should weight shoulder higher than knee
    if arm_weights.get(JointType.LEFT_SHOULDER, 0) > arm_weights.get(JointType.LEFT_KNEE, 0):
        results.add_pass("DTW: Arm raise weights shoulder > knee")
    else:
        results.add_fail("DTW: Arm raise weights", "Shoulder should be weighted higher")
    
    # Squat should weight knee higher than shoulder
    if squat_weights.get(JointType.LEFT_KNEE, 0) > squat_weights.get(JointType.LEFT_SHOULDER, 0):
        results.add_pass("DTW: Squat weights knee > shoulder")
    else:
        results.add_fail("DTW: Squat weights", "Knee should be weighted higher")
    
    # Test 3.5: Empty sequence handling
    empty_seq = []
    result_empty = compute_single_joint_dtw(empty_seq, seq1)
    
    if result_empty.similarity_score == 100.0:  # Default for empty
        results.add_pass("DTW: Empty sequence handled")
    else:
        results.add_warning("DTW: Empty sequence", f"Score={result_empty.similarity_score}")
    
    # Test 3.6: Weighted DTW with multiple joints
    user_seqs = {
        JointType.LEFT_SHOULDER: [0, 30, 60, 90, 60, 30, 0],
        JointType.LEFT_ELBOW: [150, 120, 90, 60, 90, 120, 150],
        JointType.LEFT_KNEE: [170, 170, 170, 170, 170, 170, 170],  # Doesn't move
    }
    ref_seqs = {
        JointType.LEFT_SHOULDER: [0, 35, 65, 95, 65, 35, 0],  # Slightly different
        JointType.LEFT_ELBOW: [150, 115, 85, 55, 85, 115, 150],
        JointType.LEFT_KNEE: [170, 170, 170, 170, 170, 170, 170],
    }
    
    weighted_result = compute_weighted_dtw(user_seqs, ref_seqs, arm_weights)
    
    if weighted_result.similarity_score > 0:
        results.add_pass("DTW: Multi-joint weighted DTW works")
    else:
        results.add_fail("DTW: Multi-joint", "Score should be > 0")


# =============================================
# 4. JERK METRIC TESTS
# =============================================

def test_jerk_metric():
    """Kiểm tra Jerk Metric cho phát hiện mệt mỏi"""
    print("\n[4] JERK METRIC TESTS")
    print("-" * 40)
    
    from modules.scoring import calculate_jerk, HealthScorer, FatigueLevel
    
    # Test 4.1: Smooth motion (low jerk)
    # Tạo trajectory sin mượt
    t = np.linspace(0, 2*np.pi, 100)
    smooth_positions = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.zeros_like(t)
    ])
    smooth_timestamps = np.linspace(0, 2, 100)
    
    smooth_jerk = calculate_jerk(smooth_positions, smooth_timestamps)
    
    # Test 4.2: Jerky motion (high jerk)
    # Tạo trajectory giật cục
    jerky_positions = smooth_positions.copy()
    # Thêm noise
    np.random.seed(42)
    noise = np.random.randn(100, 3) * 0.3
    jerky_positions = jerky_positions + noise
    
    jerky_jerk = calculate_jerk(jerky_positions, smooth_timestamps)
    
    if jerky_jerk > smooth_jerk:
        results.add_pass("Jerk: Jerky motion has higher jerk than smooth")
    else:
        results.add_fail("Jerk: Jerk comparison", 
                        f"Smooth={smooth_jerk:.4f}, Jerky={jerky_jerk:.4f}")
    
    # Test 4.3: Phân biệt "run sinh lý" vs "nhiễu MediaPipe"
    # Run sinh lý: biên độ nhỏ (~2-5°), tần số thấp (3-6 Hz)
    # Nhiễu MediaPipe: biên độ lớn hơn, random
    
    # Simulate tremor (run sinh lý)
    tremor_freq = 5  # Hz
    tremor_amp = 0.02  # Small amplitude
    tremor_positions = smooth_positions + tremor_amp * np.sin(2 * np.pi * tremor_freq * smooth_timestamps).reshape(-1, 1)
    tremor_jerk = calculate_jerk(tremor_positions, smooth_timestamps)
    
    # Nhiễu MediaPipe thường có jerk cao hơn nhiều
    if tremor_jerk < jerky_jerk:
        results.add_pass("Jerk: Tremor vs MediaPipe noise distinguishable")
    else:
        results.add_warning("Jerk: Tremor vs noise", 
                          f"Tremor={tremor_jerk:.4f}, Noise={jerky_jerk:.4f}")
    
    # Test 4.4: Short sequence handling
    short_positions = smooth_positions[:3]  # Only 3 points
    short_timestamps = smooth_timestamps[:3]
    short_jerk = calculate_jerk(short_positions, short_timestamps)
    
    if short_jerk == 0.0:  # Should return 0 for too few points
        results.add_pass("Jerk: Short sequence handled (returns 0)")
    else:
        results.add_warning("Jerk: Short sequence", f"Expected 0, got {short_jerk}")
    
    # Test 4.5: Zero time delta handling
    same_timestamps = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    safe_jerk = calculate_jerk(smooth_positions[:5], same_timestamps)
    
    if not np.isnan(safe_jerk) and not np.isinf(safe_jerk):
        results.add_pass("Jerk: Zero time delta handled safely")
    else:
        results.add_fail("Jerk: Zero time delta", f"Got {safe_jerk}")


# =============================================
# 5. WAIT-FOR-USER FSM TESTS
# =============================================

def test_wait_for_user_fsm():
    """Kiểm tra FSM Wait-for-User"""
    print("\n[5] WAIT-FOR-USER FSM TESTS")
    print("-" * 40)
    
    from core.synchronizer import (
        MotionSyncController,
        create_arm_raise_exercise,
        SyncStatus,
        MotionPhase,
    )
    
    # Test 5.1: Normal flow
    exercise = create_arm_raise_exercise(total_frames=100, fps=30)
    controller = MotionSyncController(exercise)
    
    # Simulate user matching targets
    states = []
    for i in range(100):
        # User matches target angle
        target = exercise.get_current_phase(i)
        user_angle = 30 if target == MotionPhase.IDLE else 90
        state = controller.update(user_angle, i, timestamp=i/30)
        states.append(state.sync_status)
    
    # Should have some PLAY states
    if SyncStatus.PLAY in states:
        results.add_pass("FSM: Normal flow has PLAY states")
    else:
        results.add_fail("FSM: Normal flow", "No PLAY states found")
    
    # Test 5.2: DEADLOCK TEST - User never reaches target
    controller2 = MotionSyncController(exercise)
    
    # Simulate user stuck at low angle (never reaches checkpoint)
    pause_count = 0
    skip_triggered = False
    
    for i in range(500):  # Long simulation
        state = controller2.update(user_angle=10.0, ref_frame=50, timestamp=i/30)
        if state.sync_status == SyncStatus.PAUSE:
            pause_count += 1
        elif state.sync_status == SyncStatus.SKIP:
            skip_triggered = True
            break
    
    if skip_triggered or pause_count < 400:
        results.add_pass("FSM: Deadlock avoided (timeout/skip works)")
    else:
        results.add_fail("FSM: DEADLOCK DETECTED", 
                        f"Paused {pause_count} times without skip, may block indefinitely")
    
    # Test 5.3: Check MAX_WAIT_TIME mechanism
    max_wait = MotionSyncController.MAX_WAIT_TIME
    if max_wait > 0:
        results.add_pass(f"FSM: Has MAX_WAIT_TIME={max_wait}s")
    else:
        results.add_fail("FSM: No MAX_WAIT_TIME", "Risk of infinite wait")
    
    # Test 5.4: Phase transitions
    controller3 = MotionSyncController(exercise)
    phases_seen = set()
    
    for i in range(100):
        state = controller3.update(user_angle=90, ref_frame=i, timestamp=i/30)
        phases_seen.add(state.current_phase)
    
    expected_phases = {MotionPhase.IDLE, MotionPhase.ECCENTRIC, MotionPhase.HOLD, MotionPhase.CONCENTRIC}
    if expected_phases.issubset(phases_seen):
        results.add_pass("FSM: All phases reachable")
    else:
        missing = expected_phases - phases_seen
        results.add_warning("FSM: Missing phases", f"{missing}")
    
    # Test 5.5: Rep counting
    controller4 = MotionSyncController(exercise)
    
    # Complete 2 full cycles
    for cycle in range(2):
        for i in range(100):
            angle = 30 + (i / 50) * 60 if i < 50 else 90 - ((i-50) / 50) * 60
            controller4.update(user_angle=angle, ref_frame=i % 100, timestamp=(cycle*100+i)/30)
    
    if controller4.state.rep_count >= 1:
        results.add_pass(f"FSM: Rep counting works (count={controller4.state.rep_count})")
    else:
        results.add_warning("FSM: Rep counting", f"Count={controller4.state.rep_count}")


# =============================================
# 6. PAIN DETECTION TESTS
# =============================================

def test_pain_detection():
    """Kiểm tra Pain Detection - đặc biệt False Positive với người già"""
    print("\n[6] PAIN DETECTION TESTS")
    print("-" * 40)
    
    from modules.pain_detection import (
        PainDetector,
        PainLevel,
        FaceLandmarkIndex,
    )
    
    detector = PainDetector()
    
    # Create mock face landmarks (468 points)
    def create_mock_landmarks(neutral=True, pain=False, wrinkles=False):
        """Tạo mock landmarks với các trạng thái khác nhau"""
        landmarks = np.random.rand(478, 3) * 0.1 + 0.5  # Base random
        
        # Set key landmarks for face structure
        landmarks[FaceLandmarkIndex.FACE_TOP] = [0.5, 0.1, 0]
        landmarks[FaceLandmarkIndex.FACE_BOTTOM] = [0.5, 0.9, 0]
        landmarks[FaceLandmarkIndex.LEFT_EYE_TOP] = [0.35, 0.35, 0]
        landmarks[FaceLandmarkIndex.LEFT_EYE_BOTTOM] = [0.35, 0.40, 0]  # Normal eye opening
        landmarks[FaceLandmarkIndex.LEFT_EYE_INNER] = [0.38, 0.37, 0]
        landmarks[FaceLandmarkIndex.LEFT_EYE_OUTER] = [0.30, 0.37, 0]
        landmarks[FaceLandmarkIndex.RIGHT_EYE_TOP] = [0.65, 0.35, 0]
        landmarks[FaceLandmarkIndex.RIGHT_EYE_BOTTOM] = [0.65, 0.40, 0]
        landmarks[FaceLandmarkIndex.RIGHT_EYE_INNER] = [0.62, 0.37, 0]
        landmarks[FaceLandmarkIndex.RIGHT_EYE_OUTER] = [0.70, 0.37, 0]
        landmarks[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE] = [0.35, 0.28, 0]
        landmarks[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE] = [0.65, 0.28, 0]
        landmarks[FaceLandmarkIndex.NOSE_TIP] = [0.5, 0.55, 0]
        landmarks[FaceLandmarkIndex.NOSE_BRIDGE] = [0.5, 0.40, 0]
        landmarks[FaceLandmarkIndex.UPPER_LIP_TOP] = [0.5, 0.70, 0]
        landmarks[FaceLandmarkIndex.UPPER_LIP_BOTTOM] = [0.5, 0.72, 0]
        landmarks[FaceLandmarkIndex.LOWER_LIP_TOP] = [0.5, 0.73, 0]
        landmarks[FaceLandmarkIndex.LOWER_LIP_BOTTOM] = [0.5, 0.78, 0]
        landmarks[FaceLandmarkIndex.MOUTH_LEFT] = [0.45, 0.74, 0]
        landmarks[FaceLandmarkIndex.MOUTH_RIGHT] = [0.55, 0.74, 0]
        
        if pain:
            # Simulate pain expression:
            # - Squinted eyes (smaller vertical distance)
            landmarks[FaceLandmarkIndex.LEFT_EYE_TOP][1] = 0.37  # Lower
            landmarks[FaceLandmarkIndex.LEFT_EYE_BOTTOM][1] = 0.38  # Higher
            landmarks[FaceLandmarkIndex.RIGHT_EYE_TOP][1] = 0.37
            landmarks[FaceLandmarkIndex.RIGHT_EYE_BOTTOM][1] = 0.38
            # - Furrowed brow (lower eyebrows)
            landmarks[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE][1] = 0.32
            landmarks[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE][1] = 0.32
            # - Raised upper lip
            landmarks[FaceLandmarkIndex.UPPER_LIP_TOP][1] = 0.68
        
        if wrinkles:
            # Người già có nhiều nếp nhăn tự nhiên
            # Khoảng cách lông mày-mắt có thể nhỏ hơn do da chảy xệ
            landmarks[FaceLandmarkIndex.LEFT_EYEBROW_MIDDLE][1] = 0.31
            landmarks[FaceLandmarkIndex.RIGHT_EYEBROW_MIDDLE][1] = 0.31
            # Mắt có thể hơi nheo do mí sụp
            landmarks[FaceLandmarkIndex.LEFT_EYE_TOP][1] = 0.36
            landmarks[FaceLandmarkIndex.RIGHT_EYE_TOP][1] = 0.36
        
        return landmarks.astype(np.float32)
    
    # Test 6.1: Neutral face - should not detect pain
    neutral_landmarks = create_mock_landmarks(neutral=True)
    detector.set_baseline(neutral_landmarks)
    
    result_neutral = detector.analyze(neutral_landmarks)
    
    if result_neutral.pain_level == PainLevel.NONE:
        results.add_pass("Pain: Neutral face = no pain")
    else:
        results.add_warning("Pain: Neutral face", 
                          f"Level={result_neutral.pain_level}, Score={result_neutral.pain_score}")
    
    # Test 6.2: Pain expression - should detect pain
    pain_landmarks = create_mock_landmarks(pain=True)
    result_pain = detector.analyze(pain_landmarks)
    
    if result_pain.pain_score > 0:
        results.add_pass(f"Pain: Pain expression detected (score={result_pain.pain_score:.1f})")
    else:
        results.add_warning("Pain: Pain expression", "Not detected")
    
    # Test 6.3: FALSE POSITIVE với người già có nhiều nếp nhăn
    detector2 = PainDetector()
    
    # Baseline là người già với nếp nhăn
    elderly_neutral = create_mock_landmarks(neutral=True, wrinkles=True)
    detector2.set_baseline(elderly_neutral)
    
    # Analyze same elderly face (should not detect pain)
    result_elderly = detector2.analyze(elderly_neutral)
    
    if result_elderly.pain_level == PainLevel.NONE or result_elderly.pain_score < 20:
        results.add_pass("Pain: Elderly wrinkles = no false positive")
    else:
        results.add_fail("Pain: FALSE POSITIVE with elderly", 
                        f"Level={result_elderly.pain_level}, Score={result_elderly.pain_score:.1f}")
    
    # Test 6.4: Duration filter - should not report pain for brief expressions
    detector3 = PainDetector()
    detector3.set_baseline(neutral_landmarks)
    
    # Flash of pain expression (< MIN_PAIN_DURATION_MS)
    brief_pain = detector3.analyze(pain_landmarks, timestamp=0.0)
    
    # Should not be marked as pain event yet (needs duration)
    if not brief_pain.is_pain_detected:
        results.add_pass("Pain: Brief expression filtered out")
    else:
        results.add_warning("Pain: Brief expression", "May cause false alarms")
    
    # Test 6.5: MIN_PAIN_DURATION_MS check
    min_duration = PainDetector.MIN_PAIN_DURATION_MS
    if min_duration >= 500:
        results.add_pass(f"Pain: Has MIN_PAIN_DURATION={min_duration}ms (good filter)")
    else:
        results.add_warning("Pain: MIN_PAIN_DURATION", f"Only {min_duration}ms, may cause false positives")


# =============================================
# 7. EDGE CASES TESTS
# =============================================

def test_edge_cases():
    """Kiểm tra các edge cases"""
    print("\n[7] EDGE CASES TESTS")
    print("-" * 40)
    
    from core.kinematics import calculate_angle, calculate_angle_safe
    import numpy as np
    
    # Test 7.1: Collinear points (góc = 0 hoặc 180)
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])  # Vertex
    c = np.array([2, 0, 0])  # Collinear
    
    angle_collinear = calculate_angle(a, b, c)
    if abs(angle_collinear - 180) < 1:  # Should be ~180°
        results.add_pass("Edge: Collinear points handled (180°)")
    else:
        results.add_warning("Edge: Collinear points", f"Got {angle_collinear}°")
    
    # Test 7.2: Coincident points (should raise error or return safe value)
    same_point = np.array([0, 0, 0])
    
    try:
        angle_same = calculate_angle(same_point, same_point, same_point)
        results.add_warning("Edge: Coincident points", f"No error, got {angle_same}")
    except ValueError:
        results.add_pass("Edge: Coincident points raise ValueError")
    
    # Safe version should return default
    angle_safe = calculate_angle_safe(same_point, same_point, same_point, default_angle=0.0)
    if angle_safe == 0.0:
        results.add_pass("Edge: Safe version returns default for bad input")
    else:
        results.add_warning("Edge: Safe version", f"Got {angle_safe}")
    
    # Test 7.3: NaN/Inf handling
    nan_point = np.array([np.nan, 0, 0])
    
    angle_nan = calculate_angle_safe(nan_point, b, c)
    if not np.isnan(angle_nan) and not np.isinf(angle_nan):
        results.add_pass("Edge: NaN input handled safely")
    else:
        results.add_fail("Edge: NaN propagation", f"Got {angle_nan}")
    
    # Test 7.4: Very large coordinates
    large = np.array([1e10, 1e10, 1e10])
    large_angle = calculate_angle_safe(large, np.array([0, 0, 0]), np.array([1e10, 0, 0]))
    
    if 0 <= large_angle <= 180:
        results.add_pass("Edge: Large coordinates handled")
    else:
        results.add_fail("Edge: Large coordinates", f"Got {large_angle}")
    
    # Test 7.5: Very small coordinates
    tiny = np.array([1e-10, 0, 0])
    tiny_angle = calculate_angle_safe(tiny, np.array([0, 0, 0]), np.array([0, 1e-10, 0]))
    
    if 0 <= tiny_angle <= 180:
        results.add_pass("Edge: Tiny coordinates handled")
    else:
        results.add_warning("Edge: Tiny coordinates", f"Got {tiny_angle}")


# =============================================
# 8. CALIBRATION TESTS
# =============================================

def test_calibration():
    """Kiểm tra Calibration Module"""
    print("\n[8] CALIBRATION TESTS")
    print("-" * 40)
    
    from modules.calibration import SafeMaxCalibrator, CalibrationState
    from core.kinematics import JointType
    from core.data_types import LandmarkSet, Point3D, LandmarkType
    
    calibrator = SafeMaxCalibrator(duration_ms=1000, min_samples=10)
    
    # Test 8.1: Start calibration
    calibrator.start_calibration(JointType.LEFT_ELBOW)
    
    if calibrator.state == CalibrationState.COLLECTING:
        results.add_pass("Calibration: Start works")
    else:
        results.add_fail("Calibration: Start", f"State={calibrator.state}")
    
    # Test 8.2: Add frames with mock landmarks
    def create_mock_pose_landmarks(elbow_angle_modifier=0):
        """Create 33 pose landmarks with adjustable elbow angle"""
        points = []
        for i in range(33):
            if i == 11:  # Left shoulder
                points.append(Point3D(0.3, 0.3, 0))
            elif i == 13:  # Left elbow
                points.append(Point3D(0.3, 0.5 + elbow_angle_modifier * 0.01, 0))
            elif i == 15:  # Left wrist
                points.append(Point3D(0.3, 0.7, 0))
            else:
                points.append(Point3D(0.5, 0.5, 0))
        return LandmarkSet(landmarks=points, landmark_type=LandmarkType.POSE)
    
    # Simulate adding frames
    for i in range(15):
        landmarks = create_mock_pose_landmarks(elbow_angle_modifier=i)
        calibrator.add_frame(landmarks, timestamp_ms=i * 100)
    
    # Test 8.3: Finish calibration
    result = calibrator.finish_calibration()
    
    if calibrator.state == CalibrationState.COMPLETED:
        results.add_pass("Calibration: Completed successfully")
    else:
        results.add_fail("Calibration: Completion", f"State={calibrator.state}")
    
    # Test 8.4: Check result
    if result is not None and result.max_angle > 0:
        results.add_pass(f"Calibration: Max angle computed ({result.max_angle:.1f}°)")
    else:
        results.add_warning("Calibration: Result", "No max angle computed")
    
    # Test 8.5: Insufficient samples
    calibrator2 = SafeMaxCalibrator(duration_ms=1000, min_samples=100)
    calibrator2.start_calibration(JointType.LEFT_ELBOW)
    
    # Only add 5 frames
    for i in range(5):
        landmarks = create_mock_pose_landmarks(i)
        calibrator2.add_frame(landmarks, timestamp_ms=i * 100)
    
    result2 = calibrator2.finish_calibration()
    
    if result2 is None or calibrator2.state == CalibrationState.ERROR:
        results.add_pass("Calibration: Insufficient samples detected")
    else:
        results.add_warning("Calibration: Insufficient samples", "Should have failed")


# =============================================
# RUN ALL TESTS
# =============================================

def main():
    print("=" * 60)
    print("MEMOTION CODE AUDIT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        test_procrustes_analysis()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Procrustes tests: {e}")
        traceback.print_exc()
    
    try:
        test_safe_max_and_target()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Safe-Max tests: {e}")
        traceback.print_exc()
    
    try:
        test_weighted_dtw()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in DTW tests: {e}")
        traceback.print_exc()
    
    try:
        test_jerk_metric()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Jerk tests: {e}")
        traceback.print_exc()
    
    try:
        test_wait_for_user_fsm()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in FSM tests: {e}")
        traceback.print_exc()
    
    try:
        test_pain_detection()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Pain Detection tests: {e}")
        traceback.print_exc()
    
    try:
        test_edge_cases()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Edge Cases tests: {e}")
        traceback.print_exc()
    
    try:
        test_calibration()
    except Exception as e:
        print(f"  ✗ CRITICAL ERROR in Calibration tests: {e}")
        traceback.print_exc()
    
    results.summary()
    return results.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)