#!/usr/bin/env python3
"""
Main Test Script for MEMOTION - Phase 1.

Test script để verify:
1. VisionDetector hoạt động với MediaPipe Tasks API
2. Procrustes Analysis chuẩn hóa skeleton đúng
3. So sánh tư thế real-time với reference pose

Usage:
    # Auto-detect mode (tự động chọn GUI/headless)
    python main_test.py --source video.mp4
    
    # Force GUI mode (local machine)
    python main_test.py --source video.mp4 --display
    
    # Force headless mode (server)
    python main_test.py --source video.mp4 --headless --output results.csv
    
    # Webcam
    python main_test.py --source webcam
    
Author: MEMOTION Team
Version: 1.1.0
"""

import argparse
import sys
import time
import csv
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

from core import (
    VisionDetector,
    DetectorConfig,
    DetectionResult,
    compute_procrustes_distance,
    compute_procrustes_similarity,
    normalize_skeleton,
    PoseLandmarkIndex,
)


def check_display_available() -> bool:
    """
    Kiểm tra xem có thể hiển thị GUI không.
    
    Returns:
        bool: True nếu có display available.
    """
    # Check DISPLAY environment variable (Linux/Mac)
    display = os.environ.get('DISPLAY')
    
    # Check if running in SSH without X forwarding
    ssh_connection = os.environ.get('SSH_CONNECTION')
    ssh_tty = os.environ.get('SSH_TTY')
    
    # Windows always has display
    if sys.platform == 'win32':
        return True
    
    # macOS typically has display
    if sys.platform == 'darwin':
        return True
    
    # Linux: check DISPLAY
    if display:
        return True
    
    # Check for Wayland
    wayland = os.environ.get('WAYLAND_DISPLAY')
    if wayland:
        return True
    
    return False


def safe_imshow(window_name: str, frame: np.ndarray) -> bool:
    """
    Hiển thị frame một cách an toàn.
    
    Args:
        window_name: Tên cửa sổ.
        frame: Frame cần hiển thị.
        
    Returns:
        bool: True nếu hiển thị thành công.
    """
    try:
        cv2.imshow(window_name, frame)
        return True
    except cv2.error as e:
        return False


def safe_waitkey(delay: int = 1) -> int:
    """
    Chờ phím nhấn một cách an toàn.
    
    Args:
        delay: Thời gian chờ (ms).
        
    Returns:
        int: Key code, -1 nếu lỗi.
    """
    try:
        return cv2.waitKey(delay) & 0xFF
    except cv2.error:
        return -1


class PoseVisualizer:
    """Visualize pose landmarks và Procrustes results."""
    
    POSE_CONNECTIONS = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 24),  # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]
    
    @staticmethod
    def draw_landmarks(
        frame: np.ndarray,
        result: DetectionResult,
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """Vẽ pose landmarks lên frame."""
        if not result.has_pose():
            return frame
        
        output = frame.copy()
        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks
        
        for i, lm in enumerate(landmarks.landmarks):
            if lm.visibility is not None and lm.visibility < 0.5:
                continue
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(output, (x, y), 5, color, -1)
        
        for start_idx, end_idx in PoseVisualizer.POSE_CONNECTIONS:
            if start_idx >= len(landmarks.landmarks) or end_idx >= len(landmarks.landmarks):
                continue
                
            start_lm = landmarks.landmarks[start_idx]
            end_lm = landmarks.landmarks[end_idx]
            
            if (start_lm.visibility is not None and start_lm.visibility < 0.5) or \
               (end_lm.visibility is not None and end_lm.visibility < 0.5):
                continue
            
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(output, start_pt, end_pt, color, 2)
        
        return output
    
    @staticmethod
    def draw_info(
        frame: np.ndarray,
        disparity: float,
        similarity: float,
        fps: float
    ) -> np.ndarray:
        """Vẽ thông tin metrics lên frame."""
        output = frame.copy()
        
        info_texts = [
            f"FPS: {fps:.1f}",
            f"Disparity: {disparity:.4f}",
            f"Similarity: {similarity:.1%}",
        ]
        
        y_offset = 30
        for i, text in enumerate(info_texts):
            y = y_offset + i * 30
            cv2.putText(
                output, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )
        
        bar_x, bar_y = 10, 120
        bar_width, bar_height = 200, 20
        filled_width = int(bar_width * similarity)
        
        cv2.rectangle(
            output, (bar_x, bar_y), 
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100), -1
        )
        
        bar_color = (0, 255, 0) if similarity > 0.7 else \
                    (0, 255, 255) if similarity > 0.4 else (0, 0, 255)
        cv2.rectangle(
            output, (bar_x, bar_y),
            (bar_x + filled_width, bar_y + bar_height),
            bar_color, -1
        )
        
        return output


def download_models(models_dir: Path) -> tuple:
    """Hướng dẫn download model files."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    pose_model = models_dir / "pose_landmarker_lite.task"
    face_model = models_dir / "face_landmarker.task"
    
    missing_models = []
    
    if not pose_model.exists():
        missing_models.append(("Pose", pose_model, 
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"))
    
    if not face_model.exists():
        missing_models.append(("Face", face_model,
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"))
    
    if missing_models:
        print("\n" + "="*60)
        print("HƯỚNG DẪN TẢI MODEL FILES")
        print("="*60)
        for name, path, url in missing_models:
            print(f"\n[!] {name} model không tìm thấy: {path}")
            print(f"    Tải từ: {url}")
        print("="*60 + "\n")
    
    return str(pose_model), str(face_model)


def run_video_process(
    detector: VisionDetector,
    video_path: str,
    display: bool = True,
    output_csv: Optional[str] = None,
    output_video: Optional[str] = None,
    sample_rate: int = 1
) -> List[Dict]:
    """
    Xử lý video với tùy chọn hiển thị GUI hoặc headless.
    
    Args:
        detector: VisionDetector đã khởi tạo.
        video_path: Đường dẫn video.
        display: True để hiển thị GUI, False cho headless mode.
        output_csv: Đường dẫn file CSV output (optional).
        output_video: Đường dẫn video output với annotations (optional).
        sample_rate: Xử lý mỗi N frames (1 = tất cả frames).
        
    Returns:
        List[Dict]: Danh sách kết quả mỗi frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    mode_str = "Display Mode" if display else "Headless Mode"
    
    print(f"\n{'='*60}")
    print(f"VIDEO PROCESSING ({mode_str})")
    print(f"{'='*60}")
    print(f"  File: {video_path}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f}s ({frame_count} frames)")
    print(f"  Sample rate: 1/{sample_rate}")
    if display:
        print(f"\n  Controls:")
        print(f"    [c] Capture new reference pose")
        print(f"    [r] Reset reference pose")
        print(f"    [SPACE] Pause/Resume")
        print(f"    [q] Quit")
    print(f"{'='*60}\n")
    
    # Video writer nếu cần
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / sample_rate if sample_rate > 1 else fps
        video_writer = cv2.VideoWriter(
            output_video, fourcc, out_fps,
            (frame_width, frame_height)
        )
    
    results = []
    reference_pose = None
    frame_idx = 0
    processed_count = 0
    start_time = time.time()
    paused = False
    display_failed = False
    
    # Window name
    window_name = "MEMOTION - Video Processing"
    
    while True:
        # Xử lý pause
        if paused and display:
            key = safe_waitkey(100)
            if key == ord(' '):
                paused = False
                print("[INFO] Resumed")
            elif key == ord('q'):
                break
            continue
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        timestamp_ms = int((frame_idx / fps) * 1000)
        result = detector.process_frame(frame, timestamp_ms)
        
        frame_result = {
            "frame": frame_idx,
            "timestamp_ms": timestamp_ms,
            "has_pose": result.has_pose(),
            "has_face": result.has_face(),
            "disparity": None,
            "similarity": None,
        }
        
        disparity = 0.0
        similarity = 0.0
        output_frame = frame.copy()
        
        if result.has_pose():
            current_skeleton = result.pose_landmarks.to_numpy()
            
            if reference_pose is None:
                reference_pose = current_skeleton
                print(f"[INFO] Reference pose captured at frame {frame_idx}")
            
            disparity = compute_procrustes_distance(current_skeleton, reference_pose)
            similarity = compute_procrustes_similarity(current_skeleton, reference_pose)
            
            frame_result["disparity"] = disparity
            frame_result["similarity"] = similarity
            
            # Vẽ skeleton và info
            output_frame = PoseVisualizer.draw_landmarks(frame, result)
            output_frame = PoseVisualizer.draw_info(
                output_frame, disparity, similarity, fps
            )
        else:
            # Vẽ thông báo không detect được pose
            cv2.putText(
                output_frame, "No pose detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        # Thêm frame info
        progress = (frame_idx / frame_count) * 100
        cv2.putText(
            output_frame,
            f"Frame: {frame_idx}/{frame_count} ({progress:.1f}%)",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Ghi video output
        if video_writer:
            video_writer.write(output_frame)
        
        # Hiển thị GUI nếu được bật và chưa fail
        if display and not display_failed:
            success = safe_imshow(window_name, output_frame)
            if not success:
                display_failed = True
                print("[WARNING] Display failed, switching to headless mode")
                display = False
            else:
                # Xử lý phím nhấn
                wait_time = max(1, int(1000 / fps))
                key = safe_waitkey(wait_time)
                
                if key == ord('q'):
                    print("[INFO] Quit by user")
                    break
                elif key == ord('c') and result.has_pose():
                    reference_pose = result.pose_landmarks.to_numpy()
                    print(f"[INFO] New reference pose captured at frame {frame_idx}")
                elif key == ord('r'):
                    reference_pose = None
                    print("[INFO] Reference pose reset")
                elif key == ord(' '):
                    paused = True
                    print("[INFO] Paused - press SPACE to resume")
        
        results.append(frame_result)
        processed_count += 1
        
        # Progress indicator (cho headless mode)
        if not display and processed_count % 50 == 0:
            elapsed = time.time() - start_time
            remaining = frame_count - frame_idx
            eta = (elapsed / processed_count) * (remaining / sample_rate)
            print(f"  Progress: {progress:.1f}% ({processed_count} frames, ETA: {eta:.1f}s)")
        
        frame_idx += 1
    
    cap.release()
    if video_writer:
        video_writer.release()
    
    if display and not display_failed:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {processed_count} frames in {elapsed:.2f}s")
    print(f"  Speed: {processed_count/elapsed:.1f} fps")
    
    # Statistics
    valid_results = [r for r in results if r["disparity"] is not None]
    if valid_results:
        disparities = [r["disparity"] for r in valid_results]
        similarities = [r["similarity"] for r in valid_results]
        
        print(f"\n  Disparity Stats:")
        print(f"    Min: {min(disparities):.4f}")
        print(f"    Max: {max(disparities):.4f}")
        print(f"    Mean: {np.mean(disparities):.4f}")
        print(f"    Std: {np.std(disparities):.4f}")
        
        print(f"\n  Similarity Stats:")
        print(f"    Min: {min(similarities):.2%}")
        print(f"    Max: {max(similarities):.2%}")
        print(f"    Mean: {np.mean(similarities):.2%}")
    
    # Save CSV
    if output_csv and results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {output_csv}")
    
    if output_video:
        print(f"  Annotated video saved to: {output_video}")
    
    print(f"{'='*60}\n")
    
    return results


def run_webcam_test(
    detector: VisionDetector,
    reference_pose: Optional[np.ndarray] = None
) -> None:
    """Chạy test với webcam real-time (cần GUI)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không thể mở webcam")
        return
    
    print("\n[INFO] Bắt đầu webcam test...")
    print("[INFO] Nhấn 'c' để capture reference pose")
    print("[INFO] Nhấn 'r' để reset reference pose")
    print("[INFO] Nhấn 'q' để thoát\n")
    
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.process_frame(frame, timestamp_ms)
        
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        output_frame = frame.copy()
        disparity = 0.0
        similarity = 0.0
        
        if result.has_pose():
            current_skeleton = result.pose_landmarks.to_numpy()
            
            output_frame = PoseVisualizer.draw_landmarks(
                output_frame, result, (0, 255, 0)
            )
            
            if reference_pose is not None:
                disparity = compute_procrustes_distance(
                    current_skeleton, reference_pose
                )
                similarity = compute_procrustes_similarity(
                    current_skeleton, reference_pose
                )
        else:
            cv2.putText(
                output_frame, "No pose detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        output_frame = PoseVisualizer.draw_info(
            output_frame, disparity, similarity, current_fps
        )
        
        ref_status = "Reference: SET" if reference_pose is not None else "Reference: NOT SET"
        cv2.putText(
            output_frame, ref_status, (10, output_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        
        cv2.imshow("MEMOTION - Phase 1 Test", output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and result.has_pose():
            reference_pose = result.pose_landmarks.to_numpy()
            print("[INFO] Reference pose captured!")
        elif key == ord('r'):
            reference_pose = None
            print("[INFO] Reference pose reset!")
    
    cap.release()
    cv2.destroyAllWindows()


def run_image_test(
    detector: VisionDetector,
    image_path: str,
    headless: bool = False
) -> None:
    """Chạy test với ảnh tĩnh."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Không thể đọc ảnh: {image_path}")
        return
    
    print(f"\n[INFO] Đang xử lý ảnh: {image_path}")
    
    result = detector.process_image(image)
    
    if result.has_pose():
        skeleton = result.pose_landmarks.to_numpy()
        normalized = normalize_skeleton(skeleton)
        
        print("\n[RESULT] Pose Detection Success!")
        print(f"  - Number of landmarks: {len(result.pose_landmarks)}")
        print(f"  - Centroid: {normalized.centroid}")
        print(f"  - Scale: {normalized.scale}")
        
        if not headless:
            output_frame = PoseVisualizer.draw_landmarks(image, result)
            cv2.imshow("MEMOTION - Image Test", output_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"[WARNING] No pose detected. Error: {result.error_message}")


def run_unit_tests() -> None:
    """Chạy unit tests cho Procrustes Analysis."""
    print("\n" + "="*60)
    print("UNIT TESTS - Procrustes Analysis")
    print("="*60)
    
    # Test 1: Identical skeletons
    print("\n[TEST 1] Identical skeletons...")
    skeleton = np.random.rand(12, 3)
    disparity = compute_procrustes_distance(skeleton, skeleton, use_core_landmarks=False)
    assert disparity < 1e-10, f"Expected ~0, got {disparity}"
    print(f"  ✓ Disparity: {disparity:.10f}")
    
    # Test 2: Translation invariance
    print("\n[TEST 2] Translation invariance...")
    skeleton_translated = skeleton + np.array([10, 20, 30])
    disparity = compute_procrustes_distance(skeleton, skeleton_translated, use_core_landmarks=False)
    assert disparity < 1e-10, f"Expected ~0, got {disparity}"
    print(f"  ✓ Disparity after translation: {disparity:.10f}")
    
    # Test 3: Scale invariance
    print("\n[TEST 3] Scale invariance...")
    skeleton_scaled = skeleton * 5.0
    disparity = compute_procrustes_distance(skeleton, skeleton_scaled, use_core_landmarks=False)
    assert disparity < 1e-10, f"Expected ~0, got {disparity}"
    print(f"  ✓ Disparity after scaling: {disparity:.10f}")
    
    # Test 4: Rotation invariance
    print("\n[TEST 4] Rotation invariance...")
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    skeleton_rotated = skeleton @ R
    disparity = compute_procrustes_distance(skeleton, skeleton_rotated, use_core_landmarks=False)
    assert disparity < 1e-10, f"Expected ~0, got {disparity}"
    print(f"  ✓ Disparity after rotation: {disparity:.10f}")
    
    # Test 5: Different skeletons
    print("\n[TEST 5] Different skeletons...")
    skeleton_different = np.random.rand(12, 3)
    disparity = compute_procrustes_distance(skeleton, skeleton_different, use_core_landmarks=False)
    assert disparity > 0.01, f"Expected >0.01, got {disparity}"
    print(f"  ✓ Disparity for different skeletons: {disparity:.4f}")
    
    # Test 6: Similarity score
    print("\n[TEST 6] Similarity score...")
    similarity_same = compute_procrustes_similarity(skeleton, skeleton, use_core_landmarks=False)
    similarity_diff = compute_procrustes_similarity(skeleton, skeleton_different, use_core_landmarks=False)
    assert similarity_same > 0.99, f"Expected >0.99, got {similarity_same}"
    assert similarity_diff < similarity_same, "Different should be less similar"
    print(f"  ✓ Similarity (same): {similarity_same:.4f}")
    print(f"  ✓ Similarity (different): {similarity_diff:.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MEMOTION Phase 1 Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect mode (tự động chọn GUI nếu có display)
  python main_test.py --source video.mp4
  
  # Force display mode (bắt buộc hiển thị GUI)
  python main_test.py --source video.mp4 --display
  
  # Force headless mode (không hiển thị, chỉ xử lý)
  python main_test.py --source video.mp4 --headless --output results.csv
  
  # Webcam (luôn cần display)
  python main_test.py --source webcam
  
  # Xuất video với annotations
  python main_test.py --source video.mp4 --output-video output.mp4
  
  # Unit tests only
  python main_test.py --mode test
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Input source: 'webcam', video path, or image path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image", "test"],
        default="video",
        help="Processing mode"
    )
    
    # Display mode group (mutually exclusive)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        action="store_true",
        help="Force display mode (show GUI window)"
    )
    display_group.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (no GUI, for servers)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video file with annotations"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Process every N frames (default: 1 = all frames)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing model files"
    )
    
    args = parser.parse_args()
    
    # Run unit tests
    if args.mode == "test":
        run_unit_tests()
        return
    
    # Determine display mode
    if args.display:
        use_display = True
        print("[INFO] Display mode: FORCED ON (--display)")
    elif args.headless:
        use_display = False
        print("[INFO] Display mode: FORCED OFF (--headless)")
    else:
        # Auto-detect
        use_display = check_display_available()
        if use_display:
            print("[INFO] Display mode: AUTO (display detected)")
        else:
            print("[INFO] Display mode: AUTO (no display, running headless)")
    
    # Webcam requires display
    if args.source.lower() == "webcam" and not use_display:
        print("[ERROR] Webcam mode requires display. Use --display or run on local machine.")
        return
    
    # Setup model paths
    models_dir = Path(args.models_dir)
    pose_model, face_model = download_models(models_dir)
    
    # Check if models exist
    if not Path(pose_model).exists():
        print("[WARNING] Pose model not found. Running unit tests only.")
        run_unit_tests()
        return
    
    # Initialize detector
    config = DetectorConfig(
        pose_model_path=pose_model,
        face_model_path=face_model if Path(face_model).exists() else None,
        running_mode="IMAGE" if args.mode == "image" else "VIDEO",
    )
    
    try:
        with VisionDetector(config) as detector:
            if args.source.lower() == "webcam":
                run_webcam_test(detector)
            elif args.mode == "image":
                run_image_test(detector, args.source, headless=not use_display)
            else:
                # Video mode - unified function
                run_video_process(
                    detector,
                    args.source,
                    display=use_display,
                    output_csv=args.output,
                    output_video=args.output_video,
                    sample_rate=args.sample_rate
                )
                    
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Running unit tests instead...")
        run_unit_tests()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()