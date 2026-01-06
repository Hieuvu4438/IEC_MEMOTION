"""
🚀 WebSocket Backend Server for Pose Estimation
Nhận video frames từ WebSocket, xử lý qua AI model, trả về frames với keypoints
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import traceback
from datetime import datetime

try:
    from rtmlib import Wholebody
    USE_RTMLIB = True
except ImportError:
    USE_RTMLIB = False
    print("⚠️ rtmlib not found. Install: pip install rtmlib")


# ==================== CONFIGURATION ====================
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

JOINT_ANGLES = {
    'left_elbow': (5, 7, 9),
    'right_elbow': (6, 8, 10),
    'left_shoulder': (7, 5, 11),
    'right_shoulder': (8, 6, 12),
    'left_hip': (5, 11, 13),
    'right_hip': (6, 12, 14),
    'left_knee': (11, 13, 15),
    'right_knee': (12, 14, 16),
}

# Colors (BGR)
COLORS = {
    'skeleton': (0, 255, 0),
    'keypoint': (255, 0, 0),
    'error_joint': (0, 0, 255),
}


# ==================== POSE ESTIMATOR ====================
class PoseEstimator:
    """AI Model wrapper for pose estimation using rtmlib."""
    
    def __init__(self, mode='balanced', device='cpu'):
        if not USE_RTMLIB:
            raise ImportError("rtmlib is required. Install: pip install rtmlib")
        
        print(f"🤖 Initializing Pose Estimator (mode={mode}, device={device})...")
        self.pose_tracker = Wholebody(
            to_openpose=False,
            mode=mode,
            backend='onnxruntime',
            device=device
        )
        print("✅ Pose Estimator ready!")
    
    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pose from frame.
        
        Returns:
            keypoints: Array of shape (17, 2) with x,y coordinates
            scores: Array of shape (17,) with confidence scores
        """
        try:
            keypoints, scores = self.pose_tracker(frame)
            
            if len(keypoints) > 0:
                kps = keypoints[0][:17]
                scrs = scores[0][:17] if len(scores) > 0 else np.ones(17)
                return kps, scrs.flatten()[:17]
            
            return np.zeros((17, 2)), np.zeros(17)
        except Exception as e:
            print(f"❌ Error in pose estimation: {e}")
            return np.zeros((17, 2)), np.zeros(17)


# ==================== MATH UTILS ====================
def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 between p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def compute_joint_angles(keypoints: np.ndarray) -> Dict[str, float]:
    """Compute angles for all joints."""
    angles = {}
    for joint_name, (idx_a, idx_b, idx_c) in JOINT_ANGLES.items():
        angle = calculate_angle(
            keypoints[idx_a], 
            keypoints[idx_b], 
            keypoints[idx_c]
        )
        angles[joint_name] = float(angle)
    return angles


# ==================== VISUALIZATION ====================
def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float = 0.3
) -> np.ndarray:
    """Draw skeleton and keypoints on frame."""
    img = frame.copy()
    
    # Draw skeleton lines
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if scores[idx1] < conf_threshold or scores[idx2] < conf_threshold:
            continue
        
        pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
        pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
        
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(img, pt1, pt2, COLORS['skeleton'], 3)
    
    # Draw keypoints
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score < conf_threshold:
            continue
        x, y = int(pt[0]), int(pt[1])
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), 5, COLORS['keypoint'], -1)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    
    return img


# ==================== WEBSOCKET HANDLER ====================
class PoseEstimationServer:
    """WebSocket server for real-time pose estimation."""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.pose_estimator = PoseEstimator(mode='balanced', device='cpu')
        self.clients = set()
        self.frame_count = 0
        
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        print(f"✅ New client connected: {client_id} (Total: {len(self.clients)})")
        
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'frame':
                        # Process frame
                        response = await self.process_frame(data)
                        await websocket.send(json.dumps(response))
                        
                    elif msg_type == 'ping':
                        # Health check
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                    
                    else:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Unknown message type: {msg_type}'
                        }))
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                    
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    traceback.print_exc()
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"⚠️ Client disconnected: {client_id}")
        finally:
            self.clients.remove(websocket)
            print(f"📊 Active clients: {len(self.clients)}")
    
    async def process_frame(self, data: Dict) -> Dict:
        """
        Process incoming frame with pose estimation.
        
        Args:
            data: Dictionary containing 'image' (base64 encoded)
        
        Returns:
            Dictionary with processed frame and keypoints
        """
        try:
            # Decode base64 image
            img_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image")
            
            # Run pose estimation
            keypoints, scores = self.pose_estimator.estimate(frame)
            
            # Calculate joint angles
            angles = compute_joint_angles(keypoints)
            
            # Draw skeleton on frame
            viz_frame = draw_skeleton_on_frame(frame, keypoints, scores)
            
            # Encode processed frame to base64
            _, buffer = cv2.imencode('.jpg', viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare keypoints data
            keypoints_data = []
            for i, (kp, score, name) in enumerate(zip(keypoints, scores, COCO_KEYPOINTS)):
                keypoints_data.append({
                    'id': i,
                    'name': name,
                    'x': float(kp[0]),
                    'y': float(kp[1]),
                    'confidence': float(score)
                })
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"📊 Processed {self.frame_count} frames")
            
            # Return response
            return {
                'type': 'result',
                'success': True,
                'frame': f"data:image/jpeg;base64,{img_base64}",
                'keypoints': keypoints_data,
                'angles': angles,
                'frame_count': self.frame_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            traceback.print_exc()
            return {
                'type': 'error',
                'success': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def start(self):
        """Start WebSocket server."""
        print(f"\n{'='*60}")
        print(f"🚀 Pose Estimation WebSocket Server")
        print(f"{'='*60}")
        print(f"📡 Server: ws://{self.host}:{self.port}")
        print(f"🤖 AI Model: RTMPose (rtmlib)")
        print(f"{'='*60}\n")
        print("⏳ Waiting for connections...")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


# ==================== MAIN ====================
async def main():
    """Main entry point."""
    if not USE_RTMLIB:
        print("\n❌ rtmlib is not installed!")
        print("📦 Install it with: pip install rtmlib")
        return
    
    server = PoseEstimationServer(host='localhost', port=8765)
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        traceback.print_exc()
