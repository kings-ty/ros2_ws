#!/usr/bin/env python3

"""
LIDAR + YOLO DQN Agent
- LIDAR for obstacle avoidance
- YOLO for object recognition and smart navigation
- Interactive goal setting via RViz
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import time
import cv2

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available, running LIDAR-only mode")

class MultiModalDQN(nn.Module):
    def __init__(self, lidar_size=24, vision_size=45, action_size=7):
        super(MultiModalDQN, self).__init__()
        
        # LIDAR encoder
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_size + 2, 64),  # +2 for goal info
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Vision encoder (YOLO features)
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Decision network  
        self.yolo_available = YOLO_AVAILABLE
        fusion_size = 128 if self.yolo_available else 64
        self.decision_net = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, lidar_input, vision_input=None):
        lidar_features = self.lidar_encoder(lidar_input)
        
        if self.yolo_available and vision_input is not None:
            vision_features = self.vision_encoder(vision_input)
            combined = torch.cat([lidar_features, vision_features], dim=1)
        else:
            combined = lidar_features
            
        return self.decision_net(combined)

class LidarYoloDQNAgent(Node):
    def __init__(self):
        super().__init__('lidar_yolo_dqn_agent')
        
        # Initialize YOLO if available
        self.yolo_available = YOLO_AVAILABLE
        if self.yolo_available:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.bridge = CvBridge()
                self.get_logger().info("🎯 YOLO initialized successfully!")
            except Exception as e:
                self.yolo_available = False
                self.get_logger().warn(f"YOLO failed to load: {e}")
        
        # Network
        self.lidar_size = 24
        self.vision_size = 45
        self.action_size = 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = MultiModalDQN(self.lidar_size, self.vision_size, self.action_size).to(self.device)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        self.yolo_detections_pub = self.create_publisher(Float32MultiArray, '/yolo/detections', 10)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        if self.yolo_available:
            self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
            # Camera image publisher for visualization
            self.camera_viz_pub = self.create_publisher(Image, '/camera_viz', 10)
        
        # State variables
        self.lidar_data = np.zeros(self.lidar_size)
        self.vision_data = np.zeros(self.vision_size)
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.goal_pos = np.array([0.0, 2.0])  # Default goal
        
        # YOLO object categories (expanded for better detection)
        self.target_objects = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        # Actions
        self.actions = {
            0: [0.0, 0.0],      # Stop
            1: [0.2, 0.0],      # Forward
            2: [0.1, 0.5],      # Forward + Left
            3: [0.1, -0.5],     # Forward + Right
            4: [0.0, 1.0],      # Rotate Left
            5: [0.0, -1.0],     # Rotate Right
            6: [-0.1, 0.0],     # Backward
        }
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.step_count = 0
        
        mode = "LIDAR + YOLO" if self.yolo_available else "LIDAR Only"
        self.get_logger().info(f"🤖 {mode} DQN Agent started!")
        self.get_logger().info(f"🎯 Current goal: ({self.goal_pos[0]}, {self.goal_pos[1]})")
        self.get_logger().info("💡 Use RViz '2D Nav Goal' tool to set new goals!")
        
        # Publish initial goal marker
        self.publish_goal_marker()
        
    def lidar_callback(self, msg):
        if len(msg.ranges) > 0:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)
            ranges = np.clip(ranges, 0.1, 10.0)
            
            # Debug: Print first few readings
            if self.step_count % 100 == 0:  # Every 10 seconds
                self.get_logger().info(f"🔍 LIDAR Debug: Min={np.min(ranges):.2f}m, Max={np.max(ranges):.2f}m, Avg={np.mean(ranges):.2f}m")
            
            # Sample evenly
            indices = np.linspace(0, len(ranges)-1, self.lidar_size).astype(int)
            self.lidar_data = ranges[indices] / 10.0  # Normalize
    
    def image_callback(self, msg):
        """Process camera images with YOLO"""
        if not self.yolo_available:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.yolo_model(cv_image, conf=0.5, verbose=False)
            
            # Create annotated image for visualization
            annotated_image = cv_image.copy()
            
            # Process detections
            detection_array = np.zeros(self.vision_size)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                height, width = cv_image.shape[:2]
                detection_count = 0
                
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    if cls in self.target_objects and detection_count < 5:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box on annotated image
                        color = (0, 255, 0) if cls in [39, 67] else (0, 0, 255)  # Green for targets, Red for obstacles
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{self.target_objects[cls]}: {score:.2f}"
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Process for AI
                        center_x = (x1 + x2) / (2 * width)  # Normalized
                        center_y = (y1 + y2) / (2 * height)
                        obj_width = (x2 - x1) / width
                        obj_height = (y2 - y1) / height
                        
                        # Estimate distance (rough)
                        distance = 3.0 - (obj_width * obj_height * 10)  # Inverse size
                        distance = max(0.5, min(distance, 5.0))
                        
                        # Object type classification
                        is_obstacle = 1.0 if cls in [0, 1, 2, 3] else 0.0  # person, bike, car, motorcycle
                        is_target = 1.0 if cls in [39, 67] else 0.0  # bottle, phone
                        
                        base_idx = detection_count * 9
                        detection_array[base_idx:base_idx+9] = [
                            center_x, center_y, obj_width, obj_height,
                            score, cls, distance, is_obstacle, is_target
                        ]
                        detection_count += 1
            
            self.vision_data = detection_array
            
            # Publish annotated image for visualization
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                annotated_msg.header = msg.header
                self.camera_viz_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish annotated image: {e}")
            
            # Publish for debugging
            detection_msg = Float32MultiArray()
            detection_msg.data = detection_array.tolist()
            self.yolo_detections_pub.publish(detection_msg)
            
        except Exception as e:
            self.get_logger().error(f"YOLO processing error: {e}")
    
    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        # Calculate yaw
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def goal_callback(self, msg):
        """Handle new goal from RViz"""
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        self.goal_pos[0] = new_x
        self.goal_pos[1] = new_y
        
        distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        self.get_logger().info(f"🎯 NEW GOAL: ({new_x:.2f}, {new_y:.2f}), Distance: {distance:.2f}m")
        self.publish_goal_marker()
    
    def publish_goal_marker(self):
        """Publish goal visualization marker"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.goal_pos[0])
        marker.pose.position.y = float(self.goal_pos[1])
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)
    
    def select_action(self):
        """Smart action selection using LIDAR + YOLO with emergency handling"""
        ranges = self.lidar_data * 10.0
        
        # Check for valid LIDAR data
        if np.max(ranges) < 0.1:  # All readings are essentially zero
            # Emergency mode: no LIDAR data, use vision-only navigation
            self.get_logger().warn("⚠️ No LIDAR data! Using vision-only emergency mode")
            return self.vision_emergency_action()
        
        # Goal direction
        goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        # LIDAR-based obstacle avoidance (primary)
        min_distance = np.min(ranges)
        front_distance = np.mean(ranges[10:14])
        
        # YOLO-based object awareness (secondary)
        vision_penalty = 0
        if self.yolo_available:
            vision_reshaped = self.vision_data.reshape(5, 9)
            for obj_data in vision_reshaped:
                if obj_data[4] > 0.5:  # Confidence > 0.5
                    is_obstacle = obj_data[7]
                    distance = obj_data[6]
                    if is_obstacle > 0.5 and distance < 2.0:
                        vision_penalty += (2.0 - distance) * 0.5
        
        # Emergency stop
        if min_distance < 0.15:
            return 0
        
        # Critical avoidance with vision input
        if min_distance < (0.3 + vision_penalty):
            left_clear = np.mean(ranges[6:10]) > np.mean(ranges[14:18])
            return 4 if left_clear else 5
        
        # Smart goal-oriented navigation
        goal_clear = ranges[max(0, min(23, int((goal_angle + np.pi) / (2 * np.pi) * 24)))] > 0.8
        
        if goal_clear and front_distance > 0.5:
            if abs(goal_angle) < 0.2:
                return 1  # Forward
            elif goal_angle > 0:
                return 2  # Forward + Left
            else:
                return 3  # Forward + Right
        
        # Find best direction
        best_score = -1
        best_action = 4
        
        for i in range(24):
            angle = -np.pi + (i / 24) * 2 * np.pi
            distance = ranges[i]
            goal_preference = 1.0 - abs(angle - goal_angle) / np.pi
            score = distance * 0.7 + goal_preference * 0.3
            
            if score > best_score and distance > 0.6:
                best_score = score
                if abs(angle) < 0.3:
                    best_action = 1
                elif angle > 0.8:
                    best_action = 4
                elif angle < -0.8:
                    best_action = 5
                elif angle > 0:
                    best_action = 2
                else:
                    best_action = 3
        
        return best_action
    
    def vision_emergency_action(self):
        """Emergency navigation using only vision data"""
        if not self.yolo_available:
            # No sensors at all - cautious exploration
            self.get_logger().warn("🚨 CRITICAL: No sensors available! Cautious mode")
            return 4  # Just turn slowly
        
        # Use YOLO data for emergency navigation
        vision_reshaped = self.vision_data.reshape(5, 9)
        
        # Check for immediate obstacles in camera view
        immediate_obstacles = []
        for obj_data in vision_reshaped:
            if obj_data[4] > 0.5:  # Confidence > 0.5
                is_obstacle = obj_data[7]
                center_x = obj_data[0]  # Normalized center X
                distance = obj_data[6]
                
                if is_obstacle > 0.5 and distance < 2.0:
                    immediate_obstacles.append((center_x, distance))
        
        if immediate_obstacles:
            # Avoid obstacles seen in camera
            avg_obstacle_x = np.mean([obs[0] for obs in immediate_obstacles])
            if avg_obstacle_x > 0.5:  # Obstacles on right side
                return 4  # Turn left
            else:  # Obstacles on left side
                return 5  # Turn right
        else:
            # No immediate obstacles seen - move forward slowly
            return 1  # Forward
    
    def control_loop(self):
        action = self.select_action()
        linear_vel, angular_vel = self.actions[action]
        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
        
        self.step_count += 1
        
        if self.step_count % 50 == 0:
            min_dist = np.min(self.lidar_data * 10.0)
            goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)
            goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                                   self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
            goal_angle = np.degrees(np.arctan2(np.sin(goal_angle), np.cos(goal_angle)))
            
            action_names = ["STOP", "FORWARD", "FWD+LEFT", "FWD+RIGHT", "LEFT", "RIGHT", "BACK"]
            
            # Count detected objects
            objects_detected = 0
            if self.yolo_available:
                vision_reshaped = self.vision_data.reshape(5, 9)
                objects_detected = sum(1 for obj in vision_reshaped if obj[4] > 0.5)
            
            mode_info = f", Objects: {objects_detected}" if self.yolo_available else ""
            
            self.get_logger().info(
                f"Step: {self.step_count}, Action: {action_names[action]}, "
                f"Min obstacle: {min_dist:.2f}m, Goal: {goal_dist:.2f}m, "
                f"Goal angle: {goal_angle:.1f}°{mode_info}"
            )
            
            if goal_dist < 0.5:
                self.get_logger().info("🎉 GOAL REACHED! Set new goal with RViz")
                stop_twist = Twist()
                self.cmd_vel_pub.publish(stop_twist)

def main(args=None):
    rclpy.init(args=args)
    agent = LidarYoloDQNAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("🛑 Shutting down...")
    finally:
        stop_twist = Twist()
        agent.cmd_vel_pub.publish(stop_twist)
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()