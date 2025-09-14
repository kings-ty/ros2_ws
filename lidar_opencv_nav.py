#!/usr/bin/env python3

"""
LIDAR + OpenCV Camera Navigation for Gazebo
- LIDAR for distance measurements
- OpenCV for visual obstacle detection (color, contours, depth)
- Goal-oriented navigation with RViz support
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import math

class LidarOpenCVNavigator(Node):
    def __init__(self):
        super().__init__('lidar_opencv_navigator')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        # Also listen to RViz's default topic
        self.rviz_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_image_pub = self.create_publisher(Image, '/opencv_debug', 10)
        
        # Navigation timer
        self.timer = self.create_timer(0.1, self.navigate)
        
        # Data storage
        self.lidar_data = None
        self.current_image = None
        self.robot_pos = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.goal_pos = [0.0, 2.0]  # Default goal
        
        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.goal_threshold = 0.3
        self.obstacle_threshold = 1.2  # Increased to detect obstacles earlier
        
        self.get_logger().info("🤖 LIDAR + OpenCV Navigator started!")
        self.get_logger().info("🎯 Use RViz '2D Nav Goal' to set destinations!")
        
    def lidar_callback(self, msg):
        """Process LIDAR scan data"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.lidar_data = ranges
        
    def image_callback(self, msg):
        """Process camera image with OpenCV"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            
            # Process image for obstacle detection
            processed_image = self.detect_visual_obstacles(cv_image)
            
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    
    def odom_callback(self, msg):
        """Update robot position"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def goal_callback(self, msg):
        """Handle new goal from RViz"""
        # If the goal is in base_link frame, convert to odom frame
        if msg.header.frame_id == 'base_link':
            # Transform from base_link to odom coordinates
            goal_x_odom = self.robot_pos[0] + msg.pose.position.x * math.cos(self.robot_yaw) - msg.pose.position.y * math.sin(self.robot_yaw)
            goal_y_odom = self.robot_pos[1] + msg.pose.position.x * math.sin(self.robot_yaw) + msg.pose.position.y * math.cos(self.robot_yaw)
            
            self.goal_pos[0] = goal_x_odom
            self.goal_pos[1] = goal_y_odom
            self.get_logger().info(f"🎯 목표 설정 (base_link->odom 변환): ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})")
        else:
            # Goal is already in odom or map frame
            self.goal_pos[0] = msg.pose.position.x
            self.goal_pos[1] = msg.pose.position.y
            self.get_logger().info(f"🎯 목표 설정 ({msg.header.frame_id}): ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})")
    
    def detect_visual_obstacles(self, image):
        """OpenCV-based obstacle detection for Gazebo"""
        debug_image = image.copy()
        height, width = image.shape[:2]
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Edge detection for walls and obstacles
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Color-based detection (for colored objects in Gazebo)
        # Detect bright/colored objects that aren't ground
        mask_ground = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 80]))  # Dark ground
        mask_obstacles = cv2.bitwise_not(mask_ground)
        
        # 3. Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze bottom half of image (closer obstacles)
        roi_height = height // 2
        obstacle_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Significant obstacle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if obstacle is in lower half (closer)
                if y + h > roi_height:
                    obstacle_detected = True
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(debug_image, "OBSTACLE", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 4. Depth estimation using image brightness
        bottom_region = gray[roi_height:, :]
        avg_brightness = np.mean(bottom_region)
        
        # Dark areas in front might be obstacles or holes
        if avg_brightness < 50:
            cv2.putText(debug_image, "DARK AREA - CAUTION", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 5. Draw analysis regions
        cv2.line(debug_image, (0, roi_height), (width, roi_height), (255, 255, 0), 2)
        cv2.putText(debug_image, "DANGER ZONE", (10, roi_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 6. Central path analysis
        center_x = width // 2
        center_width = 60
        center_region = edges[roi_height:, center_x-center_width:center_x+center_width]
        center_obstacle_density = np.sum(center_region) / (center_region.shape[0] * center_region.shape[1])
        
        if center_obstacle_density > 0.25:  # High edge density = obstacle (much less sensitive)
            cv2.rectangle(debug_image, (center_x-center_width, roi_height), 
                         (center_x+center_width, height), (0, 0, 255), 3)
            cv2.putText(debug_image, "PATH BLOCKED", (center_x-50, roi_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.rectangle(debug_image, (center_x-center_width, roi_height), 
                         (center_x+center_width, height), (0, 255, 0), 3)
            cv2.putText(debug_image, "PATH CLEAR", (center_x-50, roi_height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store visual obstacle info
        self.visual_obstacle_detected = obstacle_detected
        self.center_path_blocked = center_obstacle_density > 0.25
        
        # Debug log for visual detection
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 30 == 0:  # Log every 3 seconds (30 * 0.1s)
            self.get_logger().info(f"🔍 시각 감지: 밀도={center_obstacle_density:.3f}, 차단됨={self.center_path_blocked}")
        
        return debug_image
    
    def navigate(self):
        """Main navigation logic using LIDAR + OpenCV"""
        if self.lidar_data is None:
            return
            
        cmd = Twist()
        
        # Calculate distance to goal
        goal_distance = math.sqrt((self.goal_pos[0] - self.robot_pos[0])**2 + 
                                 (self.goal_pos[1] - self.robot_pos[1])**2)
        
        # Check if goal reached
        if goal_distance < self.goal_threshold:
            self.get_logger().info("🎯 Goal reached!")
            self.cmd_pub.publish(cmd)
            return
        
        # Calculate goal direction
        goal_angle = math.atan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0])
        angle_diff = goal_angle - self.robot_yaw
        
        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # LIDAR obstacle detection with better range handling
        # Filter out invalid readings (0.0, inf, nan)
        valid_ranges = self.lidar_data[(self.lidar_data > 0.1) & (self.lidar_data < 10.0)]
        if len(valid_ranges) == 0:
            return  # No valid LIDAR data
            
        # Get distances in different directions (more robust indexing)
        n_points = len(self.lidar_data)
        front_range = max(1, n_points // 10)  # 10% of points around front
        side_range = max(1, n_points // 20)   # 5% of points for sides
        
        # Front (center)
        front_indices = slice(n_points//2 - front_range, n_points//2 + front_range)
        front_readings = self.lidar_data[front_indices]
        front_readings = front_readings[(front_readings > 0.1) & (front_readings < 10.0)]
        front_distance = np.min(front_readings) if len(front_readings) > 0 else 10.0
        
        # Left (90 degrees)
        left_indices = slice(n_points//4 - side_range, n_points//4 + side_range)
        left_readings = self.lidar_data[left_indices]
        left_readings = left_readings[(left_readings > 0.1) & (left_readings < 10.0)]
        left_distance = np.min(left_readings) if len(left_readings) > 0 else 10.0
        
        # Right (270 degrees)
        right_indices = slice(3*n_points//4 - side_range, 3*n_points//4 + side_range)
        right_readings = self.lidar_data[right_indices]
        right_readings = right_readings[(right_readings > 0.1) & (right_readings < 10.0)]
        right_distance = np.min(right_readings) if len(right_readings) > 0 else 10.0
        
        # LIDAR-only navigation (ignore visual completely for now)
        obstacle_detected = front_distance < self.obstacle_threshold
        
        if obstacle_detected:
            # LIDAR obstacle - avoid it
            if left_distance > right_distance and left_distance > 1.0:
                cmd.angular.z = self.angular_speed  # Turn left
                cmd.linear.x = self.linear_speed * 0.1  # Move very slowly while turning
                self.get_logger().info(f"🚧 LIDAR 장애물 - 왼쪽으로 회피 (앞:{front_distance:.2f}m)")
            elif right_distance > 1.0:
                cmd.angular.z = -self.angular_speed  # Turn right
                cmd.linear.x = self.linear_speed * 0.1  # Move very slowly while turning
                self.get_logger().info(f"🚧 LIDAR 장애물 - 오른쪽으로 회피 (앞:{front_distance:.2f}m)")
            else:
                # Both sides blocked - just turn in place
                cmd.angular.z = self.angular_speed
                self.get_logger().info("🚧 양쪽 차단됨 - 제자리 회전")
        
        elif abs(angle_diff) > 0.2:
            # Need to turn toward goal
            cmd.angular.z = self.angular_speed * 0.8 if angle_diff > 0 else -self.angular_speed * 0.8
            cmd.linear.x = self.linear_speed * 0.3  # Move slowly while turning
            self.get_logger().info(f"🎯 목표 방향으로 회전 (각도차: {math.degrees(angle_diff):.1f}°)")
        
        else:
            # Clear path - move toward goal
            speed_factor = min(front_distance / 2.0, 1.0)  # Slow down near obstacles
            cmd.linear.x = self.linear_speed * max(speed_factor, 0.3)  # Minimum speed
            
            # Minor course correction toward goal
            if abs(angle_diff) > 0.05:
                cmd.angular.z = angle_diff * 1.0  # More responsive turning
            
            self.get_logger().info(f"🚀 목표로 이동 (거리: {goal_distance:.2f}m, 속도: {cmd.linear.x:.2f})")
        
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = LidarOpenCVNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()