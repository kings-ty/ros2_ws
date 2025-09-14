#!/usr/bin/env python3

"""
LIDAR-Only DQN Agent for Testing
- No camera dependency
- Pure LIDAR-based navigation
- Works with current simulation environment
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class SimpleDQN(nn.Module):
    def __init__(self, input_size=26, action_size=7):  # 24 lidar + 2 goal info
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class LidarDQNAgent(Node):
    def __init__(self):
        super().__init__('lidar_dqn_agent')
        
        # Network
        self.input_size = 26
        self.action_size = 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = SimpleDQN(self.input_size, self.action_size).to(self.device)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # State variables
        self.lidar_data = np.zeros(24)
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.goal_pos = np.array([0.0, 2.0])  # Realistic goal - move forward 2.5m
        
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
        
        self.get_logger().info("🤖 LIDAR-Only DQN Agent started!")
        self.get_logger().info(f"🎯 Current goal: ({self.goal_pos[0]}, {self.goal_pos[1]})")
        self.get_logger().info("💡 Use RViz '2D Nav Goal' tool to set new goals by clicking on the map!")
        
        # Publish initial goal marker
        self.publish_goal_marker()
        
    def lidar_callback(self, msg):
        if len(msg.ranges) > 0:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)
            ranges = np.clip(ranges, 0.1, 10.0)
            
            # Sample 24 readings
            indices = np.linspace(0, len(ranges)-1, 24).astype(int)
            self.lidar_data = ranges[indices] / 10.0  # Normalize
    
    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        # Calculate yaw
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def goal_callback(self, msg):
        """Handle new goal from RViz 2D Nav Goal"""
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        self.goal_pos[0] = new_x
        self.goal_pos[1] = new_y
        
        distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        self.get_logger().info(
            f"🎯 NEW GOAL SET: ({new_x:.2f}, {new_y:.2f}), "
            f"Distance: {distance:.2f}m from current position"
        )
        self.publish_goal_marker()
    
    def publish_goal_marker(self):
        """Publish goal visualization marker for RViz"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = float(self.goal_pos[0])
        marker.pose.position.y = float(self.goal_pos[1])
        marker.pose.position.z = 0.5  # Raise above ground
        marker.pose.orientation.w = 1.0
        
        # Set size
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Set color (bright green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)
    
    def get_state(self):
        # Goal information
        goal_distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        # Combine LIDAR + goal info
        state = np.concatenate([
            self.lidar_data,
            [goal_distance / 10.0],  # Normalized distance
            [goal_angle / np.pi]     # Normalized angle
        ])
        return state
    
    def select_action(self):
        state = self.get_state()
        ranges = self.lidar_data * 10.0  # Convert back to meters
        
        # Goal direction (in robot frame)
        goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        # Convert goal angle to LIDAR index (24 sensors, -π to π)
        goal_index = int((goal_angle + np.pi) / (2 * np.pi) * 24) % 24
        
        # LIDAR sectors: front, left, right
        front_sector = ranges[10:14]  # Front 4 sensors
        left_sector = ranges[6:10]    # Left sector
        right_sector = ranges[14:18]  # Right sector
        
        min_front = np.min(front_sector)
        min_left = np.min(left_sector)
        min_right = np.min(right_sector)
        min_overall = np.min(ranges)
        
        # Emergency stop for very close obstacles
        if min_overall < 0.15:
            return 0  # Stop
        
        # Critical obstacle avoidance (very close)
        if min_front < 0.3:
            # Choose side with more space
            if min_left > min_right:
                return 4  # Turn left
            else:
                return 5  # Turn right
        
        # Smart navigation: try to go toward goal while avoiding obstacles
        goal_clear = ranges[max(0, min(23, goal_index))] > 0.8  # Is goal direction clear?
        
        if goal_clear and min_front > 0.5:
            # Goal direction is clear - go forward toward goal
            if abs(goal_angle) < 0.2:  # Already facing goal
                return 1  # Forward
            elif goal_angle > 0:  # Goal is to the left
                return 2  # Forward + Left
            else:  # Goal is to the right
                return 3  # Forward + Right
        
        # Goal direction blocked - find the clearest path
        # Evaluate different directions
        directions = []
        for i in range(24):
            angle = -np.pi + (i / 24) * 2 * np.pi
            distance = ranges[i]
            # Prefer directions closer to goal
            goal_preference = 1.0 - abs(angle - goal_angle) / np.pi
            score = distance * 0.7 + goal_preference * 0.3
            directions.append((i, angle, distance, score))
        
        # Sort by score (higher is better)
        directions.sort(key=lambda x: x[3], reverse=True)
        
        # Choose best direction that's safe enough
        for i, angle, distance, score in directions:
            if distance > 0.6:  # Safe distance
                if abs(angle) < 0.3:  # Front-ish
                    return 1  # Forward
                elif angle > 0.8:  # Left
                    return 4  # Turn left
                elif angle < -0.8:  # Right
                    return 5  # Turn right
                elif angle > 0:  # Forward left
                    return 2  # Forward + Left
                else:  # Forward right
                    return 3  # Forward + Right
        
        # If no good direction found, just turn toward the most open space
        if min_left > min_right:
            return 4  # Turn left
        else:
            return 5  # Turn right
    
    def control_loop(self):
        action = self.select_action()
        linear_vel, angular_vel = self.actions[action]
        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
        
        self.step_count += 1
        
        if self.step_count % 50 == 0:  # Log every 5 seconds
            min_dist = np.min(self.lidar_data * 10.0)
            goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)
            goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                                   self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
            goal_angle = np.degrees(np.arctan2(np.sin(goal_angle), np.cos(goal_angle)))
            
            action_names = ["STOP", "FORWARD", "FWD+LEFT", "FWD+RIGHT", "LEFT", "RIGHT", "BACK"]
            
            self.get_logger().info(
                f"Step: {self.step_count}, Action: {action_names[action]}, "
                f"Min obstacle: {min_dist:.2f}m, Goal: {goal_dist:.2f}m, "
                f"Goal angle: {goal_angle:.1f}°, Pos: ({self.robot_pos[0]:.1f},{self.robot_pos[1]:.1f})"
            )
            
            # Check if goal reached
            if goal_dist < 0.5:  # Within 0.5m of goal
                self.get_logger().info("🎉 GOAL REACHED! Set a new goal with RViz 2D Nav Goal")
                # Stop the robot
                stop_twist = Twist()
                self.cmd_vel_pub.publish(stop_twist)

def main(args=None):
    rclpy.init(args=args)
    agent = LidarDQNAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("🛑 Stopping agent...")
    finally:
        # Stop robot
        twist = Twist()
        agent.cmd_vel_pub.publish(twist)
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()