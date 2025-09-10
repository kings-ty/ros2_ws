#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

class TurtleBot4DQNEnv(Node):
    def __init__(self):
        super().__init__('turtlebot4_dqn_env')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        
        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        
        # Service clients
        self.reset_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # Environment variables
        self.scan_data = []
        self.robot_pos_x = 0.0
        self.robot_pos_y = 0.0
        self.robot_yaw = 0.0
        self.goal_x = 3.0
        self.goal_y = 3.0
        self.previous_distance = 0.0
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # DQN parameters
        self.state_size = 26  # 24 laser readings + 2 goal info
        self.action_size = 5  # Forward, left, right, forward-left, forward-right
        self.min_range = 0.13
        self.max_range = 3.5
        
        # Reward parameters
        self.goal_distance_threshold = 0.3
        self.collision_threshold = 0.13
        
        self.get_logger().info("TurtleBot4 DQN Environment initialized!")
        
    def scan_callback(self, msg):
        """Process laser scan data"""
        if len(msg.ranges) > 0:
            # Sample 24 readings from 360 laser points
            scan_range = []
            for i in range(24):
                idx = int(len(msg.ranges) * i / 24)
                if msg.ranges[idx] == float('Inf') or msg.ranges[idx] == float('inf'):
                    scan_range.append(self.max_range)
                elif np.isnan(msg.ranges[idx]) or msg.ranges[idx] < self.min_range:
                    scan_range.append(self.min_range)
                else:
                    scan_range.append(msg.ranges[idx])
            self.scan_data = scan_range
    
    def odom_callback(self, msg):
        """Process odometry data"""
        self.robot_pos_x = msg.pose.pose.position.x
        self.robot_pos_y = msg.pose.pose.position.y
        
        # Calculate yaw from quaternion
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def get_state(self):
        """Get current state for DQN"""
        if len(self.scan_data) == 0:
            return np.zeros(self.state_size)
        
        # Normalize laser data
        scan_normalized = [(x - self.min_range) / (self.max_range - self.min_range) 
                          for x in self.scan_data]
        
        # Goal information
        goal_distance = math.sqrt((self.goal_x - self.robot_pos_x)**2 + 
                                 (self.goal_y - self.robot_pos_y)**2)
        goal_angle = math.atan2(self.goal_y - self.robot_pos_y, 
                               self.goal_x - self.robot_pos_x) - self.robot_yaw
        
        # Normalize goal distance and angle
        goal_distance_normalized = goal_distance / 10.0  # Assuming max distance of 10m
        goal_angle_normalized = goal_angle / math.pi
        
        state = scan_normalized + [goal_distance_normalized, goal_angle_normalized]
        return np.array(state, dtype=np.float32)
    
    def take_action(self, action):
        """Execute action"""
        twist = Twist()
        
        if action == 0:  # Forward
            twist.linear.x = 0.3
            twist.angular.z = 0.0
        elif action == 1:  # Left
            twist.linear.x = 0.0
            twist.angular.z = 0.3
        elif action == 2:  # Right
            twist.linear.x = 0.0
            twist.angular.z = -0.3
        elif action == 3:  # Forward-Left
            twist.linear.x = 0.2
            twist.angular.z = 0.2
        elif action == 4:  # Forward-Right
            twist.linear.x = 0.2
            twist.angular.z = -0.2
        
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)  # Small delay for action execution
    
    def calculate_reward(self):
        """Calculate reward for current state"""
        reward = 0
        done = False
        
        # Goal distance
        current_distance = math.sqrt((self.goal_x - self.robot_pos_x)**2 + 
                                    (self.goal_y - self.robot_pos_y)**2)
        
        # Reward for getting closer to goal
        if current_distance < self.previous_distance:
            reward += 10 * (self.previous_distance - current_distance)
        else:
            reward -= 5 * (current_distance - self.previous_distance)
        
        self.previous_distance = current_distance
        
        # Check for goal reached
        if current_distance < self.goal_distance_threshold:
            reward += 200
            done = True
            self.get_logger().info("Goal reached!")
        
        # Check for collision
        if len(self.scan_data) > 0 and min(self.scan_data) < self.collision_threshold:
            reward -= 100
            done = True
            self.get_logger().info("Collision detected!")
        
        # Check for timeout
        if self.episode_step >= self.max_episode_steps:
            reward -= 50
            done = True
            self.get_logger().info("Episode timeout!")
        
        # Small negative reward for each step to encourage efficiency
        reward -= 1
        
        return reward, done
    
    def reset_environment(self):
        """Reset robot position and goal"""
        # Stop robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        # Reset episode variables
        self.episode_step = 0
        
        # Set random start position
        start_x = random.uniform(-2.0, 2.0)
        start_y = random.uniform(-2.0, 2.0)
        start_yaw = random.uniform(-math.pi, math.pi)
        
        # Set random goal position
        self.goal_x = random.uniform(-3.0, 3.0)
        self.goal_y = random.uniform(-3.0, 3.0)
        
        # Reset robot position using Gazebo service
        entity_state = EntityState()
        entity_state.name = "turtlebot4"
        entity_state.pose.position.x = start_x
        entity_state.pose.position.y = start_y
        entity_state.pose.position.z = 0.01
        entity_state.pose.orientation.w = math.cos(start_yaw/2)
        entity_state.pose.orientation.z = math.sin(start_yaw/2)
        
        # Wait for service and call it
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')
        
        request = SetEntityState.Request()
        request.state = entity_state
        
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        time.sleep(1.0)  # Wait for reset to complete
        
        # Update previous distance
        self.previous_distance = math.sqrt((self.goal_x - self.robot_pos_x)**2 + 
                                          (self.goal_y - self.robot_pos_y)**2)
        
        self.get_logger().info(f"Environment reset! Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        return self.get_state()
    
    def step(self, action):
        """Perform one step in the environment"""
        self.episode_step += 1
        
        # Execute action
        self.take_action(action)
        
        # Wait a bit for the action to take effect
        time.sleep(0.2)
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward, done = self.calculate_reward()
        
        return next_state, reward, done

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBot4DQNEnv()
    
    try:
        rclpy.spin(env)
    except KeyboardInterrupt:
        pass
    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()