#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        
        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for laser scan data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state variables
        self.laser_data = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Parameters (gentler movement to prevent tipping)
        self.safe_distance = 0.8  # meters (increased for stability)
        self.max_linear_vel = 0.15  # m/s (reduced speed)
        self.max_angular_vel = 0.5  # rad/s (slower turning)
        self.acceleration_limit = 0.1  # smooth acceleration
        
        self.get_logger().info('Obstacle avoidance node started!')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def control_loop(self):
        """Main control logic"""
        if self.laser_data is None:
            return
            
        # Create twist message
        twist = Twist()
        
        # Get laser ranges and filter out invalid readings
        ranges = np.array(self.laser_data.ranges)
        ranges[ranges == float('inf')] = self.laser_data.range_max
        ranges[ranges == 0.0] = self.laser_data.range_max
        
        # Divide laser scan into regions
        front_angle = 30  # degrees
        num_readings = len(ranges)
        
        # Convert angle to array indices
        front_indices = int(front_angle * num_readings / 360)
        
        # Define regions
        front_left = ranges[:front_indices]
        front_center = ranges[num_readings//2 - front_indices//2 : num_readings//2 + front_indices//2]
        front_right = ranges[-front_indices:]
        left_side = ranges[num_readings//4 : num_readings//2 - front_indices//2]
        right_side = ranges[num_readings//2 + front_indices//2 : 3*num_readings//4]
        
        # Find minimum distances in each region
        min_front = np.min(front_center) if len(front_center) > 0 else self.laser_data.range_max
        min_front_left = np.min(front_left) if len(front_left) > 0 else self.laser_data.range_max
        min_front_right = np.min(front_right) if len(front_right) > 0 else self.laser_data.range_max
        min_left = np.min(left_side) if len(left_side) > 0 else self.laser_data.range_max
        min_right = np.min(right_side) if len(right_side) > 0 else self.laser_data.range_max
        
        # Log distances for debugging
        self.get_logger().info(f'Distances - Front: {min_front:.2f}, Left: {min_left:.2f}, Right: {min_right:.2f}')
        
        # Decision making logic with gradual movements
        if min_front > self.safe_distance:
            # Path is clear, move forward gradually
            twist.linear.x = min(self.max_linear_vel, min_front * 0.3)
            twist.angular.z = 0.0
            self.get_logger().info('Moving forward - path clear')
            
        elif min_left > min_right + 0.2:  # Add hysteresis to prevent oscillation
            # Turn left gradually
            twist.linear.x = 0.05  # Small forward motion while turning
            twist.angular.z = self.max_angular_vel * 0.7
            self.get_logger().info('Turning left')
            
        elif min_right > min_left + 0.2:
            # Turn right gradually  
            twist.linear.x = 0.05  # Small forward motion while turning
            twist.angular.z = -self.max_angular_vel * 0.7
            self.get_logger().info('Turning right')
            
        else:
            # Slow rotation when unsure
            twist.linear.x = 0.0
            twist.angular.z = 0.3
            self.get_logger().info('Slow exploration turn')
            
        # Emergency stop if too close
        if min_front < 0.2:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().warn('EMERGENCY STOP - Too close to obstacle!')
        
        # Publish movement command
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    
    obstacle_avoidance = ObstacleAvoidance()
    
    try:
        rclpy.spin(obstacle_avoidance)
    except KeyboardInterrupt:
        pass
    
    # Stop the robot before shutting down
    stop_msg = Twist()
    obstacle_avoidance.cmd_vel_pub.publish(stop_msg)
    
    obstacle_avoidance.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
