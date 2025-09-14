#!/usr/bin/env python3

"""
Simple Test Environment Validator
- Shows what robot actually sees vs what RViz shows
- Helps debug map vs reality mismatches
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan

class EnvironmentValidator(Node):
    def __init__(self):
        super().__init__('env_validator')
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        self.get_logger().info("🔍 Environment Validator started!")
        self.get_logger().info("📡 Analyzing real-time LIDAR data...")
        
    def lidar_callback(self, msg):
        if len(msg.ranges) > 0:
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)
            
            # Analyze the environment
            min_dist = np.min(ranges)
            max_dist = np.max(ranges)
            avg_dist = np.mean(ranges)
            
            # Count obstacles in different sectors
            front = ranges[len(ranges)//2-10:len(ranges)//2+10]  # Front sector
            left = ranges[len(ranges)//4:len(ranges)//2-10]      # Left sector  
            right = ranges[len(ranges)//2+10:3*len(ranges)//4]   # Right sector
            
            front_obstacles = np.sum(front < 2.0)
            left_obstacles = np.sum(left < 2.0)
            right_obstacles = np.sum(right < 2.0)
            
            self.get_logger().info(
                f"🌍 REAL Environment: Min:{min_dist:.2f}m, Avg:{avg_dist:.2f}m, Max:{max_dist:.2f}m"
            )
            self.get_logger().info(
                f"🚧 Obstacles nearby: Front:{front_obstacles}, Left:{left_obstacles}, Right:{right_obstacles}"
            )
            
            # Detect major changes
            if min_dist < 0.5:
                self.get_logger().warn("⚠️  Very close obstacle detected!")
            
            if avg_dist < 1.0:
                self.get_logger().warn("⚠️  Dense obstacle environment!")

def main(args=None):
    rclpy.init(args=args)
    validator = EnvironmentValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()