#!/usr/bin/env python3

"""
ACTUALLY WORKING Robot Demo
Step 1: Just make the robot move - no fancy stuff yet!
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

class SimpleRobotController(Node):
    def __init__(self):
        super().__init__('simple_robot_controller')
        
        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for laser data
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        # Simple timer for movement
        self.timer = self.create_timer(1.0, self.move_robot)
        
        # Robot state
        self.obstacle_detected = False
        self.step_count = 0
        
        print("🤖 Simple Robot Controller Started!")
        print("🎯 Robot will move forward and avoid obstacles")
        
    def laser_callback(self, msg):
        """Check if there's an obstacle in front"""
        if len(msg.ranges) > 0:
            # Check front 60 degrees
            front_ranges = msg.ranges[len(msg.ranges)//2-30:len(msg.ranges)//2+30]
            if len(front_ranges) > 0:
                min_distance = min(front_ranges)
                self.obstacle_detected = min_distance < 0.8
    
    def move_robot(self):
        """Simple movement logic"""
        twist = Twist()
        self.step_count += 1
        
        if self.obstacle_detected:
            # Turn right when obstacle detected
            twist.linear.x = 0.0
            twist.angular.z = -0.5
            print(f"🔄 Step {self.step_count}: TURNING - Obstacle detected!")
        else:
            # Move forward when clear
            twist.linear.x = 0.3
            twist.angular.z = 0.0
            print(f"🚀 Step {self.step_count}: FORWARD - Path clear!")
        
        # Publish the movement
        self.cmd_vel_pub.publish(twist)
        
        # Add some variety - occasionally turn even when clear
        if self.step_count % 10 == 0 and not self.obstacle_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.3
            self.cmd_vel_pub.publish(twist)
            print(f"🌟 Step {self.step_count}: EXPLORING - Random turn!")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = SimpleRobotController()
        print("✅ Controller created successfully!")
        print("📡 Waiting for laser data...")
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()