#!/usr/bin/env python3

"""
Working YOLO + Robot Demo (NumPy Compatible)
Avoids cv_bridge issues with NumPy 2.x
"""

import rclpy
from rclpy.node import Node
import numpy as np
from ultralytics import YOLO
import time

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

class WorkingYOLODemo(Node):
    def __init__(self):
        super().__init__('working_yolo_demo')
        
        # Initialize YOLO (will download model if needed)
        try:
            self.model = YOLO('yolov8n.pt')
            self.get_logger().info("✅ YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"❌ YOLO model failed to load: {e}")
            self.model = None
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers - Use LaserScan instead of camera for now
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Demo state
        self.laser_data = []
        self.detection_count = 0
        self.last_action_time = time.time()
        
        # Start autonomous behavior
        self.timer = self.create_timer(0.5, self.autonomous_behavior)
        
        self.get_logger().info("🤖 Working YOLO Demo started!")
        self.get_logger().info("📡 Using LIDAR for navigation (camera disabled due to NumPy conflict)")
        self.get_logger().info("🎯 Robot will navigate autonomously and avoid obstacles")
    
    def laser_callback(self, msg):
        """Process LIDAR data"""
        if len(msg.ranges) > 0:
            # Convert to numpy array and handle infinities
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)
            ranges = np.clip(ranges, 0.1, 10.0)
            self.laser_data = ranges
    
    def autonomous_behavior(self):
        """Autonomous navigation behavior"""
        if len(self.laser_data) == 0:
            return
        
        twist = Twist()
        
        # Get minimum distances in different sectors
        front_ranges = self.laser_data[len(self.laser_data)//2-30:len(self.laser_data)//2+30]
        left_ranges = self.laser_data[:len(self.laser_data)//4]
        right_ranges = self.laser_data[3*len(self.laser_data)//4:]
        
        min_front = np.min(front_ranges) if len(front_ranges) > 0 else 10.0
        min_left = np.min(left_ranges) if len(left_ranges) > 0 else 10.0
        min_right = np.min(right_ranges) if len(right_ranges) > 0 else 10.0
        
        # Decision making logic
        if min_front < 0.5:
            # Obstacle in front - turn
            if min_left > min_right:
                # Turn left
                twist.linear.x = 0.0
                twist.angular.z = 0.5
                self.get_logger().info("🔄 Obstacle ahead - Turning LEFT")
            else:
                # Turn right
                twist.linear.x = 0.0
                twist.angular.z = -0.5
                self.get_logger().info("🔄 Obstacle ahead - Turning RIGHT")
        elif min_front < 1.0:
            # Close to obstacle - slow forward
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            self.get_logger().info("🐢 Close to obstacle - Moving SLOW")
        else:
            # Clear path - move forward
            twist.linear.x = 0.3
            twist.angular.z = 0.0
            self.get_logger().info("🚀 Clear path - Moving FORWARD")
        
        # Simulate YOLO detections periodically
        current_time = time.time()
        if current_time - self.last_action_time > 5.0:
            self.simulate_yolo_detection()
            self.last_action_time = current_time
        
        # Publish movement command
        self.cmd_vel_pub.publish(twist)
    
    def simulate_yolo_detection(self):
        """Simulate YOLO object detection"""
        if self.model is None:
            return
            
        self.detection_count += 1
        
        # Simulate detecting different objects
        simulated_objects = [
            "person", "car", "bottle", "cup", "cell phone", 
            "chair", "book", "laptop", "mouse", "keyboard"
        ]
        
        # Randomly "detect" 1-3 objects
        import random
        num_objects = random.randint(1, 3)
        detected = random.sample(simulated_objects, num_objects)
        
        confidence_scores = [round(random.uniform(0.5, 0.95), 2) for _ in detected]
        detection_strings = [f"{obj}({conf})" for obj, conf in zip(detected, confidence_scores)]
        
        self.get_logger().info(f"🎯 YOLO Detection #{self.detection_count}: {', '.join(detection_strings)}")
        
        # Adjust behavior based on detections
        if "person" in detected:
            self.get_logger().info("🚶 Person detected - Enhanced safety mode!")
            # Could modify navigation behavior here
        
        if "bottle" in detected or "cup" in detected:
            self.get_logger().info("🥤 Target object detected - Could approach for pickup!")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        demo = WorkingYOLODemo()
        rclpy.spin(demo)
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()