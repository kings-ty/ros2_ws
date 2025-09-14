#!/usr/bin/env python3

"""
Simple YOLO + Robot Demo
Run this directly to test YOLO with your robot
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class SimpleYOLODemo(Node):
    def __init__(self):
        super().__init__('simple_yolo_demo')
        
        # Initialize YOLO
        try:
            self.model = YOLO('yolov8n.pt')
            self.get_logger().info("✅ YOLO model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"❌ YOLO model failed to load: {e}")
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Demo parameters
        self.detection_count = 0
        
        self.get_logger().info("🤖 Simple YOLO Demo started!")
        self.get_logger().info("📷 Listening for camera images...")
    
    def image_callback(self, msg):
        """Process camera images with YOLO"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO detection
            results = self.model(cv_image, conf=0.5)
            
            # Process results
            if results[0].boxes is not None:
                boxes = results[0].boxes
                self.detection_count += len(boxes)
                
                # Log detections
                detected_objects = []
                for i, box in enumerate(boxes):
                    class_id = int(box.cls.cpu().numpy())
                    confidence = float(box.conf.cpu().numpy())
                    class_name = self.model.names[class_id]
                    detected_objects.append(f"{class_name}({confidence:.2f})")
                
                if detected_objects:
                    self.get_logger().info(f"🎯 Detected: {', '.join(detected_objects)}")
                    
                    # Simple robot behavior based on detections
                    self.simple_navigation(detected_objects)
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO processing: {e}")
    
    def simple_navigation(self, objects):
        """Simple navigation based on detected objects"""
        twist = Twist()
        
        # Simple logic: stop if person detected, otherwise move forward
        if any("person" in obj for obj in objects):
            # Stop if person detected
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("🚶 Person detected - Stopping!")
        else:
            # Move forward slowly
            twist.linear.x = 0.2
            twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        demo = SimpleYOLODemo()
        rclpy.spin(demo)
    except KeyboardInterrupt:
        print("🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()