#!/usr/bin/env python3

"""
Simple Camera Viewer with YOLO annotations
Shows camera feed in OpenCV window
"""

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        self.bridge = CvBridge()
        
        # Subscribe to both original and processed camera
        self.raw_sub = self.create_subscription(Image, '/camera/image_raw', self.raw_callback, 10)
        self.viz_sub = self.create_subscription(Image, '/camera_viz', self.viz_callback, 10)
        
        self.get_logger().info("📹 Camera Viewer started!")
        self.get_logger().info("Press 'q' to quit, 'r' for raw camera, 'y' for YOLO processed")
        
        self.current_mode = "raw"
        
    def raw_callback(self, msg):
        if self.current_mode == "raw":
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.putText(cv_image, "Raw Camera Feed", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(cv_image, "Press 'y' for YOLO view", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("TurtleBot Camera", cv_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rclpy.shutdown()
                elif key == ord('y'):
                    self.current_mode = "yolo"
                    self.get_logger().info("Switched to YOLO view")
                elif key == ord('r'):
                    self.current_mode = "raw"
                    self.get_logger().info("Switched to raw camera view")
                    
            except Exception as e:
                self.get_logger().error(f"Error displaying raw image: {e}")
    
    def viz_callback(self, msg):
        if self.current_mode == "yolo":
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.putText(cv_image, "YOLO Processed", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(cv_image, "Press 'r' for raw view", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("TurtleBot Camera", cv_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rclpy.shutdown()
                elif key == ord('y'):
                    self.current_mode = "yolo"
                    self.get_logger().info("Switched to YOLO view")
                elif key == ord('r'):
                    self.current_mode = "raw"
                    self.get_logger().info("Switched to raw camera view")
                    
            except Exception as e:
                self.get_logger().error(f"Error displaying YOLO image: {e}")

def main(args=None):
    rclpy.init(args=args)
    viewer = CameraViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()