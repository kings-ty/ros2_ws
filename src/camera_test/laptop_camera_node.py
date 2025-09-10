#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LaptopCameraNode(Node):
    def __init__(self):
        super().__init__('laptop_camera_node')
        
        # Publisher for camera images
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Open camera (usually /dev/video0)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error('Could not open camera!')
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Timer to capture and publish frames
        self.timer = self.create_timer(0.1, self.capture_and_publish)  # 10 FPS
        
        self.get_logger().info('Laptop camera node started!')
        self.get_logger().info('Publishing camera frames to /camera/image_raw')

    def capture_and_publish(self):
        """Capture frame and publish as ROS message"""
        ret, frame = self.cap.read()
        
        if ret:
            try:
                # Convert OpenCV image to ROS message
                ros_image = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = 'camera_link'
                
                # Publish image
                self.image_pub.publish(ros_image)
                
            except Exception as e:
                self.get_logger().error(f'Error publishing image: {e}')
        else:
            self.get_logger().warn('Failed to capture frame')

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    camera_node = LaptopCameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()