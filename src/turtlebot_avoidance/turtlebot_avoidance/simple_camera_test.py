#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SimpleCameraTest(Node):
    def __init__(self):
        super().__init__('simple_camera_test')
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        self.get_logger().info('Camera test node started!')
        self.get_logger().info('Press Ctrl+C to stop')
        
        # Flag to log only first image
        self.first_image_received = False

    def image_callback(self, msg):
        """Process camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Display image
            cv2.imshow('TurtleBot Camera Feed', cv_image)
            cv2.waitKey(1)
            
            # Log only first image received
            if not self.first_image_received:
                height, width = cv_image.shape[:2]
                self.get_logger().info(f'First image received: {width}x{height}')
                self.get_logger().info('Camera feed is working! Check the OpenCV window.')
                self.first_image_received = True
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    camera_test = SimpleCameraTest()
    
    try:
        rclpy.spin(camera_test)
    except KeyboardInterrupt:
        pass
    
    # Clean up
    cv2.destroyAllWindows()
    camera_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
