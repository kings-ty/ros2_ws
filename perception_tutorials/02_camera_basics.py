#!/usr/bin/env python3

"""
Sensing and Perception Tutorial 2: Camera Basics
- Image data structure and properties
- Color spaces and pixel operations
- Camera calibration concepts
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraBasicsNode(Node):
    def __init__(self):
        super().__init__('camera_basics')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # Publishers
        self.processed_pub = self.create_publisher(Image, '/camera/processed', 10)
        
        self.frame_count = 0
        
        self.get_logger().info("📷 Camera Basics Tutorial Started!")
        self.get_logger().info("🎯 Learning: Image properties, color spaces, basic processing")
        
    def image_callback(self, msg):
        """
        Analyze camera image properties and demonstrate basic processing
        """
        self.frame_count += 1
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Image properties analysis
            height, width, channels = cv_image.shape
            
            # Log every 30 frames (1 second at 30fps)
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"📊 IMAGE ANALYSIS:\n"
                    f"   📏 Resolution: {width} x {height}\n"
                    f"   🎨 Channels: {channels} (BGR format)\n"
                    f"   💾 Size: {cv_image.nbytes} bytes"
                )
                
                # Color space analysis
                self.analyze_color_spaces(cv_image)
                
                # Basic image statistics
                self.analyze_image_statistics(cv_image)
            
            # Image processing demonstrations
            processed_image = self.demonstrate_processing(cv_image)
            
            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    
    def analyze_color_spaces(self, image):
        """
        Demonstrate different color space representations
        """
        # BGR (default OpenCV format)
        bgr_mean = np.mean(image, axis=(0,1))
        
        # RGB (more intuitive for humans)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_mean = np.mean(rgb_image, axis=(0,1))
        
        # HSV (Hue, Saturation, Value - good for color-based detection)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mean = np.mean(hsv_image, axis=(0,1))
        
        # Grayscale (intensity only)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_mean = np.mean(gray_image)
        
        self.get_logger().info(
            f"🎨 COLOR SPACE ANALYSIS:\n"
            f"   BGR: B={bgr_mean[0]:.1f}, G={bgr_mean[1]:.1f}, R={bgr_mean[2]:.1f}\n"
            f"   HSV: H={hsv_mean[0]:.1f}, S={hsv_mean[1]:.1f}, V={hsv_mean[2]:.1f}\n"
            f"   Gray: {gray_mean:.1f} (brightness level)"
        )
    
    def analyze_image_statistics(self, image):
        """
        Analyze image statistical properties
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        most_common_intensity = np.argmax(hist)
        
        self.get_logger().info(
            f"📈 IMAGE STATISTICS:\n"
            f"   💡 Brightness: mean={mean_brightness:.1f}, std={std_brightness:.1f}\n"
            f"   📊 Range: {min_val} to {max_val}\n"
            f"   🎯 Most common intensity: {most_common_intensity}"
        )
    
    def demonstrate_processing(self, image):
        """
        Demonstrate basic image processing techniques
        """
        # Create a multi-panel display showing different processing techniques
        height, width = image.shape[:2]
        
        # Original image (top-left)
        result = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        result[0:height, 0:width] = image
        
        # Grayscale (top-right)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result[0:height, width:width*2] = gray_bgr
        
        # Edge detection (bottom-left)
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result[height:height*2, 0:width] = edges_bgr
        
        # HSV color space (bottom-right)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result[height:height*2, width:width*2] = hsv
        
        # Add labels
        cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(result, "Grayscale", (width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(result, "Edges", (10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(result, "HSV", (width+10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        return result

def main(args=None):
    rclpy.init(args=args)
    node = CameraBasicsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()