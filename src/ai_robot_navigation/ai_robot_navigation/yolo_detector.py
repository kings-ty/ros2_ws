#!/usr/bin/env python3

"""
Professional YOLO8 Object Detection Node for Robotics
- Real-time object detection and tracking
- Spatial relationship analysis
- Multi-class detection with confidence filtering
- Integration with robotics decision making
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
import torch

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

# Custom message for detected objects
from std_msgs.msg import Float32MultiArray

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt for better accuracy
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Float32MultiArray, '/yolo/detections', 10)
        self.annotated_image_pub = self.create_publisher(
            Image, '/yolo/annotated_image', 10)
        self.objects_pub = self.create_publisher(
            String, '/yolo/objects_detected', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Target objects for navigation (COCO classes)
        self.target_objects = {
            0: 'person',      # Avoid people
            1: 'bicycle',     # Navigate around
            2: 'car',         # Major obstacle
            3: 'motorcycle',  # Obstacle
            16: 'bird',       # Ignore (small)
            17: 'cat',        # Small obstacle
            18: 'dog',        # Small obstacle
            39: 'bottle',     # Small object to grab
            41: 'cup',        # Small object to grab
            67: 'cell phone', # Target object
        }
        
        # Camera parameters (should be calibrated for your robot)
        self.camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        self.get_logger().info(f"YOLO Detector initialized on device: {self.device}")
        self.get_logger().info(f"Target objects: {list(self.target_objects.values())}")
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO detection
            results = self.model(cv_image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Process detections
            detections = self.process_detections(results[0], cv_image.shape[:2])
            
            # Publish detection data
            self.publish_detections(detections, msg.header)
            
            # Create and publish annotated image
            annotated_image = self.create_annotated_image(cv_image, results[0])
            self.publish_annotated_image(annotated_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
    
    def process_detections(self, results, image_shape):
        """Process YOLO results into structured detection data"""
        detections = []
        detected_objects = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            height, width = image_shape
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                if cls in self.target_objects:
                    x1, y1, x2, y2 = box
                    
                    # Calculate object center and size
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    
                    # Normalize coordinates (0-1)
                    norm_center_x = center_x / width
                    norm_center_y = center_y / height
                    norm_width = obj_width / width
                    norm_height = obj_height / height
                    
                    # Calculate distance estimation (rough approximation)
                    # This should be replaced with proper depth estimation
                    estimated_distance = self.estimate_distance(obj_width, obj_height, cls)
                    
                    # Object classification for navigation
                    object_type = self.classify_for_navigation(cls, estimated_distance, norm_width * norm_height)
                    
                    detection = {
                        'class_id': int(cls),
                        'class_name': self.target_objects[cls],
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center_x), float(center_y)],
                        'normalized_center': [norm_center_x, norm_center_y],
                        'size': [float(obj_width), float(obj_height)],
                        'normalized_size': [norm_width, norm_height],
                        'estimated_distance': estimated_distance,
                        'navigation_type': object_type
                    }
                    
                    detections.append(detection)
                    detected_objects.append(f"{self.target_objects[cls]}({score:.2f})")
        
        # Publish detected objects list
        objects_msg = String()
        objects_msg.data = ", ".join(detected_objects) if detected_objects else "No objects"
        self.objects_pub.publish(objects_msg)
        
        return detections
    
    def estimate_distance(self, width, height, class_id):
        """Rough distance estimation based on object size"""
        # This is a simple approximation - should be replaced with proper depth estimation
        known_sizes = {
            0: 1.7,    # person height ~1.7m
            1: 1.8,    # bicycle length ~1.8m  
            2: 4.5,    # car length ~4.5m
            39: 0.25,  # bottle height ~25cm
            67: 0.15   # phone length ~15cm
        }
        
        if class_id in known_sizes:
            # Simple distance estimation: distance = (known_size * focal_length) / pixel_size
            focal_length = 525.0  # From camera calibration
            real_size = known_sizes[class_id]
            pixel_size = max(width, height)
            distance = (real_size * focal_length) / pixel_size
            return min(max(distance, 0.5), 10.0)  # Clamp between 0.5m and 10m
        
        return 2.0  # Default distance
    
    def classify_for_navigation(self, class_id, distance, relative_size):
        """Classify objects for navigation decision making"""
        if class_id in [0, 1, 2, 3]:  # person, bicycle, car, motorcycle
            if distance < 2.0:
                return "immediate_obstacle"
            elif distance < 5.0:
                return "approaching_obstacle"
            else:
                return "distant_obstacle"
        elif class_id in [39, 41, 67]:  # bottle, cup, phone (target objects)
            if distance < 1.0 and relative_size > 0.01:
                return "reachable_target"
            else:
                return "distant_target"
        elif class_id in [17, 18]:  # cat, dog
            return "small_obstacle"
        else:
            return "neutral_object"
    
    def publish_detections(self, detections, header):
        """Publish detection data for DQN agent"""
        if not detections:
            return
            
        # Flatten detection data for DQN consumption
        detection_array = []
        for det in detections[:5]:  # Limit to 5 most confident detections
            detection_array.extend([
                det['normalized_center'][0],  # x center
                det['normalized_center'][1],  # y center  
                det['normalized_size'][0],    # width
                det['normalized_size'][1],    # height
                det['confidence'],            # confidence
                float(det['class_id']),       # class ID
                det['estimated_distance'],    # distance
                1.0 if det['navigation_type'] == 'immediate_obstacle' else 0.0,
                1.0 if det['navigation_type'] == 'reachable_target' else 0.0,
            ])
        
        # Pad with zeros if fewer than 5 objects
        while len(detection_array) < 45:  # 5 objects * 9 features
            detection_array.append(0.0)
        
        # Publish
        msg = Float32MultiArray()
        msg.data = detection_array
        self.detection_pub.publish(msg)
    
    def create_annotated_image(self, image, results):
        """Create annotated image with bounding boxes and labels"""
        annotated = image.copy()
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                if cls in self.target_objects:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Color coding for different object types
                    if cls in [0, 1, 2, 3]:  # Obstacles
                        color = (0, 0, 255)  # Red
                    elif cls in [39, 41, 67]:  # Targets
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 0, 0)  # Blue
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{self.target_objects[cls]}: {score:.2f}"
                    cv2.putText(annotated, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def publish_annotated_image(self, image, header):
        """Publish annotated image"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header = header
            self.annotated_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    detector = YOLODetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()