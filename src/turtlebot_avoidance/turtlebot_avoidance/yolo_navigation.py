#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import time

# Import YOLO - install with: pip3 install ultralytics
try:
    from ultralytics import YOLO
    import cv2
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

class YoloNavigationImproved(Node):
    def __init__(self):
        super().__init__('yolo_navigation_improved')
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        if VISION_AVAILABLE:
            self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
            
            # YOLO model setup
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.get_logger().info('YOLO model loaded successfully!')
            except Exception as e:
                self.get_logger().error(f'Failed to load YOLO model: {e}')
                self.yolo_model = None
        else:
            self.get_logger().warn('Vision not available. Running in laser-only mode.')
            self.yolo_model = None
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state variables
        self.laser_data = None
        self.current_image = None
        self.display_image = None
        self.detected_objects = []
        self.target_object = None
        
        # Navigation parameters
        self.safe_distance = 0.6  # Reduced for better movement
        self.emergency_distance = 0.3  # Very close threshold
        self.max_linear_vel = 0.15
        self.max_angular_vel = 0.8
        self.target_distance = 1.5
        
        # Robot behavior states
        self.robot_state = 'SEARCHING'  # SEARCHING, FOLLOWING, AVOIDING, EMERGENCY
        self.last_state_change = time.time()
        self.stuck_timeout = 3.0  # seconds
        
        # Object detection parameters
        self.target_classes = ['person']
        self.avoid_classes = ['chair', 'couch', 'table', 'car', 'truck']
        self.confidence_threshold = 0.5
        
        # Image processing
        self.image_width = 320
        self.image_height = 240
        self.frame_count = 0
        
        self.get_logger().info('Improved YOLO Navigation node started!')
        self.get_logger().info('States: SEARCHING -> FOLLOWING -> AVOIDING -> EMERGENCY')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS Image to OpenCV without cv_bridge"""
        try:
            height = ros_image.height
            width = ros_image.width
            encoding = ros_image.encoding
            
            if encoding == 'rgb8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((height, width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif encoding == 'bgr8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((height, width, 3))
            elif encoding == 'mono8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((height, width))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().error(f'Unsupported encoding: {encoding}')
                return None
                
            return cv_image
            
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return None

    def image_callback(self, msg):
        """Process camera images with YOLO detection"""
        if not VISION_AVAILABLE or self.yolo_model is None:
            return
            
        try:
            self.current_image = self.ros_image_to_cv2(msg)
            if self.current_image is None:
                return
                
            self.image_height, self.image_width = self.current_image.shape[:2]
            
            # Run YOLO detection (reduce frequency to save CPU)
            self.frame_count += 1
            if self.frame_count % 2 == 0:  # Process every 2nd frame
                self.process_yolo_detection()
            
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def process_yolo_detection(self):
        """Run YOLO detection on current image with visualization"""
        if self.yolo_model is None or self.current_image is None:
            return
            
        try:
            # Create display image for visualization
            self.display_image = self.current_image.copy()
            
            # Run YOLO detection
            results = self.yolo_model(self.current_image, verbose=False)
            
            self.detected_objects = []
            self.target_object = None
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Create object info
                        obj_info = {
                            'class': class_name,
                            'confidence': confidence,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'area': width * height,
                            'bbox': [x1, y1, x2, y2]
                        }
                        
                        self.detected_objects.append(obj_info)
                        
                        # Determine color based on object type
                        if class_name in self.target_classes:
                            if center_y < self.image_height * 0.7:  # Upper part of image
                                if self.target_object is None or obj_info['area'] > self.target_object['area']:
                                    self.target_object = obj_info
                                color = (0, 255, 0)  # Green for target
                                thickness = 3
                            else:
                                color = (0, 255, 255)  # Yellow for person not at head level
                                thickness = 2
                        elif class_name in self.avoid_classes:
                            color = (0, 0, 255)  # Red for obstacles
                            thickness = 2
                        else:
                            color = (255, 0, 0)  # Blue for other objects
                            thickness = 1
                        
                        # Draw bounding box with YOLO detection
                        cv2.rectangle(self.display_image, (int(x1), int(y1)), 
                                    (int(x2), int(y2)), color, thickness)
                        
                        # Draw label with background
                        label = f'{class_name}: {confidence:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(self.display_image, (int(x1), int(y1 - label_size[1] - 10)),
                                    (int(x1 + label_size[0]), int(y1)), color, -1)
                        cv2.putText(self.display_image, label, (int(x1), int(y1 - 5)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw target indicator
            if self.target_object:
                center_x = int(self.target_object['center_x'])
                center_y = int(self.target_object['center_y'])
                cv2.circle(self.display_image, (center_x, center_y), 15, (0, 255, 0), 3)
                cv2.putText(self.display_image, 'TARGET', (center_x-35, center_y-25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add status information
            status_text = [
                f"State: {self.robot_state}",
                f"Objects: {len(self.detected_objects)}",
                f"Target: {'YES' if self.target_object else 'NO'}"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(self.display_image, text, (10, 25 + i*20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add laser info if available
            if self.laser_data:
                ranges = np.array(self.laser_data.ranges)
                ranges[ranges == float('inf')] = self.laser_data.range_max
                min_dist = np.min(ranges)
                cv2.putText(self.display_image, f"Min Dist: {min_dist:.2f}m", 
                          (10, self.image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 255), 2)
            
            # Display the image with detections
            cv2.imshow('YOLO Detection and Navigation', self.display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'YOLO processing error: {e}')

    def analyze_laser_data(self):
        """Analyze laser data to understand surroundings"""
        if not self.laser_data:
            return None, None, None, None
        
        ranges = np.array(self.laser_data.ranges)
        ranges[ranges == float('inf')] = self.laser_data.range_max
        ranges[ranges == 0.0] = self.laser_data.range_max
        
        num_readings = len(ranges)
        
        # Divide into sectors
        front_sector = ranges[num_readings//3:2*num_readings//3]  # Front 120°
        left_sector = ranges[:num_readings//3]  # Left 120°
        right_sector = ranges[2*num_readings//3:]  # Right 120°
        front_narrow = ranges[5*num_readings//12:7*num_readings//12]  # Front 60°
        
        return (
            np.min(front_sector),
            np.min(left_sector), 
            np.min(right_sector),
            np.min(front_narrow)
        )

    def get_navigation_command(self):
        """Smart navigation with state machine"""
        twist = Twist()
        
        # Analyze surroundings
        if self.laser_data:
            min_front, min_left, min_right, min_front_narrow = self.analyze_laser_data()
        else:
            return twist
        
        # State machine logic
        current_time = time.time()
        
        # Emergency state - very close to obstacle
        if min_front_narrow < self.emergency_distance:
            if self.robot_state != 'EMERGENCY':
                self.robot_state = 'EMERGENCY'
                self.last_state_change = current_time
                self.get_logger().warn('Entering EMERGENCY state')
            
            return self.emergency_behavior(min_left, min_right)
        
        # If stuck in emergency too long, force recovery
        if self.robot_state == 'EMERGENCY' and (current_time - self.last_state_change) > 2.0:
            self.robot_state = 'AVOIDING'
            self.last_state_change = current_time
            self.get_logger().info('Forced recovery from EMERGENCY to AVOIDING')
        
        # Normal navigation states
        if self.target_object and min_front > self.safe_distance:
            # Switch to following if we have a target and path is clear
            if self.robot_state != 'FOLLOWING':
                self.robot_state = 'FOLLOWING'
                self.last_state_change = current_time
                self.get_logger().info('Switching to FOLLOWING state')
            
            return self.follow_target(min_front, min_left, min_right)
        
        elif min_front < self.safe_distance:
            # Switch to avoiding if obstacle detected
            if self.robot_state != 'AVOIDING':
                self.robot_state = 'AVOIDING' 
                self.last_state_change = current_time
                self.get_logger().info('Switching to AVOIDING state')
            
            return self.avoid_obstacles(min_front, min_left, min_right)
        
        else:
            # Switch to searching if no target and path is clear
            if self.robot_state != 'SEARCHING':
                self.robot_state = 'SEARCHING'
                self.last_state_change = current_time
                self.get_logger().info('Switching to SEARCHING state')
            
            return self.search_behavior(min_front, min_left, min_right)

    def emergency_behavior(self, min_left, min_right):
        """Emergency behavior - back away and turn"""
        twist = Twist()
        
        # Back away slowly
        twist.linear.x = -0.1
        
        # Turn toward more open space
        if min_left > min_right + 0.2:
            twist.angular.z = 0.8  # Turn left
        elif min_right > min_left + 0.2:
            twist.angular.z = -0.8  # Turn right
        else:
            twist.angular.z = 0.8  # Default turn left
        
        self.get_logger().warn('EMERGENCY: Backing away and turning')
        return twist

    def follow_target(self, min_front, min_left, min_right):
        """Follow target behavior"""
        twist = Twist()
        
        if not self.target_object:
            return twist
        
        # Calculate target position relative to image center
        image_center_x = self.image_width / 2
        target_x = self.target_object['center_x']
        x_error = target_x - image_center_x
        x_error_normalized = x_error / (self.image_width / 2)
        
        # Estimate distance based on object size
        object_area = self.target_object['area']
        estimated_distance = max(0.5, 8000 / object_area)
        
        # Angular control (turn toward target)
        twist.angular.z = -x_error_normalized * self.max_angular_vel * 0.8
        
        # Linear control (approach or maintain distance)
        if abs(x_error_normalized) < 0.3:  # Target is reasonably centered
            distance_error = estimated_distance - self.target_distance
            if distance_error > 0.4:  # Too far
                twist.linear.x = min(self.max_linear_vel, distance_error * 0.2)
            elif distance_error < -0.4:  # Too close
                twist.linear.x = max(-self.max_linear_vel * 0.5, distance_error * 0.2)
            else:  # Good distance
                twist.linear.x = 0.02  # Very slow approach
        else:
            # Turn in place if target is not centered
            twist.linear.x = 0.0
        
        # Safety check - don't move forward if obstacle too close
        if min_front < self.safe_distance and twist.linear.x > 0:
            twist.linear.x = 0.0
            twist.angular.z *= 1.2  # Turn faster when blocked
        
        self.get_logger().info(f'Following target, est. dist: {estimated_distance:.1f}m')
        return twist

    def avoid_obstacles(self, min_front, min_left, min_right):
        """Obstacle avoidance behavior"""
        twist = Twist()
        
        # Determine best direction to turn
        if min_left > min_right + 0.3:
            # More space on left
            twist.angular.z = self.max_angular_vel * 0.7
            twist.linear.x = 0.05 if min_left > 1.0 else 0.0
        elif min_right > min_left + 0.3:
            # More space on right  
            twist.angular.z = -self.max_angular_vel * 0.7
            twist.linear.x = 0.05 if min_right > 1.0 else 0.0
        else:
            # Equal space or both blocked - turn toward slightly more open side
            if min_left >= min_right:
                twist.angular.z = self.max_angular_vel * 0.8
            else:
                twist.angular.z = -self.max_angular_vel * 0.8
            twist.linear.x = 0.0
        
        self.get_logger().info(f'Avoiding obstacles - L:{min_left:.2f} R:{min_right:.2f} F:{min_front:.2f}')
        return twist

    def search_behavior(self, min_front, min_left, min_right):
        """Search for targets while exploring"""
        twist = Twist()
        
        # Move forward while slowly rotating to search
        twist.linear.x = self.max_linear_vel * 0.4
        twist.angular.z = 0.2  # Gentle rotation for searching
        
        self.get_logger().info('Searching for targets...', throttle_duration_sec=3.0)
        return twist

    def control_loop(self):
        """Main control loop"""
        twist = self.get_navigation_command()
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    
    yolo_nav = YoloNavigationImproved()
    
    try:
        rclpy.spin(yolo_nav)
    except KeyboardInterrupt:
        pass
    
    # Stop robot
    stop_msg = Twist()
    yolo_nav.cmd_vel_pub.publish(stop_msg)
    
    # Clean up
    cv2.destroyAllWindows()
    yolo_nav.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
