#!/usr/bin/env python3

"""
Sensing and Perception Tutorial 3: Sensor Fusion
- Combining LIDAR and Camera data
- Cross-modal validation
- Robust perception through multiple sensors
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # Publishers
        self.fusion_pub = self.create_publisher(MarkerArray, '/fusion_results', 10)
        self.overlay_pub = self.create_publisher(Image, '/camera/overlay', 10)
        
        # Data storage
        self.latest_scan = None
        self.latest_image = None
        
        # Camera parameters (should be calibrated for real robot)
        self.camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        self.get_logger().info("🔗 Sensor Fusion Tutorial Started!")
        self.get_logger().info("🎯 Learning: Multi-modal perception, data association")
        
    def lidar_callback(self, msg):
        """Store latest LIDAR data"""
        self.latest_scan = msg
        self.perform_fusion()
    
    def image_callback(self, msg):
        """Store latest image data"""
        self.latest_image = msg
        self.perform_fusion()
    
    def perform_fusion(self):
        """
        Perform sensor fusion when both sensors have data
        """
        if self.latest_scan is None or self.latest_image is None:
            return
        
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            # Get LIDAR points in camera frame
            lidar_points_3d = self.lidar_to_3d_points()
            
            # Project LIDAR points onto image
            image_with_overlay = self.project_lidar_on_image(cv_image, lidar_points_3d)
            
            # Detect obstacles using both sensors
            obstacles = self.detect_obstacles_fusion()
            
            # Visualize results
            self.visualize_fusion_results(obstacles)
            
            # Publish overlay image
            overlay_msg = self.bridge.cv2_to_imgmsg(image_with_overlay, "bgr8")
            overlay_msg.header = self.latest_image.header
            self.overlay_pub.publish(overlay_msg)
            
            # Log fusion results
            if len(obstacles) > 0:
                self.get_logger().info(f"🔗 FUSION: Detected {len(obstacles)} obstacles using both sensors")
                
        except Exception as e:
            self.get_logger().error(f"Fusion error: {e}")
    
    def lidar_to_3d_points(self):
        """
        Convert 2D LIDAR scan to 3D points
        """
        points_3d = []
        
        for i, range_val in enumerate(self.latest_scan.ranges):
            if np.isfinite(range_val) and range_val > self.latest_scan.range_min:
                # Calculate angle
                angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
                
                # Convert to 3D (assuming LIDAR is at robot center, camera slightly forward)
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0.0  # LIDAR is 2D, assume ground level
                
                # Transform to camera coordinate system
                # (This assumes camera is 0.1m forward from LIDAR)
                x_cam = x - 0.1  # Camera is 10cm forward
                y_cam = y
                z_cam = z + 0.2  # Camera is 20cm above LIDAR
                
                points_3d.append([x_cam, y_cam, z_cam, range_val])
        
        return np.array(points_3d)
    
    def project_lidar_on_image(self, image, points_3d):
        """
        Project 3D LIDAR points onto 2D image plane
        """
        overlay_image = image.copy()
        
        if len(points_3d) == 0:
            return overlay_image
        
        # Project 3D points to 2D image coordinates
        for point in points_3d:
            x, y, z, distance = point
            
            # Skip points behind camera
            if x <= 0:
                continue
                
            # Project to image plane
            image_x = int((y * self.camera_matrix[0,0]) / x + self.camera_matrix[0,2])
            image_y = int((-z * self.camera_matrix[1,1]) / x + self.camera_matrix[1,2])
            
            # Check if point is within image bounds
            if 0 <= image_x < image.shape[1] and 0 <= image_y < image.shape[0]:
                # Color based on distance (red=close, green=far)
                color_intensity = min(distance / 3.0, 1.0)
                color = (0, int(255 * color_intensity), int(255 * (1 - color_intensity)))
                
                # Draw point
                cv2.circle(overlay_image, (image_x, image_y), 3, color, -1)
        
        # Add legend
        cv2.putText(overlay_image, "LIDAR overlay: Red=Close, Green=Far", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay_image
    
    def detect_obstacles_fusion(self):
        """
        Enhanced obstacle detection using both LIDAR and vision
        """
        obstacles = []
        
        # LIDAR-based obstacle detection
        ranges = np.array(self.latest_scan.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if len(valid_ranges) == 0:
            return obstacles
        
        # Find clusters of close points (potential obstacles)
        close_threshold = 2.0  # 2 meters
        close_indices = np.where(valid_ranges < close_threshold)[0]
        
        if len(close_indices) > 0:
            # Group consecutive indices into clusters
            clusters = []
            current_cluster = [close_indices[0]]
            
            for i in range(1, len(close_indices)):
                if close_indices[i] - close_indices[i-1] <= 3:  # Within 3 readings
                    current_cluster.append(close_indices[i])
                else:
                    if len(current_cluster) > 5:  # At least 5 points
                        clusters.append(current_cluster)
                    current_cluster = [close_indices[i]]
            
            # Add last cluster
            if len(current_cluster) > 5:
                clusters.append(current_cluster)
            
            # Create obstacle data for each cluster
            for cluster in clusters:
                # Calculate cluster properties
                cluster_ranges = valid_ranges[cluster]
                cluster_angles = [self.latest_scan.angle_min + i * self.latest_scan.angle_increment 
                                for i in cluster]
                
                # Obstacle center
                center_angle = np.mean(cluster_angles)
                center_distance = np.mean(cluster_ranges)
                
                # Convert to Cartesian
                x = center_distance * np.cos(center_angle)
                y = center_distance * np.sin(center_angle)
                
                obstacle = {
                    'x': x,
                    'y': y,
                    'distance': center_distance,
                    'angle': center_angle,
                    'size': len(cluster),
                    'confidence': min(len(cluster) / 10.0, 1.0),  # More points = higher confidence
                    'sensor': 'lidar'
                }
                
                obstacles.append(obstacle)
        
        return obstacles
    
    def visualize_fusion_results(self, obstacles):
        """
        Visualize fusion results in RViz
        """
        marker_array = MarkerArray()
        
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fusion_obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = obstacle['x']
            marker.pose.position.y = obstacle['y']
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0
            
            # Size based on confidence
            size = 0.2 + obstacle['confidence'] * 0.3
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = 1.0
            
            # Color based on distance
            distance_factor = min(obstacle['distance'] / 3.0, 1.0)
            marker.color.r = 1.0 - distance_factor
            marker.color.g = distance_factor
            marker.color.b = 0.0
            marker.color.a = obstacle['confidence']
            
            marker_array.markers.append(marker)
        
        self.fusion_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()