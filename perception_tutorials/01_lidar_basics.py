#!/usr/bin/env python3

"""
Sensing and Perception Tutorial 1: LIDAR Basics
- Understanding LIDAR data structure
- Range measurements and coordinate systems
- Data preprocessing and filtering
"""

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class LidarBasicsNode(Node):
    def __init__(self):
        super().__init__('lidar_basics')
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # Publishers for visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/lidar_points', 10)
        
        # Data storage for analysis
        self.scan_data = []
        self.scan_count = 0
        
        self.get_logger().info("📡 LIDAR Basics Tutorial Started!")
        self.get_logger().info("🎯 Learning: Range measurements, angles, coordinate transforms")
        
    def lidar_callback(self, msg):
        """
        Analyze LIDAR scan data structure and properties
        """
        self.scan_count += 1
        
        # Basic scan properties
        num_readings = len(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        
        # Log every 50 scans (5 seconds)
        if self.scan_count % 50 == 0:
            self.get_logger().info(
                f"📊 SCAN ANALYSIS:\n"
                f"   📐 Readings: {num_readings}\n"
                f"   📏 Angle range: {np.degrees(angle_min):.1f}° to {np.degrees(angle_max):.1f}°\n"
                f"   🎯 Resolution: {np.degrees(angle_increment):.2f}° per reading\n"
                f"   📡 Range: {range_min:.2f}m to {range_max:.2f}m"
            )
            
            # Data quality analysis
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]
            
            if len(valid_ranges) > 0:
                self.get_logger().info(
                    f"📈 DATA QUALITY:\n"
                    f"   ✅ Valid readings: {len(valid_ranges)}/{num_readings} ({len(valid_ranges)/num_readings*100:.1f}%)\n"
                    f"   📊 Distance stats: min={np.min(valid_ranges):.2f}m, max={np.max(valid_ranges):.2f}m, avg={np.mean(valid_ranges):.2f}m\n"
                    f"   🎯 Closest obstacle: {np.min(valid_ranges):.2f}m"
                )
                
                # Convert to Cartesian coordinates
                cartesian_points = self.polar_to_cartesian(msg)
                self.visualize_lidar_points(cartesian_points)
                
                # Obstacle detection
                self.detect_obstacles(msg)
    
    def polar_to_cartesian(self, scan_msg):
        """
        Convert LIDAR polar coordinates to Cartesian (x,y)
        Essential for perception algorithms!
        """
        points = []
        
        for i, range_val in enumerate(scan_msg.ranges):
            if np.isfinite(range_val) and range_val > scan_msg.range_min:
                # Calculate angle for this reading
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                
                # Convert to Cartesian coordinates
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                
                points.append([x, y, range_val, angle])
        
        return np.array(points)
    
    def detect_obstacles(self, scan_msg):
        """
        Basic obstacle detection algorithm
        """
        ranges = np.array(scan_msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, scan_msg.range_max)
        
        # Sector analysis (divide 360° into sectors)
        sectors = {
            'Front': ranges[len(ranges)//2-15:len(ranges)//2+15],
            'Left': ranges[len(ranges)//4-15:len(ranges)//4+15],
            'Right': ranges[3*len(ranges)//4-15:3*len(ranges)//4+15],
            'Back': ranges[:15] if len(ranges[:15]) > 0 else ranges[-15:]
        }
        
        obstacles = {}
        for sector, sector_ranges in sectors.items():
            if len(sector_ranges) > 0:
                min_dist = np.min(sector_ranges)
                obstacles[sector] = min_dist
        
        # Alert for close obstacles
        close_threshold = 1.0  # 1 meter
        for sector, distance in obstacles.items():
            if distance < close_threshold:
                self.get_logger().warn(f"⚠️ {sector} obstacle at {distance:.2f}m!")
    
    def visualize_lidar_points(self, points):
        """
        Visualize LIDAR points in RViz
        """
        marker_array = MarkerArray()
        
        # Clear previous markers first
        clear_marker = Marker()
        clear_marker.header.frame_id = "base_link"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = "lidar_points"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Create marker for each point
        for i, point in enumerate(points):
            if i % 10 == 0:  # Reduce density for visualization
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "lidar_points"
                marker.id = i + 1  # Start from 1 to avoid conflict with clear marker
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(point[0])
                marker.pose.position.y = float(point[1])
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                
                # Color based on distance (blue=close, red=far)
                distance_normalized = min(point[2] / 5.0, 1.0)
                marker.color.r = float(distance_normalized)
                marker.color.g = 0.0
                marker.color.b = float(1.0 - distance_normalized)
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LidarBasicsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()