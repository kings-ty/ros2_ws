#!/usr/bin/env python3

"""
간단 직선 네비게이터 - 복잡한 A* 없이 직선으로 가면서 장애물만 회피
RViz에서 목표 찍으면 직선으로 가고, 장애물 만나면 피해서 다시 직선
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

class SimpleLineNavigator(Node):
    def __init__(self):
        super().__init__('simple_line_navigator')
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.rviz_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        # Timer
        self.timer = self.create_timer(0.05, self.navigate)  # 20Hz
        
        # Robot state
        self.robot_pos = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.goal_pos = [2.0, 2.0]  # Default goal
        self.lidar_data = None
        self.has_goal = False
        
        # Navigation parameters
        self.linear_speed = 0.4
        self.angular_speed = 0.8
        self.goal_threshold = 0.3
        
        # Obstacle avoidance state
        self.avoiding_obstacle = False
        self.avoidance_start_time = 0
        
        self.get_logger().info("🎯 간단 직선 네비게이터 시작!")
        self.get_logger().info("📍 RViz '2D Nav Goal'로 목표 설정하면 직선으로 이동!")
        
    def lidar_callback(self, msg):
        """LIDAR 데이터 처리"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.lidar_data = ranges
        
    def odom_callback(self, msg):
        """로봇 위치 업데이트"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def goal_callback(self, msg):
        """목표 설정 - RViz에서 클릭한 지점"""
        # Transform goal if needed
        if msg.header.frame_id == 'base_link':
            goal_x_odom = self.robot_pos[0] + msg.pose.position.x * math.cos(self.robot_yaw) - msg.pose.position.y * math.sin(self.robot_yaw)
            goal_y_odom = self.robot_pos[1] + msg.pose.position.x * math.sin(self.robot_yaw) + msg.pose.position.y * math.cos(self.robot_yaw)
            self.goal_pos = [goal_x_odom, goal_y_odom]
        else:
            self.goal_pos = [msg.pose.position.x, msg.pose.position.y]
        
        self.has_goal = True
        self.avoiding_obstacle = False
        
        # 직선 경로 표시
        self.draw_straight_line()
        
        self.get_logger().info(f"🎯 새 목표: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}) - 직선으로 이동!")
    
    def draw_straight_line(self):
        """목표까지 직선을 RViz에 그리기"""
        marker_array = MarkerArray()
        
        # 직선 라인
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1  # 라인 두께
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 녹색
        
        # 시작점 (로봇 위치)
        start_point = Point()
        start_point.x = self.robot_pos[0]
        start_point.y = self.robot_pos[1]
        start_point.z = 0.1
        line_marker.points.append(start_point)
        
        # 끝점 (목표 위치)
        end_point = Point()
        end_point.x = self.goal_pos[0]
        end_point.y = self.goal_pos[1]
        end_point.z = 0.1
        line_marker.points.append(end_point)
        
        marker_array.markers.append(line_marker)
        
        # 목표 마커
        goal_marker = Marker()
        goal_marker.header = line_marker.header
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.goal_pos[0]
        goal_marker.pose.position.y = self.goal_pos[1]
        goal_marker.pose.position.z = 0.2
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.4
        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 빨간색
        
        marker_array.markers.append(goal_marker)
        
        self.marker_pub.publish(marker_array)
    
    def check_obstacle(self):
        """장애물 감지 - 앞쪽과 좌우"""
        if self.lidar_data is None:
            return False, 0.0, 0.0, 0.0
        
        n_points = len(self.lidar_data)
        
        # 정면 체크 (30도 범위)
        front_range = n_points // 6
        front_start = n_points // 2 - front_range
        front_end = n_points // 2 + front_range
        front_distances = self.lidar_data[front_start:front_end]
        front_valid = front_distances[(front_distances > 0.1) & (front_distances < 8.0)]
        front_min = np.min(front_valid) if len(front_valid) > 0 else 10.0
        
        # 좌측 체크 (90도)
        left_range = n_points // 8
        left_start = n_points // 4 - left_range
        left_end = n_points // 4 + left_range
        left_distances = self.lidar_data[left_start:left_end]
        left_valid = left_distances[(left_distances > 0.1) & (left_distances < 8.0)]
        left_min = np.min(left_valid) if len(left_valid) > 0 else 10.0
        
        # 우측 체크 (270도)
        right_range = n_points // 8
        right_start = 3 * n_points // 4 - right_range
        right_end = 3 * n_points // 4 + right_range
        right_distances = self.lidar_data[right_start:right_end]
        right_valid = right_distances[(right_distances > 0.1) & (right_distances < 8.0)]
        right_min = np.min(right_valid) if len(right_valid) > 0 else 10.0
        
        # 장애물 판단 (0.6m 이내)
        obstacle_detected = front_min < 0.6
        
        return obstacle_detected, front_min, left_min, right_min
    
    def navigate(self):
        """메인 네비게이션 로직"""
        if not self.has_goal or self.lidar_data is None:
            return
            
        cmd = Twist()
        
        # 목표까지 거리와 각도 계산
        goal_distance = math.sqrt((self.goal_pos[0] - self.robot_pos[0])**2 + 
                                 (self.goal_pos[1] - self.robot_pos[1])**2)
        
        # 목표 도달 체크
        if goal_distance < self.goal_threshold:
            self.get_logger().info("🎉 목표 도달!")
            self.has_goal = False
            self.cmd_pub.publish(cmd)
            return
        
        # 목표 방향 계산
        goal_angle = math.atan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0])
        angle_diff = goal_angle - self.robot_yaw
        
        # 각도 정규화
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 장애물 체크
        obstacle_detected, front_min, left_min, right_min = self.check_obstacle()
        
        # 장애물 회피 로직
        if obstacle_detected:
            self.avoiding_obstacle = True
            self.get_logger().info(f"🚧 장애물 회피! 정면: {front_min:.2f}m")
            
            # 회피 방향 결정 (더 넓은 쪽으로)
            if left_min > right_min:
                # 왼쪽이 더 넓음
                cmd.angular.z = self.angular_speed * 0.8
                cmd.linear.x = self.linear_speed * 0.3
                self.get_logger().info("⬅️ 왼쪽으로 회피")
            else:
                # 오른쪽이 더 넓음
                cmd.angular.z = -self.angular_speed * 0.8
                cmd.linear.x = self.linear_speed * 0.3
                self.get_logger().info("➡️ 오른쪽으로 회피")
        
        else:
            # 장애물 없음 - 목표를 향해 직선 이동
            if self.avoiding_obstacle:
                self.get_logger().info("✅ 장애물 회피 완료 - 목표로 복귀")
                self.avoiding_obstacle = False
            
            # 목표 방향으로 회전
            if abs(angle_diff) > 0.1:
                cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))
                cmd.linear.x = self.linear_speed * 0.5
            else:
                # 올바른 방향 - 전진
                cmd.angular.z = angle_diff * 0.5  # 미세 조정
                cmd.linear.x = self.linear_speed * 0.9
                
            self.get_logger().info(f"🎯 목표로 직진 (거리: {goal_distance:.2f}m, 각도: {math.degrees(angle_diff):.1f}°)")
        
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleLineNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()