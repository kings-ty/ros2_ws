#!/usr/bin/env python3

"""
Pure LIDAR Navigation - 카메라 없이 LIDAR만 사용
- LIDAR 데이터로 고품질 실시간 맵핑
- A* 알고리즘으로 최적 경로 계획
- 더 넓은 영역 스캐닝 및 맵 품질 개선
- 목표 재설정 지원
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
from collections import deque
import heapq

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

class LidarOnlyNavigator(Node):
    def __init__(self):
        super().__init__('lidar_only_navigator')
        
        # Subscribers - LIDAR와 odom만
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.rviz_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/occupancy_map', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        # Timers
        self.timer = self.create_timer(0.02, self.navigate)  # 50Hz 극한 속도!
        self.map_timer = self.create_timer(0.5, self.publish_map)  # 더 자주 업데이트
        
        # Robot state
        self.robot_pos = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.goal_pos = [3.0, 3.0]  # Default goal
        self.lidar_data = None
        
        # Enhanced occupancy grid parameters
        self.map_resolution = 0.05  # 5cm per cell (더 정밀)
        self.map_size = 400  # 20m x 20m map with higher resolution
        self.map_origin = [-10.0, -10.0]
        self.occupancy_grid = np.full((self.map_size, self.map_size), 50, dtype=np.int8)  # Start with unknown
        self.inflated_grid = np.full((self.map_size, self.map_size), 50, dtype=np.int8)
        self.confidence_grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)  # Confidence scores
        self.robot_radius = 0.35  # Robot safety radius
        self.grid_updates = 0
        
        # Path planning
        self.current_path = []
        self.path_index = 0
        
        # Enhanced navigation parameters
        self.linear_speed = 0.35
        self.angular_speed = 0.7
        self.goal_threshold = 0.25
        self.lookahead_distance = 0.6
        
        # Exploration tracking
        self.explored_positions = set()
        self.exploration_radius = 0.3
        
        self.get_logger().info("🚁 Pure LIDAR Navigator 시작!")
        self.get_logger().info("📡 LIDAR만으로 고품질 맵핑 + 네비게이션")
        self.get_logger().info("🎯 RViz '2D Nav Goal'로 목표를 자유롭게 재설정하세요!")
        
    def lidar_callback(self, msg):
        """Enhanced LIDAR processing for better mapping"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.lidar_data = ranges
        
        # Update occupancy grid with improved algorithm
        self.update_occupancy_grid_enhanced(msg)
    
    def update_occupancy_grid_enhanced(self, lidar_msg):
        """Enhanced occupancy grid update with confidence tracking"""
        ranges = np.array(lidar_msg.ranges)
        angle_min = lidar_msg.angle_min
        angle_increment = lidar_msg.angle_increment
        
        robot_grid_x = int((self.robot_pos[0] - self.map_origin[0]) / self.map_resolution)
        robot_grid_y = int((self.robot_pos[1] - self.map_origin[1]) / self.map_resolution)
        
        if not (0 <= robot_grid_x < self.map_size and 0 <= robot_grid_y < self.map_size):
            return
        
        # Mark robot's current position as explored
        self.mark_explored_area(robot_grid_x, robot_grid_y)
        
        # Process each LIDAR ray with noise filtering
        for i, range_val in enumerate(ranges):
            # 더 엄격한 필터링
            if range_val < 0.08 or range_val > 8.0 or not np.isfinite(range_val):
                continue
            
            # 인근 포인트들과 비교해서 이상치 제거
            neighbors = []
            for j in [-2, -1, 1, 2]:
                neighbor_idx = (i + j) % len(ranges)
                neighbor_val = ranges[neighbor_idx]
                if 0.08 < neighbor_val < 8.0:
                    neighbors.append(neighbor_val)
            
            if neighbors:
                avg_neighbor = np.mean(neighbors)
                # 주변 값과 너무 다르면 노이즈로 판단
                if abs(range_val - avg_neighbor) > 0.5 and range_val < 1.0:
                    continue
                
            angle = angle_min + i * angle_increment + self.robot_yaw
            
            # Calculate end point of ray
            end_x = self.robot_pos[0] + range_val * math.cos(angle)
            end_y = self.robot_pos[1] + range_val * math.sin(angle)
            
            end_grid_x = int((end_x - self.map_origin[0]) / self.map_resolution)
            end_grid_y = int((end_y - self.map_origin[1]) / self.map_resolution)
            
            # Bresenham line algorithm for ray tracing
            line_points = self.bresenham_line(robot_grid_x, robot_grid_y, end_grid_x, end_grid_y)
            
            # Mark free space along the ray
            for j, (x, y) in enumerate(line_points[:-1]):  # Exclude endpoint
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    # Increase confidence for free space
                    self.confidence_grid[y, x] = min(self.confidence_grid[y, x] + 0.3, 10.0)
                    # Mark as free space with higher confidence
                    free_prob = max(0, min(20 - int(self.confidence_grid[y, x] * 2), 20))
                    self.occupancy_grid[y, x] = free_prob
            
            # Mark obstacle at endpoint (if valid)
            if (0 <= end_grid_x < self.map_size and 0 <= end_grid_y < self.map_size and 
                range_val < 7.5):  # Valid obstacle reading
                
                # Increase confidence for obstacle
                self.confidence_grid[end_grid_y, end_grid_x] = min(
                    self.confidence_grid[end_grid_y, end_grid_x] + 0.8, 10.0
                )
                
                # Mark as obstacle with confidence
                obstacle_prob = min(80 + int(self.confidence_grid[end_grid_y, end_grid_x] * 2), 100)
                self.occupancy_grid[end_grid_y, end_grid_x] = obstacle_prob
                
                # Mark surrounding cells as potential obstacles
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = end_grid_x + dx, end_grid_y + dy
                        if (0 <= nx < self.map_size and 0 <= ny < self.map_size and
                            self.occupancy_grid[ny, nx] < 70):
                            self.confidence_grid[ny, nx] = min(self.confidence_grid[ny, nx] + 0.2, 10.0)
                            partial_prob = min(40 + int(self.confidence_grid[ny, nx] * 3), 70)
                            self.occupancy_grid[ny, nx] = max(self.occupancy_grid[ny, nx], partial_prob)
        
        self.grid_updates += 1
        
        # Apply inflation every few updates
        if self.grid_updates % 3 == 0:
            self.inflate_obstacles()
    
    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def mark_explored_area(self, center_x, center_y):
        """Mark area around robot as explored"""
        radius_cells = int(self.exploration_radius / self.map_resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        self.explored_positions.add((x, y))
    
    def inflate_obstacles(self):
        """Inflate obstacles with improved algorithm"""
        self.inflated_grid = self.occupancy_grid.copy()
        
        inflation_radius = int(self.robot_radius / self.map_resolution)
        
        # Find high-confidence obstacles
        obstacle_cells = np.where(self.occupancy_grid >= 75)
        
        for i in range(len(obstacle_cells[0])):
            obs_y, obs_x = obstacle_cells[0][i], obstacle_cells[1][i]
            
            # Create circular inflation
            for dy in range(-inflation_radius, inflation_radius + 1):
                for dx in range(-inflation_radius, inflation_radius + 1):
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= inflation_radius:
                        new_y, new_x = obs_y + dy, obs_x + dx
                        
                        if (0 <= new_x < self.map_size and 0 <= new_y < self.map_size):
                            if self.occupancy_grid[new_y, new_x] <= 50:  # Only inflate free/unknown space
                                # 더 부드러운 gradient inflation
                                inflation_cost = int(70 - (distance / inflation_radius) * 45)
                                self.inflated_grid[new_y, new_x] = max(
                                    self.inflated_grid[new_y, new_x], 
                                    inflation_cost
                                )
    
    def odom_callback(self, msg):
        """Update robot position"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def goal_callback(self, msg):
        """Handle new goal and replan path"""
        # Transform goal if needed
        if msg.header.frame_id == 'base_link':
            goal_x_odom = self.robot_pos[0] + msg.pose.position.x * math.cos(self.robot_yaw) - msg.pose.position.y * math.sin(self.robot_yaw)
            goal_y_odom = self.robot_pos[1] + msg.pose.position.x * math.sin(self.robot_yaw) + msg.pose.position.y * math.cos(self.robot_yaw)
            self.goal_pos = [goal_x_odom, goal_y_odom]
        else:
            self.goal_pos = [msg.pose.position.x, msg.pose.position.y]
        
        self.get_logger().info(f"🎯 새 목표: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}) - 경로 재계획 중...")
        
        # Clear old path and reset - 이게 핵심!
        self.current_path = []
        self.path_index = 0
        
        # Plan new path
        self.plan_path()
    
    def world_to_grid(self, x, y):
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.map_resolution + self.map_origin[0]
        y = grid_y * self.map_resolution + self.map_origin[1]
        return x, y
    
    def is_valid_cell(self, grid_x, grid_y):
        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):
            return False
        return self.inflated_grid[grid_y, grid_x] <= 65  # 더 관대하게 (기둥 사이 허용)
    
    def get_cell_cost(self, grid_x, grid_y):
        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):
            return float('inf')
        
        cell_value = self.inflated_grid[grid_y, grid_x]
        if cell_value >= 85:  # 진짜 장애물만 무한대 비용
            return float('inf')
        elif cell_value > 65:  # 높은 비용이지만 통과 가능
            return 3.0 + (cell_value - 65) * 0.2
        elif cell_value > 35:  # 중간 비용
            return 1.5 + (cell_value - 35) * 0.05
        else:
            return 1.0
    
    def plan_path(self):
        """Plan path using enhanced A*"""
        start_grid = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
        goal_grid = self.world_to_grid(self.goal_pos[0], self.goal_pos[1])
        
        # 목표 위치 체크를 더 관대하게 - 기둥 사이도 허용
        goal_cell_value = self.inflated_grid[goal_grid[1], goal_grid[0]] if (0 <= goal_grid[0] < self.map_size and 0 <= goal_grid[1] < self.map_size) else 100
        if goal_cell_value >= 85:  # 진짜 장애물일 때만 거부
            self.get_logger().warn(f"❌ 목표 위치에 확실한 장애물! (값: {goal_cell_value})")
            return
        elif goal_cell_value > 65:
            self.get_logger().info(f"⚠️ 목표 위치가 약간 위험하지만 시도해봅니다 (값: {goal_cell_value})")
        
        path = self.astar(start_grid, goal_grid)
        
        if path:
            raw_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                raw_path.append([world_x, world_y])
            
            self.current_path = self.smooth_path(raw_path)
            self.path_index = 0
            
            self.get_logger().info(f"✅ 새 경로 완성! {len(self.current_path)} 포인트")
            self.publish_path_markers()
        else:
            self.get_logger().warn("❌ 경로를 찾을 수 없습니다!")
    
    def astar(self, start, goal):
        """Enhanced A* algorithm"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(node):
            x, y = node
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self.is_valid_cell(nx, ny):
                        base_cost = math.sqrt(dx*dx + dy*dy) if dx != 0 and dy != 0 else 1.0
                        cell_cost = self.get_cell_cost(nx, ny)
                        neighbors.append(((nx, ny), base_cost * cell_cost))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor, move_cost in get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def smooth_path(self, path):
        """Path smoothing"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.has_line_of_sight(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                smoothed.append(path[i + 1])
                i += 1
        
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        return smoothed
    
    def has_line_of_sight(self, start_pos, end_pos):
        """Line of sight check"""
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        samples = max(int(distance / (self.map_resolution * 2)), 5)
        
        for i in range(samples + 1):
            t = i / samples if samples > 0 else 0
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            
            grid_x, grid_y = self.world_to_grid(x, y)
            
            if not self.is_valid_cell(grid_x, grid_y):
                return False
            if self.get_cell_cost(grid_x, grid_y) > 3.0:
                return False
        
        return True
    
    def navigate(self):
        """Navigation logic"""
        if not self.current_path or self.lidar_data is None:
            return
            
        cmd = Twist()
        
        # Check if goal reached
        goal_distance = math.sqrt((self.goal_pos[0] - self.robot_pos[0])**2 + 
                                 (self.goal_pos[1] - self.robot_pos[1])**2)
        
        if goal_distance < self.goal_threshold:
            self.get_logger().info("🎉 목표 도달 완료!")
            self.current_path = []
            self.cmd_pub.publish(cmd)
            return
        
        # Get target point
        target_point = self.get_lookahead_point()
        if target_point is None:
            return
        
        # Calculate control
        target_distance = math.sqrt((target_point[0] - self.robot_pos[0])**2 + 
                                   (target_point[1] - self.robot_pos[1])**2)
        
        target_angle = math.atan2(target_point[1] - self.robot_pos[1], 
                                 target_point[0] - self.robot_pos[0])
        angle_diff = target_angle - self.robot_yaw
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # **안전 최우선**: Emergency obstacle check
        obstacle_detected, avoidance_intensity = self.check_immediate_obstacle()
        
        if obstacle_detected:
            if avoidance_intensity >= 999.0:
                # 🚨 응급정지! 25cm 이내 (제동거리 고려)
                cmd.linear.x = -0.6  # 극한 후진!
                cmd.angular.z = self.angular_speed * 2.0  # 극한 회전!
                self.get_logger().error("💀🛑 극한 응급정지!")
            elif avoidance_intensity == -1.0:
                # 양쪽 막힘 - 빠른 후진
                cmd.linear.x = -0.3  # 더 빠른 후진
                cmd.angular.z = self.angular_speed * 1.2  # 더 빠른 회전
                self.get_logger().warn("🔄 고속 후진 + 회전 탈출!")
            elif avoidance_intensity > 0.8:
                # 매우 위험 - 급격한 회피
                cmd.angular.z = self.angular_speed * 1.5 * avoidance_intensity  # 더 급격하게
                cmd.linear.x = -0.1  # 살짝 후진하면서 회전
                self.get_logger().warn("🛑 후진 + 급회전!")
            else:
                # 일반적인 회피
                cmd.angular.z = self.angular_speed * avoidance_intensity * 1.3
                cmd.linear.x = max(0.03, self.linear_speed * (0.2 - avoidance_intensity * 0.15))
                self.get_logger().info(f"🚧 장애물 회피 (강도: {avoidance_intensity:.1f})")
        else:
            # 안전 상태 - **강력한** 경로 추종
            # 목표까지 직선거리가 가까우면 더 공격적으로
            if goal_distance < 1.0:
                # 가까운 목표 - 직진 우선
                if abs(angle_diff) > 0.2:
                    cmd.angular.z = max(-self.angular_speed * 1.2, min(self.angular_speed * 1.2, angle_diff * 2.5))
                    cmd.linear.x = self.linear_speed * 0.6  # 빠르게 회전
                else:
                    cmd.angular.z = angle_diff * 1.5
                    cmd.linear.x = self.linear_speed * 0.9  # 거의 최대속도
                    
                self.get_logger().info(f"🎯 목표 근접! 직진 모드 (거리: {goal_distance:.2f}m)")
            else:
                # 일반적인 경로 추종
                if abs(angle_diff) > 0.3:
                    cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))
                    cmd.linear.x = self.linear_speed * 0.5
                else:
                    cmd.angular.z = angle_diff * 1.5
                    cmd.linear.x = self.linear_speed * 0.8
        
        self.cmd_pub.publish(cmd)
    
    def get_lookahead_point(self):
        """Enhanced lookahead point selection"""
        if not self.current_path:
            return None
        
        # Advance past close waypoints
        while self.path_index < len(self.current_path) - 1:
            current_point = self.current_path[self.path_index]
            distance = math.sqrt((current_point[0] - self.robot_pos[0])**2 + 
                               (current_point[1] - self.robot_pos[1])**2)
            
            if distance < 0.25:
                self.path_index += 1
            else:
                break
        
        # Find optimal lookahead point
        best_point = None
        best_score = float('inf')
        
        for i in range(self.path_index, len(self.current_path)):
            point = self.current_path[i]
            distance = math.sqrt((point[0] - self.robot_pos[0])**2 + 
                               (point[1] - self.robot_pos[1])**2)
            
            # Score based on distance to desired lookahead
            score = abs(distance - self.lookahead_distance)
            if score < best_score:
                best_point = point
                best_score = score
            
            if distance >= self.lookahead_distance * 0.9:
                return point
        
        return best_point if best_point else self.current_path[-1]
    
    def check_immediate_obstacle(self):
        """Enhanced emergency obstacle detection"""
        if self.lidar_data is None:
            return False, 0.0  # No obstacle, turn direction
        
        n_points = len(self.lidar_data)
        
        # Check multiple zones
        front_narrow = n_points // 10  # 18도 정면
        front_wide = n_points // 6     # 60도 앞쪽
        side_check = n_points // 8     # 양옆 체크
        
        # 1. 정면 좁은 각도 (매우 위험)
        front_start = n_points // 2 - front_narrow
        front_end = n_points // 2 + front_narrow
        front_distances = self.lidar_data[front_start:front_end]
        front_valid = front_distances[(front_distances > 0.05) & (front_distances < 8.0)]
        
        # 먼저 좌우 데이터도 계산
        left_start = n_points // 4 - side_check
        left_end = n_points // 4 + side_check
        right_start = 3 * n_points // 4 - side_check
        right_end = 3 * n_points // 4 + side_check
        
        left_distances = self.lidar_data[left_start:left_end]
        right_distances = self.lidar_data[right_start:right_end]
        
        left_valid = left_distances[(left_distances > 0.05) & (left_distances < 8.0)]
        right_valid = right_distances[(right_distances > 0.05) & (right_distances < 8.0)]
        
        left_min = np.min(left_valid) if len(left_valid) > 0 else 10.0
        right_min = np.min(right_valid) if len(right_valid) > 0 else 10.0
        
        if len(front_valid) > 0:
            min_front = np.min(front_valid)
            # 디버그: 항상 정면 거리 출력
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 20 == 0:  # 2초마다
                self.get_logger().info(f"📡 정면: {min_front:.3f}m, 좌: {left_min:.3f}m, 우: {right_min:.3f}m")
            
            if min_front < 0.25:  # 25cm - 제동거리 고려해서 더 일찍!
                self.get_logger().error(f"💀🛑 응급정지! 거리: {min_front:.3f}m")
                return True, 999.0  # 특수 신호: 응급정지
            elif min_front < 0.35:  # 35cm 매우 위험
                self.get_logger().error(f"💀 충돌 직전! 거리: {min_front:.3f}m")
                return True, 1.0  # 급격한 회전 필요
            elif min_front < 0.50:  # 50cm 위험 (더 일찍 감지)
                self.get_logger().warn(f"🚨 위험! 거리: {min_front:.3f}m")
                return True, 0.8
        
        # 2. 정면 넓은 각도 체크
        wide_start = n_points // 2 - front_wide
        wide_end = n_points // 2 + front_wide
        wide_distances = self.lidar_data[wide_start:wide_end]
        wide_valid = wide_distances[(wide_distances > 0.05) & (wide_distances < 8.0)]
        
        if len(wide_valid) > 0:
            min_wide = np.min(wide_valid)
            if min_wide < 0.3:  # 30cm 넓은 각도 위험
                return True, 0.6
        
        # 3. 좌우 사각지대 체크 (이미 위에서 계산됨)
        
        # 양쪽이 모두 가까우면 후진 필요
        if left_min < 0.25 and right_min < 0.25:
            self.get_logger().warn("🚨 양쪽 완전 막힘! 고속 후진!")
            return True, -1.0  # 특수 신호: 후진
        
        # 한쪽만 가까우면 반대쪽으로 회전 (제동거리 고려해서 더 일찍)
        if left_min < 0.30:
            self.get_logger().warn(f"⬅️ 좌측 위험 {left_min:.3f}m - 우측 회전")
            return True, -0.8  # 오른쪽으로 더 강하게
        if right_min < 0.30:
            self.get_logger().warn(f"➡️ 우측 위험 {right_min:.3f}m - 좌측 회전")
            return True, 0.8   # 왼쪽으로 더 강하게
        
        return False, 0.0
    
    def publish_map(self):
        """Publish enhanced occupancy grid"""
        if self.grid_updates == 0:
            return
            
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "odom"
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_size
        grid_msg.info.height = self.map_size
        grid_msg.info.origin.position.x = self.map_origin[0]
        grid_msg.info.origin.position.y = self.map_origin[1]
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        grid_msg.data = self.inflated_grid.flatten().tolist()
        self.map_pub.publish(grid_msg)
    
    def publish_path_markers(self):
        """Publish path visualization"""
        if not self.current_path:
            return
            
        marker_array = MarkerArray()
        
        # Path line
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.08
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        for point in self.current_path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Goal marker
        goal_marker = Marker()
        goal_marker.header = line_marker.header
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.goal_pos[0]
        goal_marker.pose.position.y = self.goal_pos[1]
        goal_marker.pose.position.z = 0.2
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.3
        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        marker_array.markers.append(goal_marker)
        
        # Current target
        if self.path_index < len(self.current_path):
            target_marker = Marker()
            target_marker.header = line_marker.header
            target_marker.id = 2
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            current_target = self.current_path[self.path_index]
            target_marker.pose.position.x = current_target[0]
            target_marker.pose.position.y = current_target[1]
            target_marker.pose.position.z = 0.15
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.2
            target_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            
            marker_array.markers.append(target_marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    navigator = LidarOnlyNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()