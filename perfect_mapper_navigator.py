#!/usr/bin/env python3

"""
완벽한 맵퍼 + 네비게이터
- LIDAR로 전체 맵을 정확하게 그리기
- 장애물 피해서 이쁜 곡선 경로 생성
- RViz에서 녹색 선으로 안전한 경로 표시
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

class PerfectMapperNavigator(Node):
    def __init__(self):
        super().__init__('perfect_mapper_navigator')
        
        # Subscribers
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
        self.timer = self.create_timer(0.1, self.navigate)
        self.map_timer = self.create_timer(0.3, self.publish_map)  # 맵 자주 업데이트
        self.exploration_timer = self.create_timer(2.0, self.auto_explore)  # 자동 탐사
        
        # Robot state
        self.robot_pos = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.goal_pos = [3.0, 3.0]
        self.lidar_data = None
        self.has_goal = False
        
        # Enhanced mapping parameters
        self.map_resolution = 0.03  # 3cm 초정밀!
        self.map_size = 600  # 18m x 18m
        self.map_origin = [-9.0, -9.0]
        
        # 맵 데이터 구조들
        self.occupancy_grid = np.full((self.map_size, self.map_size), 50, dtype=np.int8)  # Unknown
        self.hit_count = np.zeros((self.map_size, self.map_size), dtype=np.int32)  # 장애물 감지 횟수
        self.miss_count = np.zeros((self.map_size, self.map_size), dtype=np.int32)  # 빈공간 감지 횟수
        self.total_scans = 0
        
        # Path planning
        self.current_path = []
        self.path_index = 0
        self.robot_radius = 0.2  # 로봇 반지름
        
        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.6
        self.goal_threshold = 0.25
        self.lookahead_distance = 0.5
        
        # Exploration for better mapping
        self.explored_positions = set()
        self.exploration_targets = [(2, 2), (-2, 2), (-2, -2), (2, -2), (4, 0), (-4, 0), (0, 4), (0, -4)]
        self.current_exploration_target = 0
        
        self.get_logger().info("🗺️ 완벽한 맵퍼 + 네비게이터 시작!")
        self.get_logger().info("🎯 먼저 주변을 탐사해서 완벽한 맵을 만듭니다!")
        
    def lidar_callback(self, msg):
        """고품질 LIDAR 맵핑"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.lidar_data = ranges
        
        # 고품질 맵 업데이트
        self.update_high_quality_map(msg)
        
    def update_high_quality_map(self, lidar_msg):
        """초정밀 맵 생성 - 모든 장애물을 확실히 잡기"""
        ranges = np.array(lidar_msg.ranges)
        angle_min = lidar_msg.angle_min
        angle_increment = lidar_msg.angle_increment
        
        robot_grid_x = int((self.robot_pos[0] - self.map_origin[0]) / self.map_resolution)
        robot_grid_y = int((self.robot_pos[1] - self.map_origin[1]) / self.map_resolution)
        
        if not (0 <= robot_grid_x < self.map_size and 0 <= robot_grid_y < self.map_size):
            return
        
        self.total_scans += 1
        
        # 모든 LIDAR 광선 처리
        for i in range(0, len(ranges), 2):  # 모든 포인트 처리 (속도 위해 2개씩 건너뛰기)
            range_val = ranges[i]
            
            # 유효한 데이터만
            if range_val < 0.05 or range_val > 7.0:
                continue
            
            angle = angle_min + i * angle_increment + self.robot_yaw
            
            # 장애물 위치
            obs_x = self.robot_pos[0] + range_val * math.cos(angle)
            obs_y = self.robot_pos[1] + range_val * math.sin(angle)
            
            obs_grid_x = int((obs_x - self.map_origin[0]) / self.map_resolution)
            obs_grid_y = int((obs_y - self.map_origin[1]) / self.map_resolution)
            
            # Bresenham 라인 알고리즘으로 광선 경로 그리기
            line_points = self.bresenham_line(robot_grid_x, robot_grid_y, obs_grid_x, obs_grid_y)
            
            # 빈 공간 표시 (장애물까지의 경로)
            for j, (x, y) in enumerate(line_points[:-2]):  # 끝 부분 제외
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.miss_count[y, x] += 1
            
            # 장애물 표시
            if 0 <= obs_grid_x < self.map_size and 0 <= obs_grid_y < self.map_size and range_val < 6.5:
                self.hit_count[obs_grid_y, obs_grid_x] += 2  # 장애물에 더 높은 가중치
                
                # 장애물 주변도 약간 표시
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = obs_grid_x + dx, obs_grid_y + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            self.hit_count[ny, nx] += 1
        
        # 확률 기반 occupancy 업데이트
        self.update_occupancy_probabilities()
    
    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham 라인 알고리즘 - 정확한 광선 추적"""
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
    
    def update_occupancy_probabilities(self):
        """확률 기반 occupancy grid 업데이트"""
        for y in range(self.map_size):
            for x in range(self.map_size):
                total_observations = self.hit_count[y, x] + self.miss_count[y, x]
                
                if total_observations > 0:
                    hit_probability = self.hit_count[y, x] / total_observations
                    
                    if hit_probability > 0.7:  # 70% 이상 장애물
                        self.occupancy_grid[y, x] = min(100, int(hit_probability * 100))
                    elif hit_probability < 0.3:  # 30% 이하 빈공간
                        self.occupancy_grid[y, x] = max(0, int(hit_probability * 100))
                    else:  # 애매한 경우
                        self.occupancy_grid[y, x] = 50  # Unknown
    
    def auto_explore(self):
        """자동 탐사 - 더 좋은 맵을 위해"""
        if self.has_goal:
            return  # 사용자 목표가 있으면 탐사 중단
        
        if self.current_exploration_target < len(self.exploration_targets):
            target = self.exploration_targets[self.current_exploration_target]
            self.goal_pos = list(target)
            
            distance_to_target = math.sqrt((target[0] - self.robot_pos[0])**2 + 
                                         (target[1] - self.robot_pos[1])**2)
            
            if distance_to_target < 0.5:  # 목표 근처 도착
                self.current_exploration_target += 1
                self.get_logger().info(f"🗺️ 탐사 지점 {self.current_exploration_target}/{len(self.exploration_targets)} 완료")
            else:
                self.plan_path()  # 탐사 지점으로 경로 계획
    
    def odom_callback(self, msg):
        """로봇 위치 업데이트"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def goal_callback(self, msg):
        """RViz 목표 설정"""
        if msg.header.frame_id == 'base_link':
            goal_x_odom = self.robot_pos[0] + msg.pose.position.x * math.cos(self.robot_yaw) - msg.pose.position.y * math.sin(self.robot_yaw)
            goal_y_odom = self.robot_pos[1] + msg.pose.position.x * math.sin(self.robot_yaw) + msg.pose.position.y * math.cos(self.robot_yaw)
            self.goal_pos = [goal_x_odom, goal_y_odom]
        else:
            self.goal_pos = [msg.pose.position.x, msg.pose.position.y]
        
        self.has_goal = True
        self.current_path = []
        self.path_index = 0
        
        self.get_logger().info(f"🎯 사용자 목표: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}) - 안전한 경로 계획!")
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
        return self.occupancy_grid[grid_y, grid_x] < 60  # 60% 이하만 통과 가능
    
    def get_cell_cost(self, grid_x, grid_y):
        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):
            return float('inf')
        
        cell_value = self.occupancy_grid[grid_y, grid_x]
        
        # 로봇 반지름 고려한 안전 마진
        safety_cost = 0
        radius_cells = int(self.robot_radius / self.map_resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    nx, ny = grid_x + dx, grid_y + dy
                    if (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                        neighbor_value = self.occupancy_grid[ny, nx]
                        if neighbor_value > 70:
                            safety_cost += (neighbor_value - 70) * 0.1
        
        if cell_value > 80 or safety_cost > 50:
            return float('inf')
        
        base_cost = 1.0 + cell_value * 0.01 + safety_cost * 0.02
        return base_cost
    
    def plan_path(self):
        """A* 경로 계획 - 장애물 피해서 이쁜 곡선"""
        start_grid = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
        goal_grid = self.world_to_grid(self.goal_pos[0], self.goal_pos[1])
        
        # 목표 위치 체크
        if not self.is_valid_cell(*goal_grid):
            self.get_logger().warn("❌ 목표 위치 접근 불가!")
            return
        
        path = self.astar(start_grid, goal_grid)
        
        if path:
            # 그리드 경로를 월드 좌표로 변환
            raw_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                raw_path.append([world_x, world_y])
            
            # 경로 스무딩 - 이쁜 곡선 만들기
            self.current_path = self.smooth_path(raw_path)
            self.path_index = 0
            
            self.get_logger().info(f"✅ 안전한 경로 생성! {len(self.current_path)} 포인트")
            self.publish_beautiful_path()
        else:
            self.get_logger().warn("❌ 경로를 찾을 수 없습니다!")
    
    def astar(self, start, goal):
        """A* 알고리즘"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(node):
            x, y = node
            neighbors = []
            # 8방향 연결
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self.is_valid_cell(nx, ny):
                        move_cost = math.sqrt(dx*dx + dy*dy) if dx != 0 and dy != 0 else 1.0
                        cell_cost = self.get_cell_cost(nx, ny)
                        neighbors.append(((nx, ny), move_cost * cell_cost))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # 경로 재구성
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
        """경로 스무딩 - 더 이쁜 곡선"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        # Douglas-Peucker 알고리즘 간소화 버전
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_of_sight(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                smoothed.append(path[i + 1])
                i += 1
        
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        # 추가 스무딩 - 베지어 곡선 스타일
        if len(smoothed) > 3:
            extra_smooth = [smoothed[0]]
            for i in range(1, len(smoothed) - 1):
                # 중간 포인트들 사이에 보간점 추가
                prev_pt = smoothed[i - 1]
                curr_pt = smoothed[i]
                next_pt = smoothed[i + 1]
                
                # 부드러운 곡선을 위한 보간
                smooth_x = (prev_pt[0] + 2*curr_pt[0] + next_pt[0]) / 4
                smooth_y = (prev_pt[1] + 2*curr_pt[1] + next_pt[1]) / 4
                
                extra_smooth.append([smooth_x, smooth_y])
                extra_smooth.append(curr_pt)
            
            extra_smooth.append(smoothed[-1])
            return extra_smooth
        
        return smoothed
    
    def line_of_sight(self, start, end):
        """직선 경로 안전성 체크"""
        start_grid = self.world_to_grid(start[0], start[1])
        end_grid = self.world_to_grid(end[0], end[1])
        
        line_points = self.bresenham_line(start_grid[0], start_grid[1], end_grid[0], end_grid[1])
        
        for x, y in line_points:
            if not self.is_valid_cell(x, y) or self.get_cell_cost(x, y) > 5.0:
                return False
        return True
    
    def publish_beautiful_path(self):
        """아름다운 경로를 RViz에 표시"""
        if not self.current_path:
            return
        
        marker_array = MarkerArray()
        
        # 경로 라인 - 녹색 곡선
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.12  # 두꺼운 라인
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # 반투명 녹색
        
        for point in self.current_path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.05
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # 목표 마커
        goal_marker = Marker()
        goal_marker.header = line_marker.header
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.goal_pos[0]
        goal_marker.pose.position.y = self.goal_pos[1]
        goal_marker.pose.position.z = 0.3
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.4
        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        marker_array.markers.append(goal_marker)
        
        self.marker_pub.publish(marker_array)
        
        self.get_logger().info("🌈 아름다운 곡선 경로 표시 완료!")
    
    def navigate(self):
        """경로 추종"""
        if not self.current_path or self.lidar_data is None:
            return
        
        cmd = Twist()
        
        # 목표 도달 체크
        goal_distance = math.sqrt((self.goal_pos[0] - self.robot_pos[0])**2 + 
                                 (self.goal_pos[1] - self.robot_pos[1])**2)
        
        if goal_distance < self.goal_threshold:
            if self.has_goal:
                self.get_logger().info("🎉 목표 도달!")
                self.has_goal = False
            self.current_path = []
            self.cmd_pub.publish(cmd)
            return
        
        # Pure pursuit
        target_point = self.get_lookahead_point()
        if target_point is None:
            return
        
        # 제어
        target_angle = math.atan2(target_point[1] - self.robot_pos[1], 
                                 target_point[0] - self.robot_pos[0])
        angle_diff = target_angle - self.robot_yaw
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 부드러운 제어
        cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 1.5))
        cmd.linear.x = self.linear_speed * max(0.3, 1.0 - abs(angle_diff) * 0.7)
        
        self.cmd_pub.publish(cmd)
    
    def get_lookahead_point(self):
        """Lookahead 포인트"""
        if not self.current_path:
            return None
        
        # 가까운 웨이포인트 건너뛰기
        while self.path_index < len(self.current_path) - 1:
            current_point = self.current_path[self.path_index]
            distance = math.sqrt((current_point[0] - self.robot_pos[0])**2 + 
                               (current_point[1] - self.robot_pos[1])**2)
            
            if distance < 0.2:
                self.path_index += 1
            else:
                break
        
        # Lookahead 포인트 찾기
        for i in range(self.path_index, len(self.current_path)):
            point = self.current_path[i]
            distance = math.sqrt((point[0] - self.robot_pos[0])**2 + 
                               (point[1] - self.robot_pos[1])**2)
            
            if distance >= self.lookahead_distance:
                return point
        
        return self.current_path[-1] if self.current_path else None
    
    def publish_map(self):
        """고품질 맵 발행"""
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
        
        grid_msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    navigator = PerfectMapperNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()