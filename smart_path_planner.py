#!/usr/bin/env python3

"""
Smart Path Planner with LIDAR-based Occupancy Grid and A* Algorithm
- LIDAR 데이터로 실시간 장애물 지도 생성
- A* 알고리즘으로 최적 경로 계획
- RViz에서 경로를 라인으로 시각화
- Pure pursuit 알고리즘으로 부드러운 경로 추종
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

class SmartPathPlanner(Node):
    def __init__(self):
        super().__init__('smart_path_planner')
        
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
        
        # Navigation timer
        self.timer = self.create_timer(0.1, self.navigate)
        self.map_timer = self.create_timer(1.0, self.publish_map)  # Update map every second
        
        # Robot state
        self.robot_pos = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.goal_pos = [2.0, 2.0]  # Default goal
        self.lidar_data = None
        
        # Occupancy grid parameters
        self.map_resolution = 0.1  # 10cm per cell
        self.map_size = 200  # 20m x 20m map
        self.map_origin = [-10.0, -10.0]  # Map centered at robot start
        self.occupancy_grid = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  # Unknown = -1
        self.inflated_grid = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  # Inflated version
        self.robot_radius = 0.4  # Robot radius in meters (with safety margin)
        self.grid_updates = 0
        
        # Path planning
        self.current_path = []
        self.path_index = 0
        
        # Navigation parameters
        self.linear_speed = 0.4
        self.angular_speed = 0.8
        self.goal_threshold = 0.3
        self.lookahead_distance = 0.5  # Pure pursuit lookahead (reduced for better tracking)
        
        self.get_logger().info("🧠 Smart Path Planner 시작!")
        self.get_logger().info("🗺️  실시간 LIDAR 맵핑 + A* 경로 계획")
        self.get_logger().info("🎯 RViz '2D Nav Goal'로 목표 설정하세요!")
        
    def lidar_callback(self, msg):
        """Process LIDAR data and update occupancy grid"""
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.lidar_data = ranges
        
        # Update occupancy grid
        self.update_occupancy_grid(msg)
    
    def update_occupancy_grid(self, lidar_msg):
        """Update occupancy grid with LIDAR data"""
        ranges = np.array(lidar_msg.ranges)
        angle_min = lidar_msg.angle_min
        angle_increment = lidar_msg.angle_increment
        
        robot_grid_x = int((self.robot_pos[0] - self.map_origin[0]) / self.map_resolution)
        robot_grid_y = int((self.robot_pos[1] - self.map_origin[1]) / self.map_resolution)
        
        if not (0 <= robot_grid_x < self.map_size and 0 <= robot_grid_y < self.map_size):
            return
        
        for i, range_val in enumerate(ranges):
            if range_val < 0.1 or range_val > 8.0:  # Filter invalid readings
                continue
                
            angle = angle_min + i * angle_increment + self.robot_yaw
            
            # Calculate obstacle position
            obs_x = self.robot_pos[0] + range_val * math.cos(angle)
            obs_y = self.robot_pos[1] + range_val * math.sin(angle)
            
            # Convert to grid coordinates
            grid_x = int((obs_x - self.map_origin[0]) / self.map_resolution)
            grid_y = int((obs_y - self.map_origin[1]) / self.map_resolution)
            
            # Mark obstacle in grid
            if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                self.occupancy_grid[grid_y, grid_x] = 100  # Occupied
                
                # Also mark cells around obstacle as potentially occupied
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if (0 <= nx < self.map_size and 0 <= ny < self.map_size and 
                            self.occupancy_grid[ny, nx] == -1):
                            self.occupancy_grid[ny, nx] = 50  # Partially occupied
            
            # Mark free space along the ray
            steps = int(range_val / self.map_resolution)
            for step in range(0, steps, 2):  # Every other cell to reduce computation
                ray_x = self.robot_pos[0] + step * self.map_resolution * math.cos(angle)
                ray_y = self.robot_pos[1] + step * self.map_resolution * math.sin(angle)
                
                ray_grid_x = int((ray_x - self.map_origin[0]) / self.map_resolution)
                ray_grid_y = int((ray_y - self.map_origin[1]) / self.map_resolution)
                
                if (0 <= ray_grid_x < self.map_size and 0 <= ray_grid_y < self.map_size and
                    self.occupancy_grid[ray_grid_y, ray_grid_x] == -1):
                    self.occupancy_grid[ray_grid_y, ray_grid_x] = 0  # Free space
        
        self.grid_updates += 1
        
        # Apply inflation to create safety zones around obstacles
        self.inflate_obstacles()
    
    def inflate_obstacles(self):
        """Inflate obstacles to create safety margins for the robot"""
        # Start with original occupancy grid
        self.inflated_grid = self.occupancy_grid.copy()
        
        # Calculate inflation radius in grid cells
        inflation_radius = int(self.robot_radius / self.map_resolution)
        
        # Find all obstacle cells
        obstacle_cells = np.where(self.occupancy_grid >= 80)  # High occupancy cells
        
        # Inflate around each obstacle
        for i in range(len(obstacle_cells[0])):
            obs_y, obs_x = obstacle_cells[0][i], obstacle_cells[1][i]
            
            # Create circular inflation around obstacle
            for dy in range(-inflation_radius, inflation_radius + 1):
                for dx in range(-inflation_radius, inflation_radius + 1):
                    # Check if within inflation radius
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= inflation_radius:
                        new_y, new_x = obs_y + dy, obs_x + dx
                        
                        # Check bounds
                        if (0 <= new_x < self.map_size and 0 <= new_y < self.map_size):
                            # Only inflate if cell is free or unknown
                            if self.occupancy_grid[new_y, new_x] <= 25:
                                # Gradient inflation - closer to obstacle = higher cost
                                inflation_cost = int(80 - (distance / inflation_radius) * 60)
                                self.inflated_grid[new_y, new_x] = max(
                                    self.inflated_grid[new_y, new_x], 
                                    inflation_cost
                                )
    
    def odom_callback(self, msg):
        """Update robot position"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def goal_callback(self, msg):
        """Handle new goal and plan path"""
        # Transform goal to odom frame if needed
        if msg.header.frame_id == 'base_link':
            goal_x_odom = self.robot_pos[0] + msg.pose.position.x * math.cos(self.robot_yaw) - msg.pose.position.y * math.sin(self.robot_yaw)
            goal_y_odom = self.robot_pos[1] + msg.pose.position.x * math.sin(self.robot_yaw) + msg.pose.position.y * math.cos(self.robot_yaw)
            self.goal_pos = [goal_x_odom, goal_y_odom]
        else:
            self.goal_pos = [msg.pose.position.x, msg.pose.position.y]
        
        self.get_logger().info(f"🎯 새 목표 설정: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})")
        
        # Clear old path and reset
        self.current_path = []
        self.path_index = 0
        
        # Plan new path
        self.plan_path()
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.map_resolution + self.map_origin[0]
        y = grid_y * self.map_resolution + self.map_origin[1]
        return x, y
    
    def is_valid_cell(self, grid_x, grid_y):
        """Check if grid cell is valid and not in inflated obstacle zone"""
        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):
            return False
        # Use inflated grid for safer path planning
        return self.inflated_grid[grid_y, grid_x] <= 40  # Allow some inflated area but avoid high-cost zones
    
    def get_cell_cost(self, grid_x, grid_y):
        """Get the cost of moving through a cell (for A*)"""
        if not (0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size):
            return float('inf')
        
        cell_value = self.inflated_grid[grid_y, grid_x]
        if cell_value >= 80:  # Obstacle
            return float('inf')
        elif cell_value > 40:  # High inflation zone
            return 10.0 + (cell_value - 40) * 0.5  # High cost but passable in emergency
        elif cell_value > 20:  # Medium inflation zone
            return 2.0 + (cell_value - 20) * 0.2  # Medium cost
        else:  # Free space
            return 1.0  # Normal cost
    
    def plan_path(self):
        """Plan path using A* algorithm"""
        start_grid = self.world_to_grid(self.robot_pos[0], self.robot_pos[1])
        goal_grid = self.world_to_grid(self.goal_pos[0], self.goal_pos[1])
        
        if not self.is_valid_cell(*goal_grid):
            self.get_logger().warn("❌ 목표 위치가 장애물 안에 있습니다!")
            return
        
        path = self.astar(start_grid, goal_grid)
        
        if path:
            # Convert grid path to world coordinates
            raw_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                raw_path.append([world_x, world_y])
            
            # Smooth the path
            self.current_path = self.smooth_path(raw_path)
            
            self.path_index = 0
            self.get_logger().info(f"✅ 경로 계획 완료! {len(self.current_path)} 포인트 (원본: {len(raw_path)})")
            self.publish_path_markers()
        else:
            self.get_logger().warn("❌ 경로를 찾을 수 없습니다!")
    
    def astar(self, start, goal):
        """A* pathfinding algorithm"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(node):
            x, y = node
            neighbors = []
            # 8-connectivity
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self.is_valid_cell(nx, ny):
                        # Base movement cost (diagonal = sqrt(2), straight = 1)
                        base_cost = math.sqrt(dx*dx + dy*dy) if dx != 0 and dy != 0 else 1.0
                        # Add cell occupancy cost
                        cell_cost = self.get_cell_cost(nx, ny)
                        total_cost = base_cost * cell_cost
                        neighbors.append(((nx, ny), total_cost))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start->goal path
            
            for neighbor, move_cost in get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def smooth_path(self, path):
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # Always keep start point
        
        # Line-of-sight path smoothing
        i = 0
        while i < len(path) - 1:
            # Look ahead as far as possible while maintaining line of sight
            j = len(path) - 1
            while j > i + 1:
                if self.has_line_of_sight(path[i], path[j]):
                    # Skip intermediate points
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # Can't skip any points, move to next
                smoothed.append(path[i + 1])
                i += 1
        
        # Ensure goal is included
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        return smoothed
    
    def has_line_of_sight(self, start_pos, end_pos):
        """Check if there's a clear line of sight between two points"""
        x0, y0 = start_pos
        x1, y1 = end_pos
        
        # Number of samples along the line
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        samples = max(int(distance / (self.map_resolution * 0.5)), 5)
        
        # Check points along the line
        for i in range(samples + 1):
            t = i / samples if samples > 0 else 0
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            
            grid_x, grid_y = self.world_to_grid(x, y)
            
            # Check if point is in obstacle or high-cost area
            if not self.is_valid_cell(grid_x, grid_y):
                return False
            
            # Also check if cost is too high (would prefer to avoid)
            if self.get_cell_cost(grid_x, grid_y) > 5.0:
                return False
        
        return True
    
    def navigate(self):
        """Main navigation logic using path following"""
        if not self.current_path or self.lidar_data is None:
            return
            
        cmd = Twist()
        
        # Check if goal reached
        goal_distance = math.sqrt((self.goal_pos[0] - self.robot_pos[0])**2 + 
                                 (self.goal_pos[1] - self.robot_pos[1])**2)
        
        if goal_distance < self.goal_threshold:
            self.get_logger().info("🎉 목표 도달!")
            self.current_path = []
            self.cmd_pub.publish(cmd)
            return
        
        # Pure pursuit path following
        target_point = self.get_lookahead_point()
        
        if target_point is None:
            self.get_logger().warn("⚠️ 추적할 경로 포인트가 없습니다")
            return
        
        # Calculate distance to target point for debugging
        target_distance = math.sqrt((target_point[0] - self.robot_pos[0])**2 + 
                                   (target_point[1] - self.robot_pos[1])**2)
        
        # Calculate steering angle
        target_angle = math.atan2(target_point[1] - self.robot_pos[1], 
                                 target_point[0] - self.robot_pos[0])
        angle_diff = target_angle - self.robot_yaw
        
        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Emergency obstacle avoidance (close obstacles)
        if self.check_immediate_obstacle():
            cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            cmd.linear.x = 0.1
            self.get_logger().info("🚧 비상 장애물 회피!")
        else:
            # Improved path following with better control
            # Angular velocity - more responsive for large angles, smoother for small angles
            if abs(angle_diff) > 0.5:  # Large angle difference
                cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 1.5))
                cmd.linear.x = self.linear_speed * 0.4  # Slow down for sharp turns
            else:  # Small angle difference
                cmd.angular.z = angle_diff * 1.0  # Proportional control
                cmd.linear.x = self.linear_speed * max(0.5, 1.0 - abs(angle_diff) * 0.8)
            
            # Debug output every 10 cycles to avoid spam
            if hasattr(self, '_nav_counter'):
                self._nav_counter += 1
            else:
                self._nav_counter = 0
            
            if self._nav_counter % 10 == 0:
                self.get_logger().info(
                    f"🚗 경로 추종: 포인트 {self.path_index}/{len(self.current_path)}, "
                    f"거리: {target_distance:.2f}m, 각도차: {math.degrees(angle_diff):.1f}°, "
                    f"속도: {cmd.linear.x:.2f}, 회전: {cmd.angular.z:.2f}"
                )
        
        self.cmd_pub.publish(cmd)
    
    def get_lookahead_point(self):
        """Get lookahead point for pure pursuit"""
        if not self.current_path:
            return None
        
        # First, advance past any points that are too close
        while (self.path_index < len(self.current_path) - 1):
            current_point = self.current_path[self.path_index]
            distance = math.sqrt((current_point[0] - self.robot_pos[0])**2 + 
                               (current_point[1] - self.robot_pos[1])**2)
            
            if distance < 0.3:  # If we're very close to current waypoint
                self.path_index += 1
                self.get_logger().info(f"📍 웨이포인트 {self.path_index} 통과")
            else:
                break
        
        # Find point at lookahead distance
        best_point = None
        best_distance_diff = float('inf')
        
        for i in range(self.path_index, len(self.current_path)):
            point = self.current_path[i]
            distance = math.sqrt((point[0] - self.robot_pos[0])**2 + 
                               (point[1] - self.robot_pos[1])**2)
            
            # Find point closest to lookahead distance
            distance_diff = abs(distance - self.lookahead_distance)
            if distance_diff < best_distance_diff:
                best_point = point
                best_distance_diff = distance_diff
            
            # If we found a good lookahead point, use it
            if distance >= self.lookahead_distance * 0.8:  # Allow some tolerance
                return point
        
        # Return the best point we found, or the last point if at the end
        return best_point if best_point else self.current_path[-1]
    
    def check_immediate_obstacle(self):
        """Check for immediate obstacles in front"""
        if self.lidar_data is None:
            return False
        
        # Check front arc (30 degrees each side)
        n_points = len(self.lidar_data)
        front_arc = n_points // 6  # 30 degrees
        
        front_start = n_points // 2 - front_arc
        front_end = n_points // 2 + front_arc
        
        front_distances = self.lidar_data[front_start:front_end]
        valid_distances = front_distances[(front_distances > 0.1) & (front_distances < 10.0)]
        
        if len(valid_distances) > 0:
            min_distance = np.min(valid_distances)
            return min_distance < 0.8  # 80cm emergency threshold
        
        return False
    
    def publish_map(self):
        """Publish occupancy grid map"""
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
        
        # Flatten inflated grid for message (shows safety zones)
        grid_msg.data = self.inflated_grid.flatten().tolist()
        
        self.map_pub.publish(grid_msg)
    
    def publish_path_markers(self):
        """Publish path as visualization markers"""
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
        
        line_marker.scale.x = 0.1  # Line width
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        
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
        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
        
        marker_array.markers.append(goal_marker)
        
        # Current target point marker
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
            target_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue
            
            marker_array.markers.append(target_marker)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    planner = SmartPathPlanner()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()