import numpy as np

class RewardCalculator:
    """Advanced reward calculation for reinforcement learning"""
    
    def __init__(self, config):
        self.config = config
        
        # Reward weights
        self.collision_penalty = -10.0
        self.progress_reward = 2.0
        self.target_following_reward = 5.0
        self.target_centered_reward = 3.0
        self.obstacle_avoidance_reward = 1.0
        self.exploration_reward = 0.5
        self.efficiency_bonus = 1.0
        self.idle_penalty = -0.2
        
        # Distance thresholds
        self.collision_threshold = 0.25
        self.danger_threshold = 0.5
        self.safe_threshold = 1.0
        
        # Vision thresholds
        self.target_center_threshold = 0.15  # Normalized image coordinates
        self.target_distance_ideal = 1.5  # meters
        
    def calculate_reward(self, laser_data, action_index, detections=None, 
                        target_object=None, previous_position=None, 
                        current_position=None):
        """
        Calculate comprehensive reward based on multiple factors
        """
        total_reward = 0.0
        reward_breakdown = {}
        
        if laser_data is None:
            return 0.0, {}
        
        # Process laser data
        ranges = np.array(laser_data.ranges)
        ranges[ranges == float('inf')] = laser_data.range_max
        ranges[ranges == 0.0] = laser_data.range_max
        min_distance = np.min(ranges)
        
        # 1. Collision and safety rewards
        collision_reward = self._calculate_collision_reward(min_distance)
        total_reward += collision_reward
        reward_breakdown['collision'] = collision_reward
        
        # 2. Progress and movement rewards
        movement_reward = self._calculate_movement_reward(action_index, min_distance)
        total_reward += movement_reward
        reward_breakdown['movement'] = movement_reward
        
        # 3. Target following rewards
        if target_object:
            target_reward = self._calculate_target_reward(target_object, action_index)
            total_reward += target_reward
            reward_breakdown['target'] = target_reward
        
        # 4. Obstacle avoidance rewards
        avoidance_reward = self._calculate_avoidance_reward(ranges, action_index)
        total_reward += avoidance_reward
        reward_breakdown['avoidance'] = avoidance_reward
        
        # 5. Exploration rewards
        exploration_reward = self._calculate_exploration_reward(min_distance, action_index)
        total_reward += exploration_reward
        reward_breakdown['exploration'] = exploration_reward
        
        # 6. Efficiency rewards
        efficiency_reward = self._calculate_efficiency_reward(action_index, detections)
        total_reward += efficiency_reward
        reward_breakdown['efficiency'] = efficiency_reward
        
        return total_reward, reward_breakdown
    
    def _calculate_collision_reward(self, min_distance):
        """Calculate collision and safety-related rewards"""
        if min_distance < self.collision_threshold:
            return self.collision_penalty
        elif min_distance < self.danger_threshold:
            # Penalty increases as we get closer to collision
            danger_factor = (self.danger_threshold - min_distance) / self.danger_threshold
            return -2.0 * danger_factor
        elif min_distance > self.safe_threshold:
            # Small bonus for maintaining safe distance
            return 0.5
        return 0.0
    
    def _calculate_movement_reward(self, action_index, min_distance):
        """Calculate rewards for movement and progress"""
        linear_vel = self.config.ACTIONS[action_index][0]
        angular_vel = abs(self.config.ACTIONS[action_index][1])
        
        reward = 0.0
        
        # Forward movement reward (when safe)
        if linear_vel > 0 and min_distance > self.danger_threshold:
            reward += linear_vel * self.progress_reward
        
        # Penalty for stopping without reason
        if action_index == 0 and min_distance > self.safe_threshold:
            reward += self.idle_penalty
        
        # Penalty for excessive rotation
        if angular_vel > 0.5:
            reward -= 0.3
        
        return reward
    
    def _calculate_target_reward(self, target_object, action_index):
        """Calculate rewards for target following behavior"""
        reward = 0.0
        
        # Base reward for detecting target
        reward += self.target_following_reward
        
        # Bonus for keeping target centered
        image_center_x = 160  # Assuming 320px width
        target_x = target_object.get('center_x', image_center_x)
        x_error_normalized = abs(target_x - image_center_x) / image_center_x
        
        if x_error_normalized < self.target_center_threshold:
            reward += self.target_centered_reward
        
        # Distance-based reward
        estimated_distance = max(0.5, 8000 / target_object.get('area', 1000))
        distance_error = abs(estimated_distance - self.target_distance_ideal)
        
        if distance_error < 0.5:
            reward += 2.0  # Good following distance
        elif distance_error > 2.0:
            reward -= 1.0  # Too far or too close
        
        return reward
    
    def _calculate_avoidance_reward(self, ranges, action_index):
        """Calculate rewards for intelligent obstacle avoidance"""
        num_readings = len(ranges)
        
        # Analyze sectors
        front = ranges[num_readings//3:2*num_readings//3]
        left = ranges[:num_readings//3]
        right = ranges[2*num_readings//3:]
        
        min_front = np.min(front)
        min_left = np.min(left)
        min_right = np.min(right)
        
        reward = 0.0
        angular_vel = self.config.ACTIONS[action_index][1]
        
        # Reward for turning toward more open space
        if min_front < self.danger_threshold:
            if angular_vel > 0 and min_left > min_right:  # Turning left when left is clearer
                reward += self.obstacle_avoidance_reward
            elif angular_vel < 0 and min_right > min_left:  # Turning right when right is clearer
                reward += self.obstacle_avoidance_reward
        
        return reward
    
    def _calculate_exploration_reward(self, min_distance, action_index):
        """Calculate rewards for exploration behavior"""
        linear_vel = self.config.ACTIONS[action_index][0]
        
        # Reward exploration in open areas
        if min_distance > 2.0 and linear_vel > 0:
            return self.exploration_reward
        
        return 0.0
    
    def _calculate_efficiency_reward(self, action_index, detections):
        """Calculate rewards for efficient behavior"""
        reward = 0.0
        
        # Bonus for smooth, decisive actions
        if action_index in [1, 2, 3]:  # Forward movements
            reward += 0.2
        
        # Bonus for object detection engagement
        if detections and len(detections) > 0:
            reward += 0.3
        
        return reward
    
    def get_reward_summary(self, reward_breakdown):
        """Get human-readable reward summary"""
        total = sum(reward_breakdown.values())
        summary = f"Total: {total:.2f} = "
        
        for component, value in reward_breakdown.items():
            if abs(value) > 0.01:  # Only show significant components
                summary += f"{component}:{value:.2f} "
        
        return summary

