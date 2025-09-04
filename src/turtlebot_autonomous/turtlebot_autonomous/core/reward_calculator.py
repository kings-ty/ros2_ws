class RewardCalculator:
    """Calculate rewards for reinforcement learning"""
    
    def __init__(self, config):
        self.config = config
        self.collision_penalty = -10.0
        self.progress_reward = 1.0
        self.goal_reward = 10.0
    
    def calculate_reward(self, laser_data, action, detections=None):
        """Calculate reward based on current state and action"""
        if laser_data is None:
            return 0.0
        
        ranges = np.array(laser_data.ranges)
        ranges[ranges == float('inf')] = laser_data.range_max
        min_distance = np.min(ranges)
        
        reward = 0.0
        
        # Collision penalty
        if min_distance < self.config.EMERGENCY_DISTANCE:
            reward += self.collision_penalty
        
        # Forward movement reward
        linear_vel = self.config.ACTIONS[action][0]
        if linear_vel > 0 and min_distance > 0.5:
            reward += linear_vel * self.progress_reward
        
        # Target following reward
        if detections and any(det['class'] in self.config.TARGET_CLASSES for det in detections):
            reward += 2.0
        
        # Obstacle avoidance reward
        if min_distance > 1.0:
            reward += 0.1
        
        # Penalty for erratic movement
        angular_vel = abs(self.config.ACTIONS[action][1])
        if angular_vel > 0.5:
            reward -= 0.05
        
        return reward