#!/usr/bin/env python3

"""
Professional DQN Agent with Multi-Modal Sensor Fusion
- YOLO8 vision integration
- LIDAR sensor fusion  
- Advanced reward shaping
- Experience replay with prioritization
- Professional logging and metrics
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import time
import os
import json
from datetime import datetime

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class MultiModalDQN(nn.Module):
    """Advanced DQN with separate encoders for different sensor modalities"""
    
    def __init__(self, lidar_size=24, vision_size=45, action_size=7):
        super(MultiModalDQN, self).__init__()
        
        # LIDAR encoder
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Vision encoder (YOLO detections)
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128, 256),  # 64 + 64 from encoders
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Action value heads (Dueling DQN architecture)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, action_size)
        
    def forward(self, lidar_input, vision_input):
        # Encode sensor inputs
        lidar_features = self.lidar_encoder(lidar_input)
        vision_features = self.vision_encoder(vision_input)
        
        # Fuse features
        fused_features = torch.cat([lidar_features, vision_features], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        # Dueling architecture
        value = self.value_head(fused_output)
        advantage = self.advantage_head(fused_output)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for more efficient learning"""
    
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done, max_priority))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done, max_priority)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class ProfessionalDQNAgent(Node):
    def __init__(self):
        super().__init__('dqn_agent')
        
        # Network parameters
        self.lidar_size = 24
        self.vision_size = 45  # 5 objects * 9 features from YOLO
        self.action_size = 7   # Extended action set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.q_network = MultiModalDQN(self.lidar_size, self.vision_size, self.action_size).to(self.device)
        self.target_network = MultiModalDQN(self.lidar_size, self.vision_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_frequency = 1000
        self.learning_frequency = 4
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        self.step_count = 0
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.yolo_sub = self.create_subscription(Float32MultiArray, '/yolo/detections', self.yolo_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # State variables
        self.lidar_data = np.zeros(self.lidar_size)
        self.vision_data = np.zeros(self.vision_size)
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.goal_pos = np.array([5.0, 5.0])  # Default goal
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.training_metrics = {
            'episodes': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'success_rate': 0.0,
            'collision_rate': 0.0
        }
        
        # Action mapping
        self.actions = {
            0: [0.0, 0.0],      # Stop
            1: [0.3, 0.0],      # Forward
            2: [0.15, 0.3],     # Forward + Left
            3: [0.15, -0.3],    # Forward + Right
            4: [0.0, 0.5],      # Rotate Left
            5: [0.0, -0.5],     # Rotate Right
            6: [-0.1, 0.0],     # Backward
        }
        
        # Create directories
        self.model_dir = "ai_robot_models"
        self.log_dir = "training_logs"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Update target network initially
        self.update_target_network()
        
        self.get_logger().info(f"Professional DQN Agent initialized on {self.device}")
        self.get_logger().info(f"Action space: {len(self.actions)} actions")
        
    def lidar_callback(self, msg):
        """Process LIDAR data"""
        if len(msg.ranges) > 0:
            # Sample and normalize LIDAR data
            ranges = np.array(msg.ranges)
            ranges = np.where(np.isfinite(ranges), ranges, 10.0)  # Replace inf with max range
            ranges = np.clip(ranges, 0.1, 10.0)  # Clamp values
            
            # Sample 24 readings evenly distributed
            indices = np.linspace(0, len(ranges)-1, self.lidar_size).astype(int)
            sampled_ranges = ranges[indices]
            
            # Normalize to 0-1
            self.lidar_data = sampled_ranges / 10.0
    
    def yolo_callback(self, msg):
        """Process YOLO detection data"""
        if len(msg.data) == self.vision_size:
            self.vision_data = np.array(msg.data)
        else:
            self.vision_data = np.zeros(self.vision_size)
    
    def odom_callback(self, msg):
        """Process odometry data"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        
        # Calculate yaw from quaternion
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    def get_state(self):
        """Get current state combining all sensors"""
        # Goal information
        goal_distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        goal_angle = np.arctan2(self.goal_pos[1] - self.robot_pos[1], 
                               self.goal_pos[0] - self.robot_pos[0]) - self.robot_yaw
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))  # Normalize angle
        
        # Add goal information to LIDAR data
        lidar_with_goal = np.concatenate([
            self.lidar_data,
            [goal_distance / 10.0],  # Normalized distance
            [goal_angle / np.pi]     # Normalized angle
        ])
        
        return lidar_with_goal[:self.lidar_size], self.vision_data
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        lidar_state, vision_state = state
        with torch.no_grad():
            lidar_tensor = torch.FloatTensor(lidar_state).unsqueeze(0).to(self.device)
            vision_tensor = torch.FloatTensor(vision_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(lidar_tensor, vision_tensor)
            return q_values.max(1)[1].item()
    
    def execute_action(self, action):
        """Execute action on robot"""
        linear_vel, angular_vel = self.actions[action]
        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(twist)
    
    def calculate_reward(self):
        """Advanced reward function with multiple objectives"""
        reward = 0.0
        done = False
        
        # Goal reaching reward
        goal_distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        if goal_distance < 0.5:
            reward += 200  # Large reward for reaching goal
            done = True
            self.get_logger().info("🎯 Goal reached!")
        else:
            # Reward for getting closer to goal
            reward += max(0, (10.0 - goal_distance) * 5)
        
        # Collision avoidance
        min_distance = np.min(self.lidar_data * 10.0)  # Convert back to meters
        if min_distance < 0.2:
            reward -= 100  # Collision penalty
            done = True
            self.get_logger().info("💥 Collision detected!")
        elif min_distance < 0.5:
            reward -= (0.5 - min_distance) * 50  # Proximity penalty
        
        # Vision-based rewards
        vision_reward = self.calculate_vision_reward()
        reward += vision_reward
        
        # Exploration bonus (encourage movement)
        if hasattr(self, 'previous_pos'):
            movement = np.linalg.norm(self.robot_pos - self.previous_pos)
            reward += movement * 5
        self.previous_pos = self.robot_pos.copy()
        
        # Time penalty (encourage efficiency)
        reward -= 0.5
        
        # Episode timeout
        self.episode_steps += 1
        if self.episode_steps > 2000:
            done = True
            reward -= 50
        
        return reward, done
    
    def calculate_vision_reward(self):
        """Calculate reward based on YOLO detections"""
        reward = 0.0
        
        # Reshape vision data (5 objects * 9 features)
        vision_reshaped = self.vision_data.reshape(5, 9)
        
        for obj_data in vision_reshaped:
            if obj_data[4] > 0.5:  # Confidence threshold
                obj_type = int(obj_data[5])  # Class ID
                distance = obj_data[6]
                is_obstacle = obj_data[7]
                is_target = obj_data[8]
                
                if is_obstacle > 0.5:
                    # Penalty for being close to obstacles
                    if distance < 2.0:
                        reward -= (2.0 - distance) * 10
                elif is_target > 0.5:
                    # Reward for being near targets
                    if distance < 1.0:
                        reward += (1.0 - distance) * 20
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory.buffer) < self.batch_size:
            return
        
        # Sample batch with prioritized replay
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        experiences, indices, weights = batch
        
        # Prepare batch data
        lidar_states = []
        vision_states = []
        actions = []
        rewards = []
        next_lidar_states = []
        next_vision_states = []
        dones = []
        
        for exp in experiences:
            lidar_s, vision_s = exp.state
            next_lidar_s, next_vision_s = exp.next_state
            
            lidar_states.append(lidar_s)
            vision_states.append(vision_s)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_lidar_states.append(next_lidar_s)
            next_vision_states.append(next_vision_s)
            dones.append(exp.done)
        
        # Convert to tensors
        lidar_states = torch.FloatTensor(lidar_states).to(self.device)
        vision_states = torch.FloatTensor(vision_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_lidar_states = torch.FloatTensor(next_lidar_states).to(self.device)
        next_vision_states = torch.FloatTensor(next_vision_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(lidar_states, vision_states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_lidar_states, next_vision_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_lidar_states, next_vision_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss with importance sampling
        td_errors = target_q_values - current_q_values
        weighted_loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        priorities = td_errors.abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, priorities)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, episode):
        """Save model and training metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"dqn_agent_ep{episode}_{timestamp}.pth")
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_metrics': self.training_metrics
        }, model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, f"metrics_ep{episode}_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)
        
        self.get_logger().info(f"📊 Model saved: {model_path}")
    
    def run_training_episode(self):
        """Run one training episode"""
        # Reset episode variables
        self.episode_steps = 0
        self.episode_start_time = time.time()
        total_reward = 0.0
        
        # Get initial state
        state = self.get_state()
        
        while self.episode_steps < 2000:
            # Select and execute action
            action = self.select_action(state, training=True)
            self.execute_action(action)
            
            # Wait for action to take effect
            time.sleep(0.1)
            
            # Get next state and reward
            next_state = self.get_state()
            reward, done = self.calculate_reward()
            total_reward += reward
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train if enough experience
            self.step_count += 1
            if self.step_count % self.learning_frequency == 0:
                self.train_step()
            
            # Update target network
            if self.step_count % self.update_target_frequency == 0:
                self.update_target_network()
            
            # Move to next state
            state = next_state
            
            if done:
                break
                
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Stop robot
        self.execute_action(0)
        
        # Update metrics
        self.episode_rewards.append(total_reward)
        self.training_metrics['episodes'] += 1
        self.training_metrics['total_steps'] += self.episode_steps
        self.training_metrics['avg_reward'] = np.mean(self.episode_rewards)
        
        # Log episode results
        episode_time = time.time() - self.episode_start_time
        self.get_logger().info(
            f"🤖 Episode {self.training_metrics['episodes']}: "
            f"Reward={total_reward:.2f}, "
            f"Steps={self.episode_steps}, "
            f"Time={episode_time:.1f}s, "
            f"ε={self.epsilon:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    agent = ProfessionalDQNAgent()
    
    try:
        # Training loop
        for episode in range(2000):
            agent.run_training_episode()
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                agent.save_model(episode + 1)
                
    except KeyboardInterrupt:
        agent.get_logger().info("🛑 Training interrupted by user")
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()