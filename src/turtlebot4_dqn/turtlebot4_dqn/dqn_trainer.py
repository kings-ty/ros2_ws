#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import os

from .dqn_environment import TurtleBot4DQNEnv

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.update_target_frequency = 100
        self.step_count = 0
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.q_network.load_state_dict(torch.load(filepath))
            self.update_target_network()

class DQNTrainer(Node):
    def __init__(self):
        super().__init__('dqn_trainer')
        
        # Initialize environment
        self.env = TurtleBot4DQNEnv()
        
        # Initialize DQN agent
        self.agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            learning_rate=0.001
        )
        
        # Training parameters
        self.episodes = 2000
        self.max_steps_per_episode = 1000
        self.scores = deque(maxlen=100)
        
        # Create models directory
        self.model_dir = "turtlebot4_dqn_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.get_logger().info("DQN Trainer initialized!")
        self.get_logger().info(f"Device: {self.agent.device}")
        
    def train(self):
        """Main training loop"""
        self.get_logger().info("Starting DQN training...")
        
        for episode in range(self.episodes):
            # Reset environment
            state = self.env.reset_environment()
            total_reward = 0
            steps = 0
            
            while steps < self.max_steps_per_episode:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train agent
                self.agent.replay()
                
                if done:
                    break
                
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
            
            # Record score
            self.scores.append(total_reward)
            avg_score = np.mean(self.scores)
            
            # Log progress
            self.get_logger().info(
                f"Episode {episode+1}/{self.episodes}, "
                f"Score: {total_reward:.2f}, "
                f"Avg Score: {avg_score:.2f}, "
                f"Epsilon: {self.agent.epsilon:.3f}, "
                f"Steps: {steps}"
            )
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                model_path = os.path.join(self.model_dir, f"turtlebot4_dqn_episode_{episode+1}.pth")
                self.agent.save_model(model_path)
                self.get_logger().info(f"Model saved: {model_path}")
            
            # Early stopping if solved
            if avg_score >= 150 and len(self.scores) >= 100:
                self.get_logger().info(f"Environment solved in {episode+1} episodes!")
                break
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, "turtlebot4_dqn_final.pth")
        self.agent.save_model(final_model_path)
        self.get_logger().info(f"Training completed! Final model saved: {final_model_path}")

def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainer()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()