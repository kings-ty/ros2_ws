class DQNNetwork: pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
from collections import deque
from typing import Tuple, List

class DQNNetwork(nn.Module):
    """Deep Q-Network for autonomous navigation"""
    
    def __init__(self, laser_input_size=360, image_input_size=(3, 60, 80), 
                 action_size=7, hidden_size=256):
        super(DQNNetwork, self).__init__()
        
        # Laser data processing
        self.laser_fc1 = nn.Linear(laser_input_size, 128)
        self.laser_fc2 = nn.Linear(128, 64)
        self.laser_dropout = nn.Dropout(0.2)
        
        # Image processing (CNN)
        self.conv1 = nn.Conv2d(image_input_size[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        conv_output_size = self._get_conv_output_size(image_input_size)
        
        # Fusion network
        fusion_input_size = 64 + conv_output_size  # laser + vision
        self.fusion_fc1 = nn.Linear(fusion_input_size, hidden_size)
        self.fusion_fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def _get_conv_output_size(self, input_size):
        """Calculate conv layers output size"""
        dummy_input = torch.zeros(1, *input_size)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)
    
    def forward(self, laser_data, image_data):
        # Process laser data
        laser_out = F.relu(self.laser_fc1(laser_data))
        laser_out = self.laser_dropout(laser_out)
        laser_out = F.relu(self.laser_fc2(laser_out))
        
        # Process image data
        image_out = F.relu(self.conv1(image_data))
        image_out = F.relu(self.conv2(image_out))
        image_out = F.relu(self.conv3(image_out))
        image_out = image_out.view(image_out.size(0), -1)
        
        # Combine features
        combined = torch.cat([laser_out, image_out], dim=1)
        
        # Final processing
        x = F.relu(self.fusion_fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fusion_fc2(x))
        q_values = self.output_fc(x)
        
        return q_values

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, laser_size=360, image_size=(3, 60, 80), action_size=7,
                 lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, batch_size=32, memory_size=10000):
        
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(laser_size, image_size, action_size).to(self.device)
        self.target_network = DQNNetwork(laser_size, image_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory
        self.memory = ReplayBuffer(memory_size)
        
        # Training parameters
        self.update_frequency = 4
        self.target_update_frequency = 100
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
        
        print(f"DQN Agent initialized on {self.device}")
    
    def preprocess_state(self, laser_data, image_data):
        """Preprocess state data for neural network"""
        # Preprocess laser data
        if laser_data is not None:
            ranges = np.array(laser_data.ranges)
            ranges[ranges == float('inf')] = laser_data.range_max
            ranges = np.clip(ranges / laser_data.range_max, 0, 1)  # Normalize
        else:
            ranges = np.zeros(360)
        
        # Preprocess image data
        if image_data is not None:
            # Resize and normalize image
            resized_image = cv2.resize(image_data, (80, 60))
            normalized_image = resized_image.astype(np.float32) / 255.0
            # Convert to CHW format
            processed_image = np.transpose(normalized_image, (2, 0, 1))
        else:
            processed_image = np.zeros((3, 60, 80))
        
        return ranges, processed_image
    
    def act(self, laser_data, image_data, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        laser_state, image_state = self.preprocess_state(laser_data, image_data)
        
        laser_tensor = torch.FloatTensor(laser_state).unsqueeze(0).to(self.device)
        image_tensor = torch.FloatTensor(image_state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(laser_tensor, image_tensor)
        self.q_network.train()
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, laser_data, image_data, action, reward, 
                 next_laser_data, next_image_data, done):
        """Store experience in replay buffer"""
        state = self.preprocess_state(laser_data, image_data)
        next_state = self.preprocess_state(next_laser_data, next_image_data)
        
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.batch_size)
        
        # Separate batch elements
        states = [e[0] for e in batch]
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = [e[3] for e in batch]
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Convert states to tensors
        laser_states = torch.FloatTensor([s[0] for s in states]).to(self.device)
        image_states = torch.FloatTensor([s[1] for s in states]).to(self.device)
        next_laser_states = torch.FloatTensor([s[0] for s in next_states]).to(self.device)
        next_image_states = torch.FloatTensor([s[1] for s in next_states]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(laser_states, image_states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_laser_states, next_image_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

