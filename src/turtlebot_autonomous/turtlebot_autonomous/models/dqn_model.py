import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    """Modular DQN Network"""
    def __init__(self, laser_input_size=360, action_size=7, hidden_size=256):
        super(DQNNetwork, self).__init__()
        
        # Laser processing
        self.laser_branch = nn.Sequential(
            nn.Linear(laser_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Vision processing
        self.vision_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate vision feature size
        vision_feature_size = self._get_conv_output_size()
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(64 + vision_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 3, 60, 80)
        output = self.vision_branch(dummy_input)
        return output.view(1, -1).size(1)
    
    def forward(self, laser_data, image_data):
        laser_features = self.laser_branch(laser_data)
        
        vision_features = self.vision_branch(image_data)
        vision_features = vision_features.view(vision_features.size(0), -1)
        
        combined = torch.cat([laser_features, vision_features], dim=1)
        q_values = self.fusion(combined)
        
        return q_values