#!/usr/bin/env python3
"""
Action Space Definition for TurtleBot Autonomous Navigation

This module defines all possible actions the robot can take.
"""

import numpy as np
from enum import Enum
from typing import List, Tuple

class ActionType(Enum):
    """Enumeration of action types"""
    STOP = 0
    FORWARD = 1
    FORWARD_LEFT = 2
    FORWARD_RIGHT = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    BACKWARD = 6
    SLOW_FORWARD_LEFT = 7
    SLOW_FORWARD_RIGHT = 8

class ActionSpace:
    """
    Defines the action space for the TurtleBot3
    
    Each action is defined as [linear_velocity, angular_velocity]
    """
    
    def __init__(self, max_linear_vel=0.2, max_angular_vel=0.8):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        
        # Define action mappings
        self.actions = {
            ActionType.STOP: [0.0, 0.0],
            ActionType.FORWARD: [self.max_linear_vel, 0.0],
            ActionType.FORWARD_LEFT: [self.max_linear_vel * 0.5, self.max_angular_vel * 0.6],
            ActionType.FORWARD_RIGHT: [self.max_linear_vel * 0.5, -self.max_angular_vel * 0.6],
            ActionType.TURN_LEFT: [0.0, self.max_angular_vel],
            ActionType.TURN_RIGHT: [0.0, -self.max_angular_vel],
            ActionType.BACKWARD: [-self.max_linear_vel * 0.5, 0.0],
            ActionType.SLOW_FORWARD_LEFT: [self.max_linear_vel * 0.25, self.max_angular_vel * 0.3],
            ActionType.SLOW_FORWARD_RIGHT: [self.max_linear_vel * 0.25, -self.max_angular_vel * 0.3],
        }
    
    def get_action_values(self, action_type: ActionType) -> List[float]:
        """Get linear and angular velocities for an action type"""
        return self.actions.get(action_type, [0.0, 0.0])
    
    def get_action_by_index(self, index: int) -> List[float]:
        """Get action values by index"""
        action_types = list(ActionType)
        if 0 <= index < len(action_types):
            return self.actions[action_types[index]]
        return [0.0, 0.0]
    
    def get_action_name(self, index: int) -> str:
        """Get action name by index"""
        action_types = list(ActionType)
        if 0 <= index < len(action_types):
            return action_types[index].name
        return "UNKNOWN"
    
    def get_num_actions(self) -> int:
        """Get total number of actions"""
        return len(self.actions)
    
    def get_all_actions(self) -> List[List[float]]:
        """Get all action values as a list"""
        return [self.actions[action_type] for action_type in ActionType]
    
    def get_discrete_action_space(self) -> np.ndarray:
        """Get action space as numpy array for RL algorithms"""
        return np.array(self.get_all_actions())
    
    def describe_action(self, index: int) -> str:
        """Get human-readable description of action"""
        descriptions = {
            ActionType.STOP: "Stop completely",
            ActionType.FORWARD: "Move forward at full speed",
            ActionType.FORWARD_LEFT: "Move forward while turning left",
            ActionType.FORWARD_RIGHT: "Move forward while turning right", 
            ActionType.TURN_LEFT: "Turn left in place",
            ActionType.TURN_RIGHT: "Turn right in place",
            ActionType.BACKWARD: "Move backward slowly",
            ActionType.SLOW_FORWARD_LEFT: "Slow forward with gentle left turn",
            ActionType.SLOW_FORWARD_RIGHT: "Slow forward with gentle right turn",
        }
        
        action_types = list(ActionType)
        if 0 <= index < len(action_types):
            action_type = action_types[index]
            linear, angular = self.actions[action_type]
            return f"{descriptions[action_type]} (lin: {linear:.2f}, ang: {angular:.2f})"
        
        return "Unknown action"

# Example usage and testing
if __name__ == "__main__":
    # Create action space
    action_space = ActionSpace()
    
    print("=== TurtleBot Action Space ===")
    print(f"Total actions: {action_space.get_num_actions()}")
    print()
    
    # Print all actions
    for i in range(action_space.get_num_actions()):
        print(f"Action {i}: {action_space.describe_action(i)}")
    
    print("\n=== Action Values ===")
    for action_type in ActionType:
        values = action_space.get_action_values(action_type)
        print(f"{action_type.name}: {values}")
    
    print("\n=== Discrete Action Space (for RL) ===")
    discrete_actions = action_space.get_discrete_action_space()
    print(discrete_actions)
