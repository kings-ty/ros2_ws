from ..config.parameters import RobotConfig

class NavigationController:
    """Controls robot navigation decisions using your existing action space"""
    
    def __init__(self, config=None):
        self.config = config or RobotConfig()
    
    def get_search_action(self):
        """Action for searching/exploring behavior"""
        return 1  # FORWARD (from your action_space)
    
    def get_follow_action(self, target_object, image_width):
        """Action for following a target object"""
        if not target_object:
            return 0  # STOP
        
        # Calculate target position relative to image center
        image_center_x = image_width / 2
        target_x = target_object['center_x']
        x_error = target_x - image_center_x
        x_error_normalized = x_error / (image_width / 2)
        
        # Estimate distance based on object area
        estimated_distance = max(0.5, 8000 / target_object['area'])
        
        # Decision logic using your action indices
        if abs(x_error_normalized) < 0.2:  # Target is centered
            if estimated_distance > 2.0:
                return 1  # FORWARD - approach target
            elif estimated_distance < 1.0:
                return 6  # BACKWARD - too close
            else:
                return 0  # STOP - good distance
        elif x_error_normalized > 0.3:
            return 3  # FORWARD_RIGHT - target is right
        elif x_error_normalized < -0.3:
            return 2  # FORWARD_LEFT - target is left
        else:
            return 1  # FORWARD - default
    
    def get_avoid_action(self, min_front, min_left, min_right):
        """Action for obstacle avoidance"""
        if min_left > min_right + 0.3:
            return 4  # TURN_LEFT - more space on left
        elif min_right > min_left + 0.3:
            return 5  # TURN_RIGHT - more space on right
        else:
            return 4  # TURN_LEFT - default
    
    def get_emergency_action(self, min_left, min_right):
        """Action for emergency situations"""
        return 6  # BACKWARD
    
    def get_action_description(self, action_index):
        """Get human-readable description of action"""
        action_names = [
            "STOP", "FORWARD", "FORWARD_LEFT", "FORWARD_RIGHT",
            "TURN_LEFT", "TURN_RIGHT", "BACKWARD"
        ]
        
        if 0 <= action_index < len(action_names):
            return action_names[action_index]
        return "UNKNOWN"
