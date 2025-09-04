
# ==== turtlebot_autonomous/config/parameters.py ====
"""Configuration parameters for autonomous navigation"""

class RobotConfig:
    """Central configuration for TurtleBot autonomous navigation"""
    
    # Robot motion parameters
    MAX_LINEAR_VEL = 0.2      # m/s
    MAX_ANGULAR_VEL = 0.8     # rad/s
    SAFE_DISTANCE = 0.6       # meters
    EMERGENCY_DISTANCE = 0.3  # meters
    TARGET_DISTANCE = 1.5     # desired following distance
    
    # Vision parameters
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 240
    CONFIDENCE_THRESHOLD = 0.5
    
    # Object classes
    TARGET_CLASSES = ['person']
    AVOID_CLASSES = ['chair', 'couch', 'table', 'car', 'truck', 'bicycle']
    
    # Action space [linear_vel, angular_vel]
    ACTIONS = [
        [0.0, 0.0],      # 0: STOP
        [0.2, 0.0],      # 1: FORWARD
        [0.1, 0.5],      # 2: FORWARD_LEFT
        [0.1, -0.5],     # 3: FORWARD_RIGHT
        [0.0, 0.8],      # 4: TURN_LEFT
        [0.0, -0.8],     # 5: TURN_RIGHT
        [-0.1, 0.0],     # 6: BACKWARD
        [0.05, 0.3],     # 7: SLOW_LEFT
        [0.05, -0.3],    # 8: SLOW_RIGHT
    ]
    
    # Control parameters
    CONTROL_FREQUENCY = 10.0  # Hz
    YOLO_PROCESS_FREQUENCY = 3  # Process every 3rd frame
    
    # State machine timeouts
    EMERGENCY_TIMEOUT = 2.0   # seconds
    STUCK_TIMEOUT = 5.0       # seconds
