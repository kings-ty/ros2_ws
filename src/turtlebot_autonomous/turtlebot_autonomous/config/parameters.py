class RobotConfig:
    # Skid Steer Robot Configuration
    MAX_LINEAR_VEL = 0.3    # Higher speed for 4WD stability
    MAX_ANGULAR_VEL = 0.6   # Lower angular velocity for better control
    SAFE_DISTANCE = 0.65    # Increased slightly for better 0.60m detection margin
    EMERGENCY_DISTANCE = 0.25  # Decreased for more precise emergency detection
    
    # Skid steer specific parameters
    ACCELERATION_LIMIT = 0.5   # m/s² - prevent wheel slipping
    DECELERATION_LIMIT = 0.8   # m/s² - better stopping
    MIN_TURNING_RADIUS = 0.3   # meters - tighter turns possible with skid steer
    WHEEL_BASE = 0.35          # meters - distance between wheel centers
    
    # Enhanced action set optimized for skid steer dynamics
    ACTIONS = [
        [0.0, 0.0],         # 0: Stop
        [0.3, 0.0],         # 1: Forward (faster with 4WD)
        [0.15, 0.4],        # 2: Forward + Left (gentler turn)
        [0.15, -0.4],       # 3: Forward + Right (gentler turn)
        [0.0, 0.6],         # 4: Turn Left (pivot turn)
        [0.0, -0.6],        # 5: Turn Right (pivot turn)
        [-0.15, 0.0],       # 6: Backward (controlled reverse)
        [0.2, 0.2],         # 7: Slow Forward + Left
        [0.2, -0.2],        # 8: Slow Forward + Right
        [-0.1, 0.3],        # 9: Reverse + Left (escape maneuver)
        [-0.1, -0.3],       # 10: Reverse + Right (escape maneuver)
    ]
    
    # Movement smoothing parameters for skid steer
    VELOCITY_SMOOTHING = True
    VELOCITY_ALPHA = 0.7      # Smoothing factor (0.0 = no smoothing, 1.0 = max smoothing)
    
    TARGET_CLASSES = ['person']
