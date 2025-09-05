class RobotConfig:
    MAX_LINEAR_VEL = 0.2
    MAX_ANGULAR_VEL = 0.8
    SAFE_DISTANCE = 0.6
    EMERGENCY_DISTANCE = 0.3
    
    ACTIONS = [
        [0.0, 0.0],      # 0: Stop
        [0.2, 0.0],      # 1: Forward
        [0.1, 0.5],      # 2: Forward + Left
        [0.1, -0.5],     # 3: Forward + Right
        [0.0, 0.8],      # 4: Turn Left
        [0.0, -0.8],     # 5: Turn Right
        [-0.1, 0.0],     # 6: Backward
    ]
    
    TARGET_CLASSES = ['person']
