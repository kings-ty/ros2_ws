import numpy as np

class LaserProcessor:
    """Handle all laser scan processing"""
    
    @staticmethod
    def preprocess_laser(laser_msg):
        """Preprocess laser data for neural network"""
        if laser_msg is None:
            return np.zeros(360)
        
        ranges = np.array(laser_msg.ranges)
        ranges[ranges == float('inf')] = laser_msg.range_max
        ranges[ranges == 0.0] = laser_msg.range_max
        
        # Normalize to [0, 1]
        normalized_ranges = np.clip(ranges / laser_msg.range_max, 0, 1)
        return normalized_ranges
    
    @staticmethod
    def analyze_sectors(laser_msg):
        """Analyze laser data by sectors"""
        if laser_msg is None:
            return None, None, None, None
        
        ranges = np.array(laser_msg.ranges)
        ranges[ranges == float('inf')] = laser_msg.range_max
        ranges[ranges == 0.0] = laser_msg.range_max
        
        num_readings = len(ranges)
        
        # Divide into sectors
        front_sector = ranges[num_readings//3:2*num_readings//3]
        left_sector = ranges[:num_readings//3]
        right_sector = ranges[2*num_readings//3:]
        front_narrow = ranges[5*num_readings//12:7*num_readings//12]
        
        return (
            np.min(front_sector),
            np.min(left_sector),
            np.min(right_sector),
            np.min(front_narrow)
        )
