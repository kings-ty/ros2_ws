import numpy as np

class LaserProcessor:
    @staticmethod
    def analyze_sectors(laser_msg):
        if laser_msg is None:
            return None, None, None, None
        
        ranges = np.array(laser_msg.ranges)
        ranges[ranges == float('inf')] = laser_msg.range_max
        ranges[ranges == 0.0] = laser_msg.range_max
        
        num_readings = len(ranges)
        front = ranges[num_readings//3:2*num_readings//3]
        left = ranges[:num_readings//3]
        right = ranges[2*num_readings//3:]
        front_narrow = ranges[5*num_readings//12:7*num_readings//12]
        
        return np.min(front), np.min(left), np.min(right), np.min(front_narrow)
