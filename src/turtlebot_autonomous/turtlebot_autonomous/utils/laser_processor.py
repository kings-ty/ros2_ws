import numpy as np

class LaserProcessor:
    @staticmethod
    def analyze_sectors(laser_msg):
        """
        Improved laser analysis with better filtering for accurate distance detection
        """
        if laser_msg is None:
            return None, None, None, None
        
        ranges = np.array(laser_msg.ranges)
        
        # Filter out invalid readings more carefully
        # Remove inf, nan, and zero values, and readings outside sensor range
        valid_mask = (
            np.isfinite(ranges) & 
            (ranges > laser_msg.range_min) & 
            (ranges < laser_msg.range_max) &
            (ranges > 0.0)
        )
        
        # Apply noise filtering - remove readings that are outliers
        # This helps with sensor noise and reflective surfaces
        filtered_ranges = ranges.copy()
        filtered_ranges[~valid_mask] = np.nan
        
        # Additional filtering: median filter to reduce noise
        # Only apply if we have enough valid readings
        if np.sum(valid_mask) > len(ranges) * 0.5:  # At least 50% valid readings
            filtered_ranges = LaserProcessor._median_filter(filtered_ranges, window_size=3)
        
        num_readings = len(filtered_ranges)
        
        # Define sectors with better coverage for robot navigation
        # Front sector: wider coverage for better obstacle detection
        front_start = num_readings // 4  # 25% from start
        front_end = 3 * num_readings // 4  # 75% from start
        front = filtered_ranges[front_start:front_end]
        
        # Left and right sectors
        left = filtered_ranges[:num_readings//3]
        right = filtered_ranges[2*num_readings//3:]
        
        # Narrow front sector: critical area directly ahead
        # More focused than before for precise obstacle detection
        narrow_start = 5 * num_readings // 12  # ~42% from start
        narrow_end = 7 * num_readings // 12    # ~58% from start
        front_narrow = filtered_ranges[narrow_start:narrow_end]
        
        # Calculate minimum distances with NaN handling
        def safe_min(sector):
            """Calculate minimum distance ignoring NaN values"""
            valid_readings = sector[~np.isnan(sector)]
            if len(valid_readings) == 0:
                return laser_msg.range_max  # No valid readings in sector
            return np.min(valid_readings)
        
        min_front = safe_min(front)
        min_left = safe_min(left)
        min_right = safe_min(right)
        min_front_narrow = safe_min(front_narrow)
        
        return min_front, min_left, min_right, min_front_narrow
    
    @staticmethod
    def _median_filter(data, window_size=3):
        """
        Apply median filter to reduce sensor noise
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        filtered_data = data.copy()
        half_window = window_size // 2
        
        for i in range(half_window, len(data) - half_window):
            window = data[i - half_window:i + half_window + 1]
            # Only filter if we have enough valid readings in window
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= window_size // 2:
                filtered_data[i] = np.median(valid_window)
        
        return filtered_data
    
    @staticmethod
    def get_detailed_front_analysis(laser_msg, num_sectors=5):
        """
        Detailed analysis of front area divided into multiple sectors
        Useful for more precise navigation decisions
        """
        if laser_msg is None:
            return None
        
        ranges = np.array(laser_msg.ranges)
        
        # Filter invalid readings
        valid_mask = (
            np.isfinite(ranges) & 
            (ranges > laser_msg.range_min) & 
            (ranges < laser_msg.range_max) &
            (ranges > 0.0)
        )
        
        filtered_ranges = ranges.copy()
        filtered_ranges[~valid_mask] = np.nan
        
        num_readings = len(filtered_ranges)
        
        # Define front area (central 60% of scan)
        front_start = int(num_readings * 0.2)  # 20% from start
        front_end = int(num_readings * 0.8)    # 80% from start
        front_area = filtered_ranges[front_start:front_end]
        
        # Divide front area into sectors
        sector_size = len(front_area) // num_sectors
        sectors = []
        
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = start_idx + sector_size if i < num_sectors - 1 else len(front_area)
            sector = front_area[start_idx:end_idx]
            
            valid_readings = sector[~np.isnan(sector)]
            if len(valid_readings) > 0:
                min_dist = np.min(valid_readings)
                avg_dist = np.mean(valid_readings)
                sectors.append({
                    'min_distance': min_dist,
                    'avg_distance': avg_dist,
                    'valid_readings': len(valid_readings),
                    'sector_index': i
                })
            else:
                sectors.append({
                    'min_distance': laser_msg.range_max,
                    'avg_distance': laser_msg.range_max,
                    'valid_readings': 0,
                    'sector_index': i
                })
        
        return sectors
