"""
Utility Functions and Classes

This module contains helper functions and utility classes.
"""

# Import utility classes
try:
    from .image_processor import ImageProcessor
    from .laser_processor import LaserProcessor
    from .data_logger import DataLogger
    from .visualization import Visualizer
    
    # Utility registry
    UTILITY_CLASSES = {
        "image_processor": ImageProcessor,
        "laser_processor": LaserProcessor,
        "data_logger": DataLogger,
        "visualizer": Visualizer
    }
    
except ImportError as e:
    print(f"Warning: Could not import all utilities: {e}")
    UTILITY_CLASSES = {}

# Common utility functions
import numpy as np
import cv2
import time

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_timestamp():
    """Create timestamp string for logging"""
    return time.strftime("%Y%m%d_%H%M%S")

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default

# Image processing utilities
def resize_image(image, target_size):
    """Resize image to target size"""
    if image is None:
        return None
    return cv2.resize(image, target_size)

def convert_color_space(image, conversion):
    """Convert image color space"""
    if image is None:
        return None
    return cv2.cvtColor(image, conversion)

# Data validation utilities
def validate_laser_data(laser_msg):
    """Validate laser scan data"""
    if laser_msg is None:
        return False
    return len(laser_msg.ranges) > 0

def validate_image_data(image):
    """Validate image data"""
    if image is None:
        return False
    return image.shape[0] > 0 and image.shape[1] > 0

# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitoring utility"""
    def __init__(self):
        self.timers = {}
    
    def start_timer(self, name):
        self.timers[name] = time.time()
    
    def end_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        return 0.0

