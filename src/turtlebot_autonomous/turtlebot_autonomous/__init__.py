"""
TurtleBot Autonomous Navigation Package

Complete autonomous navigation system with deep learning and computer vision.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config.parameters import RobotConfig
from .core.state_machine import StateMachine, RobotState

def get_version():
    return __version__