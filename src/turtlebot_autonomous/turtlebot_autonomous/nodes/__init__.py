# ==== turtlebot_autonomous/nodes/__init__.py ====
"""
ROS2 Node implementations

This module contains all ROS2 node classes for autonomous navigation.
"""

# Import all node classes for easy access
try:
    from .main_controller import AutonomousController
    from .perception_node import YoloPerceptionNode
    from .training_node import DQNTrainerNode
    
    # List of available nodes
    AVAILABLE_NODES = [
        "AutonomousController",
        "YoloPerceptionNode", 
        "DQNTrainerNode"
    ]
    
    # Node registry for dynamic loading
    NODE_REGISTRY = {
        "autonomous_controller": AutonomousController,
        "yolo_perception": YoloPerceptionNode,
        "dqn_trainer": DQNTrainerNode
    }
    
except ImportError as e:
    print(f"Warning: Could not import all nodes: {e}")
    AVAILABLE_NODES = []
    NODE_REGISTRY = {}

def get_node_class(node_name):
    """Get node class by name"""
    return NODE_REGISTRY.get(node_name, None)

def list_available_nodes():
    """List all available node names"""
    return list(NODE_REGISTRY.keys())
