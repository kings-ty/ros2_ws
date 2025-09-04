"""
Deep Learning Models

This module contains neural network architectures and AI models.
"""

# Import model classes
try:
    from .dqn_model import DQNNetwork, DQNAgent
    from .vision_model import VisionEncoder, ObjectDetector
    
    # Model registry
    MODEL_REGISTRY = {
        "dqn": DQNNetwork,
        "vision_encoder": VisionEncoder,
        "object_detector": ObjectDetector
    }
    
    # Model configuration
    DEFAULT_MODEL_CONFIG = {
        "dqn": {
            "hidden_size": 256,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "vision": {
            "input_size": (3, 240, 320),
            "output_classes": 80
        }
    }
    
except ImportError as e:
    print(f"Warning: Could not import all models: {e}")
    MODEL_REGISTRY = {}
    DEFAULT_MODEL_CONFIG = {}

def create_model(model_type, **kwargs):
    """Factory function to create models"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    default_config = DEFAULT_MODEL_CONFIG.get(model_type, {})
    
    # Merge default config with provided kwargs
    config = {**default_config, **kwargs}
    
    return model_class(**config)

def list_available_models():
    """List all available model types"""
    return list(MODEL_REGISTRY.keys())