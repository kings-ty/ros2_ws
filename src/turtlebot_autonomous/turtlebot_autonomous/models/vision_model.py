#!/usr/bin/env python3
"""
Computer Vision Models for TurtleBot Autonomous Navigation

This module contains vision-related neural network models and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip3 install ultralytics")

class VisionEncoder(nn.Module):
    """
    CNN encoder for processing camera images
    Extracts features from raw images for downstream tasks
    """
    
    def __init__(self, input_channels=3, feature_dim=128):
        super(VisionEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate feature size after convolutions
        self.feature_size = self._get_conv_output_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, feature_dim)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output_size(self, input_channels):
        """Calculate the output size after conv layers"""
        # Assume input size is 240x320 (height x width)
        dummy_input = torch.zeros(1, input_channels, 240, 320)
        x = self.relu(self.conv1(dummy_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass through the encoder"""
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ObjectDetector:
    """
    Wrapper class for YOLO object detection
    Provides easy interface for detecting objects in images
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                print(f"YOLO model {model_name} loaded successfully!")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            print("YOLO not available. Object detection disabled.")
    
    def detect_objects(self, image, target_classes=None, avoid_classes=None):
        """
        Detect objects in the image
        
        Args:
            image: Input image (numpy array)
            target_classes: List of class names to prioritize (e.g., ['person'])
            avoid_classes: List of class names to avoid (e.g., ['car', 'truck'])
        
        Returns:
            List of detection dictionaries
        """
        if self.model is None or image is None:
            return []
        
        try:
            # Run detection
            results = self.model(image, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Determine object category
                        is_target = target_classes and class_name in target_classes
                        is_obstacle = avoid_classes and class_name in avoid_classes
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'area': area,
                            'is_target': is_target,
                            'is_obstacle': is_obstacle
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def get_target_object(self, detections, image_height=240):
        """
        Find the best target object from detections
        Prioritizes larger objects in upper part of image
        """
        target_objects = [det for det in detections if det['is_target']]
        
        if not target_objects:
            return None
        
        # Filter objects in upper part of image (head level)
        head_level_objects = [
            obj for obj in target_objects 
            if obj['center_y'] < image_height * 0.7
        ]
        
        if head_level_objects:
            # Return largest object at head level
            return max(head_level_objects, key=lambda x: x['area'])
        else:
            # Return largest target object anywhere
            return max(target_objects, key=lambda x: x['area'])
    
    def draw_detections(self, image, detections):
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        if image is None:
            return None
        
        display_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on object type
            if detection['is_target']:
                color = (0, 255, 0)  # Green for targets
                thickness = 3
            elif detection['is_obstacle']:
                color = (0, 0, 255)  # Red for obstacles
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(display_image, (int(x1), int(y1)), 
                         (int(x2), int(y2)), color, thickness)
            
            # Draw label
            label = f'{class_name}: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(display_image, 
                         (int(x1), int(y1 - label_size[1] - 10)),
                         (int(x1 + label_size[0]), int(y1)), 
                         color, -1)
            
            # Label text
            cv2.putText(display_image, label, 
                       (int(x1), int(y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return display_image

class DepthEstimator(nn.Module):
    """
    Simple depth estimation from monocular images
    (This is a basic implementation - for production use pre-trained models)
    """
    
    def __init__(self, input_channels=3):
        super(DepthEstimator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._make_encoder_block(input_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Decoder (upsampling)
        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(256, 128)
        self.dec2 = self._make_decoder_block(128, 64)
        self.dec1 = self._make_decoder_block(64, 1)  # Single channel depth
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder
        d4 = self.dec4(e4)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        depth = self.dec1(d2)
        
        return torch.sigmoid(depth)  # Normalize depth to [0, 1]

# Example usage and testing
if __name__ == "__main__":
    print("=== Vision Models Test ===")
    
    # Test VisionEncoder
    print("\n1. Testing VisionEncoder...")
    encoder = VisionEncoder(input_channels=3, feature_dim=128)
    dummy_image = torch.randn(1, 3, 240, 320)
    features = encoder(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output features shape: {features.shape}")
    
    # Test ObjectDetector (if YOLO available)
    print("\n2. Testing ObjectDetector...")
    if YOLO_AVAILABLE:
        detector = ObjectDetector(confidence_threshold=0.5)
        
        # Create dummy image
        dummy_np_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        detections = detector.detect_objects(
            dummy_np_image,
            target_classes=['person'],
            avoid_classes=['car', 'truck']
        )
        print(f"Detections: {len(detections)}")
        
        # Draw detections
        result_image = detector.draw_detections(dummy_np_image, detections)
        print(f"Result image shape: {result_image.shape if result_image is not None else 'None'}")
    else:
        print("YOLO not available - skipping ObjectDetector test")
    
    # Test DepthEstimator
    print("\n3. Testing DepthEstimator...")
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Depth map shape: {depth_map.shape}")
    
    print("\n=== All tests completed ===")
