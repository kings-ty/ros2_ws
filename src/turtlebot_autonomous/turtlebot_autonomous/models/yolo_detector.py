import cv2
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class YoloDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
                print(f"YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
        else:
            print("YOLO not available. Install: pip3 install ultralytics")
    
    def detect_objects(self, image, target_classes=None, avoid_classes=None):
        """
        Detect objects in image
        Returns: (detections_list, target_object, obstacles_list)
        """
        if not self.is_available() or image is None:
            return [], None, []
        
        target_classes = target_classes or []
        avoid_classes = avoid_classes or []
        
        try:
            results = self.model(image, verbose=False)
            detections = []
            target_object = None
            obstacles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.class_names[class_id]
                        confidence = float(box.conf[0])
                        
                        if confidence < self.confidence_threshold:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center_x': center_x,
                            'center_y': center_y,
                            'area': area,
                            'is_target': class_name in target_classes,
                            'is_obstacle': class_name in avoid_classes
                        }
                        
                        detections.append(detection)
                        
                        # Categorize detections
                        if detection['is_target']:
                            # Prioritize targets in upper part of image (head level)
                            if center_y < image.shape[0] * 0.7:
                                if target_object is None or area > target_object['area']:
                                    target_object = detection
                        
                        if detection['is_obstacle']:
                            obstacles.append(detection)
            
            return detections, target_object, obstacles
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return [], None, []
    
    def draw_detections(self, image, detections, target_object=None):
        """Draw bounding boxes and labels on image"""
        if image is None:
            return None
        
        display_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Color coding
            if det['is_target']:
                color = (0, 255, 0)  # Green for targets
                thickness = 3
            elif det['is_obstacle']:
                color = (0, 0, 255)  # Red for obstacles
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(display_image, label, (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight target with circle
        if target_object:
            center_x = int(target_object['center_x'])
            center_y = int(target_object['center_y'])
            cv2.circle(display_image, (center_x, center_y), 15, (0, 255, 0), 3)
            cv2.putText(display_image, 'TARGET', (center_x-35, center_y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_image
    
    def is_available(self):
        return self.model is not None
