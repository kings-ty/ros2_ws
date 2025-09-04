
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class YoloDetector:
    """YOLO object detection - integrates with your existing utils"""
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                print(f"✅ YOLO model loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load YOLO: {e}")
        else:
            print("❌ YOLO not available. Install: pip3 install ultralytics")
    
    def is_available(self):
        return self.model is not None
    
    def detect_objects(self, image, target_classes=None, avoid_classes=None):
        """
        Detect objects in image
        Returns: (detections_list, target_object)
        """
        if not self.is_available() or image is None:
            return [], None
        
        target_classes = target_classes or []
        avoid_classes = avoid_classes or []
        
        try:
            results = self.model(image, verbose=False)
            detections = []
            target_object = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
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
                        
                        # Select best target (upper part of image)
                        if detection['is_target'] and center_y < image.shape[0] * 0.7:
                            if target_object is None or area > target_object['area']:
                                target_object = detection
            
            return detections, target_object
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return [], None
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes - works with your image_processor"""
        if image is None:
            return None
        
        display_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            if det['is_target']:
                color = (0, 255, 0)  # Green
                thickness = 3
            elif det['is_obstacle']:
                color = (0, 0, 255)  # Red
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue
                thickness = 1
            
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(display_image, f"{det['class']}: {det['confidence']:.2f}",
                       (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_image
