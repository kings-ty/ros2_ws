import cv2
import numpy as np

class ImageProcessor:
    """Handle all image processing operations"""
    
    @staticmethod
    def ros_to_cv2(ros_image):
        """Convert ROS Image to OpenCV format"""
        try:
            if ros_image.encoding == 'rgb8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((ros_image.height, ros_image.width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif ros_image.encoding == 'bgr8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((ros_image.height, ros_image.width, 3))
            else:
                return None
            return cv_image
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None
    
    @staticmethod
    def preprocess_for_nn(image, target_size=(80, 60)):
        """Preprocess image for neural network"""
        if image is None:
            return np.zeros((3, target_size[1], target_size[0]))
        
        resized = cv2.resize(image, target_size)
        normalized = resized.astype(np.float32) / 255.0
        preprocessed = np.transpose(normalized, (2, 0, 1))
        
        return preprocessed
    
    @staticmethod
    def draw_detections(image, detections):
        """Draw YOLO detections on image"""
        display_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Color based on class
            if class_name in RobotConfig.TARGET_CLASSES:
                color = (0, 255, 0)  # Green
            elif class_name in RobotConfig.AVOID_CLASSES:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue
            
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(display_image, f'{class_name}: {confidence:.2f}', 
                       (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_image