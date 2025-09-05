import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def ros_to_cv2(ros_image):
        try:
            if ros_image.encoding == 'rgb8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                cv_image = img_array.reshape((ros_image.height, ros_image.width, 3))
                return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif ros_image.encoding == 'bgr8':
                img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                return img_array.reshape((ros_image.height, ros_image.width, 3))
            return None
        except:
            return None
    
    def draw_detections(self, image, detections):
        return image if image is not None else None
