"""
YOLOv26 Wrapper - "المترجم" الخاص بـ YOLOv26

نفس فكرة YOLOv8 بالضبط، بس هون بنستخدم YOLOv26.
الـ app والـ training ما بيفرق معهم أي موديل،
بس بسألوا بنفس الطريقة:
    model.load()
    model.predict(image)
"""

import numpy as np
import cv2
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.base.base_model import BaseDetectionModel


class YOLOv26Detector(BaseDetectionModel):
    """
    تطبيق YOLOv26 بطريقة الواجهة الموحدة.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        super().__init__(model_path, conf_threshold)
        self.model_version = "YOLOv26"

    def load(self) -> None:
        """
        تحميل موديل YOLOv26 من ملف الأوزان.
        (YOLOv26 بتشتغل عبر مكتبة ultralytics نفسها)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[YOLOv26] Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"[YOLOv26] Failed to load model: {e}")
            raise

    def predict(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        رصد كلاسيكي للصور الثابتة.
        """
        return self._inference(image, conf=conf, track=False)

    def track(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        رصد مع تتبع مستمر (التتبع يحتاج persist=True).
        """
        return self._inference(image, conf=conf, track=True)

    def _inference(self, image: np.ndarray, conf: float = None, track: bool = False) -> List[Dict[str, Any]]:
        """
        محرك الاستدلال الداخلي لـ YOLOv26.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        threshold = conf if conf is not None else self.conf_threshold
        
        # تصحيح مساحة الألوان: YOLOv26 يتوقع BGR إذا تم تمرير numpy array (نفس طريقة Colab/cv2)
        if len(image.shape) == 3 and image.shape[2] == 3:
            model_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            model_image = image
        
        if track:
            results = self.model.track(model_image, conf=threshold, persist=True, verbose=False)
        else:
            results = self.model(model_image, conf=threshold, verbose=False)

        detections = []
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                for box in result.obb:
                    class_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    points = box.xyxyxyxy[0].cpu().numpy().reshape(-1).tolist()
                    detections.append({
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'confidence': float(box.conf[0]),
                        'bbox': points,
                        'track_id': track_id
                    })
            elif hasattr(result, 'boxes'):
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    detections.append({
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                        'track_id': track_id
                    })

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """
        معلومات عن الموديل.
        """
        return {
            'version': self.model_version,
            'path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'classes': self.model.names if self.is_loaded() else None
        }
