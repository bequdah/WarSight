"""
YOLOv8 Wrapper - "المترجم" الخاص بـ YOLOv8

هذا الملف بيطبق الواجهة الموحدة (BaseDetectionModel)
باستخدام طريقة عمل YOLOv8 الخاصة.

الـ app أو الـ training ما بعرفوا شو بصير هون،
بس بعرفوا إنهم بقدروا يستخدموا:
    model.load()
    model.predict(image)
"""

import numpy as np
from typing import List, Dict, Any
import sys
import os

# إضافة مسار المشروع عشان نرث من base
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.base.base_model import BaseDetectionModel


class YOLOv8Detector(BaseDetectionModel):
    """
    تطبيق YOLOv8 بطريقة الواجهة الموحدة.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        super().__init__(model_path, conf_threshold)
        self.model_version = "YOLOv8"

    def load(self) -> None:
        """
        تحميل موديل YOLOv8 من ملف الأوزان.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[YOLOv8] Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"[YOLOv8] Failed to load model: {e}")
            raise

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        تنفيذ الـ Detection على الصورة وإرجاع النتائج بالصيغة الموحدة.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append({
                    'class_id': class_id,
                    'class_name': result.names[class_id],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
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
