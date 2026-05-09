"""
YOLOv26 Wrapper - "المترجم" الخاص بـ YOLOv26

نفس فكرة YOLOv8 بالضبط، بس هون بنستخدم YOLOv26.
الـ app والـ training ما بيفرق معهم أي موديل،
بس بسألوا بنفس الطريقة:
    model.load()
    model.predict(image)
"""

import numpy as np
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
        تنفيذ الـ Detection وإرجاع النتائج بالصيغة الموحدة.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        threshold = conf if conf is not None else self.conf_threshold
        results = self.model(image, conf=threshold, verbose=False)
        detections = []

        for result in results:
            # التحقق إذا كان هناك نتائج OBB
            if hasattr(result, 'obb') and result.obb is not None:
                for box in result.obb:
                    class_id = int(box.cls[0])
                    # نأخذ الـ 8 نقاط (xyxyxyxy) ونحولها لقائمة مسطحة
                    points = box.xyxyxyxy[0].cpu().numpy().reshape(-1).tolist()
                    detections.append({
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'confidence': float(box.conf[0]),
                        'bbox': points  # 8 coordinates for OBB
                    })
            elif hasattr(result, 'boxes'):
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detections.append({
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
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
