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
import cv2
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

    def predict(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        رصد كلاسيكي للصور الثابتة (بدون تتبع).
        """
        return self._inference(image, conf=conf, track=False)

    def track(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        رصد مع تتبع مستمر (التتبع يحتاج persist=True ليحافظ على الـ IDs).
        يستخدم في حلقات الفيديو.
        """
        return self._inference(image, conf=conf, track=True)

    def _inference(self, image: np.ndarray, conf: float = None, track: bool = False) -> List[Dict[str, Any]]:
        """
        محرك الاستدلال الداخلي.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        threshold = conf if conf is not None else self.conf_threshold
        
        # تصحيح مساحة الألوان: YOLOv8 يتوقع BGR إذا تم تمرير numpy array (نفس طريقة Colab/cv2)
        if len(image.shape) == 3 and image.shape[2] == 3:
            model_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            model_image = image
        
        if track:
            # استخدام ByteTrack مع الحفاظ على الحالة عبر الفريمات
            results = self.model.track(model_image, conf=threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = self.model(model_image, conf=threshold, verbose=False)

        detections = []
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                for box in result.obb:
                    class_id = int(box.cls[0])
                    # الحصول على الـ ID إذا كان التتبع مفعلاً
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
