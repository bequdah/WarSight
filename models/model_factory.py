"""
Model Factory - "باب الدخول الوحيد" للمشروع

هذا الملف هو "المفتاح السحري" للمشروع.
الـ app.py والـ training والـ evaluation كلهم بيجوا
لهاي الملف ويقولوا:
    "عطيني موديل"
وهو بعطيهم الموديل المناسب بدون ما يفكروا بالتفاصيل.

الاستخدام:
    from models.model_factory import load_model
    
    model = load_model("yolov8", "weights/best.pt")
    model.load()
    results = model.predict(image)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base.base_model import BaseDetectionModel


def load_model(model_type: str, model_path: str, conf_threshold: float = 0.5) -> BaseDetectionModel:
    """
    إنشاء وإرجاع الموديل المناسب.
    
    Args:
        model_type: نوع الموديل - "yolov8" أو "yolov26"
        model_path: مسار ملف الأوزان (.pt)
        conf_threshold: الحد الأدنى للثقة (افتراضي 0.5)
        
    Returns:
        موديل جاهز للاستخدام (لسا ما اتحمل، لازم تنادي .load())
        
    Example:
        model = load_model("yolov8", "models/weights/best.pt")
        model.load()
        detections = model.predict(image)
    """
    model_type = model_type.lower().strip()

    if model_type in ("yolov8", "v8"):
        from models.YOLOv8.yolov8_detector import YOLOv8Detector
        return YOLOv8Detector(model_path, conf_threshold)

    elif model_type in ("yolov26", "v26"):
        from models.YOLOv26.yolov26_detector import YOLOv26Detector
        return YOLOv26Detector(model_path, conf_threshold)

    else:
        available = ["yolov8", "yolov26"]
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available options: {available}"
        )


def get_available_models() -> list:
    """
    إرجاع قائمة بالموديلات المتاحة في المشروع.
    """
    return ["yolov8", "yolov26"]
