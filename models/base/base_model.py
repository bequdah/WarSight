"""
Base Model Interface - "الدستور" الموحد للمشروع

هذا الملف بحدد القوانين اللي أي موديل (v8 أو v26) لازم يتبعها.
أي موديل جديد بدنا نضيفه للمشروع لازم يرث من هذه الكلاس
ويطبق نفس الدوال.

المبدأ: كل المودلز بتتكلم نفس "اللغة" بداخل المشروع.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseDetectionModel(ABC):
    """
    الواجهة الموحدة (Unified Interface) لكل مودلز الـ Detection.
    
    أي موديل (YOLOv8, YOLOv26, أو غيرهم) لازم يرث من هاي الكلاس
    ويطبق الدوال المطلوبة.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        أساسيات أي موديل:
        - model_path: وين ملف الأوزان (.pt)
        - conf_threshold: الحد الأدنى للثقة (Confidence)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None  # الموديل الفعلي بيتحمل بـ load()

    @abstractmethod
    def load(self) -> None:
        """
        تحميل الموديل من ملف الأوزان.
        كل نسخة YOLO لها طريقة تحميل خاصة فيها.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        تنفيذ الـ Detection على صورة معينة.
        """
        pass

    @abstractmethod
    def track(self, image: np.ndarray, conf: float = None) -> List[Dict[str, Any]]:
        """
        تنفيذ الـ Detection مع التتبع (Tracking) المستمر (للفيديو).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        إرجاع معلومات عن الموديل (نوعه، إصداره، الكلاسات اللي يعرفها).
        """
        pass

    def is_loaded(self) -> bool:
        """
        التحقق إذا الموديل اتحمل أو لأ.
        """
        return self.model is not None

    def __repr__(self) -> str:
        status = "Loaded" if self.is_loaded() else "Not Loaded"
        return f"{self.__class__.__name__}(path='{self.model_path}', status={status})"
