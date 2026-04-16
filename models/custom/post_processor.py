"""
Custom Model Extensions - "البهارات الإضافية"

هون بنحط أي منطق custom بدنا نضيفه فوق الموديل العادي.
مثال: بعد ما الموديل بلاقي "دبابة"، بدنا نرسم حواليها
خط أحمر وبدنا نحسب مساحتها.
هذا المنطق مش موجود بـ YOLO الأصلي، هو خاص فينا.
"""

import numpy as np
import cv2
from typing import List, Dict, Any
from models.custom.iff_system import IFFSystem


class PostProcessor:
    """
    معالجة النتائج بعد الـ Detection (Post-Processing).
    """

    def identify_targets(self, detections: List[Dict]) -> List[Dict]:
        """استدعاء نظام تمييز الهوية السيادي."""
        return IFFSystem.identify_targets(detections)

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        رسم النتائج مع نظام الألوان التكتيكي (IFF Mapping).
        """
        img = image.copy()
        
        # تفعيل طبقة تحديد الهوية
        detections = self.identify_targets(detections)

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            class_name = det['class_name']
            identity = det.get('identity', 'Unknown')
            color = det.get('color', IFFSystem.COLOR_DEFAULT)

            # رسم المستطيل التكتيكي
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # كتابة التقرير التكتيكي فوق الهدف
            label = f"{class_name} [{identity}]"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    def filter_by_class(self, detections: List[Dict], target_classes: List[str]) -> List[Dict]:
        """
        فلترة وإرجاع كلاسات معينة فقط.
        مثلاً: "عطيني الدبابات بس".
        """
        return [d for d in detections if d['class_name'] in target_classes]

    def filter_by_confidence(self, detections: List[Dict], min_conf: float) -> List[Dict]:
        """
        فلترة وإرجاع النتائج اللي فوق نسبة ثقة معينة.
        """
        return [d for d in detections if d['confidence'] >= min_conf]

    def to_json(self, detections: List[Dict]) -> List[Dict]:
        """
        تحويل النتائج لصيغة تقرير أمني (Security Report Format).
        """
        return [
            {
                'target_type': d['class_name'],
                'confidence': round(d['confidence'] * 100, 1),
                'identity': d.get('identity', 'Unknown'),
                'threat_assessment': d.get('threat_level', 'LOW'),
                'coordinates': {
                    'x1': int(d['bbox'][0]), 'y1': int(d['bbox'][1]),
                    'x2': int(d['bbox'][2]), 'y2': int(d['bbox'][3])
                }
            }
            for d in detections
        ]
