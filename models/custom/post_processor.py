"""
Custom Model Extensions - "Tactical Post-Processor"

هذا الملف مسؤول عن تحويل نتائج الموديل الخام إلى بيانات تكتيكية منظمة.
تم تحديثه ليدعم الـ OBB (Oriented Bounding Boxes).
"""

import numpy as np
import cv2
from typing import List, Dict, Any


class PostProcessor:
    """
    معالجة النتائج بعد الـ Detection (Post-Processing).
    يدعم الصناديق المائلة (OBB) والمستطيلة العادية.
    """

    COLOR_DEFAULT = (0, 255, 136)  # اللون الأخضر التكتيكي

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        رسم النتائج المائلة (OBB) على الصورة بسماكة خطوط عالية ووضوح تكتيكي.
        """
        img = image.copy()
        
        for det in detections:
            # توحيد أسماء الأسلحة للرسم
            original_name = det['class_name'].lower()
            if 'gun' in original_name or 'pistol' in original_name or 'rifle' in original_name:
                class_display_name = "GUN"
            else:
                class_display_name = det['class_name'].upper()

            confidence = det['confidence']
            color = self.COLOR_DEFAULT
            
            # التحقق إذا كانت الإحداثيات OBB (8 نقاط) أو مستطيل عادي (4 نقاط)
            coords = det.get('bbox', [])
            
            if len(coords) == 8:
                # رسم OBB (صندوق مائل) بسماكة أكبر (3)
                pts = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
                # نستخدم أول نقطة لكتابة النص
                tx, ty = int(coords[0]), int(coords[1])
            elif len(coords) == 4:
                # رسم مستطيل عادي بسماكة أكبر (3)
                x1, y1, x2, y2 = [int(c) for c in coords]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                tx, ty = x1, y1
            else:
                continue

            # --- إعداد الملصق التكتيكي ---
            track_id = det.get('track_id')
            track_txt = f"#{track_id} " if track_id is not None else ""
            label = f"{track_txt}{class_display_name} {confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # حجم أكبر
            font_thickness = 2
            
            # حساب حجم النص لعمل خلفية
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # رسم مستطيل خلفية للنص لزيادة التباين (HUD Style)
            cv2.rectangle(img, (tx, ty - text_h - 15), (tx + text_w + 5, ty), color, -1)
            
            # رسم النص باللون الأسود فوق الخلفية الخضراء للوضوح
            cv2.putText(img, label, (tx + 2, ty - 7), font, font_scale, (0, 0, 0), font_thickness)

        return img

    def filter_by_confidence(self, detections: List[Dict], min_conf: float) -> List[Dict]:
        """فلترة النتائج بناءً على الحد الأدنى للثقة."""
        return [d for d in detections if d['confidence'] >= min_conf]

    def format_for_hud(self, detections: List[Dict]) -> List[Dict]:
        """
        تنسيق النتائج لتناسب واجهة الـ HUD (index.html).
        """
        formatted = []
        for d in detections:
            original_name = d['class_name'].lower()
            if 'gun' in original_name or 'pistol' in original_name or 'rifle' in original_name:
                class_display_name = "GUN"
            else:
                class_display_name = d['class_name']

            coords = d['bbox']
            if len(coords) == 8:
                cx = sum(coords[0::2]) / 4
                cy = sum(coords[1::2]) / 4
            else:
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2

            formatted.append({
                'class_name': class_display_name,
                'confidence': round(d['confidence'], 3),
                'bbox': coords,
                'center': [round(cx, 1), round(cy, 1)],
                'track_id': d.get('track_id')
            })
        return formatted
