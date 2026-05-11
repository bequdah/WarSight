"""
Custom Model Extensions - "Tactical Post-Processor"

تم التحديث:
- إضافة Class-Agnostic IoU Suppression (حل مشكلة duplicate boxes)
- تصحيح المسافات البادئة (Indentation) للميثودز داخل الكلاس.
"""

import numpy as np
import cv2
from typing import List, Dict, Any

class PostProcessor:
    THREAT_MATRIX = {
        'missile-launcher': (80, 10, 10),
        'jet-fighters': (75, 15, 10),
        'tank': (70, 20, 10),
        'artillery': (65, 15, 10),
        'helicopter': (60, 20, 10),
        'radar': (60, 10, 0),
        'drone': (40, 45, 5),
        'machine-gun': (35, 25, 10),
        'camouflaged-soldier': (30, 20, 5),
        'soldier': (25, 20, 5),
        'military-truck': (15, 15, 5),
        'handgun': (5, 25, 5),
        'civilian': (0, 0, 0)
    }

    COLORS = {
        'CRITICAL': (255, 0, 0),
        'HOSTILE': (255, 140, 0),
        'CAUTION': (255, 255, 0),
        'SAFE': (0, 255, 136),
        'NEUTRAL': (200, 200, 200)
    }

    # ============================
    # 🔥 IoU + Class-Agnostic NMS
    # ============================

    def compute_iou(self, box1, box2):
        def to_xyxy(box):
            if len(box) == 4:
                return box
            elif len(box) == 8:
                xs = box[0::2]
                ys = box[1::2]
                return [min(xs), min(ys), max(xs), max(ys)]
            return None

        b1 = to_xyxy(box1)
        b2 = to_xyxy(box2)
        if b1 is None or b2 is None:
            return 0.0

        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def class_agnostic_nms(self, detections, iou_threshold=0.4):
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []

        for det in detections:
            discard = False
            for kept in keep:
                iou = self.compute_iou(det['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    discard = True
                    break
            if not discard:
                keep.append(det)

        return keep

    # ============================
    # Threat Calculation
    # ============================

    def calculate_threat_score(self, det: Dict, img_shape: tuple) -> Dict[str, Any]:
        class_name = det['class_name'].lower()
        base_threat, prox_bonus, profile_bonus = self.THREAT_MATRIX.get(class_name, (20, 10, 0))
        conf = det['confidence']

        if class_name == 'civilian':
            return {'score': 0, 'level': 'NEUTRAL', 'color': self.COLORS['NEUTRAL']}

        coords = det.get('bbox', [])
        img_h, img_w = img_shape[:2]
        area = 0
        profile_factor = 0.0

        if len(coords) == 8:
            x = coords[0::2]
            y = coords[1::2]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            d1 = np.hypot(x[0] - x[1], y[0] - y[1])
            d2 = np.hypot(x[1] - x[2], y[1] - y[2])

            if min(d1, d2) > 0:
                aspect_ratio = max(d1, d2) / min(d1, d2)
                profile_factor = max(0.0, 2.5 - aspect_ratio) / 1.5
                profile_factor = min(1.0, profile_factor)

        elif len(coords) == 4:
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])

        normalized_area = area / (img_w * img_h)
        proximity_factor = min(1.0, normalized_area / 0.15)

        raw_score = base_threat + (proximity_factor * prox_bonus) + (profile_factor * profile_bonus)

        confidence_multiplier = 0.8 + (0.2 * conf)
        final_score = min(100.0, raw_score * confidence_multiplier)

        if final_score > 75:
            level = 'CRITICAL'
        elif final_score > 45:
            level = 'HOSTILE'
        elif final_score > 20:
            level = 'CAUTION'
        else:
            level = 'SAFE'

        return {'score': round(final_score, 1), 'level': level, 'color': self.COLORS[level]}

    # ============================
    # Drawing
    # ============================

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        # 🔥 APPLY FILTER HERE
        detections = self.class_agnostic_nms(detections, iou_threshold=0.7)

        img = image.copy()
        img_h, img_w = img.shape[:2]

        for det in detections:
            threat = self.calculate_threat_score(det, img.shape)
            color = threat['color']

            class_name = det['class_name'].upper()
            if class_name in ['MACHINE-GUN', 'HANDGUN']:
                class_name = 'GUN'

            coords = det.get('bbox', [])

            if len(coords) == 8:
                pts = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [pts], True, color, 2)
                tx, ty = int(coords[0]), int(coords[1])

            elif len(coords) == 4:
                x1, y1, x2, y2 = [int(c) for c in coords]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                tx, ty = x1, y1
            else:
                continue

            label = f"{class_name} [{det['confidence']*100:.1f}%]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)

            if tx + tw + 10 > img_w:
                tx = img_w - tw - 10
            if ty - th - 15 < 0:
                ty = th + 15

            cv2.rectangle(img, (tx, ty - th - 15), (tx + tw + 10, ty), color, -1)
            cv2.putText(img, label, (tx + 5, ty - 7), font, 0.8, (0, 0, 0), 2)

        return img

    # ============================
    # HUD Formatting
    # ============================

    def format_for_hud(self, detections: List[Dict], img_shape: tuple = (640, 640)) -> List[Dict]:
        # 🔥 APPLY FILTER HERE TOO
        detections = self.class_agnostic_nms(detections, iou_threshold=0.7)

        formatted = []
        for d in detections:
            threat = self.calculate_threat_score(d, img_shape)
            coords = d['bbox']

            if len(coords) == 8:
                cx = sum(coords[0::2]) / 4
                cy = sum(coords[1::2]) / 4
            else:
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2

            display_class = d['class_name'].upper()
            if display_class in ['MACHINE-GUN', 'HANDGUN']:
                display_class = 'GUN'

            formatted.append({
                'class_name': display_class,
                'confidence': round(d['confidence'], 3),
                'threat_score': threat['score'],
                'threat_level': threat['level'],
                'bbox': coords,
                'center': [round(cx, 1), round(cy, 1)],
                'track_id': d.get('track_id')
            })

        return sorted(formatted, key=lambda x: x['threat_score'], reverse=True)