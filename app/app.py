import os
import io
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# تحميل الموديل مرة واحدة عند بدء التشغيل
MODEL_PATH = 'weight/best.pt'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ تم تحميل الموديل من: {MODEL_PATH}")
    else:
        print(f"❌ لم يتم العثور على الموديل في: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'الموديل غير محمل'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400

    # قراءة الصورة
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # تشغيل الرصد
    conf_threshold = float(request.form.get('confidence', 0.25))
    results = model.predict(source=img, conf=conf_threshold, verbose=False)

    # استخراج النتائج
    detections = []
    for result in results:
        for box in result.boxes:
            det = {
                'class': result.names[int(box.cls[0])],
                'confidence': round(float(box.conf[0]), 3),
                'bbox': [round(float(x), 1) for x in box.xyxy[0].tolist()]
            }
            detections.append(det)

    # رسم المربعات على الصورة
    annotated = results[0].plot()
    # تحويل من BGR إلى RGB
    annotated_rgb = annotated[..., ::-1]
    pil_img = Image.fromarray(annotated_rgb)

    # تحويل الصورة الأصلية إلى Base64
    orig_buffered = io.BytesIO()
    img.save(orig_buffered, format="PNG")
    orig_base64 = base64.b64encode(orig_buffered.getvalue()).decode('utf-8')

    # تحويل الصورة المُحللة إلى Base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # إحصائيات
    class_counts = {}
    for d in detections:
        cls = d['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    response = {
        'image': img_base64,
        'original_image': orig_base64,
        'detections': detections,
        'total_targets': len(detections),
        'class_counts': class_counts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_size': f'{img.width}x{img.height}'
    }

    return jsonify(response)

if __name__ == '__main__':
    load_model()
    print("\n" + "="*50)
    print("🦅 IRON EYE SYSTEM - Tactical HUD Active")
    print("="*50)
    print("🌐 افتح المتصفح على: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
