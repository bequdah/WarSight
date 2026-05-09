import os
import io
import base64
import sys
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, Request, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn
import tempfile
import cv2

# إضافة المسار الرئيسي للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_factory import load_model
from models.custom.post_processor import PostProcessor

# إنشاء مجلد للمخرجات بمسار مطلق
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "video_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Iron Eye Tactical System")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# إعداد القوالب (Templates)
templates = Jinja2Templates(directory="app/templates")

# --- إعدادات الموديل ---
DEFAULT_MODEL_TYPE = "yolov8"
DEFAULT_WEIGHTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'results', 'YOLOv8', 'exp2', 'best.pt'
))

model_instance = None
post_processor = PostProcessor()
video_progress = {} # { session_id: percent }

def get_tactical_model():
    global model_instance
    if model_instance is None:
        if os.path.exists(DEFAULT_WEIGHTS):
            print(f"[OK] LOADING TACTICAL MODEL: {DEFAULT_WEIGHTS}")
            model_instance = load_model(DEFAULT_MODEL_TYPE, DEFAULT_WEIGHTS)
            model_instance.load()
        else:
            print(f"[ERROR] WEIGHTS NOT FOUND AT: {DEFAULT_WEIGHTS}")
    return model_instance

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    try:
        model = get_tactical_model()
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Tactical Model Offline."})

        contents = await image.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img_pil)

        raw_detections = model.predict(img_np, conf=confidence)
        annotated_img = post_processor.draw_detections(img_np, raw_detections)
        hud_data = post_processor.format_for_hud(raw_detections)

        orig_buf = io.BytesIO()
        img_pil.save(orig_buf, format="JPEG")
        orig_base64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')

        res_pil = Image.fromarray(annotated_img)
        res_buf = io.BytesIO()
        res_pil.save(res_buf, format="JPEG")
        res_base64 = base64.b64encode(res_buf.getvalue()).decode('utf-8')

        class_counts = {}
        for d in hud_data:
            cls = d['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            'image': res_base64,
            'original_image': orig_base64,
            'detections': hud_data,
            'total_targets': len(raw_detections),
            'class_counts': class_counts,
            'image_size': f"{img_pil.width}x{img_pil.height}",
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_video_task(input_path, output_path, session_id, confidence):
    """المعالج الفعلي للفيديو في الخلفية"""
    try:
        model = get_tactical_model()
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # نستخدم WebM لأنه الأكثر توافقاً مع المتصفحات (Chrome/Edge) بدون الحاجة لـ ffmpeg
        fourcc = cv2.VideoWriter_fourcc(*'VP80') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        video_progress[session_id] = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            raw_detections = model.track(frame, conf=confidence)
            annotated_frame = post_processor.draw_detections(frame, raw_detections)
            out.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 2 == 0:
                percent = int((frame_idx / total_frames) * 100)
                video_progress[session_id] = min(percent, 99)
        
        cap.release()
        out.release()
        try: os.unlink(input_path)
        except: pass
        
        video_progress[session_id] = 100
        print(f"\n[DONE] Session {session_id} completed.")
        
    except Exception as e:
        print(f"[CRITICAL] Background Task Error: {e}")
        video_progress[session_id] = -1

@app.post("/analyze-video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence: float = Form(0.25),
    session_id: str = Query(None)
):
    """استلام الفيديو وبدء المعالجة في الخلفية"""
    try:
        track_id = session_id if session_id else f"proc_{datetime.now().timestamp()}"
        video_progress[track_id] = 0
        
        # حفظ الملف المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            input_path = tmp.name

        output_filename = f"processed_{track_id}.webm"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # تشغيل المهمة في الخلفية
        background_tasks.add_task(process_video_task, input_path, output_path, track_id, confidence)
        
        return {
            'status': 'started',
            'session_id': track_id,
            'video_url': f"/outputs/{output_filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/video-status/{session_id}")
async def get_video_status(session_id: str):
    return {"percent": video_progress.get(session_id, 0)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
