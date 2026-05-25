# 🤖 Models Folder

This folder is the **Unified Model Interface** for the project.

It is **NOT** about training, configs, or results.
It is about **how we use any model in a consistent way** across the entire system.

## 🧠 Core Idea

Instead of writing YOLO-specific code everywhere in the project:

```python
# ❌ Without this folder (messy, breaks easily)
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("image.jpg")
boxes = results[0].boxes.xyxy  # YOLOv8 specific syntax
```

We write it once and use it everywhere:

```python
# ✅ With this folder (clean, consistent)
from models.model_factory import load_model

model = load_model("yolov8", "weights/best.pt")
model.load()
detections = model.predict(image)
```

If tomorrow we switch to YOLOv26, we only change `"yolov8"` → `"yolov26"`. **Nothing else in the project changes.**

## 📂 Folder Structure

```
models/
│
├── base/
│   └── base_model.py         # "The Constitution" - rules every model must follow
│
├── YOLOv8/
│   └── yolov8_detector.py    # YOLOv8 translator to the unified interface
│
├── YOLOv26/
│   └── yolov26_detector.py   # YOLOv26 translator to the unified interface
│
├── custom/
│   └── post_processor.py     # Extra logic (drawing, filtering, JSON conversion)
│
└── model_factory.py          # The single entry point - "give me a model"
```

## ⚡ How It Works in the Pipeline

```
configs/ → training/ → results/ → models/ → app.py
                                     ↑
                              This folder sits HERE
                          (after training, before the app)
```

1. **Training runs** → saves `best.pt` weights
2. **`model_factory.py`** → loads those weights via the right detector class
3. **`app.py`** → calls `model.predict(image)` without knowing any YOLO internals
4. **`post_processor.py`** → draws boxes and formats JSON for the frontend

## 🚀 Usage Example (from app.py)

```python
from models.model_factory import load_model
from models.custom.post_processor import PostProcessor

# Load the model (swap "yolov8" ↔ "yolov26" anytime)
model = load_model("yolov8", "weights/best.pt", conf_threshold=0.5)
model.load()

# Run detection
detections = model.predict(image)

# Post-process for the UI
processor = PostProcessor()
annotated_image = processor.draw_detections(image, detections)
json_results = processor.to_json(detections)
```

## 📦 Dataset Source

The training data was sourced from Roboflow Universe:
[Instance Segmentation v2.0 – Roboflow](https://universe.roboflow.com/thesis-m2-wic-by-abdelatif-boukabrine/instance-segmentation-v2-0)

## 🔑 Key Principle

> **"The models folder abstracts different YOLO implementations into a unified interface for consistent usage across the system."**
