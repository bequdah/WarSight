# 📦 Results Folder

This folder stores **all training outputs** from Kaggle experiments.

## ⚠️ Status: Placeholder — Will be populated after training runs.

## 📌 Purpose

- Store the output of each training experiment automatically
- Keep weights, metrics, and plots organized per experiment
- The final `best.pt` from here gets copied to `models/weights/` for the app

## 📂 Expected Structure (after training)

```
results/
├── yolov8/
│   ├── exp1/
│   │   ├── weights/
│   │   │   ├── best.pt       ← Best model checkpoint
│   │   │   └── last.pt       ← Last epoch checkpoint
│   │   ├── results.csv       ← Per-epoch metrics
│   │   ├── confusion_matrix.png
│   │   └── PR_curve.png
│   ├── exp2/
│   └── exp3/
│
└── yolov11/
    ├── exp1/
    ├── exp2/
    └── exp3/
```

## 🔗 Workflow

```
Kaggle Training
      ↓
results/yolov8/exp1/weights/best.pt
      ↓
Pick best experiment after evaluation
      ↓
Copy → models/weights/best.pt
      ↓
app.py uses it for inference
```

## ⚙️ Git Note

```
# Add this to .gitignore — weights are too large for GitHub:
results/**/weights/*.pt
```
