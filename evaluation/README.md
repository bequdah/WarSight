# 📊 Evaluation Folder

This folder contains everything related to **measuring and comparing model performance**.

## ⚠️ Status: Placeholder — Logic not implemented yet.

## 📌 Purpose

- Compare results between YOLOv8 and YOLOv11 experiments
- Calculate standard detection metrics (mAP, Precision, Recall)
- Generate comparison charts and summary tables

## 📂 Planned Structure

```
evaluation/
├── evaluate.py           ← Run evaluation on a trained model
├── compare_models.py     ← Side-by-side comparison of all experiments
├── metrics_summary.py    ← Export results table (CSV / Markdown)
└── README.md
```

## 🔗 Dependencies

- Input: trained weights from `results/`
- Input: validation dataset from `dataset/`
- Output: metrics tables + comparison plots

## 📏 Metrics to Track

| Metric     | Description                      |
|------------|----------------------------------|
| mAP@0.5    | Main accuracy measurement        |
| Precision  | How precise are detections       |
| Recall     | How many targets are found       |
| F1 Score   | Balance between P and R          |
| Inference Speed (ms) | How fast per image     |
