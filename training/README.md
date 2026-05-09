# 🏋️ Training Folder

This folder contains everything related to **running experiments on Kaggle**.

## ⚠️ Status: Placeholder — Logic not implemented yet.

## 📌 Purpose

- Orchestrate training runs for YOLOv8 and YOLOv26
- Each experiment is driven by a config file from `configs/`
- Training is executed on **Kaggle Notebooks** (not locally)

## 📂 Planned Structure

```
training/
├── train_yolov8.py       ← Script to trigger YOLOv8 training
├── train_yolov26.py      ← Script to trigger YOLOv26 training
└── README.md
```

## 🔗 Dependencies

- Input: `configs/yolov8/exp1.yaml` ... `exp3.yaml`
- Input: `configs/yolov26/exp1.yaml` ... `exp3.yaml`
- Output: trained weights → saved to `results/`

## 🧪 Experiments Plan

| Model   | Experiment | Description              |
|---------|------------|--------------------------|
| YOLOv8  | exp1       | Baseline configuration   |
| YOLOv8  | exp2       | Improved hyperparameters |
| YOLOv8  | exp3       | Imbalance-aware settings |
| YOLOv26 | exp1       | Baseline configuration   |
| YOLOv26 | exp2       | Tuned hyperparameters    |
| YOLOv26 | exp3       | Enhanced training setup  |
