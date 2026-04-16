# 🎯 Tactical Object Detection System

A modular AI research project for military object detection using YOLOv8 and YOLOv11, built as a full-pipeline system designed for collaborative team work.

---

## 📂 Project Structure (GitHub Optimized)

```text
CV/
├── docs/                 → Project Reports & Documentation (Reporter Role)
├── configs/              → Experiment YAML files (Trainers Role)
│   ├── yolov8/           → 3 Experiments for YOLOv8
│   └── yolov11/          → 3 Experiments for YOLOv11
├── dataset/              → Data storage (Ignored by Git)
│   ├── raw/              → Original images & OBB labels
│   └── processed/        → Output from Preprocessing shell
├── preprocessing/        → Dataset cleaning & OBB preparation (Specialist Role)
├── models/               → Unified model interface & Abstraction layer
├── training/             → Orchestration scripts for Kaggle/Local training
├── evaluation/           → Metrics, model comparison & visualization
├── results/              → Final logs and weights (Organized by model)
└── app/                  → Web UI Deployment
```

---

## 👥 Team & Responsibilities

| Name | Role | Responsibilities |
| --- | --- | --- |
| **Member 1** | **Documentation & Reports** | Tracking metrics, writing the final report, documenting results. |
| **Member 2** | **Data Preprocessing** | Cleaning data, OBB normalization, train/val/test splitting. |
| **You (User)** | **YOLOv8 Specialist** | Running 3 experimental trials with different hyperparameters on YOLOv8. |
| **Member 4** | **YOLOv11 Specialist** | Running 3 experimental trials with different hyperparameters on YOLOv11. |


---


---

## 🔬 Detection Classes (14 classes)

| ID | Class             | ID | Class           |
|----|-------------------|-----|----------------|
| 0  | Artillery         | 7  | Military-Truck  |
| 1  | Camouflaged-Soldier | 8  | Missile        |
| 2  | Civilian          | 9  | Missile-Launcher |
| 3  | Drone             | 10 | Radar           |
| 4  | Helicopter        | 11 | Soldier         |
| 5  | Jet Fighters      | 12 | Tank            |
| 6  | Machine-Gun       | 13 | Handgun         |

---

## ⚠️ Key Data Findings (from sampling/)

- **Total annotations:** 27,825 objects
- **Severe imbalance:** Jet Fighters (6013) vs Missile (138)
- **Format:** OBB (Oriented Bounding Boxes — 8 coordinates)
- **Training runs on:** Kaggle Notebooks

---

## 🚀 Pipeline

```
dataset → sampling → preprocessing → configs → training (Kaggle)
                                                    ↓
app ← models/weights ← results (best.pt) ← evaluation
```

---

## 👥 Team

University AI Project — Computer Vision
