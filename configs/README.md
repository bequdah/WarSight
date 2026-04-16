# 📌 Configurations Folder

This folder contains all configuration files that will be used for training experiments on YOLOv8 and YOLOv11 models.

At this stage, no experiments have been executed yet. The purpose of this folder is to prepare a structured and reproducible setup for future training runs.

## 🧠 Purpose of This Folder

We use configuration files to:
- Define hyperparameters for each experiment
- Ensure reproducibility across all runs
- Separate model settings from code
- Make it easy to compare different experiments

## 📂 Structure

The folder is organized as follows:
- `yolov8/` → configurations for YOLOv8 experiments
- `yolov11/` → configurations for YOLOv11 experiments

Each model will have multiple experiment files (e.g., `exp1.yaml`, `exp2.yaml`, `exp3.yaml`).

## 🧪 Future Experiments Plan

Each model will be tested using 3 configurations:

### YOLOv8
- **exp1** → baseline configuration
- **exp2** → improved hyperparameters
- **exp3** → imbalance-aware settings

### YOLOv11
- **exp1** → baseline configuration
- **exp2** → tuned hyperparameters
- **exp3** → enhanced training setup

## ⚙️ How configs will be used

Each `.yaml` file will define training parameters such as:
- learning rate
- batch size
- image size
- epochs
- optimizer

**Example usage:**
```python
config = load_yaml("configs/yolov8/exp1.yaml")
```

## 🚀 Important Note

At this stage:
- No training has been executed yet
- Config files are prepared for upcoming experiments
- This structure ensures clean and fair comparison between models

## 🔥 Goal of This Design

This setup is designed to:
- Support systematic experimentation
- Improve reproducibility
- Allow easy scaling of experiments
- Separate configuration from implementation

## 🧠 One-line summary

This folder defines all experimental setups in a structured way before training begins.
