# Preprocessing Scripts — Quick Start Guide

Get the augmented **Ultra** dataset ready in 2 commands.

---

## Requirements

```bash
pip install opencv-python numpy tqdm
```

---

## Setup

Place your raw data here (not included in the repo):
```
dataset/
└── raw/
    └── train/
        ├── images/   ← your .jpg / .png files
        └── labels/   ← your YOLO OBB .txt files
```

---

## Run

```bash
# Step 1 — Split raw data into Train / Val / Test
python preprocessing/split_dataset.py

# Step 2 — Augment, balance classes, and clean labels
python preprocessing/augment_v1.py
```

That's it. The final dataset will be in `dataset/Ultra/` with a ready `data.yaml`.

---

## Verify (Optional)

```bash
python preprocessing/quick_audit.py
```

All three splits should show `OOB: 0 / DEG: 0 / BAD: 0`.

---

## Output Structure

```
dataset/Ultra/
├── train/images/   29,921 images
├── train/labels/
├── val/images/     2,428 images
├── val/labels/
├── test/images/    3,251 images
├── test/labels/
└── data.yaml       ← pass this to YOLO
```

---

> For the full engineering breakdown (design decisions, problems solved, audit results), see [PIPELINE.md](PIPELINE.md).
