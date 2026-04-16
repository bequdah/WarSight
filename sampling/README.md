# 📊 Sampling & Data Analysis

This folder contains scripts dedicated to auditing and analyzing the dataset before training. It is crucial to understand the data quality, class distribution, and bounding box characteristics to ensure optimal model performance.

## 📂 Folder Structure

- `sample_images.py`: Randomly selects images from the dataset to verify visual quality and labels.
- `visualize_dataset.py`: A tool to overlay bounding boxes on images to ensure the ground truth is accurate.
- `class_distribution.py`: Analyzes the frequency of each class to detect and address data imbalance.
- `bbox_analysis.py`: Investigates bounding box sizes and spatial distribution across the dataset.

## 🧠 Data Analysis Findings
*Actual results from audit runs.*

- **Key Observations**:
    - **Severe Class Imbalance**: "Jet Fighters" (6013) has significantly more data than "Missile" (138) and "Radar" (242).
    - **OBB Format**: The dataset uses 8-point Oriented Bounding Boxes (OBB).
    - **High Density**: Total of 27,825 objects annotated in the training set.
- **Issues Logged**:
    - Under-represented classes like **Missile**, **Radar**, and **Machine-Gun** may lead to poor detection performance for these targets.
    - Long filenames in the dataset caused issues on Windows systems (MAX_PATH limit), resolved in sampling scripts.
- **Preprocessing steps recommended**:
    - **Oversampling**: Increase weight or instances of rare classes (Missile, Radar).
    - **Augmentation**: Focus on rotating objects to leverage the OBB format.
    - **Filtering**: Ensure Jet Fighter images are varied enough to justify the high count.

## 🚀 How to use
Run each script from the root of the project:
```bash
python sampling/class_distribution.py
```
