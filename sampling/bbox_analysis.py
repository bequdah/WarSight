import os
import matplotlib.pyplot as plt
import numpy as np

# Configurations
LABELS_DIR = "dataset/train/labels"

def analyze_bboxes():
    labels_path = os.path.abspath(LABELS_DIR)
    # Handle Windows long paths
    if os.name == 'nt' and not labels_path.startswith("\\\\?\\"):
        labels_path = "\\\\?\\" + labels_path

    label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
    
    widths = []
    heights = []
    areas = []
    error_count = 0

    for label_file in label_files:
        try:
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        w = float(parts[3])
                        h = float(parts[4])
                        widths.append(w)
                        heights.append(h)
                        areas.append(w * h)
        except Exception:
            error_count += 1
            continue
    
    if error_count > 0:
        print(f"Warning: Could not read {error_count} label files.")

    if not areas:
        print("No bounding boxes found.")
        return

    # Statistical Summary
    print(f"Total Bounding Boxes: {len(areas)}")
    print(f"Average Width: {np.mean(widths):.4f}")
    print(f"Average Height: {np.mean(heights):.4f}")
    print(f"Average Area: {np.mean(areas):.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(widths, bins=50, color='blue', alpha=0.7)
    plt.title('BBox Width Distribution')
    plt.xlabel('Normalized Width')

    plt.subplot(1, 3, 2)
    plt.hist(heights, bins=50, color='green', alpha=0.7)
    plt.title('BBox Height Distribution')
    plt.xlabel('Normalized Height')

    plt.subplot(1, 3, 3)
    plt.scatter(widths, heights, alpha=0.1, s=1)
    plt.title('Width vs Height')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.tight_layout()
    plt.savefig("sampling/bbox_analysis.png")
    print("Graphs saved as sampling/bbox_analysis.png")

if __name__ == "__main__":
    analyze_bboxes()
