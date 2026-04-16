import os
import random
import cv2
import matplotlib.pyplot as plt
import yaml
import numpy as np

# Configurations
DATA_YAML = "dataset/data.yaml"
IMAGES_DIR = "dataset/train/images"
LABELS_DIR = "dataset/train/labels"
NUM_SAMPLES = 4

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def draw_boxes(image, labels, class_names):
    h, w, _ = image.shape
    for label in labels:
        parts = label.split()
        if not parts: continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        if len(coords) == 4: # Standard YOLO
            x_center, y_center, width, height = coords
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[class_id], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif len(coords) == 8: # OBB (x1, y1, x2, y2, x3, y3, x4, y4)
            pts = []
            for j in range(0, 8, 2):
                pts.append([int(coords[j] * w), int(coords[j+1] * h)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            cv2.putText(image, class_names[class_id], (pts[0][0][0], pts[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    class_names = load_class_names(DATA_YAML)
    
    images_path = os.path.abspath(IMAGES_DIR)
    labels_path = os.path.abspath(LABELS_DIR)
    
    if os.name == 'nt':
        if not images_path.startswith("\\\\?\\"): images_path = "\\\\?\\" + images_path
        if not labels_path.startswith("\\\\?\\"): labels_path = "\\\\?\\" + labels_path

    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("No images found in images directory.")
        return

    selected_images = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    
    plt.figure(figsize=(15, 10))
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(images_path, img_name)
        label_path = os.path.join(labels_path, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            img = draw_boxes(img, labels, class_names)
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("sampling/sample_visualization.png")
    print("Sample visualization saved as sampling/sample_visualization.png")

if __name__ == "__main__":
    main()
