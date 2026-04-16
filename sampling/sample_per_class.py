import os
import shutil
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# Configurations
# Using absolute paths based on user's environment
PROJECT_ROOT = r"c:\Users\AL Qudah\Desktop\projects\CV"
DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "train", "images")
LABELS_DIR = os.path.join(PROJECT_ROOT, "dataset", "train", "labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sampling", "output", "class_samples_obb")

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def draw_annotations(image, labels, class_names):
    """Draws bounding boxes (Standard or OBB) on the image."""
    h, w, _ = image.shape
    for label in labels:
        parts = label.strip().split()
        if not parts: continue
        try:
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
        except (ValueError, IndexError):
            continue
        
        # Vibrant green for visibility
        color = (0, 255, 0) 
        
        if len(coords) == 4: # Standard YOLO (cx, cy, w, h)
            x_center, y_center, bw, bh = coords
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_names[class_id], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif len(coords) == 8: # OBB (x1, y1, x2, y2, x3, y3, x4, y4)
            pts = []
            for j in range(0, 8, 2):
                pts.append([int(coords[j] * w), int(coords[j+1] * h)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, color, 2)
            cv2.putText(image, class_names[class_id], (pts[0][0][0], pts[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    if not os.path.exists(DATA_YAML):
        print(f"Error: data.yaml not found at {DATA_YAML}")
        return

    class_names = load_class_names(DATA_YAML)
    num_classes = len(class_names)
    sampled_classes = {} # class_id: {img: img_name, label: label_name}

    if not os.path.exists(LABELS_DIR):
        print(f"Error: Labels directory not found at {LABELS_DIR}")
        return

    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    
    print(f"Scanning {len(label_files)} labels to find samples with OBB for {num_classes} classes...")

    for label_file in tqdm(label_files, desc="Searching classes"):
        if len(sampled_classes) == num_classes:
            break
            
        label_path = os.path.join(LABELS_DIR, label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except:
            continue
            
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            class_id = int(parts[0])
            
            if class_id not in sampled_classes:
                base_name = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    img_name = base_name + ext
                    if os.path.exists(os.path.join(IMAGES_DIR, img_name)):
                        sampled_classes[class_id] = {
                            'image': img_name,
                            'label': label_file
                        }
                        break
                if len(sampled_classes) == num_classes:
                    break
    
    print("\nProcessing and drawing OBBs...")
    for class_id in sorted(sampled_classes.keys()):
        data = sampled_classes[class_id]
        img_name = data['image']
        label_name = data['label']
        class_name = str(class_names[class_id]).replace(" ", "_").replace("-", "_")
        
        src_img_path = os.path.join(IMAGES_DIR, img_name)
        src_lbl_path = os.path.join(LABELS_DIR, label_name)
        
        # Load image
        img = cv2.imread(src_img_path)
        if img is None:
            print(f" [!] Error loading image: {img_name}")
            continue
            
        # Load all labels for this image to draw context
        with open(src_lbl_path, 'r') as f:
            labels = f.readlines()
            
        # Draw boxes
        img_with_boxes = draw_annotations(img, labels, class_names)
        
        # New name: classID_className.ext
        file_ext = os.path.splitext(img_name)[1]
        dest_name = f"class_{class_id:02d}_{class_name}_OBB{file_ext}"
        dest_path = os.path.join(OUTPUT_DIR, dest_name)
        
        cv2.imwrite(dest_path, img_with_boxes)
        print(f" [+] Generated: {dest_name}")

    print(f"\nDone! Samples with OBB are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
