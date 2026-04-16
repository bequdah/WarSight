import os
import cv2
import yaml
import argparse
import numpy as np

# Configurations
DATA_YAML = "dataset/data.yaml"
IMAGES_DIR = "dataset/train/images"
LABELS_DIR = "dataset/train/labels"

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def visualize(image_name):
    class_names = load_class_names(DATA_YAML)
    
    # Handle absolute paths and Windows long paths
    images_path = os.path.abspath(IMAGES_DIR)
    labels_path = os.path.abspath(LABELS_DIR)
    
    if os.name == 'nt':
        if not images_path.startswith("\\\\?\\"): images_path = "\\\\?\\" + images_path
        if not labels_path.startswith("\\\\?\\"): labels_path = "\\\\?\\" + labels_path

    img_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, os.path.splitext(image_name)[0] + ".txt")

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                if len(coords) == 4: # Standard YOLO
                    x_center, y_center, width, height = coords
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, class_names[class_id], (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif len(coords) == 8: # OBB
                    pts = []
                    for j in range(0, 8, 2):
                        pts.append([int(coords[j] * w), int(coords[j+1] * h)])
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, (255, 0, 0), 2)
                    cv2.putText(img, class_names[class_id], (pts[0][0][0], pts[0][0][1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save a preview
    cv2.imwrite("sampling/visualize_output.png", img)
    print(f"Preview saved as sampling/visualize_output.png")
    
    cv2.imshow("Dataset Visualization", img)
    print(f"Showing {image_name}. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a specific image from the dataset.")
    parser.add_argument("--img", type=str, help="Name of the image file (e.g. image1.jpg)")
    args = parser.parse_args()

    if args.img:
        visualize(args.img)
    else:
        # Resolve path for listing
        list_path = os.path.abspath(IMAGES_DIR)
        if os.name == 'nt' and not list_path.startswith("\\\\?\\"):
            list_path = "\\\\?\\" + list_path
            
        image_files = [f for f in os.listdir(list_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if image_files:
            visualize(image_files[0])
        else:
            print("No images found.")
