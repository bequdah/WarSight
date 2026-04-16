import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# Configurations
DATA_YAML = "dataset/data.yaml"
LABELS_DIR = "dataset/train/labels"

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def analyze_distribution():
    class_names = load_class_names(DATA_YAML)
    labels_path = os.path.abspath(LABELS_DIR)
    
    # Handle Windows long paths
    if os.name == 'nt' and not labels_path.startswith("\\\\?\\"):
        labels_path = "\\\\?\\" + labels_path

    label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
    
    all_classes = []
    error_count = 0
    for label_file in label_files:
        try:
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        all_classes.append(class_id)
        except Exception as e:
            error_count += 1
            continue
    
    if error_count > 0:
        print(f"Warning: Could not read {error_count} label files due to path length or other issues.")
    
    counts = Counter(all_classes)
    
    # Map back to names
    name_counts = {class_names[cid]: counts.get(cid, 0) for cid in class_names}
    
    # Sort for plotting
    sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_names)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Class Name')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training Set')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("sampling/class_distribution.png")
    print("Graph saved as sampling/class_distribution.png")
    
    print("\nSummary Counts:")
    for name, count in name_counts.items():
        print(f"{name}: {count}")

if __name__ == "__main__":
    analyze_distribution()
