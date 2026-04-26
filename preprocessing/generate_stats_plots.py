import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# --- Configuration ---
# Using \\?\ prefix for Windows Long Path support
RAW_LBL_DIR  = r"\\?\C:\Users\AL Qudah\Desktop\projects\CV\dataset\raw\train\labels"
PROC_BASE_DIR = r"\\?\C:\Users\AL Qudah\Desktop\projects\CV\dataset\processed"
ULTRA_BASE_DIR = r"\\?\C:\Users\AL Qudah\Desktop\projects\CV\dataset\Ultra"

NAMES = {
    '0':'Artillery', '1':'C-Soldier', '2':'Civilian', '3':'Drone', '4':'Heli', 
    '5':'Jet', '6':'M-Gun', '7':'Truck', '8':'Missile', '9':'Launcher', 
    '10':'Radar', '11':'Soldier', '12':'Tank', '13':'Handgun'
}

def get_class_counts(directory):
    counts = Counter()
    print(f"Scanning: {directory}...")
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return counts
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                counts[NAMES.get(parts[0], parts[0])] += 1
                except:
                    continue
    return counts

def plot_distribution(counts, title, filename, color='skyblue'):
    if not counts:
        print(f"No data found for {title}")
        return
    
    # Use consistent order for all plots (Alphabetical)
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(classes, values, color=color, edgecolor='black', alpha=0.8)
    
    plt.title(title, fontsize=18, fontweight='bold')
    plt.ylabel('Object Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # 1. Raw
    raw_counts = get_class_counts(RAW_LBL_DIR)
    plot_distribution(raw_counts, "1. Raw Dataset Distribution (Imbalanced)", "raw_distribution.png", color='#ff7f7f')

    # 2. Processed
    proc_counts = get_class_counts(PROC_BASE_DIR)
    plot_distribution(proc_counts, "2. Processed Dataset Distribution (After Split)", "processed_distribution.png", color='#7fbfbf')

    # 3. Ultra
    ultra_counts = get_class_counts(ULTRA_BASE_DIR)
    plot_distribution(ultra_counts, "3. Ultra Dataset Distribution (Final Balanced)", "ultra_distribution.png", color='#ffd700')

    print("\nAll 3 charts generated successfully in the 'preprocessing' folder!")
