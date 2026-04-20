import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
# Automatically find current project root relative to this script (inside /preprocessing)
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_DIR = BASE_DIR / "dataset" / "Ultra"
TRAIN_IMG = SOURCE_DIR / "train" / "images"
TRAIN_LBL = SOURCE_DIR / "train" / "labels"

# Deterministic behavior
SEED = 42

# Multipliers (Compensating for Smart Crop loss)
MULTIPLIERS = {
    10: 15, 8: 15, # Rare (Radar, Missile) -> Aggressive boost
    4: 10, 6: 10, 9: 10, # Weak (Heli, Machine-Gun, Launcher)
    0: 6, 7: 6, 3: 6, # Medium (Artillery, Truck, Drone)
    1: 4, 12: 4, 13: 4 # Common
}

def polygon_area(pts):
    """Calculates shoelace area of a 4-point OBB."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def order_points_clockwise(pts):
    """Sorts points clockwise around their centroid (Robust Polar Sorting)."""
    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] == 0: return pts
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sort_idx = np.argsort(angles)
    return pts[sort_idx]

def polygon_area(pts):
    """Shoelace formula for polygon area."""
    if len(pts) < 3: return 0
    x = pts[:,0]
    y = pts[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def random_photometric(image):
    """Adds variety in lighting, blur, and noise."""
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if random.random() > 0.7:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)
    if random.random() > 0.8:
        row, col, ch = image.shape
        sigma = random.uniform(5, 15)
        gauss = np.random.normal(0, sigma, (row, col, ch))
        image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return image

def rotate_image_and_obb(image, labels, angle):
    h, w = image.shape[:2]
    
    # 📐 Math Zoom Crop: Calculate perfect zoom to eliminate black corners
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    Z = c + s * max(w/h, h/w)
    
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, Z)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    new_labels = []
    for lbl in labels:
        cls_id = int(lbl[0])
        pts = np.array(lbl[1:]).reshape(4, 2)
        
        # Calculate original area in pixels
        pts_px = pts.copy()
        pts_px[:, 0] *= w
        pts_px[:, 1] *= h
        orig_area = polygon_area(pts_px)

        # Transform points
        ones = np.ones((4, 1))
        pts_aug = np.hstack([pts_px, ones])
        trans_pts = M.dot(pts_aug.T).T

        # --- Claude's Fix: Clip then Order then Check Area ---
        # 1. Normalize and Clip
        trans_pts_norm = trans_pts.copy()
        trans_pts_norm[:, 0] /= w
        trans_pts_norm[:, 1] /= h
        
        # Guard against extreme OOB
        if np.any(trans_pts_norm < -0.3) or np.any(trans_pts_norm > 1.3): continue
        
        clipped_pts_norm = np.clip(trans_pts_norm, 0, 1)
        
        # 2. Re-order AFTER clip
        final_pts_norm = order_points_clockwise(clipped_pts_norm)
        
        # 3. Check Area of CLIPPED polygon
        final_pts_px = final_pts_norm.copy()
        final_pts_px[:, 0] *= w
        final_pts_px[:, 1] *= h
        new_area = polygon_area(final_pts_px)
        
        if orig_area < 4.0 or new_area < 4.0:
            continue # Discard degenerate or microscopic boxes
        
        if orig_area > 0 and (new_area / orig_area) < 0.3:
            continue # Discard if too much object is lost (>70% cut)

        new_labels.append([cls_id] + final_pts_norm.flatten().tolist())

    return rotated_img, new_labels

def flip_image_and_obb(image, labels):
    flipped_img = cv2.flip(image, 1)
    new_labels = []
    for lbl in labels:
        cls_id = int(lbl[0])
        pts = np.array(lbl[1:]).reshape(4, 2)
        pts[:, 0] = 1.0 - pts[:, 0]
        pts = order_points_clockwise(pts) # Re-order after flip
        new_labels.append([cls_id] + pts.flatten().tolist())
    return flipped_img, new_labels

def save_datapoint(img, labels, base_name, suffix, split_dir, allow_empty=False):
    if not labels and not allow_empty: return
    img_path = split_dir / "images" / f"{base_name}_{suffix}.jpg"
    lbl_path = split_dir / "labels" / f"{base_name}_{suffix}.txt"
    
    success = cv2.imwrite(str(img_path), img)
    if not success:
        print(f"FAILED to write image: {img_path}")
        return

    with open(lbl_path, "w") as f:
        for lbl in labels:
            cls = int(lbl[0])
            # Mandatory [0, 1] clip to fix OOB labels from source data
            coords = np.clip(lbl[1:], 0, 1)
            formatted_coords = " ".join(f"{v:.6f}" for v in coords)
            f.write(f"{cls} {formatted_coords}\n")

def process_single_image(args):
    img_path, seed_inc = args
    # Re-seed for multiprocessing diversity
    random.seed(SEED + seed_inc)
    np.random.seed(SEED + seed_inc)

    base_name = img_path.stem
    lbl_path = TRAIN_LBL / f"{base_name}.txt"
    
    # --- Skip if already processed ---
    target_orig_lbl = OUTPUT_DIR / "train" / "labels" / f"{base_name}_orig.txt"
    if target_orig_lbl.exists():
        return # Skip this image
        
    img = cv2.imread(str(img_path))
    if img is None: return

    labels = []
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 9: labels.append(parts)

    mult = 1
    if labels:
        ids = [int(l[0]) for l in labels]
        mult = max([MULTIPLIERS.get(id, 1) for id in ids])

    # 1. Original
    save_datapoint(img, labels, base_name, "orig", OUTPUT_DIR / "train", allow_empty=True)

    # 2. Variants
    if mult >= 4:
        for i, ang in enumerate([90, 180, 270]):
            t_img, t_lbl = rotate_image_and_obb(img, labels, ang)
            if random.random() > 0.5: t_img, t_lbl = flip_image_and_obb(t_img, t_lbl)
            t_img = random_photometric(t_img)
            save_datapoint(t_img, t_lbl, base_name, f"var_{i}", OUTPUT_DIR / "train")

    if mult >= 10:
        # Extra angles for rare classes to hit high counts
        for i, ang in enumerate([30, 45, 60, 120, 135, 150]):
            t_img, t_lbl = rotate_image_and_obb(img, labels, ang)
            t_img = random_photometric(t_img)
            save_datapoint(t_img, t_lbl, base_name, f"extra_{i}", OUTPUT_DIR / "train")

def post_process_cleanup():
    """Final Sanitization: Clips all coords to [0,1] and removes micro-boxes (<4px area) across ALL splits."""
    print("🧹 Starting final data sanitization (Clipping & Degenerate Removal)...")
    for split in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / split / "labels"
        if not lbl_dir.exists(): continue
        
        fixed_boxes = 0
        deleted_boxes = 0
        
        for f in lbl_dir.glob("*.txt"):
            keep_lines = []
            changed = False
            for line in open(f, "r", errors="ignore"):
                parts = line.strip().split()
                if len(parts) != 9: continue
                
                try:
                    cls = int(parts[0])
                    coords = np.array(list(map(float, parts[1:])))
                    
                    # 1. Coordinate Clipping (Fix OOB)
                    if np.any(coords < 0.0) or np.any(coords > 1.0):
                        coords = np.clip(coords, 0.0, 1.0)
                        changed = True
                        fixed_boxes += 1
                    
                    # 2. Degenerate Removal (Shoelace Area Check)
                    pts = coords.reshape(4, 2) * 640 # Scale to 640 for area check
                    if polygon_area(pts) < 4.0:
                        changed = True
                        deleted_boxes += 1
                        continue
                        
                    formatted = " ".join(f"{v:.6f}" for v in coords)
                    keep_lines.append(f"{cls} {formatted}")
                except: continue

            if changed:
                with open(f, "w") as out:
                    out.write("\n".join(keep_lines) + ("\n" if keep_lines else ""))
                    
    print(f"✅ Sanitization Complete: Fixed {fixed_boxes} OOB coords, Deleted {deleted_boxes} micro-boxes.")

def main():
    # --- Claude's Safety Guard ---
    assert SOURCE_DIR.exists(), f"Source {SOURCE_DIR} not found!"
    assert SOURCE_DIR != OUTPUT_DIR, "Source and Output cannot be the same!"
    
    if not OUTPUT_DIR.exists():
        print(f"📁 Creating new output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"🔄 Resuming... Skipping files already in {OUTPUT_DIR}")
    
    for s in ["train", "val", "test"]:
        (OUTPUT_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / s / "labels").mkdir(parents=True, exist_ok=True)

    print("📋 Transferring Val/Test sets...")
    for split in ["val", "test"]:
        for ftype in ["images", "labels"]:
            src = SOURCE_DIR / split / ftype
            dst = OUTPUT_DIR / split / ftype
            if src.exists():
                for item in src.iterdir():
                    try: shutil.copy2(item, dst)
                    except Exception as e: print(f"Error copying {item}: {e}")

    # Use half of CPU cores to prevent system freeze
    workers = max(1, cpu_count() // 2)
    print(f"🚀 Starting Parallel Augmentation (Cores used: {workers} of {cpu_count()})...")
    images = sorted(list(TRAIN_IMG.glob("*")))
    # Prepare arguments for multiprocessing
    task_args = [(p, i) for i, p in enumerate(images)]
    
    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_image, task_args), total=len(images)))
        pool.close()
        pool.join()

    # Create data.yaml
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(f"path: '{OUTPUT_DIR.as_posix()}'\n")
        f.write("train: 'train/images'\nval: 'val/images'\ntest: 'test/images'\nnc: 14\nnames: {list(range(14))}")

    # --- Step 4: Final Cleanup ---
    post_process_cleanup()

    print(f"\n🚀 SUCCESS! FINAL 'ULTRA' DATASET READY AT: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
