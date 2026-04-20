import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
# Portable: Automatically resolves paths relative to this script's location
BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "dataset" / "raw" / "train"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20
SEED = 42

def validate_obb_label(label_path: Path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False, "empty"
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines: return False, "empty"
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9: return False, "invalid_format"
                [float(x) for x in parts]
        return True, "valid"
    except:
        return False, "corrupt"

def safe_reset_dir(path: Path):
    if path.exists():
        print(f"Cleaning old processed directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def split_list(items, seed):
    shuffled_items = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled_items)
    train_end = int(len(shuffled_items) * TRAIN_RATIO)
    val_end = train_end + int(len(shuffled_items) * VAL_RATIO)
    return shuffled_items[:train_end], shuffled_items[train_end:val_end], shuffled_items[val_end:]

def split_data():
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Ratios must sum up to 1.0")

    img_dir = RAW_DIR / "images"
    lbl_dir = RAW_DIR / "labels"

    # 1. Gather all images
    all_images = sorted([f.name for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"Total images found: {len(all_images)}")

    labeled_images = []
    background_images = []

    # 2. Categorization
    for img_name in all_images:
        base_name = Path(img_name).stem
        label_path = lbl_dir / f"{base_name}.txt"
        is_valid, _ = validate_obb_label(label_path)
        if is_valid:
            labeled_images.append(img_name)
        else:
            background_images.append(img_name)

    print(f"Healthy Labeled: {len(labeled_images)} | Background: {len(background_images)}")

    # 3. Splitting
    train_l, val_l, test_l = split_list(labeled_images, SEED)
    train_b, val_b, test_b = split_list(background_images, SEED)
    splits = {"train": train_l + train_b, "val": val_l + val_b, "test": test_l + test_b}

    # 4. Reset Output
    safe_reset_dir(PROCESSED_DIR)

    # 5. Copying with Error Handling
    for split_name, image_list in splits.items():
        print(f"Processing '{split_name.upper()}' set...")
        t_img_dir = PROCESSED_DIR / split_name / "images"
        t_lbl_dir = PROCESSED_DIR / split_name / "labels"
        t_img_dir.mkdir(parents=True, exist_ok=True)
        t_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        error_count = 0

        for img_name in image_list:
            try:
                # Use absolute paths to help Windows
                src_img = (img_dir / img_name).absolute()
                dst_img = (t_img_dir / img_name).absolute()
                
                shutil.copy2(src_img, dst_img)
                
                base_name = Path(img_name).stem
                src_lbl = (lbl_dir / f"{base_name}.txt").absolute()
                dst_lbl = (t_lbl_dir / f"{base_name}.txt").absolute()
                
                if src_lbl.exists() and src_lbl.stat().st_size > 0:
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.touch()
                copied_count += 1
            except Exception as e:
                error_count += 1
                if error_count < 10: # Only print first 10 errors to avoid spam
                    print(f"Failed to copy {img_name}: {e}")

        print(f"Finished {split_name}: {copied_count} copied, {error_count} failed.")

    print(f"\nREORGANIZATION DONE! Output: {PROCESSED_DIR}")

if __name__ == "__main__":
    split_data()
