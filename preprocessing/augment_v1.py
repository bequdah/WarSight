import cv2
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import shutil

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_DIR = BASE_DIR / "dataset" / "ultradata"
TRAIN_IMG = SOURCE_DIR / "train" / "images"
TRAIN_LBL = SOURCE_DIR / "train" / "labels"

# Deterministic behavior
SEED = 42

# --- Dynamic Balancing Configuration ---
TARGET_MIN = 1000
TARGET_MID = 2000
TARGET_MAX = 4000
EXCLUDED_CLASSES = {8}

# الخريطة الجديدة للترقيم المتسلسل (0-12)
REMAP_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    9: 8,   # Missile-Launcher -> 8
    10: 9,  # Radar -> 9
    11: 10, # Soldier -> 10
    12: 11, # Tank -> 11
    13: 12  # Handgun -> 12
}

CLASS_NAMES = {
    0: "Artillery", 1: "Camouflaged-Soldier", 2: "Civilian", 3: "Drone",
    4: "Helicopter", 5: "Jet-Fighters", 6: "Machine-Gun", 7: "Military-Truck",
    8: "Missile-Launcher", 9: "Radar", 10: "Soldier", 11: "Tank", 12: "Handgun",
}

def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] == 0: return pts
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

def random_photometric(image):
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
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    Z = c + s * max(w / h, h / w)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, Z)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    new_labels = []
    for lbl in labels:
        cls_id = int(lbl[0])
        pts = np.array(lbl[1:]).reshape(4, 2)
        pts_px = pts.copy()
        pts_px[:, 0] *= w; pts_px[:, 1] *= h
        orig_area = polygon_area(pts_px)
        pts_aug = np.hstack([pts_px, np.ones((4, 1))])
        trans_pts = M.dot(pts_aug.T).T
        trans_norm = trans_pts.copy()
        trans_norm[:, 0] /= w; trans_norm[:, 1] /= h
        if np.any(trans_norm < -0.3) or np.any(trans_norm > 1.3): continue
        clipped = np.clip(trans_norm, 0, 1)
        final_pts = order_points_clockwise(clipped)
        final_px = final_pts.copy()
        final_px[:, 0] *= w; final_px[:, 1] *= h
        new_area = polygon_area(final_px)
        if orig_area < 4.0 or new_area < 4.0: continue
        if orig_area > 0 and (new_area / orig_area) < 0.3: continue
        new_labels.append([cls_id] + final_pts.flatten().tolist())
    return rotated_img, new_labels

def flip_image_and_obb(image, labels):
    flipped_img = cv2.flip(image, 1)
    new_labels = []
    for lbl in labels:
        cls_id = int(lbl[0])
        pts = np.array(lbl[1:]).reshape(4, 2)
        pts[:, 0] = 1.0 - pts[:, 0]
        pts = order_points_clockwise(pts)
        new_labels.append([cls_id] + pts.flatten().tolist())
    return flipped_img, new_labels

def safe_stem(stem):
    """Return an ASCII-safe version of a file stem for output naming."""
    return stem.encode('ascii', errors='ignore').decode('ascii').replace(' ', '_') or f"img_{hash(stem) & 0xFFFFFF}"

def save_datapoint(img, labels, base_name, suffix, split_dir, allow_empty=False):
    if not labels and not allow_empty: return
    safe_name = safe_stem(base_name)
    img_path = split_dir / "images" / f"{safe_name}_{suffix}.jpg"
    lbl_path = split_dir / "labels" / f"{safe_name}_{suffix}.txt"
    try:
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        if is_success: im_buf_arr.tofile(str(img_path))
        with open(lbl_path, "w", encoding="utf-8") as f:
            for lbl in labels:
                old_cls = int(lbl[0])
                if old_cls in EXCLUDED_CLASSES: continue
                
                # تطبيق الترقيم الجديد
                new_cls = REMAP_MAP.get(old_cls, old_cls)
                
                coords = np.clip(lbl[1:], 0.0, 1.0)
                f.write(f"{new_cls} " + " ".join(f"{v:.6f}" for v in coords) + "\n")
    except Exception as e:
        print(f"[WARN] Skip write error on {img_path}: {e}")

def process_single_image(args):
    img_path, seed_inc, dyn_config, allowed_stems = args
    random.seed(SEED + seed_inc)
    np.random.seed(SEED + seed_inc)
    base_name = img_path.stem
    lbl_path = TRAIN_LBL / f"{base_name}.txt"
    if allowed_stems is not None and base_name not in allowed_stems: return
    safe_name = safe_stem(base_name)
    target_orig_lbl = OUTPUT_DIR / "train" / "labels" / f"{safe_name}_orig.txt"
    if target_orig_lbl.exists(): return
    try:
        img_array = np.fromfile(str(img_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except: img = None
    if img is None: return
    labels = []
    if lbl_path.exists():
        try:
            with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) != 9:
                        for p in (lbl_path, img_path):
                            try: os.remove(str(p))
                            except: pass
                        return
                    labels.append([float(p) for p in parts])
        except Exception as e:
            print(f"[WARN] Could not read label {lbl_path.name}: {e}")
            return
    if not labels:
        if random.random() > 0.10: return
        mult = 1
    else:
        img_classes = [int(l[0]) for l in labels]
        if all(c in EXCLUDED_CLASSES for c in img_classes): return
        best_mult = 1
        for c in img_classes:
            if c in EXCLUDED_CLASSES: continue
            cfg = dyn_config.get(c, {"mult": 1})
            best_mult = max(best_mult, cfg["mult"])
        mult = best_mult
    save_datapoint(img, labels, base_name, "orig", OUTPUT_DIR / "train", allow_empty=True)
    if mult > 1:
        num_variants = mult - 1
        angles = [30, 45, 60, 90, 120, 135, 150, 180, 210, 240, 270, 300, 330]
        for i in range(num_variants):
            ang = random.choice(angles)
            t_img, t_lbl = rotate_image_and_obb(img, labels, ang)
            if random.random() > 0.5: t_img, t_lbl = flip_image_and_obb(t_img, t_lbl)
            t_img = random_photometric(t_img)
            save_datapoint(t_img, t_lbl, base_name, f"aug_{i}", OUTPUT_DIR / "train")

def post_process_cleanup():
    for split in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / split / "labels"
        if not lbl_dir.exists(): continue
        for f in lbl_dir.glob("*.txt"):
            keep_lines = []
            changed = False
            try:
                for line in open(f, "r", encoding="utf-8", errors="ignore"):
                    parts = line.strip().split()
                    if len(parts) != 9:
                        changed = True; continue
                    old_cls = int(parts[0])
                    coords = np.array([float(v) for v in parts[1:]])
                    if np.any(coords < 0.0) or np.any(coords > 1.0):
                        coords = np.clip(coords, 0.0, 1.0); changed = True
                    if old_cls in EXCLUDED_CLASSES:
                        changed = True; continue
                    
                    # تطبيق الترقيم الجديد في الـ cleanup أيضاً
                    new_cls = REMAP_MAP.get(old_cls, old_cls)
                    if new_cls != old_cls: changed = True

                    pts = coords.reshape(4, 2) * 640
                    if polygon_area(pts) < 4.0:
                        changed = True; continue
                    keep_lines.append(f"{new_cls} " + " ".join(f"{v:.6f}" for v in coords))
                if changed:
                    with open(f, "w", encoding="utf-8") as out:
                        out.write("\n".join(keep_lines) + ("\n" if keep_lines else ""))
            except Exception as e:
                print(f"[WARN] Cleanup skip {f.name}: {e}")

def main():
    random.seed(SEED); np.random.seed(SEED)
    assert SOURCE_DIR.exists()
    for s in ["train", "val", "test"]:
        (OUTPUT_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / s / "labels").mkdir(parents=True, exist_ok=True)
    print("Scanning dataset...")
    raw_counts, class_to_images = {}, {}
    label_files = list(TRAIN_LBL.glob("*.txt"))
    for f in tqdm(label_files, desc="Scanning"):
        try:
            classes_in_file = set()
            with open(f, "r", encoding="utf-8", errors="ignore") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts: continue
                    cls = int(float(parts[0]))
                    if cls in EXCLUDED_CLASSES: continue
                    classes_in_file.add(cls)
            candidates = list(TRAIN_IMG.glob(f"{f.stem}.*"))
            if not candidates: continue
            img_path = candidates[0]
            for cls in classes_in_file:
                raw_counts[cls] = raw_counts.get(cls, 0) + 1
                class_to_images.setdefault(cls, []).append(img_path)
        except Exception as e:
            print(f"[WARN] Scan skip {f.name}: {e}")
            continue
    dyn_config, allowed_stems = {}, set()
    for cls in sorted(raw_counts.keys()):
        count = raw_counts[cls]; paths = class_to_images.get(cls, [])
        if count < TARGET_MIN: mult = min(20, math.ceil(TARGET_MIN / count)); selected = paths
        elif count < TARGET_MID: mult = min(10, math.ceil(TARGET_MID / count)); selected = paths
        elif count <= TARGET_MAX: mult = 1; selected = paths
        else: mult = 1; selected = random.sample(paths, TARGET_MAX)
        dyn_config[cls] = {"mult": mult}
        for p in selected: allowed_stems.add(p.stem)
    all_images = sorted(list(TRAIN_IMG.glob("*")))
    task_args = [(p, i, dyn_config, allowed_stems) for i, p in enumerate(all_images)]
    with Pool(max(1, cpu_count() // 2)) as pool:
        list(tqdm(pool.imap_unordered(process_single_image, task_args), total=len(all_images), desc="Augmenting"))

    print("\nCopying validation and test sets...")
    for split in ["val", "test"]:
        split_src_img = SOURCE_DIR / split / "images"
        split_src_lbl = SOURCE_DIR / split / "labels"
        
        if not split_src_img.exists():
            continue
            
        files = list(split_src_img.glob("*"))
        for img_p in tqdm(files, desc=f"Processing {split}", leave=False):
            safe_name = safe_stem(img_p.stem)
            # Use original extension or force jpg? save_datapoint uses jpg.
            # To be safe and consistent with train, we copy but keep the naming safe.
            target_img = OUTPUT_DIR / split / "images" / f"{safe_name}{img_p.suffix}"
            shutil.copy(str(img_p), str(target_img))
            
            lbl_p = split_src_lbl / f"{img_p.stem}.txt"
            if lbl_p.exists():
                target_lbl = OUTPUT_DIR / split / "labels" / f"{safe_name}.txt"
                shutil.copy(str(lbl_p), str(target_lbl))

    print("\nRunning post-process cleanup...")
    post_process_cleanup()
    
    # تحديث الـ YAML بالعدد الصحيح (13) والأسماء المتسلسلة
    nc = len(CLASS_NAMES)
    names_list = [CLASS_NAMES[i] for i in range(nc)]
    
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(f"path: {OUTPUT_DIR.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: {nc}\nnames: {names_list}\n")
    print(f"\n✅ Done! Dataset ready at: {OUTPUT_DIR} with {nc} sequential classes.")

if __name__ == "__main__":
    main()