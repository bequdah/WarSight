"""
Quick Dataset Audit - Focused on OBB integrity
Fast scan: checks OOB coords, degenerate boxes, class distribution
"""

import numpy as np
from pathlib import Path
from collections import Counter
import sys

DATASET = Path(r"c:\Users\AL Qudah\Desktop\projects\CV\dataset\v2_4_Ultra")
NAMES = {0:'Artillery',1:'Camo-Soldier',2:'Civilian',3:'Drone',4:'Helicopter',
         5:'Jet Fighters',6:'Machine-Gun',7:'Mil-Truck',8:'Missile',
         9:'Launcher',10:'Radar',11:'Soldier',12:'Tank',13:'Handgun'}

def audit_split(split):
    lbl_dir = DATASET / split / "labels"
    img_dir = DATASET / split / "images"
    if not lbl_dir.exists():
        print(f"  [SKIP] {split} not found"); return

    counts     = Counter()
    oob        = 0
    degenerate = 0
    bad_format = 0
    total      = 0
    n_labels   = 0

    for f in lbl_dir.glob("*.txt"):
        n_labels += 1
        for line in open(f, errors="ignore"):
            line = line.strip()
            if not line: continue
            parts = line.split()
            total += 1
            if len(parts) != 9:
                bad_format += 1
                continue
            try:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                bad_format += 1
                continue

            # OOB check
            if any(v < 0.0 or v > 1.0 for v in coords):
                oob += 1
                continue

            # Degenerate (shoelace area)
            pts = np.array(coords).reshape(4, 2) * 640
            x, y = pts[:,0], pts[:,1]
            area = 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
            if area < 4.0:
                degenerate += 1
                continue

            counts[cls] += 1

    n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else "?"

    print(f"\n{'='*55}")
    print(f"  SPLIT: {split.upper()}  |  Images: {n_imgs}  |  Label files: {n_labels}")
    print(f"{'='*55}")
    print(f"  Total boxes    : {total:,}")
    print(f"  Valid boxes    : {sum(counts.values()):,}")
    print(f"  [OOB]  Out-of-bounds : {oob:,}  {'<-- PROBLEM!' if oob else '<-- OK'}")
    print(f"  [DEG]  Degenerate   : {degenerate:,}  {'<-- PROBLEM!' if degenerate > 50 else '<-- OK'}")
    print(f"  [BAD]  Bad format   : {bad_format:,}  {'<-- PROBLEM!' if bad_format else '<-- OK'}")

    if counts:
        mx = max(counts.values())
        print(f"\n  Class Distribution:")
        for cls in sorted(counts):
            bar = '#' * int(counts[cls] / mx * 30)
            print(f"    {NAMES.get(cls, cls):<18} {counts[cls]:>6,}  {bar}")

if __name__ == "__main__":
    print("Quick Audit - v2_4_Ultra Dataset")
    for split in ["train", "val", "test"]:
        audit_split(split)
    print("\n  Done.")
