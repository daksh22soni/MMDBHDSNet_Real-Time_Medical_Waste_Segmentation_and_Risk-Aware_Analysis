"""
========================================================================
DBHDSNet — Dataset Inspector
Run this FIRST before training to:
  1. Print your exact 38 class names from master_data.yaml
  2. Verify folder structure and image/label counts
  3. Check class distribution across train/valid/test splits
  4. Identify empty label files and missing annotations
  5. Compute mean image dimensions and aspect ratios
  6. Show a preview of parsed label polygons

Run:  python scripts/inspect_dataset.py
========================================================================
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import cv2
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION (mirrors config.py paths)
# ════════════════════════════════════════════════════════════════════════

DATASET_ROOT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/master_dataset")   # ← UPDATE THIS
YAML_PATH    = DATASET_ROOT / "master_data.yaml"

SPLITS = {
    "train": {
        "images": DATASET_ROOT / "train" / "images",
        "labels": DATASET_ROOT / "train" / "labels",
    },
    "valid": {
        "images": DATASET_ROOT / "valid" / "images",
        "labels": DATASET_ROOT / "valid" / "labels",
    },
    "test": {
        "images": DATASET_ROOT / "test" / "images",
        "labels": DATASET_ROOT / "test" / "labels",
    },
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_class_names(yaml_data):
    names = yaml_data.get("names", {})
    if isinstance(names, list):
        return names
    return [names[i] for i in sorted(names.keys())]


def parse_label(path):
    """Returns list of (cls_id, polygon) from a YOLO seg label file."""
    annotations = []
    if not Path(path).exists():
        return annotations
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32)
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue
            poly = coords.reshape(-1, 2)
            annotations.append((cls_id, poly))
    return annotations


def sep(char="─", width=72):
    print(char * width)


# ════════════════════════════════════════════════════════════════════════
# MAIN INSPECTION
# ════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  DBHDSNet — Dataset Inspector")
    print("═" * 72 + "\n")

    # ── 1. Load YAML ──────────────────────────────────────────────────
    if not YAML_PATH.exists():
        print(f"[ERROR] YAML not found: {YAML_PATH}")
        sys.exit(1)

    data = load_yaml(YAML_PATH)
    class_names = get_class_names(data)
    nc = len(class_names)

    print(f"YAML path   : {YAML_PATH}")
    print(f"Num classes : {nc}")
    print()
    sep()
    print("  CLASS NAMES (copy-paste into config.py HAZARD_TIER_MAP):")
    sep()
    for i, name in enumerate(class_names):
        print(f"  [{i:02d}]  {name}")
    print()

    # ── 2. Folder checks ─────────────────────────────────────────────
    sep()
    print("  FOLDER STRUCTURE:")
    sep()
    for split, paths in SPLITS.items():
        img_ok = paths["images"].exists()
        lbl_ok = paths["labels"].exists()
        imgs   = list(paths["images"].glob("*")) if img_ok else []
        imgs   = [f for f in imgs if f.suffix.lower() in IMG_EXTENSIONS]
        lbls   = list(paths["labels"].glob("*.txt")) if lbl_ok else []
        print(
            f"  {split:6s}  images: {len(imgs):6d}  "
            f"labels: {len(lbls):6d}  "
            f"img_dir={'✓' if img_ok else '✗'}  "
            f"lbl_dir={'✓' if lbl_ok else '✗'}"
        )
    print()

    # ── 3. Class distribution ─────────────────────────────────────────
    sep()
    print("  CLASS DISTRIBUTION (instance counts):")
    sep()

    split_counts = {}
    issues       = defaultdict(list)

    for split, paths in SPLITS.items():
        counts   = defaultdict(int)
        n_empty  = 0
        n_missing = 0

        img_files = sorted([
            f for f in paths["images"].iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS
        ]) if paths["images"].exists() else []

        print(f"\n  Scanning {split} ({len(img_files)} images)...")

        for img_path in tqdm(img_files, desc=f"  {split}", ncols=80, leave=False):
            lbl_path = paths["labels"] / (img_path.stem + ".txt")

            if not lbl_path.exists():
                n_missing += 1
                issues[split].append(f"missing label: {img_path.name}")
                continue

            annots = parse_label(lbl_path)

            if len(annots) == 0:
                n_empty += 1
            else:
                for cls_id, _ in annots:
                    if 0 <= cls_id < nc:
                        counts[cls_id] += 1
                    else:
                        issues[split].append(
                            f"invalid class_id {cls_id} in {lbl_path.name}"
                        )

        split_counts[split] = counts
        print(f"  → {split}: {sum(counts.values())} annotations, "
              f"{n_empty} empty labels, {n_missing} missing label files")

    # Per-class table
    print()
    header = f"  {'Class':30s}  {'ID':4s}  {'Train':8s}  {'Valid':8s}  {'Test':8s}"
    print(header)
    sep("─")
    for i, name in enumerate(class_names):
        tr = split_counts.get("train", {}).get(i, 0)
        va = split_counts.get("valid", {}).get(i, 0)
        te = split_counts.get("test",  {}).get(i, 0)
        flag = "  ⚠ LOW" if (tr + va + te) < 50 else ""
        print(f"  {name:30s}  {i:4d}  {tr:8d}  {va:8d}  {te:8d}{flag}")
    print()

    # ── 4. Image dimension statistics ────────────────────────────────
    sep()
    print("  IMAGE DIMENSION STATISTICS (sampled from train):")
    sep()

    train_imgs = sorted([
        f for f in SPLITS["train"]["images"].iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    ]) if SPLITS["train"]["images"].exists() else []

    sample_n = min(500, len(train_imgs))
    widths, heights, aspects = [], [], []

    for img_path in tqdm(train_imgs[:sample_n], desc="  Reading dims", ncols=80, leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        aspects.append(w / h)

    if widths:
        print(f"  Width  — mean: {np.mean(widths):.0f}  "
              f"std: {np.std(widths):.0f}  "
              f"min: {np.min(widths)}  max: {np.max(widths)}")
        print(f"  Height — mean: {np.mean(heights):.0f}  "
              f"std: {np.std(heights):.0f}  "
              f"min: {np.min(heights)}  max: {np.max(heights)}")
        print(f"  Aspect — mean: {np.mean(aspects):.3f}  "
              f"std: {np.std(aspects):.3f}")
    print()

    # ── 5. Polygon statistics ────────────────────────────────────────
    sep()
    print("  POLYGON VERTEX STATISTICS (sampled train labels):")
    sep()

    lbl_files = list(SPLITS["train"]["labels"].glob("*.txt"))[:500] \
                if SPLITS["train"]["labels"].exists() else []
    n_verts = []
    for lf in tqdm(lbl_files, desc="  Reading labels", ncols=80, leave=False):
        for _, poly in parse_label(lf):
            n_verts.append(len(poly))

    if n_verts:
        print(f"  Polygon vertices — mean: {np.mean(n_verts):.1f}  "
              f"min: {np.min(n_verts)}  max: {np.max(n_verts)}  "
              f"median: {np.median(n_verts):.0f}")
    print()

    # ── 6. Issues report ─────────────────────────────────────────────
    total_issues = sum(len(v) for v in issues.values())
    if total_issues:
        sep()
        print(f"  ⚠  ISSUES FOUND ({total_issues} total):")
        sep()
        for split, msgs in issues.items():
            for m in msgs[:10]:
                print(f"  [{split}] {m}")
            if len(msgs) > 10:
                print(f"  ... and {len(msgs)-10} more in {split}")
    else:
        print("  ✓ No dataset issues detected.")
    print()

    # ── 7. Config snippet (with pre-filled tiers from config.py) ────
    sep()
    print("  HAZARD_TIER_MAP — current assignment from config.py:")
    print("  Tier 1=Sharps/HighHazard  2=Infectious  3=Pharma  4=General")
    sep()
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import HAZARD_TIER_MAP
        tier_labels = {1: "SHARPS/HAZARDOUS", 2: "INFECTIOUS",
                       3: "PHARMACEUTICAL",   4: "GENERAL"}
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for t in HAZARD_TIER_MAP.values():
            tier_counts[t] = tier_counts.get(t, 0) + 1
        print("\nHAZARD_TIER_MAP: dict[str, int] = {")
        for i, name in enumerate(class_names):
            tier = HAZARD_TIER_MAP.get(name, "? (MISSING)")
            lbl  = tier_labels.get(tier, "") if isinstance(tier, int) else ""
            print(f'    "{name}":{" "*(30-len(name))}{tier},   # class {i:02d} — {lbl}')
        print("}\n")
        print(f"  Tier summary: T1={tier_counts[1]} | T2={tier_counts[2]} | "
              f"T3={tier_counts[3]} | T4={tier_counts[4]}  (total={sum(tier_counts.values())})")
        missing = [n for n in class_names if n not in HAZARD_TIER_MAP]
        if missing:
            print(f"\n  ⚠ MISSING from HAZARD_TIER_MAP: {missing}")
            print("  These will default to Tier 4 (General) at runtime.")
        else:
            print("  ✓ All 38 classes present in HAZARD_TIER_MAP.")
    except Exception as e:
        print(f"  (Could not load config.py: {e})")
        print("\nHAZARD_TIER_MAP: dict[str, int] = {")
        for name in class_names:
            print(f'    "{name}": ?,    # Tier 1=Sharps 2=Infectious 3=Pharma 4=General')
        print("}\n")

    print("═" * 72)
    print("  Inspection complete. config.py HAZARD_TIER_MAP is pre-filled.")
    print("  Update DATASET_ROOT in config.py, then run:  python train.py")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
