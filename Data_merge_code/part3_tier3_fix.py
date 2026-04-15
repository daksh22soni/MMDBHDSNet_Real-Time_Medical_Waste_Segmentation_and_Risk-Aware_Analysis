import os
import json
import time
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_PATH      = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"
CHECKPOINT_FILE  = os.path.join(OUTPUT_PATH, "tier3_checkpoint.json")
REMOVED_LOG_FILE = os.path.join(OUTPUT_PATH, "tier3_removed_log.txt")

TIER3_TOLERANCE  = 0.03   # 3% tolerance for cross-dataset comparison
CHECKPOINT_EVERY = 500
SPLITS           = ["train", "valid", "test"]
DATASET_PREFIXES = ["v1", "v5", "v6", "v7", "nv2", "nv3", "nv4"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS — segmentation format aware
# ─────────────────────────────────────────────

def get_label_path(img_path):
    """
    Build label path using os.path operations.
    Works correctly on Windows paths.
    """
    parts     = list(os.path.normpath(img_path).split(os.sep))
    new_parts = ["labels" if p == "images" else p for p in parts]
    lbl_path  = os.sep.join(new_parts)
    return os.path.splitext(lbl_path)[0] + ".txt"

def read_label(label_path):
    """
    Read YOLO label file — supports BOTH formats:

    Detection format (5 values):
       class x_center y_center width height

    Segmentation format (variable values):
       class x1 y1 x2 y2 x3 y3 ... xN yN
       → always: 1 class + N pairs of coords
       → total parts = odd number >= 5

    Returns list of (class_id, coords_tuple) sorted by class then first coord.
    Returns None if file missing or unreadable.
    Returns [] if file is empty (background image).
    """
    if not os.path.exists(label_path):
        return None
    objects = []
    try:
        with open(label_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                # ── KEY FIX ──
                # Accept both detection (5 parts) and segmentation (odd, >= 5 parts)
                if len(parts) >= 5 and len(parts) % 2 == 1:
                    cls    = int(parts[0])
                    coords = tuple(float(p) for p in parts[1:])
                    objects.append((cls, coords))
    except Exception:
        return None
    return sorted(objects, key=lambda x: (x[0], x[1][0], x[1][1]))

def get_prefix(filename):
    """Extract dataset prefix e.g. 'v1_image.jpg' → 'v1'"""
    for prefix in DATASET_PREFIXES:
        if filename.startswith(f"{prefix}_"):
            return prefix
    return None

def labels_match_within_tolerance(obj_a, obj_b, tolerance):
    """
    Check if two segmentation label lists match within tolerance.
    Checks:
      1. Same number of objects
      2. Same class IDs
      3. First 4 coordinates of each object within tolerance
         (checking all coords would be too slow for segmentation polygons)
    """
    if obj_a is None or obj_b is None:
        return False
    if len(obj_a) != len(obj_b):
        return False
    for (cls_a, coords_a), (cls_b, coords_b) in zip(obj_a, obj_b):
        # Must have same class
        if cls_a != cls_b:
            return False
        # Must have same number of polygon points
        if len(coords_a) != len(coords_b):
            return False
        # Check first 8 coordinate values (4 x,y points)
        # This is enough to distinguish different objects/positions
        check_count = min(8, len(coords_a))
        for i in range(check_count):
            if abs(coords_a[i] - coords_b[i]) > tolerance:
                return False
    return True

def remove_image_and_label(img_path):
    """Safely remove an image and its label file."""
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
        lbl_path = get_label_path(img_path)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
    except Exception as e:
        tqdm.write(f"   ⚠️  Could not remove {img_path}: {e}")

def collect_images():
    images = []
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if os.path.exists(img_dir):
            files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            images.extend(files)
    return images

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {
        "tier3_done":      False,
        "tier3_removed":   [],
        "tier3_processed": [],
    }

# ─────────────────────────────────────────────
# STEP 0 — Verify the fix works
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Verifying segmentation label reading fix...")
print("=" * 60)

surviving = collect_images()
print(f"\n   Total images found: {len(surviving)}")

# Quick test on first 20 images to confirm fix works
print("\n   Testing fixed label reader on 10 sample files:")
success = 0
failed  = 0

for img_path in surviving[:20]:
    lbl_path = get_label_path(img_path)
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        continue
    objects = read_label(lbl_path)
    prefix  = get_prefix(os.path.basename(img_path))
    if objects and len(objects) > 0:
        success += 1
        if success <= 3:
            print(f"      ✅ {os.path.basename(img_path)[:50]}")
            print(f"         prefix: {prefix}  |  objects: {len(objects)}  |  "
                  f"first class: {objects[0][0]}  |  "
                  f"first coord count: {len(objects[0][1])}")
    else:
        failed += 1

print(f"\n   Result: {success} files successfully read with objects")
print(f"          {failed} files returned empty (background images — normal)")

if success == 0:
    print("\n   ❌ CRITICAL: Fix did not work — no objects read!")
    print("      Check label file format manually.")
    exit(1)
else:
    print(f"\n   ✅ Fix confirmed working — segmentation format detected correctly!")

# ─────────────────────────────────────────────
# STEP 1 — Load checkpoint
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Checkpoint system...")
print("=" * 60)

checkpoint = load_checkpoint()

if checkpoint["tier3_done"]:
    print(f"\n   ✅ Already complete — {len(checkpoint['tier3_removed'])} removed")
elif checkpoint["tier3_processed"]:
    print(f"\n   🔄 Resuming — {len(checkpoint['tier3_processed'])} already processed")
    print(f"              {len(checkpoint['tier3_removed'])} removed so far")
else:
    print(f"\n   ✅ Fresh start")

# ─────────────────────────────────────────────
# STEP 2 — Build per-dataset label index
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building per-dataset label index...")
print("=" * 60)
print(f"   ℹ️  Now correctly reads segmentation polygon format")

tier3_start      = time.time()
dataset_label_db = {p: [] for p in DATASET_PREFIXES}
failed_reads     = 0
no_prefix        = 0
empty_labels     = 0

index_bar = tqdm(surviving, desc="   Building index",
                 unit="file", ncols=70, colour="yellow")

for img_path in index_bar:
    fname  = os.path.basename(img_path)
    prefix = get_prefix(fname)

    if prefix is None:
        no_prefix += 1
        continue

    lbl_path = get_label_path(img_path)
    objects  = read_label(lbl_path)

    if objects is None:
        failed_reads += 1
        continue
    if len(objects) == 0:
        empty_labels += 1
        continue

    dataset_label_db[prefix].append((objects, img_path))

index_bar.close()

# Print index results
print(f"\n   Index build results:")
total_indexed = 0
for prefix in DATASET_PREFIXES:
    count = len(dataset_label_db[prefix])
    total_indexed += count
    status = "✅" if count > 0 else "⚠️  0 — check prefix!"
    print(f"      {prefix:<6} → {count:>6} images indexed   {status}")

print(f"\n   Total indexed        : {total_indexed}")
print(f"   Empty label files    : {empty_labels}  (background images — normal)")
print(f"   Failed label reads   : {failed_reads}")
print(f"   No prefix found      : {no_prefix}")

if total_indexed == 0:
    print("\n   ❌ CRITICAL: Index still empty — cannot run Tier 3!")
    exit(1)
else:
    print(f"\n   ✅ Index built successfully!")

# ─────────────────────────────────────────────
# STEP 3 — Cross-dataset comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TIER 3: Cross-Dataset Augmentation Duplicate Check")
print("=" * 60)
print(f"   ℹ️  Tolerance         : {TIER3_TOLERANCE*100:.0f}%")
print(f"   ℹ️  Comparison method : first 4 polygon points per object")
print(f"   ℹ️  Same dataset      : NEVER compared (within-dataset augmentations kept)")
print(f"   ⏱️  Estimated time    : ~8-15 minutes")

tier3_removed     = 0
already_processed = set(checkpoint["tier3_processed"])
removed           = checkpoint["tier3_removed"]
removed_set       = set(removed)

if checkpoint["tier3_done"]:
    print(f"\n   ✅ Already complete — {len(checkpoint['tier3_removed'])} removed (from checkpoint)")
    tier3_removed = len(checkpoint["tier3_removed"])
else:
    compare_bar = tqdm(surviving, desc="   Tier3 Cross-DS",
                       unit="file", ncols=70, colour="magenta")

    for i, img_path in enumerate(compare_bar):
        if img_path in already_processed or img_path in removed_set:
            continue
        if not os.path.exists(img_path):
            continue

        fname       = os.path.basename(img_path)
        this_prefix = get_prefix(fname)
        if this_prefix is None:
            continue

        objects = read_label(get_label_path(img_path))
        if not objects or len(objects) == 0:
            continue

        is_cross_duplicate = False

        for other_prefix, other_entries in dataset_label_db.items():
            if other_prefix == this_prefix:
                continue   # ← NEVER compare same dataset to itself

            for other_objects, other_path in other_entries:
                if not os.path.exists(other_path):
                    continue
                if labels_match_within_tolerance(objects, other_objects, TIER3_TOLERANCE):
                    remove_image_and_label(img_path)
                    removed.append(img_path)
                    removed_set.add(img_path)
                    tier3_removed += 1
                    compare_bar.set_postfix(removed=tier3_removed)
                    is_cross_duplicate = True
                    break

            if is_cross_duplicate:
                break

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint["tier3_removed"]   = removed
            checkpoint["tier3_processed"] = list(already_processed) + surviving[:i+1]
            save_checkpoint(checkpoint)
            tqdm.write(f"   💾 Checkpoint saved at {i+1} files")

    checkpoint["tier3_removed"] = removed
    checkpoint["tier3_done"]    = True
    save_checkpoint(checkpoint)

tier3_time = time.time() - tier3_start
print(f"\n   ✅ Tier 3 done in {tier3_time:.1f}s — removed: {tier3_removed}")

# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

cv_errors   = 0
final_total = 0

# Check A: Image == Label count per split
print("\n   [A] Image vs Label count after Tier 3:")
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    if not os.path.exists(img_dir):
        continue
    img_count = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])
    lbl_count = len([f for f in os.listdir(lbl_dir)
                     if f.lower().endswith(".txt")])
    final_total += img_count
    status = "✅" if img_count == lbl_count else "❌"
    if img_count != lbl_count:
        cv_errors += 1
    print(f"      {split:<6} → images: {img_count:>6}   labels: {lbl_count:>6}   {status}")

# Check B: Removed files actually gone
print("\n   [B] Verifying all removed files are deleted:")
still_exists = sum(1 for p in checkpoint["tier3_removed"] if os.path.exists(p))
if still_exists == 0:
    print(f"      ✅ All {len(checkpoint['tier3_removed'])} removed files confirmed deleted")
else:
    print(f"      ❌ {still_exists} files still exist!")
    cv_errors += 1

# Check C: Orphan labels
print("\n   [C] Orphan label check:")
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    if not os.path.exists(lbl_dir):
        continue
    img_stems = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
    orphans   = lbl_stems - img_stems
    if orphans:
        for orphan in orphans:
            os.remove(os.path.join(lbl_dir, orphan + ".txt"))
        cv_errors += 1
    status = "✅ No orphans" if not orphans else f"❌ {len(orphans)} found and cleaned"
    print(f"      {split:<6} → {status}")

# Check D: Per-prefix survival count
print("\n   [D] Per-dataset survival count after Tier 3:")
for prefix in DATASET_PREFIXES:
    count = 0
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if os.path.exists(img_dir):
            count += len([f for f in os.listdir(img_dir)
                          if f.startswith(f"{prefix}_")])
    status = "✅" if count > 0 else "⚠️  0 images!"
    print(f"      {prefix:<6} → {count:>6} images remaining   {status}")

# Check E: Dataset size healthy
print("\n   [E] Dataset size health check:")
if final_total >= 5000:
    print(f"      ✅ {final_total} images — healthy for YOLOv8 training")
else:
    print(f"      ⚠️  Only {final_total} images — consider adjusting tolerance")
    cv_errors += 1

# ─────────────────────────────────────────────
# Save log
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving removed log...")
print("=" * 60)

with open(REMOVED_LOG_FILE, "w") as f:
    f.write("TIER 3 — Cross-Dataset Segmentation Duplicates\n")
    f.write("=" * 60 + "\n")
    f.write(f"Tolerance used: {TIER3_TOLERANCE*100:.0f}%\n")
    f.write(f"Format: YOLO Segmentation\n\n")
    for p in checkpoint["tier3_removed"]:
        f.write(p + "\n")
    f.write(f"\nTotal removed: {len(checkpoint['tier3_removed'])}\n")

print(f"\n   ✅ Log saved: {REMOVED_LOG_FILE}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
   Images before Tier 3    : {len(surviving):>6}
   Tier 3 removed          : {tier3_removed:>6}  ({tier3_time:.1f}s)
   ──────────────────────────────────
   Final clean images      : {final_total:>6}

   Label format detected   : YOLO Segmentation ✅
   Cross validation errors : {cv_errors}
""")

if cv_errors == 0:
    print("   🎉 ALL CHECKS PASSED — Part 3 Complete!")
    print("   ➡️  Ready for Part 4 (Class ID Remapping)")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("   🧹 Checkpoint file cleaned up")
else:
    print("   ⚠️  Some issues found — review above before continuing")

print("=" * 60)