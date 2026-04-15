import os
import json
import time
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH        = r"C:\Users\gahan\Desktop\dbhdsnet_project"
OUTPUT_PATH      = os.path.join(BASE_PATH, "master_dataset")
CHECKPOINT_FILE  = os.path.join(OUTPUT_PATH, "part5_checkpoint.json")
LOG_FILE         = os.path.join(OUTPUT_PATH, "part5_split_log.txt")
BACKUP_DIR       = os.path.join(OUTPUT_PATH, "splits_backup_part5")

SPLITS           = ["train", "valid", "test"]
DATASET_PREFIXES = ["v1", "v5", "v6", "v7", "nv2", "nv3", "nv4"]

# Target split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO  = 0.10

RANDOM_SEED      = 42    # fixed seed for reproducibility
CHECKPOINT_EVERY = 500

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_label_path(img_path):
    parts     = list(os.path.normpath(img_path).split(os.sep))
    new_parts = ["labels" if p == "images" else p for p in parts]
    return os.path.splitext(os.sep.join(new_parts))[0] + ".txt"

def get_prefix(filename):
    for prefix in DATASET_PREFIXES:
        if filename.startswith(f"{prefix}_"):
            return prefix
    return "unknown"

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "done":         False,
        "backup_done":  False,
        "moved":        0,
        "errors":       [],
    }

def move_image_and_label(src_img, dst_split):
    """
    Move image + label from current split folder to dst_split folder.
    Works even if src and dst are the same split (no-op).
    """
    src_lbl = get_label_path(src_img)

    # Determine destination paths
    fname       = os.path.basename(src_img)
    lbl_fname   = os.path.basename(src_lbl)
    dst_img_dir = os.path.join(OUTPUT_PATH, dst_split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_PATH, dst_split, "labels")
    dst_img     = os.path.join(dst_img_dir, fname)
    dst_lbl     = os.path.join(dst_lbl_dir, lbl_fname)

    # No-op if already in correct location
    if src_img == dst_img:
        return True

    try:
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        shutil.move(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        return True
    except Exception as e:
        tqdm.write(f"   ERROR moving {fname}: {e}")
        return False

# ─────────────────────────────────────────────
# STEP 0 — Show plan
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Split Plan")
print("=" * 60)

checkpoint = load_checkpoint()

if checkpoint["done"]:
    print(f"\n   Part 5 already complete (from checkpoint)")
elif checkpoint["moved"] > 0:
    print(f"\n   Resuming -- {checkpoint['moved']} files already moved")
else:
    print(f"\n   Fresh start")

print(f"""
   Target ratios:
      train : {TRAIN_RATIO*100:.0f}%
      valid : {VALID_RATIO*100:.0f}%
      test  : {TEST_RATIO*100:.0f}%

   Strategy: STRATIFIED by dataset prefix
      Each dataset (v1, v5, v6, nv2, nv3, nv4) is split
      independently at 70/20/10 so every dataset is
      proportionally represented in all three splits.

   Random seed: {RANDOM_SEED} (fixed -- reproducible)
""")

# ─────────────────────────────────────────────
# STEP 1 — Collect all current images
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Collecting all images...")
print("=" * 60)

all_images = []
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    if not os.path.exists(img_dir):
        continue
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_images.extend(files)
    print(f"   {split:<6} -> {len(files):>6} images")

print(f"\n   Total images: {len(all_images)}")

# ─────────────────────────────────────────────
# STEP 2 — Group by prefix (stratified split)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Grouping images by dataset prefix...")
print("=" * 60)

prefix_groups = defaultdict(list)
for img_path in all_images:
    prefix = get_prefix(os.path.basename(img_path))
    prefix_groups[prefix].append(img_path)

print(f"\n   {'Prefix':<8} {'Count':>7}   {'Train':>7}   {'Valid':>7}   {'Test':>6}")
print(f"   {'-'*8} {'-'*7}   {'-'*7}   {'-'*7}   {'-'*6}")

total_train = total_valid = total_test = 0
SPLIT_ASSIGNMENTS = {"train": [], "valid": [], "test": []}

random.seed(RANDOM_SEED)

for prefix in DATASET_PREFIXES:
    images = prefix_groups.get(prefix, [])
    if not images:
        print(f"   {prefix:<8} {'0':>7}   (skipped -- 0 images)")
        continue

    # Shuffle deterministically
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)

    n       = len(images_shuffled)
    n_train = round(n * TRAIN_RATIO)
    n_test  = round(n * TEST_RATIO)
    n_valid = n - n_train - n_test   # remaining goes to valid

    train_imgs = images_shuffled[:n_train]
    valid_imgs = images_shuffled[n_train:n_train + n_valid]
    test_imgs  = images_shuffled[n_train + n_valid:]

    SPLIT_ASSIGNMENTS["train"].extend(train_imgs)
    SPLIT_ASSIGNMENTS["valid"].extend(valid_imgs)
    SPLIT_ASSIGNMENTS["test"].extend(test_imgs)

    total_train += len(train_imgs)
    total_valid += len(valid_imgs)
    total_test  += len(test_imgs)

    print(f"   {prefix:<8} {n:>7}   {len(train_imgs):>7}   "
          f"{len(valid_imgs):>7}   {len(test_imgs):>6}")

print(f"   {'-'*8} {'-'*7}   {'-'*7}   {'-'*7}   {'-'*6}")
print(f"   {'TOTAL':<8} {len(all_images):>7}   {total_train:>7}   "
      f"{total_valid:>7}   {total_test:>6}")
print(f"\n   Actual ratios:")
n = len(all_images)
print(f"      train : {total_train/n*100:.1f}%  ({total_train} images)")
print(f"      valid : {total_valid/n*100:.1f}%  ({total_valid} images)")
print(f"      test  : {total_test/n*100:.1f}%  ({total_test} images)")

# ─────────────────────────────────────────────
# STEP 3 — Backup current folder structure
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Backing up current split structure...")
print("=" * 60)
print("   INFO: Records which file was in which split before reshuffling")

if checkpoint["backup_done"] and os.path.exists(BACKUP_DIR):
    backup_count = sum(len(files) for _, _, files in os.walk(BACKUP_DIR))
    print(f"\n   Backup already exists ({backup_count} files)")
    print(f"   {BACKUP_DIR}")
elif checkpoint["done"]:
    print(f"\n   Skipping -- Part 5 already complete")
else:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    split_record = {}
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                split_record[f] = split

    record_path = os.path.join(BACKUP_DIR, "original_split_record.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(split_record, f, indent=2)

    checkpoint["backup_done"] = True
    save_checkpoint(checkpoint)
    print(f"\n   Original split recorded for {len(split_record)} files")
    print(f"   Saved to: {record_path}")
    print(f"\n   To rollback: python part5_rollback.py")

# ─────────────────────────────────────────────
# STEP 4 — Move files into new split structure
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Reshuffling files into new 70/20/10 split...")
print("=" * 60)
print(f"   INFO: Moving images + labels to correct split folders")
print(f"   Estimated time: ~3-5 minutes\n")

move_start  = time.time()
total_moved = checkpoint["moved"]
total_errors = list(checkpoint["errors"])

if checkpoint["done"]:
    print(f"   Already complete -- loaded from checkpoint")
else:
    # Build a set of already-moved filenames for resume
    already_moved = set()

    # Flatten all assignments with their target split
    all_assignments = []
    for dst_split, img_list in SPLIT_ASSIGNMENTS.items():
        for img_path in img_list:
            all_assignments.append((img_path, dst_split))

    overall_bar = tqdm(total=len(all_assignments), desc="   Moving files",
                       unit="file", ncols=70, colour="green",
                       initial=total_moved)

    for i, (img_path, dst_split) in enumerate(all_assignments):
        fname = os.path.basename(img_path)

        if fname in already_moved:
            overall_bar.update(1)
            continue

        # File may have already been moved in a previous run
        # Find current location
        current_path = None
        if os.path.exists(img_path):
            current_path = img_path
        else:
            # Search all splits for this file
            for split in SPLITS:
                candidate = os.path.join(OUTPUT_PATH, split, "images", fname)
                if os.path.exists(candidate):
                    current_path = candidate
                    break

        if current_path is None:
            tqdm.write(f"   WARNING: Could not find {fname} -- skipping")
            already_moved.add(fname)
            overall_bar.update(1)
            continue

        success = move_image_and_label(current_path, dst_split)
        if success:
            total_moved += 1
        else:
            total_errors.append(fname)

        already_moved.add(fname)
        overall_bar.update(1)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint["moved"]  = total_moved
            checkpoint["errors"] = total_errors
            save_checkpoint(checkpoint)
            tqdm.write(f"   Checkpoint saved at {i+1} files")

    overall_bar.close()

    checkpoint["moved"]  = total_moved
    checkpoint["errors"] = total_errors
    checkpoint["done"]   = True
    save_checkpoint(checkpoint)

move_time = time.time() - move_start
print(f"\n   Done in {move_time:.1f}s")
print(f"   Moved  : {total_moved}")
print(f"   Errors : {len(total_errors)}")

# ─────────────────────────────────────────────
# STEP 5 — Clean up any orphan label files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Cleaning up orphan label files...")
print("=" * 60)

total_orphans = 0
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    if not os.path.exists(lbl_dir):
        continue
    img_stems = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
    orphans   = lbl_stems - img_stems
    for orphan in orphans:
        os.remove(os.path.join(lbl_dir, orphan + ".txt"))
        total_orphans += 1
    status = "OK" if not orphans else f"{len(orphans)} removed"
    print(f"   {split:<6} -> {status}")

if total_orphans == 0:
    print(f"\n   No orphan labels found")
else:
    print(f"\n   {total_orphans} orphan labels cleaned up")

# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

cv_errors = 0

# -- Check A: Image and label counts match per split --
print("\n   [A] Image vs Label count per split:")
final_counts = {}
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    img_count = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    lbl_count = len([f for f in os.listdir(lbl_dir)
                     if f.lower().endswith(".txt")])
    final_counts[split] = img_count
    status = "OK" if img_count == lbl_count else "ERROR: MISMATCH"
    if img_count != lbl_count:
        cv_errors += 1
    print(f"      {split:<6} -> images: {img_count:>6}   labels: {lbl_count:>6}   {status}")

# -- Check B: Total image count preserved --
print("\n   [B] Total image count preserved:")
final_total = sum(final_counts.values())
if final_total == len(all_images):
    print(f"      OK: {final_total} images (same as before reshuffling)")
else:
    diff = len(all_images) - final_total
    print(f"      ERROR: {final_total} found, expected {len(all_images)} ({diff} missing!)")
    cv_errors += 1

# -- Check C: Actual split ratios --
print("\n   [C] Final split ratios:")
for split in SPLITS:
    count  = final_counts[split]
    ratio  = count / final_total * 100
    target = {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO}[split] * 100
    diff   = abs(ratio - target)
    status = "OK" if diff < 2.0 else "WARNING: >2% off target"
    print(f"      {split:<6} -> {count:>6}  ({ratio:.1f}%  target: {target:.0f}%)   {status}")

# -- Check D: Per-prefix distribution in each split --
print("\n   [D] Per-prefix representation in each split:")
print(f"   {'Prefix':<8} {'Train':>7}   {'Valid':>7}   {'Test':>6}   {'Ratio check':>14}")
for prefix in DATASET_PREFIXES:
    counts = {}
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        counts[split] = len([f for f in os.listdir(img_dir)
                              if f.startswith(f"{prefix}_")])
    total_p = sum(counts.values())
    if total_p == 0:
        print(f"   {prefix:<8} {'0':>7}   (no images -- expected for v7)")
        continue
    train_pct = counts["train"] / total_p * 100
    ratio_ok  = "OK" if abs(train_pct - 70) < 3 else "WARNING"
    print(f"   {prefix:<8} {counts['train']:>7}   {counts['valid']:>7}   "
          f"{counts['test']:>6}   train={train_pct:.1f}% {ratio_ok}")

# -- Check E: No empty split folders --
print("\n   [E] No empty split folders:")
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    img_ok  = os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0
    lbl_ok  = os.path.exists(lbl_dir) and len(os.listdir(lbl_dir)) > 0
    status  = "OK" if img_ok and lbl_ok else "ERROR: EMPTY!"
    if not (img_ok and lbl_ok):
        cv_errors += 1
    print(f"      {split:<6} -> {status}")

# ─────────────────────────────────────────────
# Save log
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving split log...")
print("=" * 60)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("PART 5 - Balanced Train/Valid/Test Split Log\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Random seed    : {RANDOM_SEED}\n")
    f.write(f"Strategy       : Stratified by dataset prefix\n")
    f.write(f"Target ratios  : {TRAIN_RATIO:.0%} / {VALID_RATIO:.0%} / {TEST_RATIO:.0%}\n\n")
    f.write("FINAL COUNTS:\n")
    for split in SPLITS:
        f.write(f"  {split:<6}: {final_counts[split]}\n")
    f.write(f"\n  Total: {final_total}\n\n")
    f.write("PER-PREFIX BREAKDOWN:\n")
    for prefix in DATASET_PREFIXES:
        counts = {}
        for split in SPLITS:
            img_dir = os.path.join(OUTPUT_PATH, split, "images")
            counts[split] = len([f for f in os.listdir(img_dir)
                                  if f.startswith(f"{prefix}_")])
        total_p = sum(counts.values())
        f.write(f"  {prefix}: total={total_p}  "
                f"train={counts['train']}  "
                f"valid={counts['valid']}  "
                f"test={counts['test']}\n")
    if total_errors:
        f.write("\nERRORS:\n")
        for e in total_errors:
            f.write(f"  {e}\n")

print(f"\n   Log saved: {LOG_FILE}")

# ─────────────────────────────────────────────
# Update master_data.yaml with correct paths
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Updating master_data.yaml with final split paths...")
print("=" * 60)

yaml_path    = os.path.join(OUTPUT_PATH, "master_data.yaml")
yaml_content = f"""# Master dataset — Medical Waste Detection (Segmentation)
# Generated by pipeline Part 5
# Total images : {final_total}
# Train        : {final_counts['train']}
# Valid        : {final_counts['valid']}
# Test         : {final_counts['test']}
# Classes      : 38

path  : {OUTPUT_PATH}
train : train/images
val   : valid/images
test  : test/images

nc    : 38

names:
  0:  bloody_objects
  1:  mask
  2:  n95
  3:  oxygen_cylinder
  4:  radioactive_objects
  5:  bandage
  6:  blade
  7:  capsule
  8:  cotton_swab
  9:  covid_buffer
  10: covid_buffer_box
  11: covid_test_case
  12: gauze
  13: glass_bottle
  14: harris_uni_core
  15: harris_uni_core_cap
  16: iodine_swab
  17: mercury_thermometer
  18: paperbox
  19: pill
  20: plastic_medical_bag
  21: plastic_medical_bottle
  22: medical_gloves
  23: reagent_tube
  24: reagent_tube_cap
  25: scalpel
  26: single_channel_pipette
  27: syringe
  28: transferpettor_glass
  29: transferpettor_plastic
  30: tweezer_metal
  31: tweezer_plastic
  32: unguent
  33: electronic_thermometer
  34: cap_plastic
  35: drug_packaging
  36: medical_infusion_bag
  37: needle
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n   master_data.yaml updated: {yaml_path}")

# ─────────────────────────────────────────────
# Write rollback script
# ─────────────────────────────────────────────
rollback_path = os.path.join(BASE_PATH, "Data_merge code", "part5_rollback.py")
rollback_content = f"""import os, json, shutil

BACKUP_DIR  = r"{BACKUP_DIR}"
OUTPUT_PATH = r"{OUTPUT_PATH}"
SPLITS      = ["train", "valid", "test"]

record_path = os.path.join(BACKUP_DIR, "original_split_record.json")
with open(record_path, "r", encoding="utf-8") as f:
    split_record = json.load(f)

print(f"Rolling back {{len(split_record)}} files to original split...")

for fname, orig_split in split_record.items():
    for split in SPLITS:
        src_img = os.path.join(OUTPUT_PATH, split, "images", fname)
        if os.path.exists(src_img):
            stem    = os.path.splitext(fname)[0]
            src_lbl = os.path.join(OUTPUT_PATH, split, "labels", stem + ".txt")
            dst_img = os.path.join(OUTPUT_PATH, orig_split, "images", fname)
            dst_lbl = os.path.join(OUTPUT_PATH, orig_split, "labels", stem + ".txt")
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            break

print("Rollback complete")
"""

os.makedirs(os.path.dirname(rollback_path), exist_ok=True)
with open(rollback_path, "w", encoding="utf-8") as f:
    f.write(rollback_content)
print(f"   Rollback script saved: {rollback_path}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
   Before Part 5 (imbalanced):
      train  : 14006  (96.9%)
      valid  :   410  ( 2.8%)
      test   :    29  ( 0.2%)

   After Part 5 (balanced):
      train  : {final_counts['train']:>6}  ({final_counts['train']/final_total*100:.1f}%)
      valid  : {final_counts['valid']:>6}  ({final_counts['valid']/final_total*100:.1f}%)
      test   : {final_counts['test']:>6}  ({final_counts['test']/final_total*100:.1f}%)
      Total  : {final_total:>6}

   Strategy       : Stratified by dataset prefix
   Random seed    : {RANDOM_SEED}
   Errors         : {len(total_errors)}
   Time taken     : {move_time:.1f}s

   Cross validation errors: {cv_errors}
""")

if cv_errors == 0 and len(total_errors) == 0:
    print("   ALL CHECKS PASSED -- Part 5 Complete!")
    print("   Pipeline COMPLETE -- dataset ready for YOLOv8 training!")
    print(f"\n   To start training:")
    print(f"      yolo task=segment mode=train")
    print(f"           model=yolov8n-seg.pt")
    print(f"           data={yaml_path}")
    print(f"           epochs=100 imgsz=640 batch=16")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n   Checkpoint file cleaned up")
else:
    print("   WARNING: Issues found -- check above before training")

print("=" * 60)
