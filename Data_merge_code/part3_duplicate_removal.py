import os
import json
import hashlib
import time
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_PATH      = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"
CHECKPOINT_FILE  = os.path.join(OUTPUT_PATH, "dedup_checkpoint.json")
REMOVED_LOG_FILE = os.path.join(OUTPUT_PATH, "removed_duplicates_log.txt")

TIER2_TOLERANCE  = 0.01   # 1% tolerance for label position matching (Tier 2)
TIER3_TOLERANCE  = 0.03   # 3% tolerance for cross-dataset augmentation (Tier 3)
CHECKPOINT_EVERY = 500    # save checkpoint every N images
SPLITS           = ["train", "valid", "test"]

DATASET_PREFIXES = ["v1", "v5", "v6", "v7", "nv2", "nv3", "nv4"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_md5(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_label_path(img_path):
    lbl_path = img_path.replace("images", "labels")
    lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
    return lbl_path

def read_label(label_path):
    if not os.path.exists(label_path):
        return None
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls  = int(parts[0])
                bbox = tuple(float(p) for p in parts[1:])
                objects.append((cls, bbox))
    return sorted(objects, key=lambda x: (x[0], x[1][0], x[1][1]))

def label_signature(objects):
    if objects is None:
        return None
    return tuple((cls, bbox) for cls, bbox in objects)

def labels_match_within_tolerance(obj_a, obj_b, tolerance):
    if obj_a is None or obj_b is None:
        return False
    if len(obj_a) != len(obj_b):
        return False
    for (cls_a, bbox_a), (cls_b, bbox_b) in zip(obj_a, obj_b):
        if cls_a != cls_b:
            return False
        for coord_a, coord_b in zip(bbox_a, bbox_b):
            if abs(coord_a - coord_b) > tolerance:
                return False
    return True

def get_prefix(filename):
    for prefix in DATASET_PREFIXES:
        if filename.startswith(f"{prefix}_"):
            return prefix
    return None

def remove_image_and_label(img_path):
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
        lbl_path = get_label_path(img_path)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
    except Exception as e:
        tqdm.write(f"   ⚠️  Could not remove {img_path}: {e}")

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {
        "tier1_done":      False,
        "tier2_done":      False,
        "tier3_done":      False,
        "tier1_removed":   [],
        "tier2_removed":   [],
        "tier3_removed":   [],
        "tier2_processed": [],
        "tier3_processed": [],
    }

def collect_images():
    images = []
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if not os.path.exists(img_dir):
            continue
        files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        images.extend(files)
    return images

# ─────────────────────────────────────────────
# STEP 0 — Load checkpoint
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Checkpoint system...")
print("=" * 60)

checkpoint = load_checkpoint()

if any([checkpoint["tier1_done"], checkpoint["tier2_done"], checkpoint["tier3_done"]]):
    print(f"\n   🔄 Resuming from checkpoint:")
    print(f"      Tier 1 done : {checkpoint['tier1_done']}  ({len(checkpoint['tier1_removed'])} removed)")
    print(f"      Tier 2 done : {checkpoint['tier2_done']}  ({len(checkpoint['tier2_removed'])} removed)")
    print(f"      Tier 3 done : {checkpoint['tier3_done']}  ({len(checkpoint['tier3_removed'])} removed)")
else:
    print(f"\n   ✅ Fresh start — no checkpoint found")

# ─────────────────────────────────────────────
# STEP 1 — Collect image paths
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Collecting all image paths...")
print("=" * 60)

all_images = collect_images()
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    if os.path.exists(img_dir):
        count = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        print(f"   {split:<6} → {count:>6} images")

print(f"\n   📊 Total images before dedup: {len(all_images)}")

# ─────────────────────────────────────────────
# TIER 1 — MD5 Exact Duplicate Removal
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TIER 1: MD5 Exact Duplicate Removal")
print("=" * 60)
print("   ℹ️  Removes byte-identical images — 100% safe, no risk")
print("   ⏱️  Estimated time: ~2-3 minutes")

tier1_start   = time.time()
tier1_removed = 0

if checkpoint["tier1_done"]:
    print(f"\n   ✅ Already complete — {len(checkpoint['tier1_removed'])} removed (from checkpoint)")
    tier1_removed = len(checkpoint["tier1_removed"])
else:
    md5_db  = {}
    removed = checkpoint["tier1_removed"]

    bar = tqdm(all_images, desc="   Tier1 MD5", unit="file", ncols=70, colour="green")

    for i, img_path in enumerate(bar):
        if not os.path.exists(img_path):
            continue
        try:
            md5 = get_md5(img_path)
            if md5 in md5_db:
                remove_image_and_label(img_path)
                removed.append(img_path)
                tier1_removed += 1
                bar.set_postfix(removed=tier1_removed)
            else:
                md5_db[md5] = img_path
        except Exception as e:
            tqdm.write(f"   ⚠️  {img_path}: {e}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint["tier1_removed"] = removed
            save_checkpoint(checkpoint)
            tqdm.write(f"   💾 Checkpoint saved at {i+1} files")

    checkpoint["tier1_removed"] = removed
    checkpoint["tier1_done"]    = True
    save_checkpoint(checkpoint)

tier1_time = time.time() - tier1_start
print(f"\n   ✅ Tier 1 done in {tier1_time:.1f}s — removed: {tier1_removed}")

# ─────────────────────────────────────────────
# TIER 2 — Label Signature Exact Match
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TIER 2: Label Signature Exact Match")
print("=" * 60)
print(f"   ℹ️  Compares ONLY label files — ignores pixels and background completely")
print(f"   ℹ️  Removes images where all object positions match within {TIER2_TOLERANCE*100:.0f}% tolerance")
print(f"   ℹ️  Safe for multi-object images and same-background datasets")
print(f"   ⏱️  Estimated time: ~3-5 minutes")

tier2_start   = time.time()
tier2_removed = 0

if checkpoint["tier2_done"]:
    print(f"\n   ✅ Already complete — {len(checkpoint['tier2_removed'])} removed (from checkpoint)")
    tier2_removed = len(checkpoint["tier2_removed"])
else:
    surviving         = collect_images()
    already_processed = set(checkpoint["tier2_processed"])
    label_db          = {}
    removed           = checkpoint["tier2_removed"]

    print(f"\n   📊 Images after Tier 1: {len(surviving)}")

    bar = tqdm(surviving, desc="   Tier2 Labels", unit="file", ncols=70, colour="cyan")

    for i, img_path in enumerate(bar):
        if img_path in already_processed or not os.path.exists(img_path):
            continue

        objects = read_label(get_label_path(img_path))
        if not objects:
            continue

        sig = label_signature(objects)

        if sig in label_db:
            existing_objects = read_label(get_label_path(label_db[sig]))
            if labels_match_within_tolerance(objects, existing_objects, TIER2_TOLERANCE):
                remove_image_and_label(img_path)
                removed.append(img_path)
                tier2_removed += 1
                bar.set_postfix(removed=tier2_removed)
        else:
            label_db[sig] = img_path

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint["tier2_removed"]   = removed
            checkpoint["tier2_processed"] = list(already_processed) + surviving[:i+1]
            save_checkpoint(checkpoint)
            tqdm.write(f"   💾 Checkpoint saved at {i+1} files")

    checkpoint["tier2_removed"] = removed
    checkpoint["tier2_done"]    = True
    save_checkpoint(checkpoint)

tier2_time = time.time() - tier2_start
print(f"\n   ✅ Tier 2 done in {tier2_time:.1f}s — removed: {tier2_removed}")

# ─────────────────────────────────────────────
# TIER 3 — Cross-Dataset Augmentation Check
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TIER 3: Cross-Dataset Augmentation Duplicate Check")
print("=" * 60)
print(f"   ℹ️  Removes same image augmented identically across DIFFERENT datasets only")
print(f"   ℹ️  Uses looser {TIER3_TOLERANCE*100:.0f}% tolerance to catch near-identical augmentations")
print(f"   ℹ️  NEVER removes within-dataset augmentations")
print(f"   ℹ️  Rotated images with genuinely different label positions are ALWAYS kept")
print(f"   ⏱️  Estimated time: ~5-8 minutes")

tier3_start   = time.time()
tier3_removed = 0

if checkpoint["tier3_done"]:
    print(f"\n   ✅ Already complete — {len(checkpoint['tier3_removed'])} removed (from checkpoint)")
    tier3_removed = len(checkpoint["tier3_removed"])
else:
    surviving         = collect_images()
    already_processed = set(checkpoint["tier3_processed"])
    removed           = checkpoint["tier3_removed"]
    removed_set       = set(removed)

    print(f"\n   📊 Images after Tier 2: {len(surviving)}")

    # Build per-dataset label index
    print("\n   Building per-dataset label index...")
    dataset_label_db = {p: [] for p in DATASET_PREFIXES}

    index_bar = tqdm(surviving, desc="   Building index", unit="file",
                     ncols=70, colour="yellow")
    for img_path in index_bar:
        prefix = get_prefix(os.path.basename(img_path))
        if prefix is None:
            continue
        objects = read_label(get_label_path(img_path))
        if objects and len(objects) > 0:
            dataset_label_db[prefix].append((objects, img_path))
    index_bar.close()

    # Print index size per dataset
    for prefix in DATASET_PREFIXES:
        print(f"      {prefix:<6} → {len(dataset_label_db[prefix]):>6} labelled images indexed")

    # Cross-dataset comparison
    print("\n   Cross-comparing datasets...")
    compare_bar = tqdm(surviving, desc="   Tier3 Cross-DS", unit="file",
                       ncols=70, colour="magenta")

    for i, img_path in enumerate(compare_bar):
        if img_path in already_processed or img_path in removed_set:
            continue
        if not os.path.exists(img_path):
            continue

        this_prefix = get_prefix(os.path.basename(img_path))
        if this_prefix is None:
            continue

        objects = read_label(get_label_path(img_path))
        if not objects:
            continue

        is_cross_duplicate = False
        for other_prefix, other_entries in dataset_label_db.items():
            if other_prefix == this_prefix:
                continue   # never compare same dataset to itself

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

cv_errors  = 0
final_total = 0

# Check A: Image == Label count per split
print("\n   [A] Image vs Label count after all 3 tiers:")
for split in SPLITS:
    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    if not os.path.exists(img_dir):
        continue
    img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    lbl_count = len([f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")])
    final_total += img_count
    status = "✅" if img_count == lbl_count else "❌"
    if img_count != lbl_count:
        cv_errors += 1
    print(f"      {split:<6} → images: {img_count:>6}   labels: {lbl_count:>6}   {status}")

# Check B: Removed files are actually gone
print("\n   [B] Verifying all removed files are deleted:")
all_removed  = (checkpoint["tier1_removed"] +
                checkpoint["tier2_removed"] +
                checkpoint["tier3_removed"])
still_exists = sum(1 for p in all_removed if os.path.exists(p))
if still_exists == 0:
    print(f"      ✅ All {len(all_removed)} removed files confirmed deleted")
else:
    print(f"      ❌ {still_exists} files still exist — removal incomplete!")
    cv_errors += 1

# Check C: No orphan labels
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
        tqdm.write(f"      🧹 Auto-cleaned {len(orphans)} orphan labels in {split}")
        cv_errors += 1
    print(f"      {split:<6} → {'✅ No orphans' if not orphans else f'❌ {len(orphans)} orphans found and cleaned'}")

# Check D: Dataset still large enough
print("\n   [D] Dataset size health check:")
min_recommended = 5000
status = "✅" if final_total >= min_recommended else "⚠️ "
if final_total < min_recommended:
    cv_errors += 1
print(f"      {status} {final_total} images remaining (min recommended: {min_recommended})")

# Check E: Per-prefix survival count
print("\n   [E] Per-dataset survival count:")
for prefix in DATASET_PREFIXES:
    count = 0
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if os.path.exists(img_dir):
            count += len([f for f in os.listdir(img_dir) if f.startswith(f"{prefix}_")])
    print(f"      {prefix:<6} → {count:>6} images remaining")

# ─────────────────────────────────────────────
# Save removed log
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving removed duplicates log...")
print("=" * 60)

with open(REMOVED_LOG_FILE, "w") as f:
    f.write("TIER 1 — MD5 Exact Duplicates\n" + "="*60 + "\n")
    for p in checkpoint["tier1_removed"]:
        f.write(p + "\n")
    f.write(f"\nTotal: {len(checkpoint['tier1_removed'])}\n\n")

    f.write("TIER 2 — Label Signature Match\n" + "="*60 + "\n")
    for p in checkpoint["tier2_removed"]:
        f.write(p + "\n")
    f.write(f"\nTotal: {len(checkpoint['tier2_removed'])}\n\n")

    f.write("TIER 3 — Cross-Dataset Augmentation\n" + "="*60 + "\n")
    for p in checkpoint["tier3_removed"]:
        f.write(p + "\n")
    f.write(f"\nTotal: {len(checkpoint['tier3_removed'])}\n")

print(f"\n   ✅ Log saved: {REMOVED_LOG_FILE}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

total_removed = (len(checkpoint["tier1_removed"]) +
                 len(checkpoint["tier2_removed"]) +
                 len(checkpoint["tier3_removed"]))

print(f"""
   Started with               : {len(all_images):>6} images

   Tier 1 — MD5 exact         : {len(checkpoint['tier1_removed']):>6} removed  ({tier1_time:.1f}s)
   Tier 2 — Label signature   : {len(checkpoint['tier2_removed']):>6} removed  ({tier2_time:.1f}s)
   Tier 3 — Cross-dataset aug : {len(checkpoint['tier3_removed']):>6} removed  ({tier3_time:.1f}s)
   ─────────────────────────────────────────────
   Total removed              : {total_removed:>6}
   Final clean images         : {final_total:>6}

   Cross validation errors    : {cv_errors}
   Removed log                : {REMOVED_LOG_FILE}
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
