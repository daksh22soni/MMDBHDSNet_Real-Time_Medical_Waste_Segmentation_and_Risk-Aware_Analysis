import os
import shutil
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH   = r"C:\Users\gahan\Desktop\dbhdsnet_project"
OUTPUT_PATH = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"

DATASET_PREFIXES = {
    "MedBin_dataset.v1i.yolov8": "v1",
    "MedBin_dataset.v5i.yolov8": "v5",
    "MedBin_dataset.v6i.yolov8": "v6",
    "MedBin_dataset.v7i.yolov8": "v7",
    "New.v2i.yolov8":            "nv2",
    "New.v3i.yolov8":            "nv3",
    "New.v4i.yolov8":            "nv4",
}

SPLITS = ["train", "valid", "test"]

# ─────────────────────────────────────────────
# STEP 1 — Create output folder structure
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Creating master_dataset folder structure...")
print("=" * 60)

for split in SPLITS:
    for sub in ["images", "labels"]:
        folder = os.path.join(OUTPUT_PATH, split, sub)
        os.makedirs(folder, exist_ok=True)
        print(f"   ✅ Created: {os.path.join(split, sub)}")

# ─────────────────────────────────────────────
# STEP 2 — Scan all datasets and count files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Scanning all datasets...")
print("=" * 60)

scan_stats     = {}
total_expected = 0

for dataset_name, prefix in DATASET_PREFIXES.items():
    dataset_path = os.path.join(BASE_PATH, dataset_name)
    scan_stats[dataset_name] = {}
    print(f"\n   📁 {dataset_name}  (prefix: {prefix})")

    for split in SPLITS:
        img_path = os.path.join(dataset_path, split, "images")
        lbl_path = os.path.join(dataset_path, split, "labels")

        if not os.path.exists(img_path):
            print(f"      ⚠️  '{split}' folder not found — skipping")
            scan_stats[dataset_name][split] = {"images": 0, "labels": 0}
            continue

        images = [f for f in os.listdir(img_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = [f for f in os.listdir(lbl_path) if f.lower().endswith(".txt")] if os.path.exists(lbl_path) else []

        scan_stats[dataset_name][split] = {"images": len(images), "labels": len(labels)}
        total_expected += len(images)
        print(f"      {split:<6} → {len(images):>5} images   {len(labels):>5} labels")

print(f"\n   📊 Total images to copy: {total_expected}")

# ─────────────────────────────────────────────
# STEP 3 — Copy and rename files with tqdm
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Copying and renaming files...")
print("=" * 60)

total_copied   = {"train": 0, "valid": 0, "test": 0}
skipped_labels = []
copy_errors    = []

# Overall progress bar across ALL datasets and splits
overall_bar = tqdm(total=total_expected, desc="   Overall Progress",
                   unit="file", ncols=70, colour="green")

for dataset_name, prefix in DATASET_PREFIXES.items():
    dataset_path = os.path.join(BASE_PATH, dataset_name)

    for split in SPLITS:
        src_img_dir = os.path.join(dataset_path, split, "images")
        src_lbl_dir = os.path.join(dataset_path, split, "labels")
        dst_img_dir = os.path.join(OUTPUT_PATH, split, "images")
        dst_lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")

        if not os.path.exists(src_img_dir):
            continue

        images = [f for f in os.listdir(src_img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Per dataset+split progress bar
        split_bar = tqdm(images, desc=f"   {prefix}/{split}",
                         unit="file", ncols=70, leave=False, colour="cyan")

        for img_file in split_bar:
            new_img_name = f"{prefix}_{img_file}"
            base_name    = os.path.splitext(img_file)[0]
            lbl_file     = base_name + ".txt"
            new_lbl_name = f"{prefix}_{lbl_file}"

            src_img = os.path.join(src_img_dir, img_file)
            dst_img = os.path.join(dst_img_dir, new_img_name)
            src_lbl = os.path.join(src_lbl_dir, lbl_file)
            dst_lbl = os.path.join(dst_lbl_dir, new_lbl_name)

            try:
                shutil.copy2(src_img, dst_img)
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    skipped_labels.append(f"{dataset_name}/{split}/{img_file}")

                total_copied[split] += 1
                overall_bar.update(1)

            except Exception as e:
                copy_errors.append(f"{dataset_name}/{split}/{img_file} → {str(e)}")
                overall_bar.update(1)

        split_bar.close()

overall_bar.close()
print(f"\n   ✅ Copying complete!")

# ─────────────────────────────────────────────
# STEP 4 — Cross Validation (5 checks)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Cross Validation")
print("=" * 60)

cv_errors = 0

# ── Check 4a: Output image count matches input count ──
print("\n   [4a] Output vs Input image count per split:")
for split in SPLITS:
    expected    = sum(scan_stats[d][split]["images"] for d in scan_stats if split in scan_stats[d])
    dst_img_dir = os.path.join(OUTPUT_PATH, split, "images")
    actual      = len([f for f in os.listdir(dst_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    status      = "✅" if actual == expected else "❌"
    if actual != expected:
        cv_errors += 1
    print(f"      {split:<6} → expected: {expected:>6}   actual: {actual:>6}   {status}")

# ── Check 4b: Image count == Label count per split ──
print("\n   [4b] Image vs Label count match per split:")
for split in SPLITS:
    dst_img_dir = os.path.join(OUTPUT_PATH, split, "images")
    dst_lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    img_count   = len([f for f in os.listdir(dst_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    lbl_count   = len([f for f in os.listdir(dst_lbl_dir) if f.lower().endswith(".txt")])
    diff        = img_count - lbl_count
    status      = "✅" if diff == 0 else f"⚠️  {diff} images have no label"
    if diff != 0:
        cv_errors += 1
    print(f"      {split:<6} → images: {img_count:>6}   labels: {lbl_count:>6}   {status}")

# ── Check 4c: No filename conflicts ──
print("\n   [4c] Filename conflict check:")
for split in SPLITS:
    dst_img_dir = os.path.join(OUTPUT_PATH, split, "images")
    all_files   = os.listdir(dst_img_dir)
    unique      = set(all_files)
    if len(all_files) != len(unique):
        dupes = len(all_files) - len(unique)
        print(f"      ❌ {split} — {dupes} duplicate filenames found!")
        cv_errors += 1
    else:
        print(f"      ✅ {split} — no filename conflicts")

# ── Check 4d: Every file starts with a known prefix ──
print("\n   [4d] Prefix integrity check:")
known_prefixes = tuple(f"{p}_" for p in DATASET_PREFIXES.values())
for split in SPLITS:
    dst_img_dir = os.path.join(OUTPUT_PATH, split, "images")
    bad_files   = [f for f in os.listdir(dst_img_dir) if not f.startswith(known_prefixes)]
    if bad_files:
        print(f"      ❌ {split} — {len(bad_files)} files missing valid prefix!")
        cv_errors += 1
    else:
        print(f"      ✅ {split} — all files have correct prefix")

# ── Check 4e: Per-dataset file count in output matches scan ──
print("\n   [4e] Per-dataset file count spot check:")
for dataset_name, prefix in DATASET_PREFIXES.items():
    for split in SPLITS:
        dst_img_dir  = os.path.join(OUTPUT_PATH, split, "images")
        prefix_files = [f for f in os.listdir(dst_img_dir) if f.startswith(f"{prefix}_")]
        expected     = scan_stats.get(dataset_name, {}).get(split, {}).get("images", 0)
        status       = "✅" if len(prefix_files) == expected else "❌"
        if len(prefix_files) != expected:
            cv_errors += 1
        print(f"      {prefix}/{split:<6} → expected: {expected:>5}   found: {len(prefix_files):>5}   {status}")

# ─────────────────────────────────────────────
# STEP 5 — Issues report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Issues Report")
print("=" * 60)

if skipped_labels:
    print(f"\n   ⚠️  {len(skipped_labels)} images had no matching label file:")
    for item in skipped_labels[:10]:
        print(f"      - {item}")
    if len(skipped_labels) > 10:
        print(f"      ... and {len(skipped_labels) - 10} more")
else:
    print("\n   ✅ Every image has a matching label file")

if copy_errors:
    print(f"\n   ❌ {len(copy_errors)} copy errors occurred:")
    for e in copy_errors[:10]:
        print(f"      - {e}")
else:
    print("   ✅ Zero copy errors")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

grand_total = sum(total_copied.values())

print(f"""
   Datasets processed      : {len(DATASET_PREFIXES)}
   Files copied:
      train                : {total_copied['train']:>6} images
      valid                : {total_copied['valid']:>6} images
      test                 : {total_copied['test']:>6} images
      ──────────────────────────────
      GRAND TOTAL          : {grand_total:>6} images

   Images missing labels   : {len(skipped_labels)}
   Copy errors             : {len(copy_errors)}
   Cross validation errors : {cv_errors}
""")

if cv_errors == 0 and len(copy_errors) == 0:
    print("   🎉 ALL CHECKS PASSED — Part 2 Complete!")
    print("   ➡️  Ready for Part 3 (Duplicate Removal)")
else:
    print("   ⚠️  Some issues found — review above before continuing")

print("=" * 60)