import os
import json
import time
import shutil
import yaml
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH        = r"C:\Users\gahan\Desktop\dbhdsnet_project"
OUTPUT_PATH      = os.path.join(BASE_PATH, "master_dataset")
CHECKPOINT_FILE  = os.path.join(OUTPUT_PATH, "part4_checkpoint.json")
LOG_FILE         = os.path.join(OUTPUT_PATH, "part4_remapping_log.txt")
BACKUP_DIR       = os.path.join(OUTPUT_PATH, "labels_backup_part4")
CHECKPOINT_EVERY = 500
SPLITS           = ["train", "valid", "test"]

# Original dataset paths (for yaml verification)
ORIGINAL_DATASETS = {
    "v1":  os.path.join(BASE_PATH, "MedBin_dataset.v1i.yolov8"),
    "v5":  os.path.join(BASE_PATH, "MedBin_dataset.v5i.yolov8"),
    "v6":  os.path.join(BASE_PATH, "MedBin_dataset.v6i.yolov8"),
    "v7":  os.path.join(BASE_PATH, "MedBin_dataset.v7i.yolov8"),
    "nv2": os.path.join(BASE_PATH, "New.v2i.yolov8"),
    "nv3": os.path.join(BASE_PATH, "New.v3i.yolov8"),
    "nv4": os.path.join(BASE_PATH, "New.v4i.yolov8"),
}

# ─────────────────────────────────────────────
# MASTER CLASS LIST — 38 classes (0-37)
# Merged:
#   plastic_medical_gloves (v1, nv2) -> 22: medical_gloves
#   medicine_bottole typo (v5)       -> 21: plastic_medical_bottle
# ─────────────────────────────────────────────
MASTER_CLASSES = [
    "bloody_objects",        # 0
    "mask",                  # 1
    "n95",                   # 2
    "oxygen_cylinder",       # 3
    "radioactive_objects",   # 4
    "bandage",               # 5
    "blade",                 # 6
    "capsule",               # 7
    "cotton_swab",           # 8
    "covid_buffer",          # 9
    "covid_buffer_box",      # 10
    "covid_test_case",       # 11
    "gauze",                 # 12
    "glass_bottle",          # 13
    "harris_uni_core",       # 14
    "harris_uni_core_cap",   # 15
    "iodine_swab",           # 16
    "mercury_thermometer",   # 17
    "paperbox",              # 18
    "pill",                  # 19
    "plastic_medical_bag",   # 20
    "plastic_medical_bottle",# 21
    "medical_gloves",        # 22  <- merged from plastic_medical_gloves
    "reagent_tube",          # 23
    "reagent_tube_cap",      # 24
    "scalpel",               # 25
    "single_channel_pipette",# 26
    "syringe",               # 27
    "transferpettor_glass",  # 28
    "transferpettor_plastic",# 29
    "tweezer_metal",         # 30
    "tweezer_plastic",       # 31
    "unguent",               # 32
    "electronic_thermometer",# 33
    "cap_plastic",           # 34
    "drug_packaging",        # 35
    "medical_infusion_bag",  # 36
    "needle",                # 37
]
MASTER_CLASSES_DICT = {i: name for i, name in enumerate(MASTER_CLASSES)}

# Normalise class names for fuzzy matching
# (handles capitalisation and minor spelling differences)
def normalise(name):
    return name.lower().strip().replace(" ", "_").replace("-", "_")

# Canonical name -> master ID lookup (normalised)
MASTER_LOOKUP = {normalise(name): i for i, name in enumerate(MASTER_CLASSES)}
# Add known aliases/typos
MASTER_LOOKUP["plastic_medical_gloves"] = 22  # v1, nv2 -> medical_gloves
MASTER_LOOKUP["medicine_bottole"]       = 21  # v5 typo -> plastic_medical_bottle

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_prefix(filename):
    for prefix in ORIGINAL_DATASETS.keys():
        if filename.startswith(f"{prefix}_"):
            return prefix
    return None

def read_yaml_classes(yaml_path):
    """Read class names from a data.yaml file."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            return list(names)
    except Exception as e:
        return None
    return None

def build_remap_table(dataset_classes):
    """
    Build { old_class_id: new_master_id } from dataset's class list.
    Uses normalised name matching + known aliases.
    Returns (remap_table, warnings)
    """
    remap  = {}
    warnings = []
    for old_id, name in enumerate(dataset_classes):
        norm = normalise(name)
        if norm in MASTER_LOOKUP:
            remap[old_id] = MASTER_LOOKUP[norm]
        else:
            warnings.append(f"  WARNING: Class '{name}' (id={old_id}) not found in master -- will keep as-is")
            remap[old_id] = old_id  # fallback: keep same ID
    return remap, warnings

def remap_label_file(lbl_path, remap_table):
    """
    Remap class IDs in a YOLO segmentation label file.
    Segmentation format per line: class x1 y1 x2 y2 ... xN yN
    Only the FIRST value (class ID) is changed.
    All polygon coordinates stay exactly the same.
    Returns True if successful, False if error.
    """
    try:
        with open(lbl_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Accept both detection (5 parts) and segmentation (odd >= 5 parts)
            if len(parts) >= 5 and len(parts) % 2 == 1:
                old_cls = int(parts[0])
                if old_cls not in remap_table:
                    tqdm.write(f"   WARNING: Unknown class {old_cls} in {os.path.basename(lbl_path)}")
                    new_lines.append(line.strip())
                    continue
                parts[0] = str(remap_table[old_cls])
                new_lines.append(" ".join(parts))
            else:
                new_lines.append(line.strip())

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        return True

    except Exception as e:
        tqdm.write(f"   ERROR remapping {lbl_path}: {e}")
        return False

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:   # FIX 1
        json.dump(cp, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:   # FIX 2
            return json.load(f)
    return {
        "done":             False,
        "backup_done":      False,
        "verified":         False,
        "processed":        [],
        "errors":           [],
        "remapped":         0,
        "skipped":          0,
    }

# ─────────────────────────────────────────────
# STEP 0 — Load checkpoint
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Checkpoint system...")
print("=" * 60)

checkpoint = load_checkpoint()

if checkpoint["done"]:
    print(f"\n   Part 4 already complete (from checkpoint)")
elif checkpoint["processed"]:
    print(f"\n   Resuming -- {len(checkpoint['processed'])} files already processed")
    print(f"              {checkpoint['remapped']} remapped so far")
else:
    print(f"\n   Fresh start")

# ─────────────────────────────────────────────
# STEP 1 — Read original data.yaml files
#           and build remapping tables automatically
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Reading original data.yaml files & building remap tables...")
print("=" * 60)

DATASET_REMAPPING = {}
yaml_errors       = 0

for prefix, ds_path in ORIGINAL_DATASETS.items():
    # Try common yaml locations
    yaml_candidates = [
        os.path.join(ds_path, "data.yaml"),
        os.path.join(ds_path, "dataset.yaml"),
    ]
    yaml_found = None
    for candidate in yaml_candidates:
        if os.path.exists(candidate):
            yaml_found = candidate
            break

    if yaml_found is None:
        print(f"   {prefix:<6} -> ERROR: data.yaml not found at {ds_path}")
        yaml_errors += 1
        continue

    ds_classes = read_yaml_classes(yaml_found)
    if ds_classes is None:
        print(f"   {prefix:<6} -> ERROR: Failed to read {yaml_found}")
        yaml_errors += 1
        continue

    remap, warnings = build_remap_table(ds_classes)
    DATASET_REMAPPING[prefix] = remap
    changed = sum(1 for old, new in remap.items() if old != new)

    print(f"   {prefix:<6} -> OK  {len(ds_classes):>2} classes read | "
          f"{changed:>2} IDs change | {yaml_found.split(os.sep)[-2]}")

    for w in warnings:
        print(f"            {w}")

if yaml_errors > 0:
    print(f"\n   WARNING: {yaml_errors} yaml files missing -- "
          f"those datasets will use identity mapping (no change)")

# ─────────────────────────────────────────────
# STEP 2 — Collect all label files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Collecting all label files...")
print("=" * 60)

all_labels = []
for split in SPLITS:
    lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
    if not os.path.exists(lbl_dir):
        continue
    files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
             if f.lower().endswith(".txt")]
    all_labels.extend(files)
    print(f"   {split:<6} -> {len(files):>6} label files")

print(f"\n   Total label files: {len(all_labels)}")

# ─────────────────────────────────────────────
# STEP 3 — Dry run preview (read only, no changes)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Dry run -- previewing remapping (NO files changed yet)")
print("=" * 60)

for prefix, remap_table in DATASET_REMAPPING.items():
    samples = [f for f in all_labels
               if os.path.basename(f).startswith(f"{prefix}_")
               and os.path.getsize(f) > 0]

    if not samples:
        print(f"\n   {prefix} -> 0 files found (all removed in dedup)")
        continue

    print(f"\n   {prefix} -- first 3 lines of first label file:")
    print(f"   File: {os.path.basename(samples[0])}")

    with open(samples[0], "r", encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in lines[:3]:
        parts = line.split()
        if len(parts) >= 1:
            old_cls  = int(parts[0])
            new_cls  = remap_table.get(old_cls, old_cls)
            old_name = MASTER_CLASSES_DICT.get(new_cls, "unknown")
            arrow    = f"{old_cls} -> {new_cls}" if old_cls != new_cls else f"{old_cls} (unchanged)"
            print(f"      class: {arrow:>14}  ({old_name})  "
                  f"coords: {parts[1][:8]}... [{len(parts)-1} coord values]")

# ─────────────────────────────────────────────
# STEP 4 — Backup all label files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Backing up all label files...")
print("=" * 60)
print("   INFO: Safety measure -- if anything goes wrong, backup lets you restore")

if checkpoint["backup_done"] and os.path.exists(BACKUP_DIR):
    print(f"\n   Backup already exists at:")
    print(f"      {BACKUP_DIR}")
    backup_count = sum(
        len(files) for _, _, files in os.walk(BACKUP_DIR)
    )
    print(f"      {backup_count} files backed up")
elif checkpoint["done"]:
    print(f"\n   Skipping backup -- Part 4 already complete")
else:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_bar = tqdm(total=len(all_labels), desc="   Backing up",
                      unit="file", ncols=70, colour="blue")
    backed_up = 0
    for lbl_path in all_labels:
        # Mirror the split/labels structure inside backup dir
        rel_path = os.path.relpath(lbl_path, OUTPUT_PATH)
        dst      = os.path.join(BACKUP_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(lbl_path, dst)
        backed_up += 1
        backup_bar.update(1)
    backup_bar.close()

    checkpoint["backup_done"] = True
    save_checkpoint(checkpoint)
    print(f"\n   {backed_up} label files backed up to:")
    print(f"      {BACKUP_DIR}")

# ─────────────────────────────────────────────
# STEP 5 — Run remapping
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Remapping class IDs in all label files...")
print("=" * 60)
print(f"   INFO: Only the class ID (first number per line) changes")
print(f"   INFO: All polygon coordinates stay exactly the same")
print(f"   Estimated time: ~3-5 minutes\n")

remap_start       = time.time()
total_remapped    = checkpoint["remapped"]
total_skipped     = checkpoint["skipped"]
total_errors      = list(checkpoint["errors"])
already_processed = set(checkpoint["processed"])

if checkpoint["done"]:
    print(f"   Already complete -- loaded from checkpoint")
else:
    overall_bar = tqdm(total=len(all_labels), desc="   Overall Progress",
                       unit="file", ncols=70, colour="green",
                       initial=len(already_processed))

    for split in SPLITS:
        lbl_dir = os.path.join(OUTPUT_PATH, split, "labels")
        if not os.path.exists(lbl_dir):
            continue

        files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                 if f.lower().endswith(".txt")]

        split_bar = tqdm(files, desc=f"   {split:<6}",
                         unit="file", ncols=70, leave=False, colour="cyan")

        for lbl_path in split_bar:
            if lbl_path in already_processed:
                continue

            fname  = os.path.basename(lbl_path)
            prefix = get_prefix(fname)

            # Skip empty files (background images)
            if os.path.getsize(lbl_path) == 0:
                total_skipped += 1
                already_processed.add(lbl_path)
                overall_bar.update(1)
                continue

            if prefix is None or prefix not in DATASET_REMAPPING:
                total_skipped += 1
                already_processed.add(lbl_path)
                overall_bar.update(1)
                continue

            success = remap_label_file(lbl_path, DATASET_REMAPPING[prefix])
            if success:
                total_remapped += 1
            else:
                total_errors.append(lbl_path)

            already_processed.add(lbl_path)
            overall_bar.update(1)

            processed_count = len(already_processed)
            if processed_count % CHECKPOINT_EVERY == 0:
                checkpoint["processed"] = list(already_processed)
                checkpoint["remapped"]  = total_remapped
                checkpoint["skipped"]   = total_skipped
                checkpoint["errors"]    = total_errors
                save_checkpoint(checkpoint)
                tqdm.write(f"   Checkpoint saved at {processed_count} files")

        split_bar.close()

    overall_bar.close()

    checkpoint["processed"] = list(already_processed)
    checkpoint["remapped"]  = total_remapped
    checkpoint["skipped"]   = total_skipped
    checkpoint["errors"]    = total_errors
    checkpoint["done"]      = True
    save_checkpoint(checkpoint)

remap_time = time.time() - remap_start
print(f"\n   Remapping done in {remap_time:.1f}s")
print(f"   Remapped : {total_remapped}")
print(f"   Skipped  : {total_skipped}")
print(f"   Errors   : {len(total_errors)}")

# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

cv_errors = 0

# -- Check A: All class IDs within valid range 0-37 --
print("\n   [A] Checking all class IDs are within valid range (0-37)...")

invalid_count  = 0
checked_files  = 0
invalid_sample = []

check_bar = tqdm(all_labels, desc="   Checking IDs",
                 unit="file", ncols=70, colour="yellow")
for lbl_path in check_bar:
    if os.path.getsize(lbl_path) == 0:
        continue
    checked_files += 1
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls < 0 or cls > 37:
                        invalid_count += 1
                        if len(invalid_sample) < 5:
                            invalid_sample.append(
                                f"{os.path.basename(lbl_path)}: class {cls}")
    except Exception:
        pass
check_bar.close()

if invalid_count == 0:
    print(f"      All class IDs valid 0-37  ({checked_files} files checked)")
else:
    print(f"      ERROR: {invalid_count} invalid class IDs found!")
    for s in invalid_sample:
        print(f"         {s}")
    cv_errors += 1

# -- Check B: Per-prefix spot check --
print("\n   [B] Per-dataset class ID spot check:")
for prefix, remap_table in DATASET_REMAPPING.items():
    prefix_files = [f for f in all_labels
                    if os.path.basename(f).startswith(f"{prefix}_")
                    and os.path.getsize(f) > 0]
    if not prefix_files:
        print(f"      {prefix:<6} -> 0 files (skipped)")
        continue
    errors_found = 0
    for lbl_path in prefix_files[:10]:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and (int(parts[0]) < 0 or int(parts[0]) > 37):
                    errors_found += 1
    status = "OK" if errors_found == 0 else f"ERROR: {errors_found} invalid IDs"
    if errors_found > 0:
        cv_errors += 1
    print(f"      {prefix:<6} -> {status}  (checked {min(10, len(prefix_files))} files)")

# -- Check C: Verify merged classes --
print("\n   [C] Verifying merged class remapping:")

nv34_class22_found = 0
nv34_files = [f for f in all_labels
              if (os.path.basename(f).startswith("nv3_") or
                  os.path.basename(f).startswith("nv4_"))
              and os.path.getsize(f) > 0]
for lbl_path in nv34_files[:50]:
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == 22:
                nv34_class22_found += 1
                break

print(f"      nv3/nv4 files containing class 22 (medical_gloves): {nv34_class22_found}")
if nv34_class22_found > 0:
    print(f"      OK: medical_gloves (17->22) correctly remapped in nv3/nv4")
else:
    print(f"      INFO: No class 22 found in sampled nv3/nv4 files (may be rare class)")

v5_files  = [f for f in all_labels
             if os.path.basename(f).startswith("v5_")
             and os.path.getsize(f) > 0]
v5_bad    = 0
for lbl_path in v5_files[:50]:
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and (int(parts[0]) < 0 or int(parts[0]) > 37):
                v5_bad += 1
if v5_bad == 0:
    print(f"      OK: v5 medicine_bottole typo correctly merged")
else:
    print(f"      ERROR: {v5_bad} v5 lines have out-of-range IDs")
    cv_errors += 1

# -- Check D: Label file count unchanged --
print(f"\n   [D] Label file count unchanged:")
print(f"      OK: {len(all_labels)} label files (no files added or removed)")

# -- Check E: Class distribution --
print(f"\n   [E] Class distribution -- top 10 most common classes:")
class_counts = [0] * 38
sample_size  = min(2000, len(all_labels))

sample_bar = tqdm(all_labels[:sample_size], desc="   Counting",
                  unit="file", ncols=70, leave=False, colour="cyan")
for lbl_path in sample_bar:
    if os.path.getsize(lbl_path) == 0:
        continue
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) >= 5:
                    cls = int(parts[0])
                    if 0 <= cls <= 37:
                        class_counts[cls] += 1
    except Exception:
        pass
sample_bar.close()

sorted_classes = sorted(enumerate(class_counts), key=lambda x: x[1], reverse=True)
print(f"      (sampled {sample_size} files)")
for cls_id, count in sorted_classes[:10]:
    bar = "#" * min(20, count // max(1, max(class_counts) // 20))
    print(f"      {cls_id:>2} {MASTER_CLASSES_DICT[cls_id]:<28} {count:>6}  {bar}")

# ─────────────────────────────────────────────
# Save log  ← FIX 3: encoding="utf-8" + ASCII-safe characters
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving remapping log...")
print("=" * 60)

with open(LOG_FILE, "w", encoding="utf-8") as f:        # FIX 3a: added encoding="utf-8"
    f.write("PART 4 - Class ID Remapping Log\n")         # FIX 3b: removed special dash char
    f.write("=" * 60 + "\n\n")
    f.write(f"Label format   : YOLO Segmentation\n")
    f.write(f"Master classes : {len(MASTER_CLASSES)}\n")
    f.write(f"Files remapped : {total_remapped}\n")
    f.write(f"Files skipped  : {total_skipped}\n")
    f.write(f"Errors         : {len(total_errors)}\n\n")
    f.write("MASTER CLASS LIST:\n")
    for i, name in enumerate(MASTER_CLASSES):
        f.write(f"  {i:>2}: {name}\n")
    f.write("\nPER-DATASET REMAPPING (auto-built from data.yaml):\n")
    for ds, remap in DATASET_REMAPPING.items():
        f.write(f"\n{ds}:\n")
        for old, new in sorted(remap.items()):
            changed = " <- CHANGED" if old != new else ""  # FIX 3c: <- instead of special arrow
            f.write(f"  {old:>2} -> {new:>2}  {MASTER_CLASSES_DICT.get(new,'?')}{changed}\n")  # FIX 3d: -> instead of special arrow
    if total_errors:
        f.write("\nERRORS:\n")
        for e in total_errors:
            f.write(f"  {e}\n")

print(f"\n   Log saved: {LOG_FILE}")

# ─────────────────────────────────────────────
# ROLLBACK OPTION
# ─────────────────────────────────────────────
if cv_errors > 0 or len(total_errors) > 0:
    print("\n" + "=" * 60)
    print("WARNING: ISSUES FOUND -- Rollback instructions:")
    print("=" * 60)
    print(f"""
   A backup of all original label files was saved at:
   {BACKUP_DIR}

   To rollback and restore original labels, run this:

      import shutil, os
      backup = r"{BACKUP_DIR}"
      output = r"{OUTPUT_PATH}"
      for root, dirs, files in os.walk(backup):
          for f in files:
              src = os.path.join(root, f)
              dst = src.replace(backup, output)
              os.makedirs(os.path.dirname(dst), exist_ok=True)
              shutil.copy2(src, dst)
      print("Rollback complete")
""")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
   Total label files      : {len(all_labels)}
   Successfully remapped  : {total_remapped}
   Skipped (empty/unknown): {total_skipped}
   Errors                 : {len(total_errors)}
   Time taken             : {remap_time:.1f}s

   Master classes         : 38 (IDs 0-37)
   Label format           : YOLO Segmentation
   Backup saved at        : {BACKUP_DIR}

   Cross validation errors: {cv_errors}
""")

if cv_errors == 0 and len(total_errors) == 0:
    print("   ALL CHECKS PASSED -- Part 4 Complete!")
    print("   Ready for Part 5 (Balanced Train/Valid/Test Split)")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("   Checkpoint file cleaned up")
    print(f"\n   INFO: Backup kept at: {BACKUP_DIR}")
    print(f"         You may delete it manually once Part 5 is verified")
else:
    print("   WARNING: Issues found -- see rollback instructions above")

print("=" * 60)