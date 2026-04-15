import os, json, shutil

BACKUP_DIR  = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset\splits_backup_part5"
OUTPUT_PATH = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"
SPLITS      = ["train", "valid", "test"]

record_path = os.path.join(BACKUP_DIR, "original_split_record.json")
with open(record_path, "r", encoding="utf-8") as f:
    split_record = json.load(f)

print(f"Rolling back {len(split_record)} files to original split...")

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
