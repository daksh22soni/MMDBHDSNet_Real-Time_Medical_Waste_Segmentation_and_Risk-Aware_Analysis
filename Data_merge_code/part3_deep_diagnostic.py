import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_PATH      = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"
SPLITS           = ["train", "valid", "test"]

def get_label_path(img_path):
    parts     = list(os.path.normpath(img_path).split(os.sep))
    new_parts = ["labels" if p == "images" else p for p in parts]
    lbl_path  = os.sep.join(new_parts)
    return os.path.splitext(lbl_path)[0] + ".txt"

def collect_images():
    images = []
    for split in SPLITS:
        img_dir = os.path.join(OUTPUT_PATH, split, "images")
        if os.path.exists(img_dir):
            files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            images.extend(files)
    return images

# ─────────────────────────────────────────────
# DIAGNOSTIC 1 — Raw bytes of first 10 label files
# ─────────────────────────────────────────────
print("=" * 60)
print("DIAGNOSTIC 1: Raw content of first 10 label files")
print("=" * 60)

all_images = collect_images()

checked = 0
for img_path in all_images:
    lbl_path = get_label_path(img_path)
    if not os.path.exists(lbl_path):
        continue

    # Read raw bytes
    with open(lbl_path, "rb") as f:
        raw_bytes = f.read(200)   # first 200 bytes only

    # Read as text with different encodings
    try:
        text_utf8 = open(lbl_path, "r", encoding="utf-8").read(200)
    except Exception as e:
        text_utf8 = f"UTF-8 ERROR: {e}"

    try:
        text_utf8_bom = open(lbl_path, "r", encoding="utf-8-sig").read(200)
    except Exception as e:
        text_utf8_bom = f"UTF-8-BOM ERROR: {e}"

    try:
        text_latin = open(lbl_path, "r", encoding="latin-1").read(200)
    except Exception as e:
        text_latin = f"LATIN-1 ERROR: {e}"

    print(f"\n   File    : {os.path.basename(lbl_path)}")
    print(f"   Size    : {os.path.getsize(lbl_path)} bytes")
    print(f"   Raw hex : {raw_bytes[:50].hex()}")
    print(f"   UTF-8   : {repr(text_utf8[:100])}")
    print(f"   UTF-BOM : {repr(text_utf8_bom[:100])}")
    print(f"   LATIN-1 : {repr(text_latin[:100])}")

    checked += 1
    if checked >= 10:
        break

# ─────────────────────────────────────────────
# DIAGNOSTIC 2 — Count empty vs non-empty labels
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Empty vs Non-empty label count")
print("=" * 60)

empty_count    = 0
nonempty_count = 0
missing_count  = 0
total_checked  = 0

for img_path in all_images:
    lbl_path = get_label_path(img_path)
    if not os.path.exists(lbl_path):
        missing_count += 1
        continue
    size = os.path.getsize(lbl_path)
    if size == 0:
        empty_count += 1
    else:
        nonempty_count += 1
    total_checked += 1

print(f"\n   Total images checked : {total_checked}")
print(f"   Non-empty labels     : {nonempty_count}  ✅")
print(f"   Empty labels (0 KB)  : {empty_count}   {'✅ normal' if empty_count < total_checked * 0.3 else '⚠️ too many!'}")
print(f"   Missing labels       : {missing_count}")

# ─────────────────────────────────────────────
# DIAGNOSTIC 3 — Show first non-empty label content
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Content of first 5 NON-EMPTY label files")
print("=" * 60)

found = 0
for img_path in all_images:
    lbl_path = get_label_path(img_path)
    if not os.path.exists(lbl_path):
        continue
    if os.path.getsize(lbl_path) == 0:
        continue

    with open(lbl_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    lines = [l for l in content.strip().split("\n") if l.strip()]
    print(f"\n   File    : {os.path.basename(lbl_path)}")
    print(f"   Size    : {os.path.getsize(lbl_path)} bytes")
    print(f"   Lines   : {len(lines)}")
    for line in lines[:3]:
        print(f"   Content : {repr(line)}")
        parts = line.strip().split()
        print(f"   Parts   : {parts}  (count: {len(parts)})")

    found += 1
    if found >= 5:
        break

# ─────────────────────────────────────────────
# DIAGNOSTIC 4 — Check a specific nv2 file
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Checking the specific nv2 file from earlier")
print("=" * 60)

target = None
for img_path in all_images:
    if "nv2_10mixed1__png.rf.0cd1e3f23bf73ceca6f3c367abdd6688" in img_path:
        target = img_path
        break

if target:
    lbl_path = get_label_path(target)
    print(f"\n   Image : {os.path.basename(target)}")
    print(f"   Label : {lbl_path}")
    print(f"   Exists: {os.path.exists(lbl_path)}")
    if os.path.exists(lbl_path):
        size = os.path.getsize(lbl_path)
        print(f"   Size  : {size} bytes")
        with open(lbl_path, "rb") as f:
            raw = f.read()
        print(f"   Raw   : {repr(raw[:200])}")
else:
    print("   ⚠️  Target file not found in current dataset")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE — Share this full output!")
print("=" * 60)
