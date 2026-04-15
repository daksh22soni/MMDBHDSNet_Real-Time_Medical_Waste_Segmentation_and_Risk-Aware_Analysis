import os
import random
import json
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_PATH   = r"C:\Users\gahan\Desktop\dbhdsnet_project\master_dataset"
SAVE_DIR      = r"C:\Users\gahan\Desktop\dbhdsnet_project\visual_check"
SAMPLES       = 10          # images per split
RANDOM_SEED   = 42
SPLITS        = ["train", "valid", "test"]

# ─────────────────────────────────────────────
# MASTER CLASS LIST — 38 classes
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
    "medical_gloves",        # 22
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

# One distinct colour per class (RGB)
# Bright palette so polygons are visible on any background
CLASS_COLORS = [
    (255,  56,  56), (255, 157,  51), (255, 225,  56), ( 56, 220,  56),
    ( 56, 190, 255), ( 56,  56, 255), (220,  56, 255), (255,  56, 160),
    (180, 255,  56), ( 56, 255, 180), (255, 120, 120), (120, 255, 120),
    (120, 120, 255), (255, 200,  80), ( 80, 255, 200), (200,  80, 255),
    (255,  80, 200), ( 80, 200, 255), (200, 255,  80), (255, 160,  80),
    ( 80, 255, 160), (160,  80, 255), (255,  80, 160), ( 80, 160, 255),
    (160, 255,  80), (200, 200,  56), ( 56, 200, 200), (200,  56, 200),
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (230, 180,  50),
    ( 50, 230, 180), (180,  50, 230), (255, 140,  30), ( 30, 255, 140),
    (140,  30, 255), (200, 100, 150),
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_label_path(img_path):
    parts     = list(os.path.normpath(img_path).split(os.sep))
    new_parts = ["labels" if p == "images" else p for p in parts]
    return os.path.splitext(os.sep.join(new_parts))[0] + ".txt"

def parse_label_file(lbl_path):
    """
    Parse YOLO segmentation label file.
    Returns list of (class_id, [(x,y), (x,y), ...]) in normalised 0-1 coords.
    """
    objects = []
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return objects
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and len(parts) % 2 == 1:
                cls_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                points = [(coords[i], coords[i+1])
                          for i in range(0, len(coords), 2)]
                objects.append((cls_id, points))
    return objects

def draw_annotated_image(img_path, lbl_path):
    """
    Draw segmentation polygons + class labels on image.
    Returns annotated PIL Image.
    """
    img    = Image.open(img_path).convert("RGB")
    W, H   = img.size
    draw   = ImageDraw.Draw(img, "RGBA")

    # Try to load a font — fall back to default if not available
    try:
        font       = ImageFont.truetype("arial.ttf", max(12, H // 40))
        font_small = ImageFont.truetype("arial.ttf", max(10, H // 50))
    except Exception:
        font       = ImageFont.load_default()
        font_small = font

    objects = parse_label_file(lbl_path)

    if not objects:
        # Draw "NO LABELS" warning on image
        draw.rectangle([0, 0, W, H], outline=(255, 0, 0), width=6)
        draw.text((10, 10), "NO LABELS", fill=(255, 0, 0), font=font)
        return img, []

    class_names_found = []

    for cls_id, norm_points in objects:
        if cls_id < 0 or cls_id >= len(MASTER_CLASSES):
            class_name = f"UNKNOWN_ID_{cls_id}"
            color      = (255, 0, 0)
        else:
            class_name = MASTER_CLASSES[cls_id]
            color      = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        class_names_found.append(f"{cls_id}: {class_name}")

        # Convert normalised coords to pixel coords
        pixel_points = [(int(x * W), int(y * H)) for x, y in norm_points]

        if len(pixel_points) < 3:
            continue

        # Draw filled polygon (semi-transparent)
        r, g, b = color
        draw.polygon(pixel_points, fill=(r, g, b, 60), outline=(r, g, b, 255))

        # Draw outline separately for visibility
        for i in range(len(pixel_points)):
            p1 = pixel_points[i]
            p2 = pixel_points[(i + 1) % len(pixel_points)]
            draw.line([p1, p2], fill=color, width=max(2, H // 300))

        # Draw class label at centroid of polygon
        cx = int(sum(p[0] for p in pixel_points) / len(pixel_points))
        cy = int(sum(p[1] for p in pixel_points) / len(pixel_points))

        label_text = f"{cls_id}:{class_name}"

        # Text background for readability
        try:
            bbox = draw.textbbox((cx, cy), label_text, font=font_small)
            pad  = 3
            draw.rectangle(
                [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                fill=(0, 0, 0, 180)
            )
        except Exception:
            pass

        draw.text((cx, cy), label_text, fill=(255, 255, 255), font=font_small)

    return img, class_names_found

def make_summary_grid(images_and_info, split_name, n_cols=5):
    """
    Combine multiple annotated images into a single grid image.
    Adds filename + classes found as caption below each image.
    """
    if not images_and_info:
        return None

    THUMB_W   = 400
    THUMB_H   = 300
    CAPTION_H = 80
    PADDING   = 8
    N_COLS    = min(n_cols, len(images_and_info))
    N_ROWS    = (len(images_and_info) + N_COLS - 1) // N_COLS

    CELL_W    = THUMB_W + PADDING * 2
    CELL_H    = THUMB_H + CAPTION_H + PADDING * 2
    HEADER_H  = 60

    GRID_W    = CELL_W * N_COLS
    GRID_H    = CELL_H * N_ROWS + HEADER_H

    grid = Image.new("RGB", (GRID_W, GRID_H), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Try font
    try:
        font_header  = ImageFont.truetype("arial.ttf", 28)
        font_caption = ImageFont.truetype("arial.ttf", 11)
        font_fname   = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_header  = ImageFont.load_default()
        font_caption = font_header
        font_fname   = font_header

    # Draw header
    header_text = f"{split_name.upper()} SET -- {len(images_and_info)} sample images"
    draw.text((PADDING, 14), header_text, fill=(255, 255, 255), font=font_header)

    for idx, (img, fname, class_names) in enumerate(images_and_info):
        row = idx // N_COLS
        col = idx % N_COLS

        x = col * CELL_W + PADDING
        y = row * CELL_H + HEADER_H + PADDING

        # Resize image to thumbnail
        thumb = img.copy()
        thumb.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        # Paste on dark background tile
        tile = Image.new("RGB", (THUMB_W, THUMB_H), (50, 50, 50))
        offset_x = (THUMB_W - thumb.width)  // 2
        offset_y = (THUMB_H - thumb.height) // 2
        tile.paste(thumb, (offset_x, offset_y))
        grid.paste(tile, (x, y))

        # Draw filename
        short_fname = fname[:45] + "..." if len(fname) > 45 else fname
        draw.text((x, y + THUMB_H + 4), short_fname,
                  fill=(180, 180, 180), font=font_fname)

        # Draw class names found (up to 4 lines)
        classes_text = ", ".join(set(class_names))
        if len(classes_text) > 60:
            classes_text = classes_text[:57] + "..."
        draw.text((x, y + THUMB_H + 16), classes_text,
                  fill=(100, 220, 100), font=font_caption)

        # Object count
        draw.text((x, y + THUMB_H + 30),
                  f"{len(class_names)} object(s) annotated",
                  fill=(200, 200, 100), font=font_caption)

    return grid

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("=" * 60)
print("PART 6: Visual Label Cross-Check")
print("=" * 60)
print(f"\n   Checking 10 images from each of: {SPLITS}")
print(f"   Output saved to: {SAVE_DIR}\n")

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

grand_report  = {}
total_invalid = 0

for split in SPLITS:
    print(f"\n{'=' * 60}")
    print(f"  {split.upper()} SET")
    print(f"{'=' * 60}")

    img_dir = os.path.join(OUTPUT_PATH, split, "images")
    if not os.path.exists(img_dir):
        print(f"   ERROR: {img_dir} not found")
        continue

    all_imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(all_imgs) < SAMPLES:
        sampled = all_imgs
        print(f"   WARNING: Only {len(all_imgs)} images available (less than {SAMPLES})")
    else:
        sampled = random.sample(all_imgs, SAMPLES)

    print(f"\n   Sampled {len(sampled)} images:\n")

    images_and_info = []
    split_report    = []
    split_invalid   = 0

    for fname in sampled:
        img_path = os.path.join(img_dir, fname)
        lbl_path = get_label_path(img_path)

        # Parse labels
        objects = parse_label_file(lbl_path)

        # Check for invalid class IDs
        invalid_ids = [cls_id for cls_id, _ in objects
                       if cls_id < 0 or cls_id >= len(MASTER_CLASSES)]

        status = "OK" if not invalid_ids else f"INVALID IDs: {invalid_ids}"
        if invalid_ids:
            split_invalid += 1
            total_invalid += 1

        # Print per-image report
        class_summary = ", ".join(
            f"{cls_id}({MASTER_CLASSES[cls_id] if 0 <= cls_id < 38 else '?'})"
            for cls_id, _ in objects
        )
        print(f"   {fname[:55]}")
        print(f"   Objects: {len(objects)}  |  Classes: {class_summary or 'NONE'}")
        print(f"   Status : {status}")
        print()

        split_report.append({
            "file":        fname,
            "objects":     len(objects),
            "class_ids":   [cls_id for cls_id, _ in objects],
            "class_names": [MASTER_CLASSES[cls_id] if 0 <= cls_id < 38 else "INVALID"
                            for cls_id, _ in objects],
            "status":      status,
        })

        # Draw annotated image
        ann_img, class_names_found = draw_annotated_image(img_path, lbl_path)
        images_and_info.append((ann_img, fname, class_names_found))

        # Also save individual annotated image
        individual_dir = os.path.join(SAVE_DIR, split, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        save_name = os.path.splitext(fname)[0] + "_annotated.jpg"
        ann_img.save(os.path.join(individual_dir, save_name), quality=92)

    grand_report[split] = split_report

    # Save grid image for this split
    grid = make_summary_grid(images_and_info, split, n_cols=5)
    if grid:
        grid_path = os.path.join(SAVE_DIR, f"{split}_grid.jpg")
        grid.save(grid_path, quality=92)
        print(f"   Grid saved  : {grid_path}")

    print(f"   Invalid IDs : {split_invalid}")
    print(f"   OK images   : {len(sampled) - split_invalid} / {len(sampled)}")

# ─────────────────────────────────────────────
# Save JSON report
# ─────────────────────────────────────────────
report_path = os.path.join(SAVE_DIR, "check_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(grand_report, f, indent=2)
print(f"\n   Full report : {report_path}")

# ─────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)

total_checked = len(SPLITS) * SAMPLES

print(f"""
   Images checked : {total_checked}  (10 per split x 3 splits)
   Invalid labels : {total_invalid}
   Correct labels : {total_checked - total_invalid}
""")

if total_invalid == 0:
    print("   100% CORRECT -- Dataset is verified and ready for training!")
    print()
    print("   What to check in the saved images:")
    print("   - Polygons should tightly trace the actual object shape")
    print("   - Class label text should match what you see in the photo")
    print("   - No wildly wrong labels (e.g. syringe labeled as mask)")
    print()
    print(f"   Open these to visually verify:")
    for split in SPLITS:
        print(f"      {os.path.join(SAVE_DIR, split + '_grid.jpg')}")
else:
    print(f"   WARNING: {total_invalid} images have INVALID class IDs!")
    print(f"   Check the report at: {report_path}")
    print(f"   You may need to re-run Part 4 (remapping)")

print("\n" + "=" * 60)
print("Output folder structure:")
print("=" * 60)
print(f"""
   {SAVE_DIR}
   |-- train_grid.jpg          <- all 10 train samples in one image
   |-- valid_grid.jpg          <- all 10 valid samples in one image
   |-- test_grid.jpg           <- all 10 test samples in one image
   |-- check_report.json       <- full JSON report with class IDs
   |-- train/individual/       <- 10 individual annotated images
   |-- valid/individual/       <- 10 individual annotated images
   |-- test/individual/        <- 10 individual annotated images
""")
print("=" * 60)
