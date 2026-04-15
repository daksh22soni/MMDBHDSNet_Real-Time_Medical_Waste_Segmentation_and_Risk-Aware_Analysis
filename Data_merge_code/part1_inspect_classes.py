import os
import yaml

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH = r"C:\Users\gahan\Desktop\dbhdsnet_project"

# ─────────────────────────────────────────────
# FINAL 38 MASTER CLASSES (corrected & merged)
# ─────────────────────────────────────────────
# Changes made:
#   - plastic_medical_gloves (22) + medical_gloves (36) → merged into 22: medical_gloves
#   - medicine_bottole (37, typo) + plastic_medical_bottle (21) → merged into 21: plastic_medical_bottle
#   - Removed duplicate entries → total 40 → 38 classes

MASTER_CLASSES = {
    0:  "bloody_objects",
    1:  "mask",
    2:  "n95",
    3:  "oxygen_cylinder",
    4:  "radioactive_objects",
    5:  "bandage",
    6:  "blade",
    7:  "capsule",
    8:  "cotton_swab",
    9:  "covid_buffer",
    10: "covid_buffer_box",
    11: "covid_test_case",
    12: "gauze",
    13: "glass_bottle",
    14: "harris_uni_core",
    15: "harris_uni_core_cap",
    16: "iodine_swab",
    17: "mercury_thermometer",
    18: "paperbox",
    19: "pill",
    20: "plastic_medical_bag",
    21: "plastic_medical_bottle",
    22: "medical_gloves",
    23: "reagent_tube",
    24: "reagent_tube_cap",
    25: "scalpel",
    26: "single_channel_pipette",
    27: "syringe",
    28: "transferpettor_glass",
    29: "transferpettor_plastic",
    30: "tweezer_metal",
    31: "tweezer_plastic",
    32: "unguent",
    33: "electronic_thermometer",
    34: "cap_plastic",
    35: "drug_packaging",
    36: "medical_infusion_bag",
    37: "needle",
}

# ─────────────────────────────────────────────
# STEP 1 — Show final master class list
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Final Master Class List (38 classes)")
print("=" * 60)
for idx, name in MASTER_CLASSES.items():
    print(f"   {idx:>2}: {name}")

# ─────────────────────────────────────────────
# STEP 2 — Show what was merged/fixed
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Changes from original 40 → 38 classes")
print("=" * 60)
print("""
   MERGE 1 (Gloves):
      OLD 22: plastic_medical_gloves  (v1, v2)      ─┐
      OLD 36: medical_gloves          (v3,v4,v5,v6,v7)─┘
      NEW 22: medical_gloves  ✅

   MERGE 2 (Bottle):
      OLD 21: plastic_medical_bottle  (all datasets) ─┐
      OLD 37: medicine_bottole        (v5, typo)     ─┘
      NEW 21: plastic_medical_bottle  ✅

   Total: 40 → 38 classes
""")

# ─────────────────────────────────────────────
# STEP 3 — Show per-dataset remapping with merges applied
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Per-dataset remapping (with merges applied)")
print("=" * 60)

# Helper: name → master ID
name_to_id = {v: k for k, v in MASTER_CLASSES.items()}

# Special merge rules (old name → corrected master name)
MERGE_MAP = {
    "plastic_medical_gloves": "medical_gloves",
    "medical_gloves":         "medical_gloves",
    "medicine_bottole":       "plastic_medical_bottle",
    "plastic_medical_bottle": "plastic_medical_bottle",
}

datasets_classes = {}
for folder in os.listdir(BASE_PATH):
    yaml_path = os.path.join(BASE_PATH, folder, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        datasets_classes[folder] = names

final_remapping = {}  # { dataset: { old_id: new_master_id } }

for dataset, classes in datasets_classes.items():
    print(f"\n   📁 {dataset}:")
    remap = {}
    for old_id, cls in enumerate(classes):
        cls_lower = cls.strip().lower()
        resolved = MERGE_MAP.get(cls_lower, cls_lower)
        new_id = name_to_id.get(resolved)
        if new_id is None:
            print(f"      ⚠️  {old_id} ({cls}) → NOT FOUND IN MASTER — check manually!")
        else:
            status = "✅ same" if old_id == new_id else "🔄 changed"
            merge_note = " ← MERGED" if cls_lower in ["plastic_medical_gloves", "medicine_bottole"] else ""
            print(f"      {old_id:>2} ({cls:<30}) → {new_id:>2} ({MASTER_CLASSES[new_id]}){merge_note}  {status}")
        remap[old_id] = new_id
    final_remapping[dataset] = remap

# ─────────────────────────────────────────────
# STEP 4 — Save corrected master_data.yaml
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Saving corrected master_data.yaml...")
print("=" * 60)

master_yaml = {
    "path":  BASE_PATH + r"\MedicalWasteDataset",
    "train": "train/images",
    "val":   "valid/images",
    "test":  "test/images",
    "nc":    len(MASTER_CLASSES),
    "names": MASTER_CLASSES
}

output_path = os.path.join(BASE_PATH, "master_data.yaml")
with open(output_path, "w") as f:
    yaml.dump(master_yaml, f, default_flow_style=False, sort_keys=False)

print(f"\n✅ Saved at: {output_path}")

# ─────────────────────────────────────────────
# STEP 5 — CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: CROSS VALIDATION")
print("=" * 60)

with open(output_path, "r") as f:
    verify = yaml.safe_load(f)

errors = 0

# Check 1: nc matches class count
if verify["nc"] == len(verify["names"]):
    print(f"\n   ✅ nc ({verify['nc']}) matches number of class names ({len(verify['names'])})")
else:
    print(f"\n   ❌ nc MISMATCH — nc={verify['nc']} but names count={len(verify['names'])}")
    errors += 1

# Check 2: No duplicate class names
class_names = list(verify["names"].values())
if len(class_names) == len(set(class_names)):
    print(f"   ✅ No duplicate class names found")
else:
    dupes = [x for x in class_names if class_names.count(x) > 1]
    print(f"   ❌ Duplicate class names found: {set(dupes)}")
    errors += 1

# Check 3: IDs are sequential 0 to nc-1
expected_ids = list(range(verify["nc"]))
actual_ids   = list(verify["names"].keys())
if actual_ids == expected_ids:
    print(f"   ✅ Class IDs are sequential (0 to {verify['nc']-1})")
else:
    print(f"   ❌ Class IDs are NOT sequential — check master list")
    errors += 1

# Check 4: merged classes are gone
for removed in ["plastic_medical_gloves", "medicine_bottole"]:
    if removed in class_names:
        print(f"   ❌ '{removed}' still exists — merge failed!")
        errors += 1
    else:
        print(f"   ✅ '{removed}' correctly removed after merge")

# Final verdict
print("\n" + "-" * 60)
if errors == 0:
    print("   🎉 ALL CHECKS PASSED — master_data.yaml is correct!")
else:
    print(f"   ⚠️  {errors} issue(s) found — review above before continuing")
print("-" * 60)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
   Total datasets found   : {len(datasets_classes)}
   Final class count      : {len(MASTER_CLASSES)} (was 40, merged 2 pairs)
   master_data.yaml saved : {output_path}

   ✅ Part 1 COMPLETE — master_data.yaml is ready!
   ➡️  You can now proceed to Part 2 (Rename & Merge files)
""")
print("=" * 60)