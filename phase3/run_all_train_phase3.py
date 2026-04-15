# ============================================================
# FILE 2 — VALIDATE & TEST YOLOv8 SEGMENTATION MODEL
# Run: python file2_validate.py
#
# pip install requirements:
#   pip install ultralytics tqdm torch torchvision
#   pip install matplotlib seaborn numpy pandas
# ============================================================

import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
BASE_PATH    = r"C:\Users\gahan\Desktop\dbhdsnet_project"
DATASET_YAML = os.path.join(BASE_PATH, "master_dataset", "master_data.yaml")
RUNS_DIR     = os.path.join(BASE_PATH, "runs")
RUN_NAME     = "medwaste_seg_v16"    # must match what you used in file1_train.py
RESULTS_DIR  = os.path.join(BASE_PATH, "validation_results")

# Point to your trained model
BEST_MODEL = os.path.join(RUNS_DIR, "segment", RUN_NAME, "weights", "best.pt")

# Validation settings
IMG_SIZE    = 640
BATCH_SIZE  = 16
CONF_THRESH = 0.25   # confidence threshold for predictions
IOU_THRESH  = 0.6    # IoU threshold for NMS

# ─────────────────────────────────────────────
# MASTER CLASSES
# ─────────────────────────────────────────────
MASTER_CLASSES = [
    "bloody_objects", "mask", "n95", "oxygen_cylinder", "radioactive_objects",
    "bandage", "blade", "capsule", "cotton_swab", "covid_buffer",
    "covid_buffer_box", "covid_test_case", "gauze", "glass_bottle",
    "harris_uni_core", "harris_uni_core_cap", "iodine_swab", "mercury_thermometer",
    "paperbox", "pill", "plastic_medical_bag", "plastic_medical_bottle",
    "medical_gloves", "reagent_tube", "reagent_tube_cap", "scalpel",
    "single_channel_pipette", "syringe", "transferpettor_glass",
    "transferpettor_plastic", "tweezer_metal", "tweezer_plastic", "unguent",
    "electronic_thermometer", "cap_plastic", "drug_packaging",
    "medical_infusion_bag", "needle",
]



if __name__ == "__main__":
    # ─────────────────────────────────────────────
    # STEP 0 — GPU and model check
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 0: Hardware & Model Check")
    print("=" * 60)

    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"\n   GPU : {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n   Running on CPU")

    print(f"   Device : {device}")

    if not os.path.exists(BEST_MODEL):
        print(f"\n   ERROR: Trained model not found at:")
        print(f"   {BEST_MODEL}")
        print(f"   Run file1_train.py first.")
        exit(1)

    print(f"\n   Model  : {BEST_MODEL}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─────────────────────────────────────────────
    # STEP 1 — Load model
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading trained model...")
    print("=" * 60)

    model = YOLO(BEST_MODEL)
    print(f"\n   Loaded: {BEST_MODEL}")
    model.info()

    # ─────────────────────────────────────────────
    # STEP 2 — Validation set evaluation
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running validation set evaluation...")
    print("=" * 60)
    print(f"\n   This computes mAP, precision, recall for all 38 classes\n")

    val_start = time.time()

    val_results = model.val(
        data      = DATASET_YAML,
        split     = "val",
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = device,
        conf      = CONF_THRESH,
        iou       = IOU_THRESH,
        project   = RESULTS_DIR,
        name      = "val_run",
        plots     = True,
        save_json = True,
        verbose   = True,
    )

    val_time = time.time() - val_start

    print(f"\n   Validation completed in {val_time:.1f}s")

    # Extract key metrics
    map50    = val_results.seg.map50   if hasattr(val_results, 'seg') else val_results.box.map50
    map5095  = val_results.seg.map     if hasattr(val_results, 'seg') else val_results.box.map
    precision = val_results.seg.mp    if hasattr(val_results, 'seg') else val_results.box.mp
    recall    = val_results.seg.mr    if hasattr(val_results, 'seg') else val_results.box.mr

    print(f"""
       ─────────────────────────────────────────
       VALIDATION RESULTS (Segmentation)
       ─────────────────────────────────────────
       mAP@0.5        : {map50:.4f}   ({map50*100:.1f}%)
       mAP@0.5:0.95   : {map5095:.4f}   ({map5095*100:.1f}%)
       Precision      : {precision:.4f}   ({precision*100:.1f}%)
       Recall         : {recall:.4f}   ({recall*100:.1f}%)
       ─────────────────────────────────────────
    """)

    # Grade the model
    if map50 >= 0.85:
        grade = "Excellent -- ready for deployment"
    elif map50 >= 0.70:
        grade = "Good -- consider more epochs or larger model"
    elif map50 >= 0.50:
        grade = "Fair -- try more epochs, data augmentation, or yolov8m-seg"
    else:
        grade = "Poor -- check data quality, class balance, increase epochs"

    print(f"   Grade: {grade}")

    # ─────────────────────────────────────────────
    # STEP 3 — Test set evaluation
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Running TEST set evaluation...")
    print("=" * 60)
    print(f"\n   This is the FINAL unbiased accuracy on held-out test data\n")

    test_start = time.time()

    test_results = model.val(
        data      = DATASET_YAML,
        split     = "test",
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = device,
        conf      = CONF_THRESH,
        iou       = IOU_THRESH,
        project   = RESULTS_DIR,
        name      = "test_run",
        plots     = True,
        save_json = True,
        verbose   = True,
    )

    test_time = time.time() - test_start

    test_map50   = test_results.seg.map50 if hasattr(test_results, 'seg') else test_results.box.map50
    test_map5095 = test_results.seg.map   if hasattr(test_results, 'seg') else test_results.box.map
    test_prec    = test_results.seg.mp    if hasattr(test_results, 'seg') else test_results.box.mp
    test_rec     = test_results.seg.mr    if hasattr(test_results, 'seg') else test_results.box.mr

    print(f"""
       ─────────────────────────────────────────
       TEST SET RESULTS (Segmentation)
       ─────────────────────────────────────────
       mAP@0.5        : {test_map50:.4f}   ({test_map50*100:.1f}%)
       mAP@0.5:0.95   : {test_map5095:.4f}   ({test_map5095*100:.1f}%)
       Precision      : {test_prec:.4f}   ({test_prec*100:.1f}%)
       Recall         : {test_rec:.4f}   ({test_rec*100:.1f}%)
       ─────────────────────────────────────────
    """)

    # ─────────────────────────────────────────────
    # STEP 4 — Per-class accuracy breakdown
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Per-class accuracy breakdown (validation set)")
    print("=" * 60)
    print(f"\n   {'Class':<28} {'AP@0.5':>8}   {'Grade':>10}")
    print(f"   {'-'*28} {'-'*8}   {'-'*10}")

    per_class_ap = None
    try:
        # Try to get per-class AP from results
        if hasattr(val_results, 'seg') and hasattr(val_results.seg, 'ap_class_index'):
            ap_values       = val_results.seg.ap50
            class_indices   = val_results.seg.ap_class_index
            per_class_ap    = {int(idx): float(ap)
                               for idx, ap in zip(class_indices, ap_values)}
        elif hasattr(val_results, 'box') and hasattr(val_results.box, 'ap_class_index'):
            ap_values       = val_results.box.ap50
            class_indices   = val_results.box.ap_class_index
            per_class_ap    = {int(idx): float(ap)
                               for idx, ap in zip(class_indices, ap_values)}
    except Exception as e:
        print(f"   WARNING: Could not extract per-class AP: {e}")

    if per_class_ap:
        poor_classes = []
        for cls_id, cls_name in enumerate(MASTER_CLASSES):
            ap    = per_class_ap.get(cls_id, None)
            if ap is None:
                print(f"   {cls_name:<28} {'N/A':>8}   {'No samples':>10}")
                continue
            if ap >= 0.80:
                grade = "Good"
            elif ap >= 0.50:
                grade = "Fair"
            else:
                grade = "Poor"
                poor_classes.append((cls_name, ap))
            print(f"   {cls_name:<28} {ap:>8.3f}   {grade:>10}")

        if poor_classes:
            print(f"\n   Classes needing attention (AP < 0.5):")
            for name, ap in poor_classes:
                print(f"      {name}: {ap:.3f} -- consider adding more samples")
        else:
            print(f"\n   All classes performing well!")
    else:
        print(f"   Per-class breakdown not available from this ultralytics version.")
        print(f"   Check plots in: {RESULTS_DIR}")

    # ─────────────────────────────────────────────
    # STEP 5 — Cross validation summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Cross Validation Summary")
    print("=" * 60)

    overfit_gap = abs(map50 - test_map50)

    print(f"""
       ─────────────────────────────────────────
       METRIC           VAL       TEST      GAP
       ─────────────────────────────────────────
       mAP@0.5        {map50:>6.3f}    {test_map50:>6.3f}    {overfit_gap:>5.3f}
       mAP@0.5:0.95   {map5095:>6.3f}    {test_map5095:>6.3f}
       Precision      {precision:>6.3f}    {test_prec:>6.3f}
       Recall         {recall:>6.3f}    {test_rec:>6.3f}
       ─────────────────────────────────────────
    """)

    if overfit_gap < 0.05:
        overfit_status = "No overfitting detected -- model generalises well"
    elif overfit_gap < 0.10:
        overfit_status = "Slight gap -- acceptable, minor overfitting"
    else:
        overfit_status = "Large gap -- model is overfitting, try more augmentation or dropout"

    print(f"   Overfit check: {overfit_status}")

    # ─────────────────────────────────────────────
    # STEP 6 — Save full report to JSON
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Saving results report...")
    print("=" * 60)

    report = {
        "model":       BEST_MODEL,
        "run_name":    RUN_NAME,
        "conf_thresh": CONF_THRESH,
        "iou_thresh":  IOU_THRESH,
        "validation": {
            "map50":     round(float(map50),    4),
            "map5095":   round(float(map5095),  4),
            "precision": round(float(precision),4),
            "recall":    round(float(recall),   4),
        },
        "test": {
            "map50":     round(float(test_map50),    4),
            "map5095":   round(float(test_map5095),  4),
            "precision": round(float(test_prec),     4),
            "recall":    round(float(test_rec),      4),
        },
        "overfitting_gap": round(float(overfit_gap), 4),
        "per_class_ap": {MASTER_CLASSES[k]: round(v, 4)
                         for k, v in (per_class_ap or {}).items()
                         if k < len(MASTER_CLASSES)},
    }

    report_path = os.path.join(RESULTS_DIR, "full_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n   Report saved: {report_path}")

    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"""
       Model         : {os.path.basename(BEST_MODEL)}
       Run name      : {RUN_NAME}

       Validation mAP@0.5   : {map50*100:.1f}%
       Test mAP@0.5         : {test_map50*100:.1f}%

       Overfitting          : {overfit_status}
       Grade                : {grade}

       Plots saved to       : {RESULTS_DIR}
       Full report          : {report_path}

       Next step:
          Run file3_realtime.py for live detection
    """)

    print("=" * 60)