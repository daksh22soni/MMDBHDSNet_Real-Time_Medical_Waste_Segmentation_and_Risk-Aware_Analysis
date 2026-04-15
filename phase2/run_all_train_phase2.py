# ============================================================
# FILE 1 — TRAIN YOLOv8 SEGMENTATION MODEL
# Run: python train.py
#
# pip install requirements:
#   pip install ultralytics tqdm torch torchvision torchaudio
#   pip install --upgrade ultralytics
#
# For NVIDIA GPU (CUDA):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# For CPU only:
#   pip install torch torchvision torchaudio
# ============================================================

import os
import time
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
BASE_PATH    = r"C:\Users\gahan\Desktop\dbhdsnet_project"
DATASET_YAML = os.path.join(BASE_PATH, "master_dataset", "master_data.yaml")
RUNS_DIR     = os.path.join(BASE_PATH, "runs")           # where all runs are saved
RUN_NAME     = "medwaste_seg_v1"                          # change for each new run

# Model choice — pick ONE:
#   "yolov8n-seg.pt"  — nano    (fastest, least accurate, good for testing)
#   "yolov8s-seg.pt"  — small   (good balance for limited GPU)
#   "yolov8m-seg.pt"  — medium  (recommended for this dataset size)
#   "yolov8l-seg.pt"  — large   (best accuracy, needs strong GPU)
MODEL_WEIGHTS = "yolov8n-seg.pt"

# Training hyperparameters
EPOCHS      = 100
IMG_SIZE    = 640
BATCH_SIZE  = 16     # reduce to 8 if GPU runs out of memory
WORKERS     = 4      # data loader workers (reduce to 2 if errors occur)
PATIENCE    = 20     # early stopping — stops if no improvement for 20 epochs



# Checkpoint resume — set to path of last.pt to resume, or None for fresh start
RESUME_FROM = None
# Example: RESUME_FROM = r"C:\...\runs\medwaste_seg_v1\weights\last.pt"



if __name__ == "__main__":
    # ─────────────────────────────────────────────
    # STEP 0 — GPU CHECK
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 0: Hardware Check")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device     = "0"   # GPU index
        print(f"\n   GPU detected  : {gpu_name}")
        print(f"   GPU memory    : {gpu_mem_gb:.1f} GB")
        print(f"   CUDA version  : {torch.version.cuda}")

        # Auto-adjust batch size based on GPU memory
        if gpu_mem_gb < 4:
            BATCH_SIZE = 4
            print(f"   WARNING: Low GPU memory -- batch size reduced to {BATCH_SIZE}")
        elif gpu_mem_gb < 8:
            BATCH_SIZE = 8
            print(f"   INFO: Moderate GPU memory -- batch size set to {BATCH_SIZE}")
        else:
            print(f"   Batch size    : {BATCH_SIZE}")
    else:
        device = "cpu"
        print(f"\n   No GPU found -- training on CPU")
        print(f"   WARNING: CPU training is very slow (hours vs minutes on GPU)")
        print(f"   Consider using Google Colab for free GPU access")
        BATCH_SIZE = 4
        WORKERS    = 0

    print(f"\n   Device  : {device}")
    print(f"   Model   : {MODEL_WEIGHTS}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Img size: {IMG_SIZE}")
    print(f"   Batch   : {BATCH_SIZE}")

    # ─────────────────────────────────────────────
    # STEP 1 — Verify dataset yaml exists
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Verifying dataset...")
    print("=" * 60)

    if not os.path.exists(DATASET_YAML):
        print(f"\n   ERROR: master_data.yaml not found at:")
        print(f"   {DATASET_YAML}")
        print(f"   Run part5_balanced_split.py first to generate it.")
        exit(1)

    print(f"\n   YAML    : {DATASET_YAML}")

    # Count images per split
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(BASE_PATH, "master_dataset", split, "images")
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"   {split:<6}  : {count} images")

    # ─────────────────────────────────────────────
    # STEP 2 — Load model
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading model...")
    print("=" * 60)

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"\n   Resuming from checkpoint: {RESUME_FROM}")
        model = YOLO(RESUME_FROM)
    else:
        print(f"\n   Loading pretrained weights: {MODEL_WEIGHTS}")
        print(f"   (Will auto-download if not cached)")
        model = YOLO(MODEL_WEIGHTS)

    model.info()

    # ─────────────────────────────────────────────
    # STEP 3 — Train
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training...")
    print("=" * 60)
    print(f"\n   Checkpoints saved every epoch to:")
    print(f"   {os.path.join(RUNS_DIR, 'segment', RUN_NAME, 'weights')}")
    print(f"\n   best.pt  = best model overall")
    print(f"   last.pt  = most recent epoch (use for resuming)")
    print(f"\n   Starting training...\n")

    train_start = time.time()

    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = device,
        workers   = WORKERS,
        project   = os.path.join(RUNS_DIR, "segment"),
        name      = RUN_NAME,
        patience  = PATIENCE,       # early stopping
        save      = True,           # save best.pt and last.pt
        save_period = 10,           # also save checkpoint every 10 epochs
        resume    = bool(RESUME_FROM and os.path.exists(RESUME_FROM)),
        plots     = True,           # save training plots
        verbose   = True,
        # Augmentation — good defaults for medical images
        flipud    = 0.3,            # vertical flip (medical items can be any orientation)
        fliplr    = 0.5,            # horizontal flip
        degrees   = 15,             # rotation
        translate = 0.1,
        scale     = 0.5,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        mosaic    = 1.0,
        mixup     = 0.1,
    )

    train_time = time.time() - train_start
    hours      = int(train_time // 3600)
    minutes    = int((train_time % 3600) // 60)

    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    best_model_path = os.path.join(RUNS_DIR, "segment", RUN_NAME, "weights", "best.pt")
    last_model_path = os.path.join(RUNS_DIR, "segment", RUN_NAME, "weights", "last.pt")

    print(f"""
       Training time : {hours}h {minutes}m
       Device used   : {device}
       Total epochs  : {EPOCHS}  (early stop patience: {PATIENCE})

       Saved models:
          Best  : {best_model_path}
          Last  : {last_model_path}

       Training plots saved in:
          {os.path.join(RUNS_DIR, "segment", RUN_NAME)}

       Next steps:
          Run file2_validate.py to check model accuracy
          Run file3_realtime.py for live detection
    """)

    if os.path.exists(best_model_path):
        print(f"   best.pt confirmed at: {best_model_path}")
    else:
        print(f"   WARNING: best.pt not found -- check training output above")

    print("=" * 60)