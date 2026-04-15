# DBHDSNet — Phase 2: Setup & Run Guide

## Project Structure

```
phase2_dbhdsnet/
├── config.py                  ← ALL PATHS & HYPERPARAMETERS (edit this first)
├── train.py                   ← Main entry point
├── requirements.txt
├── SETUP_AND_RUN.md           ← This file
│
├── src/
│   ├── dataset.py             ← YOLO seg dataset loader + augmentation
│   ├── losses.py              ← Novel composite + hierarchy-aware loss
│   ├── metrics.py             ← mAP, hazard accuracy, ECE
│   ├── trainer.py             ← Training loop (tqdm, AMP, EMA, checkpoints)
│   ├── utils.py               ← Logging, EarlyStopping, ModelEMA, helpers
│   └── models/
│       ├── backbone.py        ← Shared ResNet-50 backbone
│       ├── branch_a.py        ← CNN-FPN segmentation branch
│       ├── branch_b.py        ← Transformer hazard branch (LoRA)
│       ├── fusion.py          ← Bidirectional cross-attention fusion
│       ├── heads.py           ← Detection + mask assembly + UQ heads
│       ├── lora.py            ← LoRA layer injection
│       └── dbhdsnet.py        ← Full DBHDSNet model
│
├── scripts/
│   └── inspect_dataset.py     ← Run BEFORE training to verify dataset
│
├── checkpoints/               ← Auto-created: best.pt, last.pt, epoch_XXXX.pt
├── logs/                      ← Auto-created: .log files + TensorBoard
└── outputs/                   ← Auto-created: evaluation outputs
```

---

## Step 1 — Install Dependencies

```bash
# Create a fresh conda environment (recommended)
conda create -n dbhdsnet python=3.10 -y
conda activate dbhdsnet

# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
pip install -r requirements.txt
```

---

## Step 2 — Configure Paths

Open `config.py` and update **Section A**:

```python
# Line to update:
DATASET_ROOT = Path("/absolute/path/to/your/master_dataset")
```

Your dataset must have this structure (which you already have):
```
master_dataset/
├── train/
│   ├── images/    ← .jpg / .png files
│   └── labels/    ← .txt files (YOLO seg format)
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── master_data.yaml
```

---

## Step 3 — Inspect Dataset (Tiers Pre-Filled — Verify Only)

The `HAZARD_TIER_MAP` in `config.py` is **already filled** for all 38 classes
from your `master_data.yaml` (WHO/CPCB aligned). No manual tier entry needed.

```bash
# Update DATASET_ROOT in scripts/inspect_dataset.py, then:
python scripts/inspect_dataset.py
```

This prints your 38 classes with pre-assigned tiers and a validation summary:
- **T1 Sharps/Hazardous (7)**: blade, harris_uni_core, mercury_thermometer, needle, radioactive_objects, scalpel, syringe
- **T2 Infectious (18)**: bloody_objects, mask, n95, bandage, cotton_swab, covid_buffer, covid_test_case, gauze, iodine_swab, plastic_medical_bag, medical_gloves, reagent_tube, single_channel_pipette, transferpettor_glass, transferpettor_plastic, tweezer_metal, tweezer_plastic, medical_infusion_bag
- **T3 Pharmaceutical (11)**: oxygen_cylinder, capsule, covid_buffer_box, glass_bottle, harris_uni_core_cap, pill, plastic_medical_bottle, reagent_tube_cap, unguent, electronic_thermometer, drug_packaging
- **T4 General (2)**: paperbox, cap_plastic

Tier definitions:
- `1` = Sharps + High Hazard (needle-stick, mercury, radioactive) ← highest risk
- `2` = Infectious (used gloves, bandages, bloodied items)
- `3` = Pharmaceutical (medicine bottles, vials, IV bags)
- `4` = General (non-contaminated packaging)

---

## Step 4 — Smoke Test (Dry Run)

Before committing to full training, verify everything runs end-to-end:

```bash
python train.py --dry-run
```

This runs:
1. Dataset loading
2. Model forward pass
3. Loss computation
4. Backward pass
5. 10-step overfit test on one batch

Expected output:
```
  Step  1/10  loss = 8.4321
  Step  2/10  loss = 7.9812
  ...
  Step 10/10  loss = 4.2301
  ✓ PASS — Initial: 8.4321 → Final: 4.2301
```

If you see `⚠ WARN`, check your `HAZARD_TIER_MAP` and class names.

---

## Step 5 — Train

```bash
# Full training (150 epochs by default)
python train.py

# All terminal output is saved simultaneously to:
# logs/dbhdsnet_YYYYMMDD_HHMMSS.log
```

### Progress Bar Behaviour

Each epoch shows **two separate continuous bars**:
```
Epoch 0001/0150 [TRAIN] 100%|████████| 1265/1265 [08:23<00:00, loss=3.241, box=1.12, seg=0.89, haz=0.31, hier=0.44, lr=1.0e-04]
Epoch 0001/0150 [VAL]    100%|████████|  361/361  [02:11<00:00, loss=2.987]

  Epoch 0001/0150  │ Train: loss=3.241  seg=0.891  haz=0.312  hier=0.441
                   │ Val mAP@0.5=0.4821  │ LR=1.00e-03  │ ⏱ 0:10:34
```

Each bar is **continuous** — no reprinting per step, only the postfix updates.

---

## Step 6 — Resume After Interruption

```bash
# Resume from last.pt (automatic)
python train.py --resume

# Resume from a specific checkpoint
python train.py --resume --ckpt checkpoints/epoch_0060.pt
```

The trainer restores: epoch number, model weights, optimizer state,
scheduler state, EMA weights, early stopping counter, and best mAP.

---

## Step 7 — Monitor with TensorBoard

```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

Tracks per-epoch:
- `train/loss_total`, `train/loss_box`, `train/loss_seg`, `train/loss_hazard`, `train/loss_hierarchy`
- `val/mAP@0.5`, `val/mAP@0.75`, `val/loss_total`
- `lr`

---

## Overfitting Controls (built-in)

| Guard | Config key | Default |
|---|---|---|
| Weight decay (L2 reg) | `WEIGHT_DECAY` | 5e-4 |
| Label smoothing | `LABEL_SMOOTHING` | 0.1 |
| Mosaic augmentation | `AUG_MOSAIC` | 0.5 |
| MixUp augmentation | `AUG_MIXUP` | 0.1 |
| EMA model weights | `EMA` | True |
| Early stopping | `PATIENCE` | 25 epochs |
| Gradient clipping | `GRAD_CLIP` | 10.0 |
| BatchNorm frozen | `FREEZE_BN` | True |
| Backbone frozen (warm-up) | `WARMUP_EPOCHS` | 5 epochs |
| LoRA (reduced trainable params) | `LORA_RANK` | 16 |
| Focal loss (hard example mining) | `FOCAL_GAMMA` | 1.5 |
| Dropout in hazard head | `MC_DROPOUT_RATE` | 0.3 |

---

## Checkpoints Saved

```
checkpoints/
├── best.pt             ← Best validation mAP (used for final eval)
├── last.pt             ← Most recent epoch (used for resume)
├── epoch_0005.pt       ← Periodic save (every SAVE_EVERY epochs)
├── epoch_0010.pt
└── epoch_0015.pt       ← Only last 3 periodic kept (auto-pruned)
```

---

## GPU Memory Guide

| GPU VRAM | Recommended batch size | Image size |
|---|---|---|
| 8 GB  | 4 | 640 |
| 16 GB | 8 | 640 |
| 24 GB | 16 | 640 |
| 40 GB | 32 | 640 or 960 |

If you get OOM errors, reduce `BATCH_SIZE` in `config.py` → `TrainConfig`.

---

## Common Errors

**`ModuleNotFoundError: No module named 'timm'`**
→ `pip install timm`

**`CUDA out of memory`**
→ Reduce `BATCH_SIZE` in config.py, or set `AMP = True` (already default)

**`AssertionError: seq_len must be a perfect square`**
→ Your C4 feature map is not square. Ensure `IMG_SIZE` is divisible by 32.

**Loss is NaN**
→ Usually a learning rate issue. Check `BASE_LR` is not too high, and ensure `AMP = True` for numerical stability.

**`KeyError: 'names'` from YAML**
→ Your master_data.yaml uses a different key. Open it and check — it may use `nc:` and a list.
