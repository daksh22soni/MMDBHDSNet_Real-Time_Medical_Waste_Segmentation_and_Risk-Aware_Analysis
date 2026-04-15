"""
========================================================================
DBHDSNet — Phase 2 Configuration
ALL paths and hyperparameters are defined here.
Update DATASET_ROOT and HAZARD_TIER_MAP before first run.
========================================================================
"""

import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# SECTION A — PATHS  (UPDATE THESE TO MATCH YOUR SYSTEM)
# ════════════════════════════════════════════════════════════════════════

# Root of your master_dataset folder (the one containing train/valid/test + yaml)
DATASET_ROOT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/master_dataset")

# YAML file with class names (master_data.yaml)
YAML_PATH = DATASET_ROOT / "master_data.yaml"

# Output directories (auto-created at runtime)
PROJECT_ROOT  = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"

# Specific split paths (leave as-is if you followed standard layout)
TRAIN_IMG_DIR  = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR  = DATASET_ROOT / "train" / "labels"
VALID_IMG_DIR  = DATASET_ROOT / "valid" / "images"
VALID_LBL_DIR  = DATASET_ROOT / "valid" / "labels"
TEST_IMG_DIR   = DATASET_ROOT / "test"  / "images"
TEST_LBL_DIR   = DATASET_ROOT / "test"  / "labels"

# ════════════════════════════════════════════════════════════════════════
# SECTION B — HAZARD TIER MAPPING
# Map each of your 38 class names to a hazard tier (1–4).
# ── Tier 1 : SHARPS       (syringes, needles, scalpels, blades)
# ── Tier 2 : INFECTIOUS   (used bandages, gloves, swabs, gauze)
# ── Tier 3 : PHARMACEUTICAL (medicine bottles, vials, IV bags)
# ── Tier 4 : GENERAL      (packaging, cardboard, non-contaminated)
#
# ACTION REQUIRED: run  `python scripts/inspect_dataset.py`  first to
# print your exact 38 class names, then fill them in below.
# ════════════════════════════════════════════════════════════════════════

HAZARD_TIER_MAP: dict[str, int] = {
    # ════════════════════════════════════════════════════════════════
    # All 38 class names are taken VERBATIM from master_data.yaml.
    # Tier assignment follows WHO Safe Management of Health-Care Waste
    # (2014) and India CPCB Bio-Medical Waste Rules (2016/2018).
    #
    # ── Tier 1 — SHARPS + HIGHLY HAZARDOUS  ──────────────────────
    # Criteria: puncture risk (needle-stick), or extreme chemical /
    # radiological toxicity requiring Category-A disposal.
    # ── Tier 2 — INFECTIOUS  ──────────────────────────────────────
    # Criteria: direct blood / body-fluid contact OR laboratory use
    # with biological specimens (WHO Category 1 / 3).
    # ── Tier 3 — PHARMACEUTICAL / CHEMICAL  ──────────────────────
    # Criteria: chemical or pharmaceutical content — expired / used
    # medicines, reagent packaging, pressurised gas (WHO Category 4).
    # ── Tier 4 — GENERAL (lowest risk)  ──────────────────────────
    # Criteria: no biological, chemical, or sharps hazard; standard
    # municipal or recyclable waste stream.
    # ════════════════════════════════════════════════════════════════

    # ── Tier 1 (7 classes) ────────────────────────────────────────
    "radioactive_objects":        1,   # class 4  — WHO Cat.8 radiological;
                                       #            highest possible hazard
    "blade":                      1,   # class 6  — sharps; laceration risk
    "harris_uni_core":            1,   # class 14 — biopsy core needle sampler
    "mercury_thermometer":        1,   # class 17 — mercury; neurotoxic heavy metal
    "scalpel":                    1,   # class 25 — surgical sharp
    "syringe":                    1,   # class 27 — needle-stick risk
    "needle":                     1,   # class 37 — highest needle-stick risk

    # ── Tier 2 (18 classes) ───────────────────────────────────────
    "bloody_objects":             2,   # class 0  — blood-contaminated; direct infectious
    "mask":                       2,   # class 1  — used surgical PPE; respiratory droplets
    "n95":                        2,   # class 2  — used N95; aerosol/pathogen exposure
    "bandage":                    2,   # class 5  — wound-contact; blood/exudate
    "cotton_swab":                2,   # class 8  — specimen / wound-contact
    "covid_buffer":               2,   # class 9  — COVID-19 viral transport medium
    "covid_test_case":            2,   # class 11 — used COVID rapid-test cassette
    "gauze":                      2,   # class 12 — wound dressing; blood contact
    "iodine_swab":                2,   # class 16 — pre/post-procedure antiseptic swab
    "plastic_medical_bag":        2,   # class 20 — specimen / biohazard collection bag
    "medical_gloves":             2,   # class 22 — used gloves; body-fluid contact
    "reagent_tube":               2,   # class 23 — blood/specimen tube (e.g. Vacutainer)
    "single_channel_pipette":     2,   # class 26 — lab pipette; biological specimen contact
    "transferpettor_glass":       2,   # class 28 — glass transfer pipette; lab specimen
    "transferpettor_plastic":     2,   # class 29 — plastic transfer pipette; lab specimen
    "tweezer_metal":              2,   # class 30 — metal forceps; wound/tissue contact
    "tweezer_plastic":            2,   # class 31 — plastic forceps; lab/clinical use
    "medical_infusion_bag":       2,   # class 36 — used IV/infusion bag; patient contact

    # ── Tier 3 (11 classes) ───────────────────────────────────────
    "oxygen_cylinder":            3,   # class 3  — pressurised gas; WHO Cat.4 chemical
    "capsule":                    3,   # class 7  — oral pharmaceutical
    "covid_buffer_box":           3,   # class 10 — reagent kit outer packaging
    "glass_bottle":               3,   # class 13 — pharmaceutical / reagent glass vessel
    "harris_uni_core_cap":        3,   # class 15 — cap of biopsy sampler (no sharps risk)
    "pill":                       3,   # class 19 — loose tablet/pill; pharmaceutical waste
    "plastic_medical_bottle":     3,   # class 21 — drug bottle; pharmaceutical container
    "reagent_tube_cap":           3,   # class 24 — tube cap; minimal biological risk
    "unguent":                    3,   # class 32 — topical ointment/cream
    "electronic_thermometer":     3,   # class 33 — battery/electronic component waste
    "drug_packaging":             3,   # class 35 — blister packs, foil strips, cartons

    # ── Tier 4 (2 classes) ────────────────────────────────────────
    "paperbox":                   4,   # class 18 — cardboard outer box; no hazard
    "cap_plastic":                4,   # class 34 — plain plastic cap; recyclable
}
# VERIFICATION: 7 + 18 + 11 + 2 = 38 entries (matches nc in master_data.yaml)

# ════════════════════════════════════════════════════════════════════════
# SECTION C — MODEL HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════

class ModelConfig:
    # Input resolution (must be divisible by 32)
    IMG_SIZE        = 640

    # Number of visual classes (matches your YAML nc field)
    NUM_CLASSES     = 38

    # Number of hazard tiers
    NUM_HAZARD_TIERS = 4

    # Shared backbone
    BACKBONE        = "resnet50"       # resnet50 | resnet101
    BACKBONE_PRETRAINED = True
    FREEZE_BN       = True             # freeze BatchNorm stats

    # LoRA settings (applied to transformer branch attention layers)
    LORA_RANK       = 16
    LORA_ALPHA      = 32
    LORA_DROPOUT    = 0.05

    # FPN (Branch A)
    FPN_OUT_CHANNELS = 256
    FPN_SCALES       = [0, 1, 2]       # indices into [C3, C4, C5]

    # Transformer hazard branch (Branch B)
    VIT_MODEL        = "vit_small_patch16_224"
    VIT_PRETRAINED   = True
    VIT_EMBED_DIM    = 384             # ViT-Small hidden dim

    # Proto-mask head
    NUM_PROTO_MASKS  = 32              # K prototype masks

    # Cross-attention fusion
    FUSION_DIM       = 256
    FUSION_HEADS     = 8
    FUSION_DROPOUT   = 0.1

    # Uncertainty (MC-Dropout)
    MC_DROPOUT_RATE  = 0.3
    MC_FORWARD_PASSES = 20            # N forward passes for UQ

    # Detection head
    OBJ_PRIOR        = 0.01           # prior probability of objectness


# ════════════════════════════════════════════════════════════════════════
# SECTION D — TRAINING HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════

class TrainConfig:
    EPOCHS          = 150
    BATCH_SIZE      = 8               # reduce to 4 if OOM on <8GB GPU
    VAL_BATCH_SIZE  = 4
    NUM_WORKERS     = 4               # DataLoader workers
    PIN_MEMORY      = True

    # Optimizer (AdamW)
    BASE_LR         = 1e-3
    MIN_LR          = 1e-6
    WEIGHT_DECAY    = 5e-4
    MOMENTUM        = 0.937           # for SGD (unused with AdamW)
    BETAS           = (0.937, 0.999)

    # Scheduler (cosine annealing with warm-up)
    WARMUP_EPOCHS   = 5
    WARMUP_LR_START = 1e-6

    # Gradient
    GRAD_CLIP       = 10.0            # max gradient norm

    # Mixed precision (AMP)
    AMP             = True            # set False if GPU <8GB causes issues

    # Exponential moving average of model weights
    EMA             = True
    EMA_DECAY       = 0.9999
    EMA_UPDATES     = 0               # internal counter

    # Early stopping
    PATIENCE        = 25              # epochs without improvement
    MIN_DELTA       = 1e-4            # minimum mAP improvement

    # Checkpointing
    SAVE_EVERY      = 5               # save checkpoint every N epochs
    KEEP_LAST_N     = 3              # number of periodic checkpoints to keep

    # Label smoothing (for classification)
    LABEL_SMOOTHING = 0.1

    # Augmentation probabilities
    AUG_FLIP_H      = 0.5
    AUG_FLIP_V      = 0.0
    AUG_MOSAIC      = 0.5             # mosaic augmentation probability
    AUG_MIXUP       = 0.1
    AUG_SCALE       = (0.5, 1.5)      # random scale range
    AUG_HSV_H       = 0.015
    AUG_HSV_S       = 0.7
    AUG_HSV_V       = 0.4

    # Class weights (set None to compute from dataset; or provide manually)
    CLASS_WEIGHTS   = None            # auto-computed if None


# ════════════════════════════════════════════════════════════════════════
# SECTION E — LOSS FUNCTION WEIGHTS
# ════════════════════════════════════════════════════════════════════════

class LossConfig:
    # Box regression (CIoU)
    LAMBDA_BOX      = 7.5

    # Classification (Focal + label smoothing)
    LAMBDA_CLS      = 0.5

    # Objectness / confidence
    LAMBDA_OBJ      = 1.0

    # Segmentation mask (Dice + BCE)
    LAMBDA_SEG      = 2.0

    # Hazard tier classification (cross-entropy)
    LAMBDA_HAZARD   = 1.5

    # Hazard hierarchy penalty (cost-sensitive)
    LAMBDA_HIERARCHY = 3.0

    # Focal loss parameters
    FOCAL_GAMMA     = 1.5
    FOCAL_ALPHA     = 0.25

    # Hazard penalty matrix P[i,j] = cost of predicting tier j when true tier is i
    # Rows and cols: [Tier1=Sharps, Tier2=Infectious, Tier3=Pharma, Tier4=General]
    HAZARD_PENALTY_MATRIX = [
        # pred→  T1    T2    T3    T4
        [0.0,   2.0,  4.0,  10.0],  # true T1 (Sharps)
        [1.0,   0.0,  2.0,   8.0],  # true T2 (Infectious)
        [1.0,   1.5,  0.0,   6.0],  # true T3 (Pharmaceutical)
        [1.5,   1.5,  1.5,   0.0],  # true T4 (General)
    ]


# ════════════════════════════════════════════════════════════════════════
# SECTION F — EVALUATION THRESHOLDS
# ════════════════════════════════════════════════════════════════════════

class EvalConfig:
    CONF_THRESH     = 0.25
    NMS_IOU_THRESH  = 0.45
    MAX_DETECTIONS  = 300

    # Uncertainty flag threshold (predictions with U > this go to human review)
    UNCERTAINTY_THRESH = 0.20

    # mAP IoU thresholds
    MAP_IOU_50      = 0.50
    MAP_IOU_75      = 0.75


# ════════════════════════════════════════════════════════════════════════
# SECTION G — CONVENIENCE ACCESSOR
# ════════════════════════════════════════════════════════════════════════

class Config:
    DATA   = type("DATA", (), {
        "ROOT": DATASET_ROOT, "YAML": YAML_PATH,
        "TRAIN_IMG": TRAIN_IMG_DIR, "TRAIN_LBL": TRAIN_LBL_DIR,
        "VALID_IMG": VALID_IMG_DIR, "VALID_LBL": VALID_LBL_DIR,
        "TEST_IMG":  TEST_IMG_DIR,  "TEST_LBL":  TEST_LBL_DIR,
        "HAZARD_MAP": HAZARD_TIER_MAP,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LOG_DIR": LOG_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
    })()
    MODEL  = ModelConfig()
    TRAIN  = TrainConfig()
    LOSS   = LossConfig()
    EVAL   = EvalConfig()

    @staticmethod
    def make_dirs():
        for d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
