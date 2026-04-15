"""
========================================================================
DBHDSNet — Phase 4 Configuration
Federated Learning with Differential Privacy

Novel PhD contributions:
  1. LoRA-only federation — only LoRA + hazard-head parameters are
     communicated; backbone, FPN, BN are frozen locally at every
     client. This cuts communication cost by ~95% vs full-model FL.

  2. FedProx + FedNova hybrid aggregation — FedProx proximal term
     prevents client drift on non-IID medical waste data; FedNova
     normalised averaging corrects for heterogeneous local steps.

  3. Per-client DP-SGD with RDP accountant — each hospital client
     clips gradients to sensitivity S, adds calibrated Gaussian noise
     σ·S, and tracks its own (ε, δ)-budget via Rényi DP accounting.

  4. HIPAA / India DPDPA compliance report — auto-generated PDF
     documenting each client's privacy spend, aggregation protocol,
     and data residency guarantees.

  5. Client-drift-aware model selection — global model is evaluated
     on a held-out federated test partition; per-client contribution
     is re-weighted by a drift score so low-quality clients are
     down-weighted without being excluded.

Update SECTION A paths before first run.
========================================================================
"""

from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# SECTION A — PATHS  (UPDATE THESE)
# ════════════════════════════════════════════════════════════════════════

# Root of your master_dataset (same as Phase 2 / 3)
DATASET_ROOT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/master_dataset")
YAML_PATH    = DATASET_ROOT / "master_data.yaml"

# Path to Phase 2 best checkpoint (starting point for all clients)
PHASE2_CHECKPOINT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/phase2/checkpoints/best.pt")

# Phase 4 output root
PROJECT_ROOT   = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"

# Dataset split paths
TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATASET_ROOT / "train" / "labels"
VALID_IMG_DIR = DATASET_ROOT / "valid" / "images"
VALID_LBL_DIR = DATASET_ROOT / "valid" / "labels"
TEST_IMG_DIR  = DATASET_ROOT / "test"  / "images"
TEST_LBL_DIR  = DATASET_ROOT / "test"  / "labels"

# ════════════════════════════════════════════════════════════════════════
# SECTION B — FEDERATED CLIENT REGISTRY
# Each entry represents one hospital/lab site.
# Prefixes match the dataset prefix used in master_dataset
# (set in Phase 1 Part 2 pipeline: v1_, v5_, v6_, nv2_, nv3_, nv4_).
# v7 was entirely duplicate of v6 and was removed in Phase 1.
# ════════════════════════════════════════════════════════════════════════

CLIENTS: list[dict] = [
    {
        "client_id":   "hospital_A",
        "prefix":      "v1",
        "description": "General hospital — mixed clinical waste",
        "country":     "IN",
        "n_images_approx": 2500,   # approximate; exact count from dataset
    },
    {
        "client_id":   "hospital_B",
        "prefix":      "v5",
        "description": "COVID-19 reference lab — high infectious load",
        "country":     "IN",
        "n_images_approx": 2800,
    },
    {
        "client_id":   "hospital_C",
        "prefix":      "v6",
        "description": "Surgical centre — sharps-dominant",
        "country":     "IN",
        "n_images_approx": 2200,
    },
    {
        "client_id":   "lab_A",
        "prefix":      "nv2",
        "description": "Research laboratory — reagent tubes & pipettes",
        "country":     "IN",
        "n_images_approx": 1900,
    },
    {
        "client_id":   "lab_B",
        "prefix":      "nv3",
        "description": "Diagnostics lab — scalpel & specimen tubes",
        "country":     "IN",
        "n_images_approx": 1800,
    },
    {
        "client_id":   "clinic_A",
        "prefix":      "nv4",
        "description": "Out-patient clinic — pharmaceutical dominant",
        "country":     "IN",
        "n_images_approx": 1600,
    },
]

# Number of active clients (set to len(CLIENTS) to use all)
N_CLIENTS = len(CLIENTS)   # 6

# ════════════════════════════════════════════════════════════════════════
# SECTION C — HAZARD TIER MAP (verbatim from master_data.yaml)
# ════════════════════════════════════════════════════════════════════════

HAZARD_TIER_MAP: dict[str, int] = {
    # T1 — SHARPS / HIGH-HAZARD (7)
    "radioactive_objects":        1,
    "blade":                      1,
    "harris_uni_core":            1,
    "mercury_thermometer":        1,
    "scalpel":                    1,
    "syringe":                    1,
    "needle":                     1,
    # T2 — INFECTIOUS (18)
    "bloody_objects":             2,
    "mask":                       2,
    "n95":                        2,
    "bandage":                    2,
    "cotton_swab":                2,
    "covid_buffer":               2,
    "covid_test_case":            2,
    "gauze":                      2,
    "iodine_swab":                2,
    "plastic_medical_bag":        2,
    "medical_gloves":             2,
    "reagent_tube":               2,
    "single_channel_pipette":     2,
    "transferpettor_glass":       2,
    "transferpettor_plastic":     2,
    "tweezer_metal":              2,
    "tweezer_plastic":            2,
    "medical_infusion_bag":       2,
    # T3 — PHARMACEUTICAL (11)
    "oxygen_cylinder":            3,
    "capsule":                    3,
    "covid_buffer_box":           3,
    "glass_bottle":               3,
    "harris_uni_core_cap":        3,
    "pill":                       3,
    "plastic_medical_bottle":     3,
    "reagent_tube_cap":           3,
    "unguent":                    3,
    "electronic_thermometer":     3,
    "drug_packaging":             3,
    # T4 — GENERAL (2)
    "paperbox":                   4,
    "cap_plastic":                4,
}
# T1=7, T2=18, T3=11, T4=2  →  total 38 (matches nc in master_data.yaml)


# ════════════════════════════════════════════════════════════════════════
# SECTION D — PHASE 2 MODEL CONFIG (must match trained model exactly)
# ════════════════════════════════════════════════════════════════════════

class ModelConfig:
    IMG_SIZE          = 640
    NUM_CLASSES       = 38
    NUM_HAZARD_TIERS  = 4
    BACKBONE          = "resnet50"
    BACKBONE_PRETRAINED = True
    FREEZE_BN         = True
    LORA_RANK         = 16
    LORA_ALPHA        = 32
    LORA_DROPOUT      = 0.05
    FPN_OUT_CHANNELS  = 256
    FPN_SCALES        = [0, 1, 2]
    VIT_MODEL         = "vit_small_patch16_224"
    VIT_PRETRAINED    = True
    VIT_EMBED_DIM     = 384
    NUM_PROTO_MASKS   = 32
    FUSION_DIM        = 256
    FUSION_HEADS      = 8
    FUSION_DROPOUT    = 0.1
    MC_DROPOUT_RATE   = 0.3
    MC_FORWARD_PASSES = 20
    OBJ_PRIOR         = 0.01


# ════════════════════════════════════════════════════════════════════════
# SECTION E — FEDERATED LEARNING CONFIG
# ════════════════════════════════════════════════════════════════════════

class FedConfig:
    # ── Global training rounds ────────────────────────────────────────
    NUM_ROUNDS         = 100       # total federated rounds
    CLIENTS_PER_ROUND  = 6        # how many clients participate per round
                                   # set equal to N_CLIENTS for full participation
    MIN_CLIENTS        = 2         # minimum clients to proceed with aggregation

    # ── Local training per client per round ───────────────────────────
    LOCAL_EPOCHS       = 3        # local SGD epochs before aggregation
    LOCAL_BATCH_SIZE   = 16        # reduce to 4 if <8 GB VRAM
    LOCAL_LR           = 1e-4     # local fine-tune LR (lower than Phase 2)
    LOCAL_WEIGHT_DECAY = 5e-4
    LOCAL_GRAD_CLIP    = 10.0

    # ── LoRA-only communication (PhD novel contribution) ──────────────
    # Only LoRA matrices (lora_A, lora_B) + hazard head + fusion gating
    # are communicated. Backbone / BN are always frozen locally.
    # This reduces communication payload by ~95% vs full-model FL.
    LORA_ONLY_COMM     = True
    COMM_PARAM_PATTERNS = [        # param name substrings to communicate
        "lora_A", "lora_B",
        "hazard_head",
        "fusion.gate", "fusion.refine",
    ]

    # ── Aggregation algorithm ─────────────────────────────────────────
    # "fedavg"    — McMahan et al. (2017)  — weighted by n_samples
    # "fedprox"   — Li et al. (2020)       — proximal regularisation
    # "fednova"   — Wang et al. (2021)     — normalised averaging
    # "fedadam"   — Reddi et al. (2021)    — server-side Adam
    AGGREGATION        = "fedprox"   # recommended for non-IID medical data

    # FedProx proximal coefficient μ — larger μ = closer to FedAvg
    # Tune: 0.001 (loose) → 0.1 (tight). Start with 0.01.
    FEDPROX_MU         = 0.01

    # FedNova correction — τ effective steps per client
    FEDNOVA_RHO        = 0.9       # momentum for local objective normalisation

    # FedAdam server learning rate and momentum
    FEDADAM_LR         = 1e-3
    FEDADAM_BETA1      = 0.9
    FEDADAM_BETA2      = 0.999
    FEDADAM_EPS        = 1e-8
    FEDADAM_TAU        = 1e-3      # gradient adaptivity

    # ── Client weighting ─────────────────────────────────────────────
    # "uniform"      — equal weight for all clients
    # "n_samples"    — weighted by local dataset size (standard FedAvg)
    # "drift_aware"  — re-weight by inverse client drift score (novel)
    CLIENT_WEIGHTING   = "drift_aware"

    # Drift score smoothing factor (EMA across rounds)
    DRIFT_EMA_ALPHA    = 0.3

    # ── Evaluation ───────────────────────────────────────────────────
    EVAL_EVERY         = 5        # evaluate global model every N rounds
    SAVE_EVERY         = 10       # save checkpoint every N rounds
    KEEP_LAST_N        = 3        # periodic checkpoints to keep

    # ── Early stopping ────────────────────────────────────────────────
    PATIENCE           = 20       # rounds without improvement
    MIN_DELTA          = 1e-4


# ════════════════════════════════════════════════════════════════════════
# SECTION F — DIFFERENTIAL PRIVACY CONFIG
# ════════════════════════════════════════════════════════════════════════

class DPConfig:
    # ── Enable / disable DP ──────────────────────────────────────────
    ENABLE_DP          = True

    # ── Per-client DP-SGD parameters ─────────────────────────────────
    # Gradient clipping threshold (L2 sensitivity S)
    MAX_GRAD_NORM      = 1.0      # S — clip each per-sample gradient to this L2 norm

    # Noise multiplier σ: noise_std = σ × S / batch_size
    NOISE_MULTIPLIER   = 1.1      # σ — higher = more privacy, less accuracy

    # ── Privacy budget ────────────────────────────────────────────────
    # Target (ε, δ) per client per experiment
    TARGET_EPSILON     = 8.0      # ε — privacy loss bound (lower = stronger privacy)
    TARGET_DELTA       = 1e-5     # δ — failure probability (≤ 1/n_train recommended)

    # ── RDP Accountant ────────────────────────────────────────────────
    # Orders α for Rényi Differential Privacy accounting
    RDP_ORDERS         = list(range(2, 512))  + [float("inf")]

    # Accountant backend: "rdp" | "prv"
    ACCOUNTANT         = "rdp"

    # ── Secure aggregation ───────────────────────────────────────────
    # In production: replace with cryptographic secure aggregation.
    # In simulation: use masked aggregation (additive secret sharing).
    SECURE_AGG         = True     # simulate secure aggregation

    # ── Per-tier DP tightening ────────────────────────────────────────
    # For T1/T2 images, apply tighter DP (lower noise, more local steps)
    # since these are most sensitive for model performance.
    # T1 images from sharps/radioactive are also most likely PHI-adjacent.
    TIER_DP_MULTIPLIERS = {
        1: 1.5,    # Sharps: extra noise (higher sensitivity label)
        2: 1.2,    # Infectious
        3: 1.0,    # Pharmaceutical: baseline
        4: 0.8,    # General: less noise needed
    }

    # ── Compliance ────────────────────────────────────────────────────
    COMPLIANCE_STANDARDS = ["HIPAA", "India_DPDPA_2023", "ISO_27701"]
    AUDIT_REPORT_FORMAT  = "pdf"   # "pdf" | "json"


# ════════════════════════════════════════════════════════════════════════
# SECTION G — LOCAL TRAINING CONFIG (per client, per round)
# ════════════════════════════════════════════════════════════════════════

class LocalTrainConfig:
    # Augmentation (lighter than Phase 2 — clients have smaller shards)
    AUG_FLIP_H      = 0.5
    AUG_FLIP_V      = 0.0
    AUG_MOSAIC      = 0.2         # reduced mosaic (smaller local datasets)
    AUG_MIXUP       = 0.05
    AUG_SCALE       = (0.7, 1.2)
    AUG_HSV_H       = 0.015
    AUG_HSV_S       = 0.7
    AUG_HSV_V       = 0.4

    # Loss weights (identical to Phase 2)
    LAMBDA_BOX      = 7.5
    LAMBDA_CLS      = 0.5
    LAMBDA_OBJ      = 1.0
    LAMBDA_SEG      = 2.0
    LAMBDA_HAZARD   = 1.5
    LAMBDA_HIERARCHY = 3.0
    FOCAL_GAMMA     = 1.5
    FOCAL_ALPHA     = 0.25

    LABEL_SMOOTHING = 0.1
    AMP             = True
    PIN_MEMORY      = True
    NUM_WORKERS     = 2           # lower than Phase 2 (parallel client sims)


# ════════════════════════════════════════════════════════════════════════
# SECTION H — EVALUATION CONFIG
# ════════════════════════════════════════════════════════════════════════

class EvalConfig:
    CONF_THRESH     = 0.25
    NMS_IOU_THRESH  = 0.45
    MAX_DETECTIONS  = 300
    MAP_IOU_50      = 0.50
    MAP_IOU_75      = 0.75


# ════════════════════════════════════════════════════════════════════════
# SECTION I — CONVENIENCE ACCESSOR
# ════════════════════════════════════════════════════════════════════════

class Config:
    DATA  = type("DATA", (), {
        "ROOT":          DATASET_ROOT,
        "YAML":          YAML_PATH,
        "TRAIN_IMG":     TRAIN_IMG_DIR,  "TRAIN_LBL": TRAIN_LBL_DIR,
        "VALID_IMG":     VALID_IMG_DIR,  "VALID_LBL": VALID_LBL_DIR,
        "TEST_IMG":      TEST_IMG_DIR,   "TEST_LBL":  TEST_LBL_DIR,
        "HAZARD_MAP":    HAZARD_TIER_MAP,
        "CLIENTS":       CLIENTS,
        "N_CLIENTS":     N_CLIENTS,
        "PHASE2_CKPT":   PHASE2_CHECKPOINT,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LOG_DIR":       LOG_DIR,
        "OUTPUT_DIR":    OUTPUT_DIR,
    })()
    MODEL = ModelConfig()
    FED   = FedConfig()
    DP    = DPConfig()
    LOCAL = LocalTrainConfig()
    EVAL  = EvalConfig()

    # Phase 2 compatibility shims
    TRAIN = type("TRAIN", (), {
        "BATCH_SIZE":    FedConfig.LOCAL_BATCH_SIZE,
        "VAL_BATCH_SIZE": 4,
        "NUM_WORKERS":   LocalTrainConfig.NUM_WORKERS,
        "PIN_MEMORY":    True,
        "AUG_MOSAIC":    LocalTrainConfig.AUG_MOSAIC,
        "AUG_FLIP_H":    LocalTrainConfig.AUG_FLIP_H,
        "AUG_FLIP_V":    LocalTrainConfig.AUG_FLIP_V,
        "LABEL_SMOOTHING": LocalTrainConfig.LABEL_SMOOTHING,
    })()
    LOSS = type("LOSS", (), {
        "LAMBDA_BOX":      LocalTrainConfig.LAMBDA_BOX,
        "LAMBDA_CLS":      LocalTrainConfig.LAMBDA_CLS,
        "LAMBDA_OBJ":      LocalTrainConfig.LAMBDA_OBJ,
        "LAMBDA_SEG":      LocalTrainConfig.LAMBDA_SEG,
        "LAMBDA_HAZARD":   LocalTrainConfig.LAMBDA_HAZARD,
        "LAMBDA_HIERARCHY": LocalTrainConfig.LAMBDA_HIERARCHY,
        "FOCAL_GAMMA":     LocalTrainConfig.FOCAL_GAMMA,
        "FOCAL_ALPHA":     LocalTrainConfig.FOCAL_ALPHA,
        "HAZARD_PENALTY_MATRIX": [
            [0.0, 2.0, 4.0, 10.0],
            [1.0, 0.0, 2.0,  8.0],
            [1.0, 1.5, 0.0,  6.0],
            [1.5, 1.5, 1.5,  0.0],
        ],
    })()

    @staticmethod
    def make_dirs():
        for d in [
            CHECKPOINT_DIR / "global",
            CHECKPOINT_DIR / "clients",
            LOG_DIR,
            OUTPUT_DIR / "convergence",
            OUTPUT_DIR / "privacy_audit",
            OUTPUT_DIR / "per_client",
            OUTPUT_DIR / "visualisations",
        ]:
            d.mkdir(parents=True, exist_ok=True)


CFG4 = Config()
