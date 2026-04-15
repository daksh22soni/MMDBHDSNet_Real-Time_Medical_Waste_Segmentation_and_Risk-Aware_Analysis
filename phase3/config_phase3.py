"""
========================================================================
DBHDSNet — Phase 3 Configuration
Phase 3a : Uncertainty Quantification (MC-Dropout + Deep Ensembles
           + Temperature Scaling + Human-in-the-Loop protocol)
Phase 3b : ContamRisk-GNN (scene graph → bin-level risk score)

ALL paths and hyperparameters defined here.
Update Section A paths before first run.
========================================================================
"""

from pathlib import Path
import torch

# ════════════════════════════════════════════════════════════════════════
# SECTION A — PATHS  (UPDATE THESE)
# ════════════════════════════════════════════════════════════════════════

# Root of your master_dataset (same as Phase 2)
DATASET_ROOT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/master_dataset")
YAML_PATH    = DATASET_ROOT / "master_data.yaml"

# Path to your best Phase 2 checkpoint (best.pt from Phase 2 training)
PHASE2_CHECKPOINT = Path("C:/Users/gahan/Documents/Daksh/sem 6/Deep Learning/dbhdsnet_project/phase2/checkpoints/best.pt")

# Phase 3 output directories
PROJECT_ROOT   = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"

# Dataset split paths (identical structure to Phase 2)
TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATASET_ROOT / "train" / "labels"
VALID_IMG_DIR = DATASET_ROOT / "valid" / "images"
VALID_LBL_DIR = DATASET_ROOT / "valid" / "labels"
TEST_IMG_DIR  = DATASET_ROOT / "test"  / "images"
TEST_LBL_DIR  = DATASET_ROOT / "test"  / "labels"

# ════════════════════════════════════════════════════════════════════════
# SECTION B — HAZARD TIER MAP (same as Phase 2 — fill with your classes)
# ════════════════════════════════════════════════════════════════════════

HAZARD_TIER_MAP: dict[str, int] = {
    # ════════════════════════════════════════════════════════════════
    # All 38 class names taken VERBATIM from master_data.yaml (nc=38).
    # Tier assignment: WHO Safe Management of Health-Care Waste (2014)
    #                  + India CPCB Bio-Medical Waste Rules (2016/2018).
    #
    # T1 SHARPS/HIGH-HAZARD  — needle-stick, toxic metal, radioactive
    # T2 INFECTIOUS          — blood/body-fluid or specimen contact
    # T3 PHARMACEUTICAL      — drug/chemical/pressurised gas content
    # T4 GENERAL             — no biological, chemical, sharps hazard
    # ════════════════════════════════════════════════════════════════

    # ── Tier 1 — SHARPS / HIGH-HAZARD (7 classes) ───────────────────
    "radioactive_objects":        1,   # class  4 — WHO Cat.8 radiological
    "blade":                      1,   # class  6 — surgical sharp / laceration
    "harris_uni_core":            1,   # class 14 — biopsy core-needle sampler
    "mercury_thermometer":        1,   # class 17 — mercury; neurotoxic heavy metal
    "scalpel":                    1,   # class 25 — surgical sharp
    "syringe":                    1,   # class 27 — needle-stick risk
    "needle":                     1,   # class 37 — highest needle-stick risk

    # ── Tier 2 — INFECTIOUS (18 classes) ────────────────────────────
    "bloody_objects":             2,   # class  0 — blood-contaminated; direct infectious
    "mask":                       2,   # class  1 — used surgical PPE; respiratory droplets
    "n95":                        2,   # class  2 — used N95; aerosol/pathogen exposure
    "bandage":                    2,   # class  5 — wound-contact; blood/exudate
    "cotton_swab":                2,   # class  8 — specimen/wound-contact
    "covid_buffer":               2,   # class  9 — COVID-19 viral transport medium
    "covid_test_case":            2,   # class 11 — used COVID rapid-test cassette
    "gauze":                      2,   # class 12 — wound dressing; blood contact
    "iodine_swab":                2,   # class 16 — pre/post-procedure antiseptic swab
    "plastic_medical_bag":        2,   # class 20 — specimen/biohazard collection bag
    "medical_gloves":             2,   # class 22 — used gloves; body-fluid contact
    "reagent_tube":               2,   # class 23 — blood/specimen tube (Vacutainer)
    "single_channel_pipette":     2,   # class 26 — lab pipette; biological specimen
    "transferpettor_glass":       2,   # class 28 — glass transfer pipette; lab specimen
    "transferpettor_plastic":     2,   # class 29 — plastic transfer pipette; lab specimen
    "tweezer_metal":              2,   # class 30 — metal forceps; wound/tissue contact
    "tweezer_plastic":            2,   # class 31 — plastic forceps; clinical/lab use
    "medical_infusion_bag":       2,   # class 36 — used IV/infusion bag; patient contact

    # ── Tier 3 — PHARMACEUTICAL / CHEMICAL (11 classes) ─────────────
    "oxygen_cylinder":            3,   # class  3 — pressurised gas; WHO Cat.4 chemical
    "capsule":                    3,   # class  7 — oral pharmaceutical
    "covid_buffer_box":           3,   # class 10 — reagent kit outer packaging
    "glass_bottle":               3,   # class 13 — pharmaceutical/reagent glass vessel
    "harris_uni_core_cap":        3,   # class 15 — biopsy sampler cap (no sharps risk)
    "pill":                       3,   # class 19 — loose tablet/pill; pharmaceutical
    "plastic_medical_bottle":     3,   # class 21 — drug bottle; pharmaceutical container
    "reagent_tube_cap":           3,   # class 24 — tube cap; minimal biological risk
    "unguent":                    3,   # class 32 — topical ointment/cream
    "electronic_thermometer":     3,   # class 33 — battery/electronic component waste
    "drug_packaging":             3,   # class 35 — blister packs, foil strips, cartons

    # ── Tier 4 — GENERAL (2 classes) ────────────────────────────────
    "paperbox":                   4,   # class 18 — cardboard outer box; no hazard
    "cap_plastic":                4,   # class 34 — plain plastic cap; recyclable
}
# VERIFICATION: T1=7 + T2=18 + T3=11 + T4=2 = 38 (matches nc in master_data.yaml)

# ════════════════════════════════════════════════════════════════════
# WHO/CPCB CROSS-CONTAMINATION RULES — used in ContamRisk-GNN
# ════════════════════════════════════════════════════════════════════
#
# Format: (tier_a, tier_b) → risk_multiplier (symmetric lookup)
# The ContamRisk-GNN uses this to compute edge-level contamination
# risk when two waste items co-occur in the same disposal bin.
#
# Clinical basis (WHO 2014 / CPCB 2016):
#   T1 (blade, syringe, scalpel, needle, mercury_thermometer,
#       harris_uni_core, radioactive_objects)
#   ↕  next to T4 (paperbox, cap_plastic)
#   → 3.0× : A sharps item contaminates adjacent general waste,
#             requiring upgrade of entire bin to Category-A disposal.
#
#   T1 ↕ T3 → 2.5× : Pharmaceutical containers near sharps risk
#             mercury/radioactive cross-leaching into drug packaging.
#
#   T1 ↕ T2 → 2.0× : Two high-hazard categories co-located; combined
#             needle-stick + biological exposure risk.
#
#   T2 ↕ T4 → 2.0× : Used gloves/gauze/swabs contaminate general
#             waste — entire bin must be treated as bio-hazardous.
#
#   T2 ↕ T3 → 1.5× : Infectious items near drug bottles; body fluids
#             may enter pharmaceutical containers.
#
#   T3 ↕ T4 → 1.2× : Drug packaging near general waste; low but
#             non-zero pharmaceutical leachate risk.
#
#   Same tier → 1.0× : Baseline risk (no escalation multiplier).
# ════════════════════════════════════════════════════════════════════
CROSS_CONTAMINATION_RULES: dict[tuple, float] = {
    (1, 4): 3.0,   # Sharps/Hazardous near General        → Category-A bin upgrade
    (1, 3): 2.5,   # Sharps near Pharmaceutical           → mercury/radioactive leach risk
    (1, 2): 2.0,   # Sharps near Infectious               → combined needle + biohazard
    (2, 4): 2.0,   # Infectious near General              → full bin bio-contamination
    (2, 3): 1.5,   # Infectious near Pharmaceutical       → body-fluid + drug exposure
    (3, 4): 1.2,   # Pharmaceutical near General          → pharmaceutical leachate
    (1, 1): 1.0,   # Same tier — no additional multiplier
    (2, 2): 1.0,
    (3, 3): 1.0,
    (4, 4): 1.0,
}

# ════════════════════════════════════════════════════════════════════
# CLASS-SPECIFIC HIGH-PRIORITY PAIRS (used in GNN loss weighting)
# These specific class combinations from YOUR dataset carry the
# highest clinical contamination risk and are up-weighted in training.
# ════════════════════════════════════════════════════════════════════
HIGH_RISK_CLASS_PAIRS: list[tuple] = [
    # (class_name_a, class_name_b, risk_weight)
    # — Sharps + General waste (T1 + T4): mandatory Category-A upgrade
    ("needle",              "paperbox",           3.0),
    ("syringe",             "paperbox",           3.0),
    ("blade",               "cap_plastic",        3.0),
    ("scalpel",             "paperbox",           3.0),
    # — Sharps + Infectious (T1 + T2): combined needle-stick + biohazard
    ("needle",              "bloody_objects",     2.8),
    ("syringe",             "bloody_objects",     2.8),
    ("needle",              "medical_gloves",     2.5),
    ("syringe",             "gauze",              2.5),
    ("blade",               "bandage",            2.5),
    # — Radioactive + any (T1 T1): requires specialist radiological disposal
    ("radioactive_objects", "syringe",            3.0),
    ("radioactive_objects", "covid_buffer",       3.0),
    # — Mercury thermometer near pharmaceutical (T1 + T3): chemical cross-contamination
    ("mercury_thermometer", "plastic_medical_bottle", 2.5),
    ("mercury_thermometer", "glass_bottle",       2.5),
    # — Used COVID items + general (T2 + T4): bio-contamination escalation
    ("covid_buffer",        "paperbox",           2.0),
    ("covid_test_case",     "cap_plastic",        2.0),
    ("covid_test_case",     "paperbox",           2.0),
    # — Infectious + pharmaceutical (T2 + T3): body-fluid in drug containers
    ("bloody_objects",      "plastic_medical_bottle", 2.0),
    ("reagent_tube",        "drug_packaging",     1.8),
    ("medical_gloves",      "pill",               1.8),
]

# ════════════════════════════════════════════════════════════════════════
# SECTION C — PHASE 2 MODEL CONFIG (must match what was trained)
# ════════════════════════════════════════════════════════════════════════

class Phase2ModelConfig:
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
    VIT_EMBED_DIM     = 384
    NUM_PROTO_MASKS   = 32
    FUSION_DIM        = 256
    FUSION_HEADS      = 8
    FUSION_DROPOUT    = 0.1
    MC_DROPOUT_RATE   = 0.3
    MC_FORWARD_PASSES = 20

# ════════════════════════════════════════════════════════════════════════
# SECTION D — PHASE 3a: UNCERTAINTY QUANTIFICATION CONFIG
# ════════════════════════════════════════════════════════════════════════

class UQConfig:
    # ── MC-Dropout settings ───────────────────────────────────────────
    MC_DROPOUT_RATE      = 0.3     # dropout rate during stochastic inference
    MC_N_PASSES          = 30     # number of stochastic forward passes
    MC_BATCH_SIZE        = 4      # batch size during MC inference (memory bound)

    # ── Deep Ensemble settings ────────────────────────────────────────
    N_ENSEMBLE_MEMBERS   = 5      # number of independently trained models
    ENSEMBLE_LR          = 5e-5   # fine-tune each member from Phase 2 best.pt
    ENSEMBLE_EPOCHS      = 30     # epochs per ensemble member fine-tune
    ENSEMBLE_SEED_BASE   = 100    # seeds: 100, 101, 102, 103, 104

    # ── Temperature Scaling (post-hoc calibration) ────────────────────
    TEMP_INIT            = 1.5    # initial temperature (>1 softens predictions)
    TEMP_LR              = 0.01
    TEMP_MAX_ITER        = 1000
    TEMP_CONVERGENCE     = 1e-5

    # ── Human-in-the-Loop protocol ────────────────────────────────────
    # Predictions with epistemic uncertainty > threshold → flagged for review
    UNCERTAINTY_THRESH_LOW  = 0.10  # green zone: auto-accept
    UNCERTAINTY_THRESH_HIGH = 0.25  # red zone: mandatory human review
    # Between low and high = "amber zone": log but continue

    # Tier-conditional flagging thresholds — tighter for higher-risk tiers.
    # Calibrated against your dataset's class distribution:
    #   T1 (7 classes)  : blade, scalpel, needle, syringe, harris_uni_core,
    #                     mercury_thermometer, radioactive_objects
    #   T2 (18 classes) : bloody_objects, mask, n95, bandage, cotton_swab,
    #                     covid_buffer, covid_test_case, gauze, iodine_swab,
    #                     plastic_medical_bag, medical_gloves, reagent_tube,
    #                     single_channel_pipette, transferpettor_glass/plastic,
    #                     tweezer_metal/plastic, medical_infusion_bag
    #   T3 (11 classes) : oxygen_cylinder, capsule, covid_buffer_box,
    #                     glass_bottle, harris_uni_core_cap, pill,
    #                     plastic_medical_bottle, reagent_tube_cap,
    #                     unguent, electronic_thermometer, drug_packaging
    #   T4 (2 classes)  : paperbox, cap_plastic
    TIER_THRESHOLDS = {
        1: {"low": 0.05, "high": 0.12},  # T1 Sharps/Hazardous — tightest (7 classes)
        2: {"low": 0.08, "high": 0.18},  # T2 Infectious       — tight   (18 classes)
        3: {"low": 0.10, "high": 0.22},  # T3 Pharmaceutical   — moderate (11 classes)
        4: {"low": 0.15, "high": 0.30},  # T4 General          — lenient  (2 classes)
    }

    # ── Calibration metrics ───────────────────────────────────────────
    ECE_N_BINS           = 15
    RELIABILITY_DIAGRAM_BINS = 15

    # ── Training (temperature head fine-tune) ─────────────────────────
    BATCH_SIZE           = 8
    NUM_WORKERS          = 4

# ════════════════════════════════════════════════════════════════════════
# SECTION E — PHASE 3b: ContamRisk-GNN CONFIG
# ════════════════════════════════════════════════════════════════════════

class GNNConfig:
    # ── Scene graph construction ──────────────────────────────────────
    # Two detected items are connected if their bounding boxes overlap
    # by more than this IoU threshold OR their centres are within
    # proximity_distance_norm of each other (normalised image coords).
    PROXIMITY_IOU_THRESH    = 0.05   # box IoU threshold for edge creation
    PROXIMITY_DIST_NORM     = 0.15   # centre-distance threshold (0–1)

    # ── Node features ─────────────────────────────────────────────────
    # Each node = one detected waste item.
    # Node feature vector:
    #   [class_one_hot(38), hazard_tier_one_hot(4),
    #    cx, cy, w, h,
    #    confidence_score,
    #    uncertainty_score]
    # Total: 38 + 4 + 4 + 1 + 1 = 48
    NODE_FEAT_DIM       = 48

    # ── Edge features ─────────────────────────────────────────────────
    # Edge feature vector:
    #   [relative_cx, relative_cy, log(w1/w2), log(h1/h2),
    #    iou, centre_distance,
    #    cross_contamination_risk_scalar]
    # Total: 7
    EDGE_FEAT_DIM       = 7

    # ── GNN architecture ──────────────────────────────────────────────
    GNN_HIDDEN_DIM      = 128
    GNN_N_LAYERS        = 4      # number of message-passing layers
    GNN_HEADS           = 4      # attention heads (GAT layers)
    GNN_DROPOUT         = 0.2
    GNN_LAYER_TYPE      = "GAT"  # "GAT" | "GCN" | "GraphSAGE"
    GNN_RESIDUAL        = True   # skip connections between layers
    GNN_NORM            = "batch"  # "batch" | "layer" | "none"

    # ── Risk score head ───────────────────────────────────────────────
    # Outputs:
    #   item_risk   : per-node risk score  (0–1)
    #   bin_risk    : global graph-level risk (0–1) via readout
    #   risk_class  : 4-class risk category (Low/Medium/High/Critical)
    RISK_CLASSES        = 4      # Low | Medium | High | Critical
    RISK_THRESHOLDS     = [0.25, 0.50, 0.75]  # Low<0.25, Med<0.50, High<0.75, Crit≥0.75

    # ── Training ─────────────────────────────────────────────────────
    EPOCHS              = 100
    BATCH_SIZE          = 16     # number of scene graphs per batch
    LR                  = 3e-4
    WEIGHT_DECAY        = 1e-4
    PATIENCE            = 20     # early stopping
    WARMUP_EPOCHS       = 5
    GRAD_CLIP           = 5.0
    AMP                 = True
    SAVE_EVERY          = 10
    KEEP_LAST_N         = 3
    NUM_WORKERS         = 4

    # ── Loss weights ─────────────────────────────────────────────────
    LAMBDA_ITEM_RISK    = 1.0   # per-node risk regression (MSE)
    LAMBDA_BIN_RISK     = 2.0   # graph-level risk (MSE, higher weight)
    LAMBDA_RISK_CLS     = 1.5   # risk category classification (CE)
    LAMBDA_CONTAM       = 1.0   # cross-contamination auxiliary loss

    # ── Scene graph generation (synthetic training data from Phase 2) ─
    # How many synthetic scenes to generate per real image
    # (augmentation: randomise item positions within a scene)
    SYNTHETIC_SCENES_PER_IMAGE = 3
    MIN_ITEMS_PER_SCENE = 2
    MAX_ITEMS_PER_SCENE = 20

# ════════════════════════════════════════════════════════════════════════
# SECTION F — EVALUATION CONFIG
# ════════════════════════════════════════════════════════════════════════

class EvalConfig:
    CONF_THRESH       = 0.25
    NMS_IOU_THRESH    = 0.45
    MAX_DETECTIONS    = 300
    # UQ flag threshold (used for HITL protocol at inference)
    UQ_FLAG_THRESH    = 0.20

# ════════════════════════════════════════════════════════════════════════
# SECTION G — CONVENIENCE ACCESSOR
# ════════════════════════════════════════════════════════════════════════

class Config:
    DATA  = type("DATA", (), {
        "ROOT":          DATASET_ROOT,
        "YAML":          YAML_PATH,
        "TRAIN_IMG":     TRAIN_IMG_DIR,
        "TRAIN_LBL":     TRAIN_LBL_DIR,
        "VALID_IMG":     VALID_IMG_DIR,
        "VALID_LBL":     VALID_LBL_DIR,
        "TEST_IMG":      TEST_IMG_DIR,
        "TEST_LBL":      TEST_LBL_DIR,
        "HAZARD_MAP":    HAZARD_TIER_MAP,
        "CONTAMINATION_RULES": CROSS_CONTAMINATION_RULES,
        "HIGH_RISK_PAIRS":    HIGH_RISK_CLASS_PAIRS,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LOG_DIR":       LOG_DIR,
        "OUTPUT_DIR":    OUTPUT_DIR,
        "PHASE2_CKPT":   PHASE2_CHECKPOINT,
    })()
    MODEL = Phase2ModelConfig()
    UQ    = UQConfig()
    GNN   = GNNConfig()
    EVAL  = EvalConfig()

    # Shims for Phase 2 compatibility
    TRAIN = type("TRAIN", (), {
        "BATCH_SIZE":    UQConfig.BATCH_SIZE,
        "VAL_BATCH_SIZE":4,
        "NUM_WORKERS":   UQConfig.NUM_WORKERS,
        "PIN_MEMORY":    True,
        "AUG_MOSAIC":    0.0,   # no mosaic during UQ eval
        "AUG_FLIP_H":    0.0,
        "AUG_FLIP_V":    0.0,
        "LABEL_SMOOTHING": 0.0,
    })()

    @staticmethod
    def make_dirs():
        for d in [
            CHECKPOINT_DIR / "uq",
            CHECKPOINT_DIR / "gnn",
            LOG_DIR, OUTPUT_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
