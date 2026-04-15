# DBHDSNet — Phase 3: Setup & Run Guide

## Project Structure

```
phase3_dbhdsnet/
├── config_phase3.py              ← ALL PATHS & HYPERPARAMETERS (edit first)
├── run_phase3a.py                ← Phase 3a entry point (UQ + Calibration)
├── run_phase3b.py                ← Phase 3b entry point (ContamRisk-GNN)
├── requirements_phase3.txt       ← Additional dependencies
├── SETUP_PHASE3.md               ← This file
│
├── src/
│   ├── utils.py                  ← Shared utilities (copied from Phase 2)
│   ├── metrics_base.py           ← Base mAP metrics
│   │
│   ├── uq/                       ← Phase 3a — Uncertainty Quantification
│   │   ├── mc_dropout.py         ← MC-Dropout sampler + BALD scorer
│   │   ├── calibration.py        ← TemperatureScaling, VectorScaling, ECE
│   │   ├── uq_trainer.py         ← UQ fine-tuning + calibration workflow
│   │   └── visualiser.py         ← Reliability diagrams + UQ dashboards
│   │
│   └── gnn/                      ← Phase 3b — ContamRisk-GNN
│       ├── scene_graph.py        ← Scene graph construction from detections
│       ├── contamrisk_gnn.py     ← GATv2-based ContamRisk-GNN architecture
│       ├── gnn_losses.py         ← Novel composite GNN loss
│       ├── gnn_trainer.py        ← GNN training loop + evaluation
│       └── visualiser_gnn.py     ← Risk scatter, scene graph, violin plots
│
├── checkpoints/
│   ├── phase3a/                  ← UQ fine-tuning checkpoints
│   └── phase3b/                  ← GNN checkpoints
│
├── logs/                         ← TensorBoard + text logs
└── outputs/
    ├── calibration/              ← Reliability diagrams, UQ reports
    ├── contamrisk/               ← GNN evaluation reports, scene graphs
    └── visualisations/           ← All generated figures
```

---

## Step 1 — Install Additional Dependencies

```bash
# Activate your Phase 2 conda environment
conda activate dbhdsnet

# Install Phase 3 extras
pip install -r requirements_phase3.txt

# torch-geometric (critical — version depends on your CUDA + PyTorch)
# Find the right command at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# Example for PyTorch 2.1 + CUDA 11.8:
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Verify
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

---

## Step 2 — Configure Phase 3

Open `config_phase3.py` and update two fields:

```python
# Section A
DATASET_ROOT      = Path("/absolute/path/to/your/master_dataset")
PHASE2_CHECKPOINT = Path("/absolute/path/to/phase2_dbhdsnet/checkpoints/best.pt")
```

All other paths auto-resolve from these two.

---

## Phase 3a — Uncertainty Quantification

### What it does:
1. Loads your Phase 2 DBHDSNet
2. Fine-tunes ONLY the hazard head + LoRA + fusion gating with `UncertaintyAwareLoss`
3. Fits Temperature Scaling + Vector Scaling calibrators on val set
4. Runs 30-pass MC-Dropout on test set → computes ECE, BALD, flag rate
5. Generates: reliability diagrams, uncertainty histograms, flagging dashboard

### Run order:

```bash
# Smoke test (30 seconds)
python run_phase3a.py --dry-run

# Full pipeline
python run_phase3a.py

# Resume after interruption
python run_phase3a.py --resume

# Only calibration (skip fine-tuning)
python run_phase3a.py --calibrate-only

# Only UQ report
python run_phase3a.py --uq-only
```

### Outputs:
```
outputs/calibration/
├── reliability_diagram.png        ← Before/after calibration curves
├── uncertainty_histograms.png     ← Per-tier epistemic uncertainty
├── bald_distribution.png          ← BALD scores (flagged vs accepted)
├── flagging_dashboard.png         ← Clinical flagging summary
├── calibration_report.json        ← ECE, ACE, Brier, NLL per method
├── uq_report.json                 ← Flag rates, BALD stats, per-tier rates
└── reliability_*.npy              ← Raw diagram data
```

### Expected calibration metrics:
| Metric | Before calibration | After temperature scaling |
|---|---|---|
| ECE | 0.12–0.18 | 0.04–0.08 |
| Brier | 0.15–0.25 | 0.10–0.18 |
| NLL | 0.8–1.2 | 0.6–0.9 |

---

## Phase 3b — ContamRisk-GNN

### What it does:
1. Runs DBHDSNet inference on all images → builds scene graphs
2. Each scene graph: nodes=detected items, edges=spatial proximity
3. Node features: class onehot(38) + tier onehot(4) + confidence + geometry
4. Edge features: distance + IoU + tier_diff + WHO_risk
5. Trains GAT-based GNN to predict scene-level + per-item contamination risk
6. Loss: scene_MSE + node_BCE + tier_consistency + contrastive

### Run order:

```bash
# Smoke test (builds 2 graphs + 1 GNN forward pass)
python run_phase3b.py --dry-run

# Full pipeline (first run: builds + caches scene graphs, then trains GNN)
python run_phase3b.py

# Resume GNN training
python run_phase3b.py --resume

# Force rebuild scene graphs (re-run DBHDSNet inference)
python run_phase3b.py --rebuild-graphs

# Evaluate only (load best checkpoint)
python run_phase3b.py --eval-only
```

### Scene graph cache:
On first run, DBHDSNet inference takes ~15–30 min (on 14,445 images).
Results are cached as pickle files in `outputs/contamrisk/graphs_*.pkl`.
Subsequent runs load from cache (~2 seconds).
Use `--rebuild-graphs` to force re-inference.

### Outputs:
```
outputs/contamrisk/
├── graphs_train.pkl               ← Cached scene graphs
├── graphs_valid.pkl
├── graphs_test.pkl
├── test_report.json               ← MAE, RMSE, Spearman, sensitivity
├── risk_scatter.png               ← Predicted vs true risk scatter
├── risk_confusion_matrix.png      ← LOW/MEDIUM/HIGH confusion matrix
├── node_risk_by_tier.png          ← Per-tier node risk violin plot
└── scene_graph_sample_*.png       ← Example scene graph visualisations
```

### Expected GNN metrics:
| Metric | Expected range |
|---|---|
| MAE | 0.06–0.12 |
| Spearman r | 0.75–0.90 |
| Risk-level accuracy | 80–90% |
| HIGH sensitivity | 85–95% |

---

## TensorBoard (both phases)

```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

Phase 3a tracks:
- `phase3a/train/loss_nll`, `phase3a/train/loss_ent`, `phase3a/train/loss_tier`
- `phase3a/val/val_ece`, `phase3a/val/val_brier`, `phase3a/val/val_acc`

Phase 3b tracks:
- `gnn/train/loss_scene`, `gnn/train/loss_node`, `gnn/train/loss_tier`, `gnn/train/loss_contrastive`
- `gnn/val/mae`, `gnn/val/high_sensitivity`, `gnn/val/risk_level_acc`

---

## Key Hyperparameters to Tune

### Phase 3a
| Parameter | Location | Notes |
|---|---|---|
| `MC_N_PASSES` | `UQConfig` | 30 is good; 50 for publication |
| `UNCERTAINTY_THRESH_HIGH` | `UQConfig` | Set empirically from reliability diagram |
| `TIER_THRESHOLDS` | `UQConfig` | Pre-set: T1=0.12, T2=0.18, T3=0.22, T4=0.30 |
| `ENSEMBLE_EPOCHS` | `UQConfig` | 30 per member; total 5× fine-tunes from best.pt |

**Your dataset tier assignment (already configured):**
- T1 tight threshold (0.05/0.12): `blade`, `scalpel`, `needle`, `syringe`, `harris_uni_core`, `mercury_thermometer`, `radioactive_objects`
- T2 (0.08/0.18): `bloody_objects`, `mask`, `n95`, `bandage`, `cotton_swab`, `covid_buffer`, `covid_test_case`, `gauze`, `iodine_swab`, `plastic_medical_bag`, `medical_gloves`, `reagent_tube`, `single_channel_pipette`, `transferpettor_glass`, `transferpettor_plastic`, `tweezer_metal`, `tweezer_plastic`, `medical_infusion_bag`
- T3 (0.10/0.22): `oxygen_cylinder`, `capsule`, `covid_buffer_box`, `glass_bottle`, `harris_uni_core_cap`, `pill`, `plastic_medical_bottle`, `reagent_tube_cap`, `unguent`, `electronic_thermometer`, `drug_packaging`
- T4 lenient (0.15/0.30): `paperbox`, `cap_plastic`

### Phase 3b
| Parameter | Location | Notes |
|---|---|---|
| `GNN_N_LAYERS` | `GNNConfig` | 4 is good; 6 for larger graphs |
| `PROXIMITY_DIST_NORM` | `GNNConfig` | 0.15 default; increase for sparse detections |
| `LAMBDA_PAIR` | `GNNConfig` | Novel pair loss; 0.5 default |
| `RISK_THRESHOLDS` | `GNNConfig` | [0.25, 0.50, 0.75] — Low/Med/High/Crit |

**HIGH_RISK_CLASS_PAIRS (pre-configured, WHO/CPCB):**
Key pairs your model is specifically trained to flag as dangerous co-occurrences:
`needle+paperbox`, `syringe+bloody_objects`, `blade+cap_plastic`, `radioactive_objects+covid_buffer`, `mercury_thermometer+glass_bottle` and 14 more — all in `config_phase3.py`.

---

## Common Errors

**`ImportError: torch_geometric`**
→ Follow PyG install guide exactly for your PyTorch + CUDA version.

**`RuntimeError: CUDA out of memory` (Phase 3b)**
→ Reduce `GNN_BATCH_SIZE` in `GNNConfig` (try 16 or 8).

**`No graphs built` from scene graph builder**
→ Your Phase 2 model is not producing detections above `CONF_THRESH=0.25`.
  Either lower the threshold or use the Phase 2 EMA checkpoint.

**`AttributeError: 'NoneType' object has no attribute 'ema'`**
→ Pass `use_ema=False` in `collect_logits()` if Phase 2 checkpoint has no EMA.

**Scene graphs have 0 edges**
→ Increase `PROXIMITY_THRESH` in `GNNConfig` (try 0.50 for sparse detections).
