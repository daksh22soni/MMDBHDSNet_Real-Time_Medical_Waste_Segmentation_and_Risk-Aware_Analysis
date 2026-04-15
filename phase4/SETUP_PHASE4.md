# DBHDSNet — Phase 4: Federated Learning Setup & Run Guide

## Project Structure

```
phase4_dbhdsnet/
├── config_phase4.py              ← ALL paths & hyperparameters (edit first)
├── train_phase4.py               ← Main entry point
├── requirements_phase4.txt       ← Additional dependencies
├── SETUP_PHASE4.md               ← This file
│
├── src/
│   ├── utils.py                  ← Shared utilities (copied from Phase 2)
│   ├── dataset.py                ← Dataset (copied from Phase 2)
│   ├── metrics.py                ← mAP metrics (copied from Phase 2)
│   │
│   ├── federation/               ← Federated Learning core
│   │   ├── aggregators.py        ← FedAvg, FedProx, FedNova, FedAdam
│   │   ├── client.py             ← Hospital client local trainer
│   │   └── server.py             ← Federated server orchestrator
│   │
│   ├── privacy/                  ← Differential Privacy
│   │   ├── dp_engine.py          ← DP-SGD + RDP accountant per client
│   │   └── dp_audit.py           ← HIPAA/DPDPA compliance report
│   │
│   └── evaluation/               ← Federated evaluation
│       ├── fed_metrics.py        ← Global + per-client mAP, fairness metrics
│       └── visualiser.py         ← Convergence, radar, DP budget plots
│
├── scripts/
│   └── inspect_clients.py        ← Run FIRST: validates client shards
│
├── checkpoints/
│   ├── global/                   ← Global model: best.pt, last.pt, epoch_*.pt
│   └── clients/                  ← Per-client local checkpoints (optional)
│
├── logs/                         ← TensorBoard + text logs + history JSON
└── outputs/
    ├── convergence/              ← Convergence curve plots
    ├── privacy_audit/            ← dp_audit_report.json + PDF
    ├── per_client/               ← Per-client mAP tables
    └── visualisations/           ← All generated figures
```

---

## Step 1 — Install Dependencies

```bash
conda activate dbhdsnet   # same env as Phase 2

pip install -r requirements_phase4.txt

# Verify
python -c "import prv_accountant; print('PRV accountant OK')"
```

---

## Step 2 — Configure

Open `config_phase4.py` and update **Section A** only:

```python
DATASET_ROOT      = Path("/absolute/path/to/your/master_dataset")
PHASE2_CHECKPOINT = Path("/absolute/path/to/phase2_dbhdsnet/checkpoints/best.pt")
```

Everything else (HAZARD_TIER_MAP, CLIENTS, FedConfig, DPConfig) is pre-configured.

---

## Step 3 — Inspect Client Shards

```bash
python scripts/inspect_clients.py
```

This prints:
- How many images each client prefix finds in your training set
- Per-client hazard tier distribution (non-IID characterisation)
- Top-5 classes per client
- Cross-client comparison table

**Expected output:**
```
  [hospital_A]  prefix=v1  →  ~2500 images (24.7% of train set)
  Hazard tier distribution:
    T1 Sharps/Hazardous  ████░░░░░░░░░░░░░░░░░░░░░░░░░░   800  (12.3%)
    T2 Infectious        ████████████████████░░░░░░░░░░  3100  (47.8%)
    ...
```

If any client shows 0 images, your filename prefixes don't match. The pipeline
(Phase 1 Part 2) should have produced files like `v1_originalname.jpg`.

---

## Step 4 — Dry Run

```bash
python train_phase4.py --dry-run
```

Tests forward pass + 1 local training step per client. Exits immediately.
Should print `✓ Dry run complete` in < 60 seconds.

---

## Step 5 — Full Federated Training

```bash
python train_phase4.py
```

Or with specific overrides:
```bash
# Use FedNova instead of FedProx
python train_phase4.py --aggregation fednova

# Disable DP for ablation
python train_phase4.py --no-dp

# Resume after interruption
python train_phase4.py --resume

# Custom rounds for quick experiment
python train_phase4.py --rounds 20
```

---

## TensorBoard

TensorBoard is not yet integrated in Phase 4 (rounds-based logging goes to JSON history).
View convergence via generated figures in `outputs/visualisations/convergence.png`, or
load the history JSON manually:

```python
import json
with open("logs/phase4_fed_TIMESTAMP_history.json") as f:
    history = json.load(f)
```

---

## Aggregation Algorithm Guide

| Algorithm | When to use | Key parameter |
|---|---|---|
| `fedavg` | Baseline; IID-ish data | — |
| `fedprox` | **Recommended** for non-IID medical data | `FEDPROX_MU=0.01` |
| `fednova` | Variable local steps per client | `FEDNOVA_RHO=0.9` |
| `fedadam` | Large-scale, many clients | `FEDADAM_LR=1e-3` |

Your dataset is strongly non-IID (hospital_B is COVID-dominant, hospital_C is
sharps-dominant). **FedProx is the recommended default.**

---

## Differential Privacy Guide

Default configuration: σ=1.1, S=1.0, target ε=8.0, δ=1e-5.

**Interpretation:** With ε=8.0, an adversary's ability to determine whether any
individual training image was used is bounded to ≤ e⁸ ≈ 2981× random guessing.
For medical waste images (which are not direct patient data), this is acceptable.

**Tuning σ:**
| σ | Expected ε after 100 rounds | mAP impact |
|---|---|---|
| 0.7 | ~4.0 (tighter privacy) | −3–5% |
| 1.1 | ~8.0 (default)         | −1–2% |
| 1.5 | ~12.0 (looser)         | minimal |

To run a privacy-utility sweep (for the PhD paper):
```bash
python train_phase4.py --no-dp              # baseline (no DP)
python train_phase4.py --rounds 50          # with default σ=1.1
# Manually change NOISE_MULTIPLIER in config and re-run for each σ
```

---

## Novel PhD Contributions in Phase 4

1. **LoRA-only federation** — Only ~5% of parameters communicated per round.
   Communication savings: `comm_bytes / full_bytes` printed at startup.
   Baseline comparison: full-model FedAvg (set `LORA_ONLY_COMM=False`).

2. **FedProx + FedNova hybrid** — FedProx proximal term limits client drift;
   FedNova normalises heterogeneous local steps. Compare via `--aggregation` flag.

3. **Per-client DP-SGD with RDP accounting** — Each hospital independently tracks
   its privacy budget. T1 (sharps/radioactive) items receive 1.5× noise amplification.

4. **Drift-aware client weighting** — Clients with high cosine drift from the global
   model are down-weighted without exclusion. Track via `drift_scores` in history JSON.

5. **HIPAA/DPDPA compliance report** — Auto-generated in `outputs/privacy_audit/`.
   Includes formal (ε,δ)-DP guarantees, data residency attestation, and compliance
   control mapping — a first in federated medical waste classification literature.

---

## Expected Results

| Metric | Expected range |
|---|---|
| Global mAP@50 (final) | 0.82–0.88 |
| Global mAP@75 (final) | 0.65–0.75 |
| Hazard tier accuracy  | 0.88–0.93 |
| Convergence gap       | +0.03–0.07 (FL beats local-only training) |
| Fairness CV           | < 0.08 (low variance across hospitals) |
| Max ε after 100 rounds| 7–9 (within target ε=8.0 budget) |

---

## Common Errors

**`0 images for client hospital_A`**
→ Filenames in your train/images folder don't start with `v1_`.
  Run Phase 1 pipeline again, or check your prefix assignment.

**`Phase 2 checkpoint not found`**
→ Update `PHASE2_CHECKPOINT` in `config_phase4.py`.

**`CUDA out of memory`**
→ Reduce `LOCAL_BATCH_SIZE` in `FedConfig` (try 4 for <8GB GPU).

**`Module 'prv_accountant' not found`**
→ `pip install prv-accountant>=0.3.1`

**`KeyError in MedWasteDataset`**
→ Ensure Phase 2's `dataset.py` is accessible. Set sys.path or install as package.
