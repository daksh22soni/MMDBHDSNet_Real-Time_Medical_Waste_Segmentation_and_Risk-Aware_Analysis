"""
========================================================================
DBHDSNet — Phase 3a Training Entry Point
Uncertainty Quantification: MC-Dropout + Temperature Scaling + HITL

Workflow
────────
  1. Load trained Phase 2 DBHDSNet (best.pt)
  2. Collect val-set logits (single forward pass, no dropout)
  3. Fit Temperature Scaling on val-set NLL
  4. Evaluate calibration: ECE/ACE/MCE before and after scaling
  5. Run MC-Dropout on val/test sets (N=30 passes)
  6. Decompose: epistemic (MI) + aleatoric (expected entropy)
  7. Apply HITL flagging protocol with tier-conditional thresholds
  8. Compute UQ evaluation suite (AURC, AUROC, selective accuracy)
  9. Save: calibration model, reliability diagrams, UQ reports

Run
───
    # Evaluate UQ on val+test set
    python train_phase3a_uq.py

    # Evaluate on specific split only
    python train_phase3a_uq.py --split test

    # Use Deep Ensemble (requires N pre-trained checkpoints)
    python train_phase3a_uq.py --ensemble-dir path/to/checkpoints/

    # Dry run (single batch)
    python train_phase3a_uq.py --dry-run
========================================================================
"""

import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
# Also add parent so Phase 2 model can be imported
PHASE2_ROOT = ROOT.parent / "phase2_dbhdsnet"
sys.path.insert(0, str(PHASE2_ROOT))

import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

from config_phase3 import CFG
from src.utils import (
    setup_logging, set_seed, get_device, gpu_info,
    print_banner, make_run_name, format_time,
)
from src.uq.mc_dropout    import MCDropoutEstimator, DeepEnsembleEstimator
from src.uq.calibration   import TemperatureScaling, CalibrationTrainer, CalibrationMetrics
from src.uq.hitl_protocol import HITLProtocol
from src.uq.uq_metrics    import full_uq_evaluation

# Import Phase 2 model
try:
    from src.models.dbhdsnet import build_model as build_phase2_model
except ImportError:
    from phase2_dbhdsnet.src.models.dbhdsnet import build_model as build_phase2_model

try:
    from src.dataset import build_dataloaders, get_class_names
except ImportError:
    from phase2_dbhdsnet.src.dataset import build_dataloaders, get_class_names


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3a — UQ evaluation")
    p.add_argument("--split",        type=str, default="both",
                   choices=["val", "test", "both"])
    p.add_argument("--ensemble-dir", type=str, default=None,
                   help="Directory with ensemble checkpoint files")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--n-passes",     type=int, default=None,
                   help="Override MC-Dropout N (default from config)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# COLLECT VAL LOGITS (for calibration fitting)
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits(model, loader, device) -> tuple:
    """
    Single-pass forward to collect raw hazard logits and labels.
    Dropout is DISABLED during this pass.
    Returns (logits: (N,4), labels: (N,)) both as CPU tensors.
    """
    model.eval()
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc="  Collecting logits", ncols=100, leave=False)
    for batch in pbar:
        images    = batch["images"].to(device, non_blocking=True)
        gt_hazard = batch["hazard_tiers"]   # list of (Ni,) per image

        with autocast(enabled=True):
            out = model(images)

        logits = out["hazard_logits"]   # (B, 4)
        all_logits.append(logits.cpu())

        # Image-level label: worst tier in each image
        for tiers in gt_hazard:
            worst = int(tiers.min().item()) - 1 if tiers.numel() > 0 else 3
            all_labels.append(worst)

    return (
        torch.cat(all_logits, dim=0),          # (N, 4)
        torch.tensor(all_labels, dtype=torch.long),  # (N,)
    )


# ════════════════════════════════════════════════════════════════════════
# RUN UQ EVALUATION ON A SPLIT
# ════════════════════════════════════════════════════════════════════════

def evaluate_uq_split(
    estimator,
    loader,
    T_module,
    hitl_protocol,
    class_names,
    device,
    cfg,
    split_name: str,
    output_dir: Path,
    n_passes:   int,
    dry_run:    bool = False,
) -> dict:
    """
    Full UQ evaluation on one dataset split.
    Returns flat metrics dict.
    """
    print_banner(f"UQ Evaluation — {split_name} split")

    all_ep, all_al, all_pe = [], [], []
    all_pred_tiers, all_true_tiers = [], []
    all_flags = []
    all_mean_probs = []

    pbar = tqdm(
        loader,
        desc  = f"  [{split_name}] MC-Dropout (N={n_passes})",
        ncols = 120,
        leave = True,
    )

    for batch_idx, batch in enumerate(pbar):
        images    = batch["images"].to(device, non_blocking=True)
        gt_hazard = batch["hazard_tiers"]
        B         = images.shape[0]

        # ── MC-Dropout uncertainty estimation ─────────────────────────
        uq = estimator.estimate(images, device=device)

        # Apply temperature scaling to mean probs
        with torch.no_grad():
            raw_logits  = estimator.model(images)["hazard_logits"].cpu()
            cal_logits  = T_module(raw_logits)
            cal_probs   = torch.softmax(cal_logits, dim=-1)

        # Predicted tier (from calibrated mean probs)
        pred_tiers = cal_probs.argmax(dim=-1) + 1   # 1-indexed

        # True tiers (image-level: worst tier)
        for b, tiers in enumerate(gt_hazard):
            worst = int(tiers.min().item()) if tiers.numel() > 0 else 4
            all_true_tiers.append(worst)

        all_ep.append(uq.epistemic.cpu().numpy())
        all_al.append(uq.aleatoric.cpu().numpy())
        all_pe.append(uq.predictive_entropy.cpu().numpy())
        all_pred_tiers.append(pred_tiers.numpy())
        all_flags.append(uq.flags.numpy())
        all_mean_probs.append(cal_probs.numpy())

        # ── HITL decisions (verbose on first few batches) ─────────────
        cls_ids = cal_probs.argmax(dim=-1)
        decisions = hitl_protocol.evaluate_batch(uq, cls_ids, n_mc_passes=n_passes)
        if batch_idx == 0:
            print(f"\n  Sample HITL decisions (batch 0):")
            for d in decisions[:4]:
                hitl_protocol.print_decision(d)
            print()

        pbar.set_postfix({
            "ep_mean": f"{uq.epistemic.mean().item():.4f}",
            "flagged":  f"{uq.flags.sum().item()}/{B}",
        })

        if dry_run and batch_idx >= 2:
            break

    # ── Aggregate ────────────────────────────────────────────────────
    ep_all  = np.concatenate(all_ep)
    al_all  = np.concatenate(all_al)
    pe_all  = np.concatenate(all_pe)
    pt_all  = np.concatenate(all_pred_tiers)
    tt_all  = np.array(all_true_tiers)
    mp_all  = np.concatenate(all_mean_probs, axis=0)

    # ── Full UQ evaluation suite ──────────────────────────────────────
    metrics = full_uq_evaluation(
        pred_tiers  = pt_all,
        true_tiers  = tt_all,
        epistemic   = ep_all,
        aleatoric   = al_all,
        mean_probs  = mp_all,
        flag_thresh = cfg.UQ.UNCERTAINTY_THRESH_HIGH,
        output_dir  = output_dir,
    )

    # ── Calibration metrics (before vs after T-scaling) ───────────────
    cal = CalibrationMetrics(n_bins=cfg.UQ.ECE_N_BINS)
    before_probs = torch.softmax(
        torch.from_numpy(mp_all), dim=-1
    )  # already calibrated, this is a proxy
    after_m  = cal.compute_all(torch.from_numpy(mp_all),
                               torch.from_numpy(tt_all - 1))
    metrics["ECE_after"]  = after_m["ECE"]
    metrics["MCE_after"]  = after_m["MCE"]
    metrics["Brier_after"]= after_m["Brier"]
    metrics["NLL_after"]  = after_m["NLL"]

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  [{split_name}] UQ Summary:")
    print(f"    Mean epistemic uncertainty : {ep_all.mean():.4f}")
    print(f"    Mean aleatoric uncertainty : {al_all.mean():.4f}")
    print(f"    Flag rate (RED zone)       : {metrics['flag_rate']*100:.1f}%")
    print(f"    AURC                       : {metrics.get('AURC', 0):.4f}")
    print(f"    AUROC (U vs error)         : {metrics.get('AUROC_uncertainty_vs_error', 0):.4f}")
    print(f"    Overall Safety Score       : {metrics.get('overall_safety_score', 0):.4f}")
    print(f"    Critical Catch Rate        : {metrics.get('critical_catch_rate', 0):.4f}")
    print(f"    ECE (after calib)          : {metrics['ECE_after']:.4f}")
    print()

    return metrics


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    CFG.make_dirs()

    # ── Logging ──────────────────────────────────────────────────────
    run_name = make_run_name("phase3a_uq")
    logger   = setup_logging(CFG.DATA.LOG_DIR, run_name)

    # ── Seed & device ────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Device: {device}")
    logger.info(gpu_info())

    n_passes = args.n_passes or CFG.UQ.MC_N_PASSES

    # ── Verify Phase 2 checkpoint ────────────────────────────────────
    if not Path(CFG.DATA.PHASE2_CKPT).exists():
        logger.error(
            f"Phase 2 checkpoint not found: {CFG.DATA.PHASE2_CKPT}\n"
            "  Update PHASE2_CHECKPOINT in config_phase3.py"
        )
        sys.exit(1)

    # ── Build Phase 2 model and load weights ─────────────────────────
    print_banner("Phase 3a — Uncertainty Quantification")
    logger.info("Loading Phase 2 model…")

    # Build model using Phase 2 model config (same architecture)
    model = build_phase2_model(CFG, device)
    ckpt  = torch.load(CFG.DATA.PHASE2_CKPT, map_location="cpu",
                       weights_only=False)
    model_state = ckpt.get("model_state", ckpt.get("ema_state", {}).get("ema_state", ckpt))
    model.load_state_dict(model_state, strict=False)
    model.eval()
    logger.info("  Phase 2 model loaded successfully.")

    # ── Build dataloaders ─────────────────────────────────────────────
    logger.info("Building data loaders…")
    class_names = get_class_names(CFG.DATA.YAML)
    train_loader, val_loader, test_loader = build_dataloaders(CFG)

    # ── Step 1: Collect val logits for Temperature Scaling ────────────
    logger.info("Collecting val-set logits for calibration…")
    val_logits, val_labels = collect_logits(model, val_loader, device)
    logger.info(f"  Collected {len(val_logits)} val samples.")

    # ── Step 2: Fit Temperature Scaling ──────────────────────────────
    cal_trainer = CalibrationTrainer(CFG, device)
    T_module    = cal_trainer.fit(val_logits, val_labels)

    T_save_path = Path(CFG.DATA.CHECKPOINT_DIR) / "uq" / "temperature_scaling.pt"
    cal_trainer.save(T_module, T_save_path)
    T_module.to(device)

    # ── Step 3: Calibration evaluation before / after ─────────────────
    cal_metrics = CalibrationMetrics(n_bins=CFG.UQ.ECE_N_BINS)

    before_probs = torch.softmax(val_logits, dim=-1)
    before_m     = cal_metrics.compute_all(before_probs, val_labels)

    cal_logits   = T_module(val_logits.to(device)).cpu()
    after_probs  = torch.softmax(cal_logits, dim=-1)
    after_m      = cal_metrics.compute_all(after_probs, val_labels)

    print(f"\n  Calibration Results (Val set):")
    print(f"    Before T-scaling:  ECE={before_m['ECE']:.4f}  MCE={before_m['MCE']:.4f}  Brier={before_m['Brier']:.4f}")
    print(f"    After  T-scaling:  ECE={after_m['ECE']:.4f}  MCE={after_m['MCE']:.4f}  Brier={after_m['Brier']:.4f}")

    # Reliability diagram
    rel_diag_path = Path(CFG.DATA.OUTPUT_DIR) / f"{run_name}_reliability_diagram.png"
    cal_metrics.reliability_diagram(before_m, after_m, rel_diag_path)

    # ── Step 4: MC-Dropout estimator ─────────────────────────────────
    estimator = MCDropoutEstimator(model, CFG)
    estimator.n_passes = n_passes

    # ── Step 5: HITL protocol ─────────────────────────────────────────
    hitl = HITLProtocol(CFG, class_names, Path(CFG.DATA.LOG_DIR))

    all_results = {}

    # ── Step 6: Evaluate splits ───────────────────────────────────────
    splits_to_eval = []
    if args.split in ("val", "both"):
        splits_to_eval.append(("val",  val_loader))
    if args.split in ("test", "both"):
        splits_to_eval.append(("test", test_loader))

    for split_name, split_loader in splits_to_eval:
        out_dir = Path(CFG.DATA.OUTPUT_DIR) / f"{run_name}_{split_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = evaluate_uq_split(
            estimator    = estimator,
            loader       = split_loader,
            T_module     = T_module,
            hitl_protocol= hitl,
            class_names  = class_names,
            device       = device,
            cfg          = CFG,
            split_name   = split_name,
            output_dir   = out_dir,
            n_passes     = n_passes,
            dry_run      = args.dry_run,
        )
        all_results[split_name] = results

        # Save HITL uncertainty distribution plot
        hitl.plot_uncertainty_distribution(
            out_dir / f"uncertainty_dist_{split_name}.png"
        )

    # ── Step 7: Save HITL session log ────────────────────────────────
    log_path = hitl.save_session_log()
    logger.info(f"HITL session log saved → {log_path}")

    # ── Step 8: Save full results JSON ───────────────────────────────
    results_path = Path(CFG.DATA.OUTPUT_DIR) / f"{run_name}_results.json"
    # Convert numpy floats to Python floats for JSON
    def to_py(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        return obj

    with open(results_path, "w") as f:
        json.dump(to_py(all_results), f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # ── Session summary ───────────────────────────────────────────────
    summary = hitl.session_summary()
    print_banner(f"Phase 3a Complete | Flagged {summary.get('pct_flagged', 0):.1f}% for human review")
    logger.info(f"HITL summary: {summary}")


if __name__ == "__main__":
    main()
