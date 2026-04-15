"""
========================================================================
DBHDSNet — Phase 3a: UQ Fine-Tuning & Calibration Entry Point

Usage:
    # Full Phase 3a pipeline (fine-tune + calibrate + UQ report)
    python run_phase3a.py

    # Resume fine-tuning from last checkpoint
    python run_phase3a.py --resume

    # Skip fine-tuning: only run calibration on Phase 2 model
    python run_phase3a.py --calibrate-only

    # Only generate UQ report (requires Phase 3a checkpoint)
    python run_phase3a.py --uq-only

    # Dry run (1 batch smoke test)
    python run_phase3a.py --dry-run
========================================================================
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "phase2"))  # Phase 2 src

import torch

from config_phase3 import CFG3
from src.utils import (
    setup_logging, set_seed, get_device, gpu_info,
    count_parameters, print_banner, make_run_name
)

# ── Phase 2 imports (dataset + model) ────────────────────────────────
# These are imported from the Phase 2 codebase.
# Adjust the path if Phase 2 lives elsewhere.
try:
    from src.dataset import build_dataloaders
    from src.models  import build_model
except ImportError:
    # Fallback: assume Phase 2 is on sys.path already
    from src.dataset import build_dataloaders
    from src.models  import build_model

from src.uq.uq_trainer  import Phase3aTrainer
from src.uq.visualiser  import generate_all_uq_figures


# ════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3a — UQ Fine-Tuning")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--calibrate-only",  action="store_true",
                   help="Skip UQ fine-tuning; only run calibration")
    p.add_argument("--uq-only",         action="store_true",
                   help="Only run MC-Dropout UQ report")
    p.add_argument("--dry-run",         action="store_true")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--ckpt",            type=str, default=None)
    p.add_argument("--device",          type=str, default=None)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    CFG3.make_dirs()

    logger = setup_logging(CFG3.DATA.LOG_DIR, "phase3a")
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    logger.info(f"Phase 3a | Device: {device}")
    logger.info(gpu_info())

    # ── Verify Phase 2 checkpoint ─────────────────────────────────────
    p2_ckpt = Path(CFG3.DATA.PHASE2_CKPT)
    if not p2_ckpt.exists():
        logger.error(f"Phase 2 checkpoint not found: {p2_ckpt}")
        logger.error("Update PHASE2_CHECKPOINT in config_phase3.py")
        sys.exit(1)

    # ── Build data loaders ────────────────────────────────────────────
    logger.info("Building data loaders…")
    train_loader, val_loader, test_loader = build_dataloaders(CFG3)

    # ── Load Phase 2 model ────────────────────────────────────────────
    logger.info(f"Loading Phase 2 model from {p2_ckpt}")
    model = build_model(CFG3, device)
    state = torch.load(p2_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=False)
    logger.info("  Phase 2 model loaded.")

    params = count_parameters(model)
    logger.info(f"  Total params: {params['total']:,}")

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print_banner("Phase 3a — DRY RUN")
        batch   = next(iter(train_loader))
        images  = batch["images"].to(device)
        with torch.no_grad():
            out = model(images)
        print(f"  hazard_logits shape : {out['hazard_logits'].shape}")
        print(f"  proto_masks   shape : {out['proto_masks'].shape}")
        print("  ✓ Forward pass OK. Dry run complete.")
        return

    # ── Phase 3a Trainer ─────────────────────────────────────────────
    trainer = Phase3aTrainer(model, CFG3, device)

    if args.resume:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        trainer.resume(ckpt_path)

    # ── UQ Fine-tuning ────────────────────────────────────────────────
    if not args.calibrate_only and not args.uq_only:
        logger.info("Starting UQ fine-tuning…")
        trainer.fit(train_loader, val_loader)

    # ── Load best UQ checkpoint for calibration / UQ report ──────────
    best = trainer.ckpt_mgr.load_best()
    if best:
        model.load_state_dict(best["model_state"], strict=False)
        logger.info(f"  Loaded best UQ checkpoint (ECE={best.get('best_ece','-'):.4f})")

    # ── Calibration ───────────────────────────────────────────────────
    if not args.uq_only:
        logger.info("Running post-hoc calibration…")
        cal_results = trainer.run_calibration(val_loader, test_loader)

    # ── MC-Dropout UQ Report ──────────────────────────────────────────
    logger.info("Running MC-Dropout UQ estimation on test set…")
    uq_summary, aggregator = trainer.run_mc_uncertainty(test_loader)

    # ── Visualisations ────────────────────────────────────────────────
    logger.info("Generating Phase 3a visualisations…")
    uq_np   = aggregator.to_numpy()
    out_dir = Path(CFG3.DATA.OUTPUT_DIR) / "calibration"

    # Load reliability diagram data
    import numpy as np
    diag_data  = {}
    cal_metrics_vis = {}
    if not args.uq_only:
        for method in ["uncalibrated", "temp_scaling", "vector_scaling"]:
            fp = out_dir / f"reliability_{method}.npy"
            if fp.exists():
                diag_data[method]      = dict(np.load(fp, allow_pickle=True).item())
                cal_metrics_vis[method] = cal_results.get(method, {})

    if diag_data:
        generate_all_uq_figures(
            uq_data        = uq_np,
            uq_summary     = uq_summary,
            diag_data      = diag_data,
            cal_metrics    = cal_metrics_vis,
            tier_thresholds = CFG3.UQ.UNCERTAINTY_THRESH_BY_TIER,
            output_dir     = out_dir,
        )

    print_banner("Phase 3a Complete!")


if __name__ == "__main__":
    main()
