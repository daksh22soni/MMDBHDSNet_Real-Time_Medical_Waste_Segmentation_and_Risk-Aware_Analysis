"""
========================================================================
DBHDSNet — Main Training Entry Point
Usage:
    # Fresh training
    python train.py

    # Resume from last checkpoint
    python train.py --resume

    # Resume from specific checkpoint
    python train.py --resume --ckpt checkpoints/epoch_0050.pt

    # Dry run (1 batch, no save) to verify everything works
    python train.py --dry-run
========================================================================
"""

import sys
import argparse
import time
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch

from config import CFG
from src.utils   import (
    setup_logging, set_seed, get_device,
    gpu_info, count_parameters, print_banner
)
from src.dataset import build_dataloaders
from src.models  import build_model
from src.trainer import Trainer


# ════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train DBHDSNet")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--ckpt",    type=str, default=None,
                   help="Path to specific checkpoint to resume from")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 1 batch then exit (smoke test)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--device",  type=str, default=None,
                   help="Device override: 'cuda', 'cpu', 'cuda:1' etc.")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# OVERFITTING GUARD — sanity checks before training begins
# ════════════════════════════════════════════════════════════════════════

def overfit_guard(model, train_loader, device, cfg):
    """
    Runs a 10-step overfit test on a single batch.
    If the model cannot overfit a single batch, something is wrong
    with the architecture or loss before you waste GPU hours.
    Returns True if loss decreases monotonically, False otherwise.
    """
    print_banner("Overfit Guard — 10-step single-batch test")
    from src.losses import DBHDSNetLoss
    criterion = DBHDSNetLoss(cfg).to(device)
    opt       = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    batch = next(iter(train_loader))
    images    = batch["images"].to(device)
    gt_boxes  = [b.to(device) for b in batch["boxes"]]
    gt_masks  = [m.to(device) for m in batch["masks"]]
    gt_hazard = [h.to(device) for h in batch["hazard_tiers"]]

    model.train()
    losses = []
    for step in range(10):
        opt.zero_grad(set_to_none=True)
        out  = model(images)
        loss, _ = criterion(out, gt_boxes, gt_masks, gt_hazard, device)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  Step {step+1:2d}/10  loss = {loss.item():.4f}")

    decreased = losses[-1] < losses[0]
    status    = "✓ PASS" if decreased else "⚠ WARN"
    print(f"\n  {status} — Initial: {losses[0]:.4f} → Final: {losses[-1]:.4f}")
    if not decreased:
        print("  WARNING: Loss did not decrease. Check architecture + data pipeline.")
    print()
    return decreased


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Create output dirs ────────────────────────────────────────────
    CFG.make_dirs()

    # ── Logging (stdout → terminal + file simultaneously) ─────────────
    logger = setup_logging(CFG.DATA.LOG_DIR, run_name="dbhdsnet_train")

    # ── Reproducibility ───────────────────────────────────────────────
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # ── Device ───────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Device: {device}")
    logger.info(gpu_info())

    # ── Print config summary ──────────────────────────────────────────
    print_banner("DBHDSNet — Phase 2 Training")
    print(f"  Dataset root  : {CFG.DATA.ROOT}")
    print(f"  Classes       : {CFG.MODEL.NUM_CLASSES}")
    print(f"  Image size    : {CFG.MODEL.IMG_SIZE}")
    print(f"  Epochs        : {CFG.TRAIN.EPOCHS}")
    print(f"  Batch size    : {CFG.TRAIN.BATCH_SIZE}")
    print(f"  AMP           : {CFG.TRAIN.AMP}")
    print(f"  EMA           : {CFG.TRAIN.EMA}")
    print(f"  Early stopping: patience={CFG.TRAIN.PATIENCE}")
    print()

    # ── Verify dataset paths exist ────────────────────────────────────
    for split_name, path in [
        ("train/images", CFG.DATA.TRAIN_IMG),
        ("train/labels", CFG.DATA.TRAIN_LBL),
        ("valid/images", CFG.DATA.VALID_IMG),
        ("valid/labels", CFG.DATA.VALID_LBL),
        ("YAML",         CFG.DATA.YAML),
    ]:
        if not Path(path).exists():
            logger.error(
                f"Path not found: {path}\n"
                f"  → Update DATASET_ROOT in config.py, then re-run."
            )
            sys.exit(1)

    # ── Build data loaders ────────────────────────────────────────────
    logger.info("Building data loaders…")
    train_loader, val_loader, test_loader = build_dataloaders(CFG)
    logger.info(
        f"  Train: {len(train_loader.dataset)} images / "
        f"{len(train_loader)} batches"
    )
    logger.info(
        f"  Valid: {len(val_loader.dataset)} images / "
        f"{len(val_loader)} batches"
    )
    logger.info(
        f"  Test : {len(test_loader.dataset)} images"
    )

    # ── Build model ───────────────────────────────────────────────────
    logger.info("Building DBHDSNet…")
    model = build_model(CFG, device)

    # Freeze backbone during warm-up
    model.freeze_backbone()
    logger.info("  Backbone frozen for warm-up epochs.")

    param_info = count_parameters(model)
    logger.info(
        f"  Total params   : {param_info['total']:,}"
    )
    logger.info(
        f"  Trainable      : {param_info['trainable']:,} "
        f"({100*param_info['trainable']/max(param_info['total'],1):.1f}%)"
    )
    logger.info(
        f"  Frozen         : {param_info['frozen']:,}"
    )

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print_banner("DRY RUN — 1 batch forward + backward")
        overfit_guard(model, train_loader, device, CFG)
        print("Dry run complete. No checkpoints saved.")
        return

    # ── Overfit guard (always run before full training) ───────────────
    overfit_guard(model, train_loader, device, CFG)

    # Re-build model (overfit guard modified optimizer state; start clean)
    model = build_model(CFG, device)
    model.freeze_backbone()

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(model, CFG, device)

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        resumed   = trainer.resume(ckpt_path)
        if not resumed:
            logger.warning("Resume requested but no checkpoint found. Training from scratch.")

    # ── Training ─────────────────────────────────────────────────────
    logger.info("Starting training loop…")
    trainer.fit(train_loader, val_loader)

    # ── Final test evaluation ─────────────────────────────────────────
    logger.info("\nRunning final evaluation on test set…")
    test_metrics = trainer._val_epoch(-1, test_loader)
    logger.info(f"  Test mAP@0.5  = {test_metrics.get('mAP@0.5',  0):.4f}")
    logger.info(f"  Test mAP@0.75 = {test_metrics.get('mAP@0.75', 0):.4f}")
    logger.info(f"  Test loss     = {test_metrics.get('loss_total', 0):.4f}")

    print_banner(
        f"All done!  Best val mAP@0.5 = {trainer.best_map:.4f}  "
        f"| Test mAP@0.5 = {test_metrics.get('mAP@0.5', 0):.4f}"
    )


if __name__ == "__main__":
    main()
