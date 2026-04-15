"""
========================================================================
DBHDSNet — Trainer
Full training loop with:
  • Continuous per-epoch tqdm progress bars
  • Mixed-precision (AMP) training
  • Gradient clipping
  • EMA model weights
  • Checkpoint save/resume
  • EarlyStopping
  • TensorBoard logging
  • Terminal + file dual output
========================================================================
"""

import time
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import (
    AverageMeter, MetricTracker, ModelEMA, EarlyStopping,
    CheckpointManager, TrainingHistory,
    gpu_info, format_time, print_banner, make_run_name,
)
from .losses import DBHDSNetLoss
from .metrics import compute_map


# ════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER (cosine + warm-up)
# ════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warm-up for `warmup_epochs`, then cosine decay to `min_lr`.
    One step = one epoch.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs:  int,
        total_epochs:   int,
        warmup_lr:      float = 1e-6,
        min_lr:         float = 1e-6,
        last_epoch:     int   = -1,
    ):
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.warmup_lr      = warmup_lr
        self.min_lr         = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
            return [
                self.warmup_lr + scale * (base - self.warmup_lr)
                for base in self.base_lrs
            ]
        progress = (e - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cos_val  = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base - self.min_lr) * cos_val
            for base in self.base_lrs
        ]


# ════════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Orchestrates the full DBHDSNet training pipeline.

    Usage
    -----
    trainer = Trainer(model, cfg, device)
    trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model, cfg, device: torch.device):
        self.model  = model
        self.cfg    = cfg
        self.device = device
        self.tc     = cfg.TRAIN

        # ── Run identity ──────────────────────────────────────────────
        self.run_name = make_run_name("dbhdsnet")

        # ── Directories ───────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.DATA.CHECKPOINT_DIR)
        self.log_dir  = Path(cfg.DATA.LOG_DIR)

        # ── Loss function ─────────────────────────────────────────────
        self.criterion = DBHDSNetLoss(cfg).to(device)

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr           = self.tc.BASE_LR,
            weight_decay = self.tc.WEIGHT_DECAY,
            betas        = self.tc.BETAS,
        )

        # ── Scheduler ────────────────────────────────────────────────
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs = self.tc.WARMUP_EPOCHS,
            total_epochs  = self.tc.EPOCHS,
            warmup_lr     = self.tc.WARMUP_LR_START,
            min_lr        = self.tc.MIN_LR,
        )

        # ── AMP scaler ────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=self.tc.AMP)

        # ── EMA ───────────────────────────────────────────────────────
        self.ema = ModelEMA(model, decay=self.tc.EMA_DECAY) if self.tc.EMA else None

        # ── Early stopping ────────────────────────────────────────────
        self.early_stop = EarlyStopping(
            patience=self.tc.PATIENCE, min_delta=self.tc.MIN_DELTA
        )

        # ── Checkpoint manager ────────────────────────────────────────
        self.ckpt_mgr = CheckpointManager(
            self.ckpt_dir, keep_last_n=self.tc.KEEP_LAST_N
        )

        # ── TensorBoard ───────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.log_dir / self.run_name))

        # ── Training history ──────────────────────────────────────────
        self.history = TrainingHistory(
            self.log_dir / f"{self.run_name}_history.json"
        )

        # ── State ─────────────────────────────────────────────────────
        self.start_epoch = 0
        self.best_map    = -1.0

    # ------------------------------------------------------------------
    # RESUME FROM CHECKPOINT
    # ------------------------------------------------------------------

    def resume(self, ckpt_path: Optional[Path] = None):
        """Load the last (or specified) checkpoint and restore state."""
        state = self.ckpt_mgr.load(ckpt_path)
        if state is None:
            print("[Trainer] No checkpoint found — starting from scratch.")
            return False

        self.model.load_state_dict(state["model_state"], strict=False)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        if self.ema and "ema_state" in state:
            self.ema.load_state_dict(state["ema_state"])
        if "early_stop_state" in state:
            self.early_stop.load_state_dict(state["early_stop_state"])

        self.start_epoch = state["epoch"] + 1
        self.best_map    = state.get("best_map", -1.0)

        tqdm.write(
            f"[Trainer] Resumed from epoch {state['epoch']:04d} "
            f"(best mAP={self.best_map:.4f})"
        )
        return True

    # ------------------------------------------------------------------
    # MAIN FIT LOOP
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print_banner(
            f"DBHDSNet Training — run: {self.run_name} | "
            f"Device: {self.device} | Epochs: {self.tc.EPOCHS}"
        )
        print(gpu_info())
        print()

        t0 = time.time()

        for epoch in range(self.start_epoch, self.tc.EPOCHS):

            # ── Unfreeze backbone after warm-up ───────────────────────
            if epoch == self.tc.WARMUP_EPOCHS:
                self.model.unfreeze_backbone()
                tqdm.write("[Trainer] Backbone layers 3+4 unfrozen.")

            # ── Train one epoch ───────────────────────────────────────
            train_metrics = self._train_epoch(epoch, train_loader)

            # ── Validate ──────────────────────────────────────────────
            val_metrics = self._val_epoch(epoch, val_loader)

            # ── Scheduler step ────────────────────────────────────────
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Log to TensorBoard ────────────────────────────────────
            self._tensorboard_log(epoch, train_metrics, val_metrics, current_lr)

            # ── Print epoch summary ───────────────────────────────────
            val_map = val_metrics.get("mAP@0.5", 0.0)
            elapsed = format_time(time.time() - t0)
            tqdm.write(
                f"\n  Epoch {epoch+1:04d}/{self.tc.EPOCHS:04d}  "
                f"│ Train: {_fmt(train_metrics)}  "
                f"│ Val mAP@0.5={val_map:.4f}  "
                f"│ LR={current_lr:.2e}  │ ⏱ {elapsed}"
            )

            # ── Checkpoint ────────────────────────────────────────────
            is_best = val_map > self.best_map
            if is_best:
                self.best_map = val_map
                tqdm.write(f"  ★ New best mAP: {self.best_map:.4f}")

            periodic = ((epoch + 1) % self.tc.SAVE_EVERY == 0)
            state    = self._build_state(epoch)
            self.ckpt_mgr.save(state, is_best, epoch + 1, periodic)

            # ── History ───────────────────────────────────────────────
            self.history.append({
                "epoch": epoch + 1,
                "lr":    current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}":   v for k, v in val_metrics.items()},
            })

            # ── Early stopping check ───────────────────────────────────
            if self.early_stop(val_map):
                tqdm.write(
                    f"\n[Trainer] EarlyStopping triggered after "
                    f"{self.early_stop.patience} epochs without improvement."
                )
                break

        # ── Final summary ─────────────────────────────────────────────
        total_time = format_time(time.time() - t0)
        print_banner(
            f"Training complete!  Best mAP@0.5 = {self.best_map:.4f}  "
            f"│  Total time: {total_time}"
        )
        self.writer.close()

    # ------------------------------------------------------------------
    # TRAIN ONE EPOCH
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, loader: DataLoader) -> dict:
        self.model.train()
        tracker = MetricTracker()

        # ── Continuous per-epoch tqdm bar ────────────────────────────
        pbar = tqdm(
            loader,
            desc     = f"Epoch {epoch+1:04d}/{self.tc.EPOCHS:04d} [TRAIN]",
            ncols    = 130,
            leave    = True,                 # bar stays after epoch ends
            dynamic_ncols = False,
        )

        for batch_idx, batch in enumerate(pbar):
            images      = batch["images"].to(self.device, non_blocking=True)
            gt_boxes    = [b.to(self.device) for b in batch["boxes"]]
            gt_masks    = [m.to(self.device) for m in batch["masks"]]
            gt_hazard   = [h.to(self.device) for h in batch["hazard_tiers"]]

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward + Loss (mixed precision) ─────────────────────
            with autocast(enabled=self.tc.AMP):
                model_out = self.model(images)
                loss, loss_dict = self.criterion(
                    model_out, gt_boxes, gt_masks, gt_hazard, self.device
                )

            # ── Backward ─────────────────────────────────────────────
            self.scaler.scale(loss).backward()

            # ── Gradient clipping ─────────────────────────────────────
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.GRAD_CLIP
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── EMA update ────────────────────────────────────────────
            if self.ema:
                self.ema.update(self.model)

            # ── Update tracker & progress bar postfix ────────────────
            bs = images.size(0)
            tracker.update(loss_dict, n=bs)
            avgs = tracker.averages()

            pbar.set_postfix(OrderedDict([
                ("loss",  f"{avgs.get('loss_total',  0):.3f}"),
                ("box",   f"{avgs.get('loss_box',    0):.3f}"),
                ("seg",   f"{avgs.get('loss_seg',    0):.3f}"),
                ("haz",   f"{avgs.get('loss_hazard', 0):.3f}"),
                ("hier",  f"{avgs.get('loss_hierarchy', 0):.3f}"),
                ("lr",    f"{self.optimizer.param_groups[0]['lr']:.2e}"),
            ]))

        return tracker.averages()

    # ------------------------------------------------------------------
    # VALIDATE ONE EPOCH
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int, loader: DataLoader) -> dict:
        eval_model = self.ema.ema if self.ema else self.model
        eval_model.eval()
        tracker = MetricTracker()

        all_preds, all_targets = [], []

        pbar = tqdm(
            loader,
            desc   = f"Epoch {epoch+1:04d}/{self.tc.EPOCHS:04d} [VAL]  ",
            ncols  = 130,
            leave  = False,     # validation bar clears after each epoch
            dynamic_ncols = False,
        )

        for batch in pbar:
            images      = batch["images"].to(self.device, non_blocking=True)
            gt_boxes    = [b.to(self.device) for b in batch["boxes"]]
            gt_masks    = [m.to(self.device) for m in batch["masks"]]
            gt_hazard   = [h.to(self.device) for h in batch["hazard_tiers"]]

            with autocast(enabled=self.tc.AMP):
                model_out = eval_model(images)
                loss, loss_dict = self.criterion(
                    model_out, gt_boxes, gt_masks, gt_hazard, self.device
                )

            bs = images.size(0)
            tracker.update(loss_dict, n=bs)
            pbar.set_postfix(loss=f"{loss_dict['loss_total']:.3f}")

            # Collect decoded predictions for mAP
            decoded = eval_model(images, return_decoded=True)
            for b in range(bs):
                all_preds.append({
                    "boxes":    decoded["boxes"][b],
                    "scores":   decoded["scores"][b],
                    "labels":   decoded["class_ids"][b],
                })
                all_targets.append({
                    "boxes":    gt_boxes[b][:, 1:],    # [cx,cy,w,h]
                    "labels":   gt_boxes[b][:, 0].long(),
                })

        # ── Compute mAP ──────────────────────────────────────────────
        map_50 = compute_map(all_preds, all_targets, iou_thresh=0.5)
        map_75 = compute_map(all_preds, all_targets, iou_thresh=0.75)

        val_metrics = tracker.averages()
        val_metrics["mAP@0.5"]  = map_50
        val_metrics["mAP@0.75"] = map_75

        return val_metrics

    # ------------------------------------------------------------------
    # TENSORBOARD LOGGING
    # ------------------------------------------------------------------

    def _tensorboard_log(self, epoch, train_m, val_m, lr):
        for k, v in train_m.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_m.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("lr", lr, epoch)

    # ------------------------------------------------------------------
    # BUILD CHECKPOINT STATE DICT
    # ------------------------------------------------------------------

    def _build_state(self, epoch: int) -> dict:
        return {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "ema_state":       self.ema.state_dict() if self.ema else None,
            "early_stop_state": self.early_stop.state_dict(),
            "best_map":        self.best_map,
            "run_name":        self.run_name,
        }


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def _fmt(metrics: dict) -> str:
    """Format a metric dict as a compact string."""
    key_map = {
        "loss_total":     "loss",
        "loss_seg":       "seg",
        "loss_hazard":    "haz",
        "loss_hierarchy": "hier",
    }
    parts = []
    for full_k, short_k in key_map.items():
        if full_k in metrics:
            parts.append(f"{short_k}={metrics[full_k]:.3f}")
    return "  ".join(parts)
