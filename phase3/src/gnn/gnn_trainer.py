"""
========================================================================
DBHDSNet Phase 3b — ContamRisk-GNN Trainer
Full training loop with:
  • Continuous per-epoch tqdm progress bars (separate train/val bars)
  • AMP (automatic mixed precision)
  • Cosine LR with warm-up
  • Gradient clipping
  • Checkpoint manager (best + last + periodic)
  • Early stopping on validation bin_MAE
  • TensorBoard logging
  • Terminal + file dual output (via DualOutput from utils.py)
========================================================================
"""

import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import (
    MetricTracker, EarlyStopping, CheckpointManager,
    TrainingHistory, gpu_info, format_time, print_banner, make_run_name,
)
from .gnn_losses import ContamRiskLoss, GNNMetrics


# ════════════════════════════════════════════════════════════════════════
# WARM-UP COSINE SCHEDULER
# ════════════════════════════════════════════════════════════════════════

class GNNScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 warmup_lr=1e-6, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.warmup_lr     = warmup_lr
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(self.warmup_epochs, 1)
            return [self.warmup_lr + scale * (b - self.warmup_lr)
                    for b in self.base_lrs]
        progress = (e - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (b - self.min_lr) * cos
                for b in self.base_lrs]


# ════════════════════════════════════════════════════════════════════════
# GNN TRAINER
# ════════════════════════════════════════════════════════════════════════

class GNNTrainer:
    """
    Training orchestrator for ContamRisk-GNN.

    Usage
    -----
    trainer = GNNTrainer(model, cfg, device)
    trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model, cfg, device: torch.device):
        self.model  = model
        self.cfg    = cfg
        self.gc     = cfg.GNN
        self.device = device

        self.run_name = make_run_name("contamrisk_gnn")

        # ── Loss ──────────────────────────────────────────────────────
        self.criterion = ContamRiskLoss(cfg)

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = self.gc.LR,
            weight_decay = self.gc.WEIGHT_DECAY,
        )

        # ── Scheduler ────────────────────────────────────────────────
        self.scheduler = GNNScheduler(
            self.optimizer,
            warmup_epochs = self.gc.WARMUP_EPOCHS,
            total_epochs  = self.gc.EPOCHS,
            min_lr        = 1e-6,
        )

        # ── AMP ───────────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=self.gc.AMP)

        # ── Early stopping ────────────────────────────────────────────
        self.early_stop = EarlyStopping(
            patience  = self.gc.PATIENCE,
            min_delta = 1e-4,
        )

        # ── Checkpoint manager ────────────────────────────────────────
        self.ckpt_mgr = CheckpointManager(
            cfg.DATA.CHECKPOINT_DIR / "gnn",
            keep_last_n = self.gc.KEEP_LAST_N,
        )

        # ── TensorBoard ───────────────────────────────────────────────
        self.writer = SummaryWriter(
            log_dir=str(Path(cfg.DATA.LOG_DIR) / self.run_name)
        )

        # ── History ───────────────────────────────────────────────────
        self.history = TrainingHistory(
            Path(cfg.DATA.LOG_DIR) / f"{self.run_name}_history.json"
        )

        self.start_epoch  = 0
        self.best_bin_mae = float("inf")

    # ------------------------------------------------------------------

    def resume(self, ckpt_path: Optional[Path] = None) -> bool:
        state = self.ckpt_mgr.load(ckpt_path)
        if state is None:
            print("[GNNTrainer] No checkpoint found — starting from scratch.")
            return False

        self.model.load_state_dict(state["model_state"], strict=False)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        if "early_stop_state" in state:
            self.early_stop.load_state_dict(state["early_stop_state"])

        self.start_epoch  = state["epoch"] + 1
        self.best_bin_mae = state.get("best_bin_mae", float("inf"))

        tqdm.write(
            f"[GNNTrainer] Resumed from epoch {state['epoch']:04d} "
            f"(best bin_MAE={self.best_bin_mae:.5f})"
        )
        return True

    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader):
        print_banner(
            f"ContamRisk-GNN Training — run: {self.run_name} | "
            f"Device: {self.device} | Epochs: {self.gc.EPOCHS}"
        )
        print(gpu_info())
        print()
        t0 = time.time()

        for epoch in range(self.start_epoch, self.gc.EPOCHS):

            # ── Train ─────────────────────────────────────────────────
            train_m = self._train_epoch(epoch, train_loader)

            # ── Validate ──────────────────────────────────────────────
            val_m   = self._val_epoch(epoch, val_loader)

            # ── Scheduler ────────────────────────────────────────────
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # ── TensorBoard ──────────────────────────────────────────
            for k, v in train_m.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", lr, epoch)

            # ── Epoch summary ─────────────────────────────────────────
            bin_mae = val_m.get("bin_MAE", 999.0)
            elapsed = format_time(time.time() - t0)
            tqdm.write(
                f"\n  Epoch {epoch+1:04d}/{self.gc.EPOCHS:04d}  "
                f"│ Train loss={train_m.get('loss_total', 0):.4f}  "
                f"│ Val bin_MAE={bin_mae:.5f}  "
                f"cls_acc={val_m.get('cls_accuracy', 0):.4f}  "
                f"│ LR={lr:.2e}  │ ⏱ {elapsed}"
            )

            # ── Checkpoint ───────────────────────────────────────────
            # Note: for GNN we minimise MAE (lower = better)
            is_best  = bin_mae < self.best_bin_mae
            if is_best:
                self.best_bin_mae = bin_mae
                tqdm.write(f"  ★ New best bin_MAE: {self.best_bin_mae:.5f}")

            periodic = (epoch + 1) % self.gc.SAVE_EVERY == 0
            self.ckpt_mgr.save(
                self._build_state(epoch), is_best, epoch + 1, periodic
            )

            # ── History ───────────────────────────────────────────────
            self.history.append({
                "epoch": epoch + 1, "lr": lr,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}":   v for k, v in val_m.items()},
            })

            # ── Early stopping (on bin_MAE — minimise) ────────────────
            # Invert for EarlyStopping which maximises
            if self.early_stop(-bin_mae):
                tqdm.write(
                    f"\n[GNNTrainer] EarlyStopping triggered after "
                    f"{self.gc.PATIENCE} epochs without improvement."
                )
                break

        total_time = format_time(time.time() - t0)
        print_banner(
            f"GNN Training complete!  Best bin_MAE = {self.best_bin_mae:.5f}"
            f"  │  Total: {total_time}"
        )
        self.writer.close()

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, loader) -> dict:
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(
            loader,
            desc  = f"Epoch {epoch+1:04d}/{self.gc.EPOCHS:04d} [GNN TRAIN]",
            ncols = 130,
            leave = True,
            dynamic_ncols = False,
        )

        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.gc.AMP):
                out              = self.model(batch)
                loss, loss_dict  = self.criterion(out, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gc.GRAD_CLIP
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = int(batch.num_graphs)
            tracker.update(loss_dict, n=bs)
            avgs = tracker.averages()

            pbar.set_postfix(OrderedDict([
                ("total",  f"{avgs.get('loss_total',  0):.4f}"),
                ("item",   f"{avgs.get('loss_item',   0):.4f}"),
                ("bin",    f"{avgs.get('loss_bin',    0):.4f}"),
                ("cls",    f"{avgs.get('loss_cls',    0):.4f}"),
                ("contam", f"{avgs.get('loss_contam', 0):.4f}"),
                ("lr",     f"{self.optimizer.param_groups[0]['lr']:.2e}"),
            ]))

        return tracker.averages()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int, loader) -> dict:
        self.model.eval()
        tracker  = MetricTracker()
        gnn_eval = GNNMetrics()

        pbar = tqdm(
            loader,
            desc  = f"Epoch {epoch+1:04d}/{self.gc.EPOCHS:04d} [GNN VAL]  ",
            ncols = 130,
            leave = False,
            dynamic_ncols = False,
        )

        for batch in pbar:
            batch = batch.to(self.device)
            with autocast(enabled=self.gc.AMP):
                out             = self.model(batch)
                loss, loss_dict = self.criterion(out, batch)

            bs = int(batch.num_graphs)
            tracker.update(loss_dict, n=bs)
            gnn_eval.update(out, batch)
            pbar.set_postfix(loss=f"{loss_dict['loss_total']:.4f}")

        val_metrics = tracker.averages()
        val_metrics.update(gnn_eval.compute())
        return val_metrics

    # ------------------------------------------------------------------

    def _build_state(self, epoch: int) -> dict:
        return {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "early_stop_state": self.early_stop.state_dict(),
            "best_bin_mae":    self.best_bin_mae,
            "run_name":        self.run_name,
        }
