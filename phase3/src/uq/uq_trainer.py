"""
========================================================================
DBHDSNet Phase 3a — UQ Trainer

Fine-tunes DBHDSNet hazard head with an asymmetric uncertainty-aware
loss, then fits post-hoc calibrators and runs MC-Dropout evaluation.

Novel UQ Loss:
  L_uq = L_nll  +  λ_ent · L_entropy_reg  +  λ_tier · L_tier_certainty

  L_nll            : risk-weighted NLL on hazard tier predictions
  L_entropy_reg    : penalise low entropy on WRONG predictions
                     (wrong + overconfident = bad calibration)
  L_tier_certainty : push model CONFIDENT on CORRECT high-risk preds
                     (correct sharps/infectious should be certain)

Asymmetry principle:
  correct + high-risk  → REWARD certainty
  wrong   + any risk   → REWARD calibrated uncertainty

Only fusion gating + hazard MLP + LoRA matrices are trainable.
Backbone, FPN, and Branch A remain frozen throughout.
========================================================================
"""

import time
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import (
    AverageMeter, MetricTracker, ModelEMA, EarlyStopping,
    CheckpointManager, TrainingHistory, gpu_info,
    format_time, print_banner, make_run_name,
)
from .mc_dropout import MCDropoutSampler, UncertaintyAggregator
from .calibration import (
    CalibratorManager, collect_logits,
    compute_calibration_metrics, compute_reliability_diagram,
)


# ════════════════════════════════════════════════════════════════════════
# 1 — UNCERTAINTY-AWARE LOSS
# ════════════════════════════════════════════════════════════════════════

class UncertaintyAwareLoss(nn.Module):
    """
    Novel Phase 3a loss for DBHDSNet hazard head.
    See module docstring for full derivation.
    """

    TIER_RISK_WEIGHT = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0}

    def __init__(self, lambda_ent: float = 0.5, lambda_tier: float = 1.0):
        super().__init__()
        self.lambda_ent  = lambda_ent
        self.lambda_tier = lambda_tier

    def forward(
        self,
        hazard_logits: torch.Tensor,       # (B, 4)
        gt_hazard:     List[torch.Tensor],
        mc_samples:    Optional[torch.Tensor] = None,   # (N, B, 4)
    ) -> Tuple[torch.Tensor, Dict]:
        device = hazard_logits.device
        B      = hazard_logits.shape[0]

        # Image-level worst tier (0-indexed)
        tier_labels = torch.tensor(
            [int(t.min().item()) - 1 if t.numel() > 0 else 3
             for t in gt_hazard],
            dtype=torch.long, device=device,
        )

        # ── 1. Risk-weighted NLL ──────────────────────────────────────
        w = torch.tensor(
            [self.TIER_RISK_WEIGHT.get(int(l.item()) + 1, 1.0) for l in tier_labels],
            dtype=torch.float32, device=device,
        )
        nll      = F.nll_loss(F.log_softmax(hazard_logits, dim=-1),
                              tier_labels, reduction="none")
        loss_nll = (nll * w).mean()

        # ── 2. Entropy regularisation on wrong predictions ────────────
        loss_ent = torch.tensor(0., device=device)
        if mc_samples is not None and self.lambda_ent > 0:
            mean_p   = mc_samples.mean(dim=0)                   # (B, T)
            H        = -(mean_p * (mean_p + 1e-10).log()).sum(dim=-1)  # (B,)
            wrong    = (hazard_logits.argmax(dim=-1) != tier_labels).float()
            loss_ent = (wrong * (-H)).mean()   # penalise low-entropy wrong preds

        # ── 3. Tier certainty on correct high-risk preds ──────────────
        loss_tier = torch.tensor(0., device=device)
        if self.lambda_tier > 0:
            probs   = F.softmax(hazard_logits, dim=-1)
            correct = (hazard_logits.argmax(dim=-1) == tier_labels)
            for b in range(B):
                tl = int(tier_labels[b].item()) + 1   # 1-indexed
                if tl in (1, 2) and correct[b]:
                    loss_tier = loss_tier - probs[b, tier_labels[b]].log()
            loss_tier = loss_tier / max(B, 1)

        total = (loss_nll
                 + self.lambda_ent  * loss_ent
                 + self.lambda_tier * loss_tier)

        return total, {
            "loss_nll":      loss_nll.item(),
            "loss_ent":      loss_ent.item(),
            "loss_tier":     loss_tier.item(),
            "loss_uq_total": total.item(),
        }


# ════════════════════════════════════════════════════════════════════════
# 2 — PARAMETER GROUPS (freeze everything except UQ-relevant layers)
# ════════════════════════════════════════════════════════════════════════

def get_uq_param_groups(model, lr: float) -> List[Dict]:
    for p in model.parameters():
        p.requires_grad = False

    uq_params = []
    # Hazard head (Branch B MLP)
    for p in model.branch_b.hazard_head.parameters():
        p.requires_grad = True
        uq_params.append(p)
    # LoRA matrices in Branch B transformer blocks
    for name, p in model.branch_b.blocks.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            uq_params.append(p)
    # Fusion gating layers
    for name, p in model.fusion.named_parameters():
        if "gate" in name or "refine" in name:
            p.requires_grad = True
            uq_params.append(p)

    n = sum(p.numel() for p in uq_params)
    print(f"[UQ Trainer] Trainable params: {n:,}  "
          f"({100*n/sum(p.numel() for p in model.parameters()):.2f}% of model)")
    return [{"params": uq_params, "lr": lr}]


# ════════════════════════════════════════════════════════════════════════
# 3 — COSINE + WARMUP SCHEDULER
# ════════════════════════════════════════════════════════════════════════

class UQScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup: int, total: int, min_lr: float = 1e-7):
        self.warmup  = warmup
        self.total   = total
        self.min_lr  = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            return [b * (e + 1) / max(self.warmup, 1) for b in self.base_lrs]
        prog = (e - self.warmup) / max(self.total - self.warmup, 1)
        return [self.min_lr + (b - self.min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
                for b in self.base_lrs]


# ════════════════════════════════════════════════════════════════════════
# 4 — PHASE 3a TRAINER
# ════════════════════════════════════════════════════════════════════════

class Phase3aTrainer:
    """
    Orchestrates UQ fine-tuning → calibration → MC-Dropout evaluation.

    Step 1: fine-tune with UncertaintyAwareLoss  (20 epochs)
    Step 2: fit Temperature + Vector scaling on val set
    Step 3: MC-Dropout N=30 passes on test set → flagging report
    Step 4: generate all visualisation data
    """

    def __init__(self, model, cfg, device: torch.device):
        self.model    = model
        self.cfg      = cfg
        self.uqcfg    = cfg.UQ
        self.device   = device
        self.run_name = make_run_name("phase3a_uq")

        ckpt_dir = Path(cfg.DATA.PHASE3A_CKPT_DIR)
        log_dir  = Path(cfg.DATA.LOG_DIR)

        self.criterion  = UncertaintyAwareLoss(
            lambda_ent  = 0.5,
            lambda_tier = 1.0,
        ).to(device)
        self.scaler_amp = GradScaler(enabled=True)

        param_groups = get_uq_param_groups(model, self.uqcfg.UQ_LR)
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.uqcfg.UQ_WEIGHT_DECAY,
        )
        self.scheduler  = UQScheduler(
            self.optimizer,
            warmup = self.uqcfg.UQ_WARMUP_EPOCHS,
            total  = self.uqcfg.UQ_FINETUNE_EPOCHS,
        )
        self.ema        = ModelEMA(model, decay=0.9999)
        self.early_stop = EarlyStopping(patience=self.uqcfg.UQ_PATIENCE)
        self.ckpt_mgr   = CheckpointManager(ckpt_dir, keep_last_n=3)
        self.writer     = SummaryWriter(log_dir=str(log_dir / self.run_name))
        self.history    = TrainingHistory(log_dir / f"{self.run_name}_history.json")

        self.mc_sampler = MCDropoutSampler(
            model             = model,
            n_passes          = self.uqcfg.MC_FORWARD_PASSES,
            device            = device,
            tier_thresholds   = self.uqcfg.UNCERTAINTY_THRESH_BY_TIER,
            default_threshold = self.uqcfg.UNCERTAINTY_FLAG_THRESH,
        )
        self.calibrator = CalibratorManager(
            num_classes = cfg.MODEL.NUM_HAZARD_TIERS,
            save_dir    = ckpt_dir / "calibrators",
        )

        self.start_epoch = 0
        self.best_ece    = float("inf")

    # ------------------------------------------------------------------

    def resume(self, ckpt_path: Optional[Path] = None) -> bool:
        state = self.ckpt_mgr.load(ckpt_path)
        if state is None:
            return False
        self.model.load_state_dict(state["model_state"], strict=False)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if self.ema and "ema_state" in state:
            self.ema.load_state_dict(state["ema_state"])
        if "early_stop_state" in state:
            self.early_stop.load_state_dict(state["early_stop_state"])
        self.start_epoch = state.get("epoch", 0) + 1
        self.best_ece    = state.get("best_ece", float("inf"))
        tqdm.write(f"[Phase3a] Resumed epoch {state['epoch']:04d} | ECE={self.best_ece:.4f}")
        return True

    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print_banner(f"Phase 3a — UQ Fine-tuning  |  Run: {self.run_name}")
        print(gpu_info())
        t0 = time.time()

        for epoch in range(self.start_epoch, self.uqcfg.UQ_FINETUNE_EPOCHS):
            tr = self._train_epoch(epoch, train_loader)
            vl = self._val_epoch(epoch, val_loader)
            self.scheduler.step()

            lr      = self.optimizer.param_groups[0]["lr"]
            val_ece = vl.get("val_ece", float("inf"))
            is_best = val_ece < self.best_ece
            if is_best:
                self.best_ece = val_ece

            for k, v in tr.items():
                self.writer.add_scalar(f"phase3a/train/{k}", v, epoch)
            for k, v in vl.items():
                self.writer.add_scalar(f"phase3a/val/{k}", v, epoch)
            self.writer.add_scalar("phase3a/lr", lr, epoch)

            self.ckpt_mgr.save({
                "epoch":            epoch,
                "model_state":      self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "scheduler_state":  self.scheduler.state_dict(),
                "ema_state":        self.ema.state_dict() if self.ema else None,
                "early_stop_state": self.early_stop.state_dict(),
                "best_ece":         self.best_ece,
            }, is_best, epoch + 1,
            periodic=((epoch + 1) % self.uqcfg.UQ_SAVE_EVERY == 0))

            flag = "★ " if is_best else "  "
            tqdm.write(
                f"\n  {flag}Ep {epoch+1:03d}/{self.uqcfg.UQ_FINETUNE_EPOCHS:03d}  "
                f"│ {_fmt(tr)}  │ ECE={val_ece:.4f}  LR={lr:.2e}  "
                f"⏱ {format_time(time.time()-t0)}"
            )
            self.history.append({"epoch": epoch+1, "lr": lr,
                                  **{f"tr_{k}": v for k, v in tr.items()},
                                  **{f"vl_{k}": v for k, v in vl.items()}})

            if self.early_stop(1.0 - val_ece):
                tqdm.write("[Phase3a] Early stopping triggered.")
                break

        print_banner(f"UQ Fine-tuning done!  Best ECE = {self.best_ece:.4f}")
        self.writer.close()

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, loader: DataLoader) -> Dict:
        self.model.train()
        tracker = MetricTracker()
        E = self.uqcfg.UQ_FINETUNE_EPOCHS
        pbar = tqdm(loader, desc=f"UQ {epoch+1:03d}/{E:03d} [TRAIN]",
                    ncols=130, leave=True)

        for batch in pbar:
            images    = batch["images"].to(self.device, non_blocking=True)
            gt_hazard = [h.to(self.device) for h in batch["hazard_tiers"]]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                out = self.model(images)
                with torch.no_grad():
                    mc = self.mc_sampler.sample_batch(images).to(self.device)
                loss, ld = self.criterion(out["hazard_logits"], gt_hazard, mc)

            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()
            if self.ema:
                self.ema.update(self.model)

            tracker.update(ld, n=images.size(0))
            a = tracker.averages()
            pbar.set_postfix(OrderedDict([
                ("total", f"{a.get('loss_uq_total',0):.4f}"),
                ("nll",   f"{a.get('loss_nll',     0):.4f}"),
                ("ent",   f"{a.get('loss_ent',     0):.4f}"),
                ("tier",  f"{a.get('loss_tier',    0):.4f}"),
            ]))
        return tracker.averages()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int, loader: DataLoader) -> Dict:
        em = self.ema.ema if self.ema else self.model
        em.eval()
        tracker = MetricTracker()
        logits_list, labels_list = [], []
        E = self.uqcfg.UQ_FINETUNE_EPOCHS
        pbar = tqdm(loader, desc=f"UQ {epoch+1:03d}/{E:03d} [VAL]  ",
                    ncols=130, leave=False)

        for batch in pbar:
            images    = batch["images"].to(self.device, non_blocking=True)
            gt_hazard = [h.to(self.device) for h in batch["hazard_tiers"]]

            with autocast(enabled=True):
                out  = em(images)
                hl   = out["hazard_logits"]
                loss, ld = self.criterion(hl, gt_hazard, None)

            tracker.update(ld, n=images.size(0))
            logits_list.append(hl.cpu())
            for t in gt_hazard:
                labels_list.append(int(t.min().item()) - 1 if t.numel() > 0 else 3)
            pbar.set_postfix(nll=f"{ld['loss_nll']:.4f}")

        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.tensor(labels_list, dtype=torch.long)
        cal = compute_calibration_metrics(all_logits, all_labels, self.uqcfg.N_BINS)

        out_d = tracker.averages()
        out_d["val_ece"]   = cal["ece"]
        out_d["val_ace"]   = cal["ace"]
        out_d["val_brier"] = cal["brier"]
        out_d["val_nll"]   = cal["nll"]
        out_d["val_acc"]   = cal["accuracy"]
        return out_d

    # ------------------------------------------------------------------

    def run_calibration(self, val_loader, test_loader) -> Dict:
        print_banner("Post-Hoc Calibration")
        em = self.ema.ema if self.ema else self.model
        lv, lv_lbl = collect_logits(em, val_loader,  self.device,
                                    use_ema=False, desc="Val logits")
        lt, lt_lbl = collect_logits(em, test_loader, self.device,
                                    use_ema=False, desc="Test logits")

        self.calibrator.fit(lv, lv_lbl, verbose=True)
        results = self.calibrator.evaluate(lt, lt_lbl, self.uqcfg.N_BINS)

        print("\n  Test calibration:")
        for method, m in results.items():
            print(f"    {method:20s}  ECE={m['ece']:.4f}  "
                  f"ACE={m['ace']:.4f}  Brier={m['brier']:.4f}")

        self.calibrator.save()

        # Save reliability diagram data
        out_dir = Path(self.cfg.DATA.OUTPUT_DIR) / "calibration"
        for name, logits in [
            ("uncalibrated",  lt),
            ("temp_scaling",  self.calibrator.temp_scaler(lt).detach()),
            ("vector_scaling", self.calibrator.vec_scaler(lt).detach()),
        ]:
            diag = compute_reliability_diagram(logits, lt_lbl, self.uqcfg.N_BINS)
            import numpy as np
            np.save(out_dir / f"reliability_{name}.npy", diag)

        import json
        with open(out_dir / "calibration_report.json", "w") as f:
            json.dump({k: {m: float(v) for m, v in vals.items()
                           if isinstance(v, (int, float))}
                       for k, vals in results.items()}, f, indent=2)
        return results

    # ------------------------------------------------------------------

    def run_mc_uncertainty(self, test_loader) -> Tuple[Dict, UncertaintyAggregator]:
        print_banner("MC-Dropout Uncertainty Estimation (Test Set)")
        results    = self.mc_sampler.estimate_loader(test_loader)
        aggregator = UncertaintyAggregator()
        aggregator.add(results)
        summary    = aggregator.summary()

        print(f"\n  UQ Summary ({summary['n_images']} images):")
        print(f"    Mean epistemic  : {summary['mean_epistemic']:.4f}")
        print(f"    P90  epistemic  : {summary['p90_epistemic']:.4f}")
        print(f"    Mean BALD       : {summary['mean_mutual_info']:.4f}")
        print(f"    Flag rate (all) : {summary['flag_rate']*100:.1f}%  "
              f"({summary['n_flagged']} / {summary['n_images']})")
        for t, rate in summary["tier_flag_rates"].items():
            lbl = {1:"Sharps",2:"Infectious",3:"Pharma",4:"General"}.get(t,"?")
            print(f"    Flag rate T{t} {lbl:12s}: {rate*100:.1f}%")

        import json
        rpt = Path(self.cfg.DATA.OUTPUT_DIR) / "calibration" / "uq_report.json"
        with open(rpt, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  UQ report → {rpt}")
        return summary, aggregator


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def _fmt(m: Dict) -> str:
    return "  ".join(
        f"{k.split('_')[-1]}={v:.3f}" for k, v in m.items() if "loss" in k
    )

from collections import OrderedDict
