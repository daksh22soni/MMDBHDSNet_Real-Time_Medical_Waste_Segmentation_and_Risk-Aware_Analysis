"""
========================================================================
DBHDSNet Phase 3a — Post-Hoc Calibration
Implements Temperature Scaling (Guo et al., 2017) as a post-hoc
calibration method on top of the trained DBHDSNet hazard head.

A single scalar temperature T is learned on the validation set by
minimising NLL. T > 1 softens overconfident predictions; T < 1
sharpens underconfident ones. No model weights are changed.

Also provides:
  • Expected Calibration Error (ECE)
  • Adaptive Calibration Error (ACE)
  • Maximum Calibration Error (MCE)
  • Reliability diagram (confidence vs accuracy per bin)
  • Platt Scaling (alternative to Temperature Scaling)

Reference:
  Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file save
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ════════════════════════════════════════════════════════════════════════
# TEMPERATURE SCALING MODULE
# ════════════════════════════════════════════════════════════════════════

class TemperatureScaling(nn.Module):
    """
    Learnable scalar temperature T applied to logits before softmax.
        p(y|x) = softmax(logits / T)

    Only T is trained; the backbone and head are frozen.
    Optimised on validation-set NLL (cross-entropy).

    Can also be applied to the MC-Dropout mean logit for joint
    aleatoric + epistemic calibration.
    """

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        # log(T) is the actual parameter for numerical stability
        self.log_temp = nn.Parameter(
            torch.tensor([init_temp], dtype=torch.float32).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits and return calibrated probabilities."""
        return logits / self.temperature.clamp(min=0.1)

    def calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calibrate already-computed softmax probabilities.
        Re-derive logits via log, scale, and re-softmax.
        """
        eps    = 1e-8
        logits = (probs + eps).log()
        return torch.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

    def extra_repr(self) -> str:
        return f"T={self.temperature.item():.4f}"


# ════════════════════════════════════════════════════════════════════════
# CALIBRATION TRAINER
# ════════════════════════════════════════════════════════════════════════

class CalibrationTrainer:
    """
    Fits TemperatureScaling on validation logits using L-BFGS.

    Usage
    -----
    trainer = CalibrationTrainer(cfg)
    T_module = trainer.fit(val_logits, val_labels)
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.uq_cfg = cfg.UQ

    # ------------------------------------------------------------------

    def fit(
        self,
        val_logits: torch.Tensor,   # (N, num_tiers) raw logits on val set
        val_labels: torch.Tensor,   # (N,) true tier indices (0-indexed)
    ) -> TemperatureScaling:
        """
        Learn T via L-BFGS on validation NLL.
        Returns the fitted TemperatureScaling module.
        """
        T_module = TemperatureScaling(
            init_temp=self.uq_cfg.TEMP_INIT
        ).to(self.device)

        logits = val_logits.to(self.device)
        labels = val_labels.to(self.device).long()

        optimizer = torch.optim.LBFGS(
            [T_module.log_temp],
            lr           = self.uq_cfg.TEMP_LR,
            max_iter     = self.uq_cfg.TEMP_MAX_ITER,
            tolerance_grad = self.uq_cfg.TEMP_CONVERGENCE,
            line_search_fn = "strong_wolfe",
        )

        prev_loss = float("inf")

        def closure():
            optimizer.zero_grad()
            scaled_logits = T_module(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        print(f"[Calibration] Fitting Temperature Scaling "
              f"(init T={self.uq_cfg.TEMP_INIT:.3f})…")

        for step in range(50):
            loss = optimizer.step(closure)
            delta = abs(prev_loss - loss.item())
            if step % 10 == 0:
                print(f"  Step {step+1:3d}  NLL={loss.item():.6f}  "
                      f"T={T_module.temperature.item():.4f}  Δ={delta:.2e}")
            if delta < self.uq_cfg.TEMP_CONVERGENCE:
                print(f"  Converged at step {step+1}.")
                break
            prev_loss = loss.item()

        print(f"[Calibration] Fitted T = {T_module.temperature.item():.4f}")
        return T_module

    # ------------------------------------------------------------------

    def save(self, T_module: TemperatureScaling, path: Path):
        torch.save({
            "log_temp":   T_module.log_temp.data,
            "temperature": T_module.temperature.item(),
        }, path)
        print(f"[Calibration] Saved to {path}")

    def load(self, path: Path) -> TemperatureScaling:
        state = torch.load(path, map_location="cpu", weights_only=False)
        T_module = TemperatureScaling()
        T_module.log_temp.data = state["log_temp"]
        return T_module


# ════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ════════════════════════════════════════════════════════════════════════

class CalibrationMetrics:
    """
    Computes ECE, ACE, MCE and generates reliability diagrams.

    All metrics compare model confidence (max predicted probability)
    against empirical accuracy within confidence bins.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    # ------------------------------------------------------------------

    def compute_all(
        self,
        probs:  torch.Tensor,   # (N, C) softmax probabilities
        labels: torch.Tensor,   # (N,) true class indices
    ) -> Dict[str, float]:
        """
        Returns dict with ECE, ACE, MCE, Brier Score, NLL.
        """
        probs  = probs.cpu().numpy()
        labels = labels.cpu().numpy().astype(int)

        confs   = probs.max(axis=1)             # (N,) max confidence
        preds   = probs.argmax(axis=1)          # (N,) predicted class
        correct = (preds == labels).astype(float)

        bins = np.linspace(0, 1, self.n_bins + 1)
        N    = len(confs)

        bin_acc  = np.zeros(self.n_bins)
        bin_conf = np.zeros(self.n_bins)
        bin_n    = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (confs > bins[i]) & (confs <= bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc[i]  = correct[mask].mean()
            bin_conf[i] = confs[mask].mean()
            bin_n[i]    = mask.sum()

        # ECE — frequency-weighted mean gap
        ece = float(np.sum(bin_n / N * np.abs(bin_acc - bin_conf)))

        # ACE — equal-mass binning (adaptive)
        ace = self._ace(confs, correct)

        # MCE — maximum gap across bins
        nonempty = bin_n > 0
        mce = float(np.max(np.abs(bin_acc - bin_conf)[nonempty])) \
              if nonempty.any() else 0.0

        # Brier score (multi-class)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), labels] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

        # NLL
        nll = float(-np.log(probs[np.arange(N), labels] + 1e-8).mean())

        return {
            "ECE":         ece,
            "ACE":         ace,
            "MCE":         mce,
            "Brier":       brier,
            "NLL":         nll,
            # Bin arrays for plotting
            "_bin_acc":    bin_acc,
            "_bin_conf":   bin_conf,
            "_bin_n":      bin_n,
        }

    # ------------------------------------------------------------------

    def _ace(self, confs: np.ndarray, correct: np.ndarray) -> float:
        """Adaptive Calibration Error with equal-mass bins."""
        order  = np.argsort(confs)
        confs  = confs[order]
        correct = correct[order]
        N      = len(confs)
        bin_sz = N // self.n_bins
        if bin_sz == 0:
            return 0.0
        ace = 0.0
        for i in range(self.n_bins):
            sl   = slice(i * bin_sz, (i + 1) * bin_sz)
            gap  = abs(correct[sl].mean() - confs[sl].mean())
            ace += gap / self.n_bins
        return float(ace)

    # ------------------------------------------------------------------

    def reliability_diagram(
        self,
        before_metrics: dict,
        after_metrics:  dict,
        save_path:      Path,
        title:          str = "Hazard Classification — Reliability Diagram",
    ):
        """
        Saves a 2-panel reliability diagram comparing pre- and
        post-calibration predictions.
        """
        fig = plt.figure(figsize=(14, 6))
        gs  = gridspec.GridSpec(1, 2, figure=fig)

        for ax_idx, (metrics, label) in enumerate([
            (before_metrics, "Before Calibration"),
            (after_metrics,  "After Temperature Scaling"),
        ]):
            ax   = fig.add_subplot(gs[ax_idx])
            bacc = metrics["_bin_acc"]
            bcon = metrics["_bin_conf"]
            bn   = metrics["_bin_n"]
            bins = np.linspace(0, 1, self.n_bins + 1)
            centres = (bins[:-1] + bins[1:]) / 2

            nonempty = bn > 0
            ax.bar(
                centres[nonempty],
                bacc[nonempty],
                width   = 1.0 / self.n_bins,
                alpha   = 0.75,
                color   = "#4C9BE8",
                label   = "Accuracy",
                edgecolor = "white",
            )
            ax.plot(
                [0, 1], [0, 1],
                "k--",
                linewidth = 1.5,
                label     = "Perfect calibration",
            )
            ax.plot(
                centres[nonempty],
                bcon[nonempty],
                "o-",
                color     = "#E84C4C",
                linewidth = 1.5,
                markersize = 5,
                label     = "Confidence",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence", fontsize=12)
            ax.set_ylabel("Accuracy",   fontsize=12)
            ax.set_title(
                f"{label}\n"
                f"ECE={metrics['ECE']:.4f}  "
                f"MCE={metrics['MCE']:.4f}  "
                f"Brier={metrics['Brier']:.4f}",
                fontsize = 11,
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Calibration] Reliability diagram saved → {save_path}")
