"""
========================================================================
DBHDSNet Phase 3a — MC-Dropout Uncertainty Estimator
Implements Gal & Ghahramani (2016) Bayesian approximation via dropout.

Computes per-prediction:
  • Predictive entropy     H[y|x]    — total uncertainty
  • Expected entropy       E[H]      — aleatoric (data) uncertainty
  • Mutual information     MI        — epistemic (model) uncertainty ★
  • Mean prediction        p̄(y|x)   — final probabilistic output
  • Predictive variance    Var[p]    — per-class variance

The epistemic uncertainty (MI) is the key safety signal:
  high MI → model has not seen similar data → flag for human review.

Reference:
  Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
  Kendall & Gal (2017). What Uncertainties Do We Need in Bayesian
    Deep Learning for Computer Vision?
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ════════════════════════════════════════════════════════════════════════
# OUTPUT DATA CLASS
# ════════════════════════════════════════════════════════════════════════

@dataclass
class UQEstimate:
    """
    Container for all uncertainty estimates from one forward batch.

    Shapes assume B = batch size, T = num hazard tiers (4), N = MC passes.
    """
    mean_probs:          torch.Tensor   # (B, T)   — mean softmax probability
    predictive_entropy:  torch.Tensor   # (B,)     — total uncertainty H[y|x]
    aleatoric:           torch.Tensor   # (B,)     — E_θ[H[y|x,θ]]
    epistemic:           torch.Tensor   # (B,)     — mutual information (MI)
    variance:            torch.Tensor   # (B, T)   — per-class variance
    all_probs:           torch.Tensor   # (N, B, T) — raw stochastic samples
    flags:               torch.Tensor   # (B,)     — bool: True = human review
    flag_reason:         list           # (B,)     — reason strings per sample
    tier_thresholds:     dict           # tier → {low, high} used for flagging


# ════════════════════════════════════════════════════════════════════════
# DROPOUT ACTIVATION HELPER
# ════════════════════════════════════════════════════════════════════════

def enable_dropout(module: nn.Module):
    """Set all Dropout layers to train mode (active at inference)."""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def disable_dropout(module: nn.Module):
    """Set all Dropout layers back to eval mode."""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()


# ════════════════════════════════════════════════════════════════════════
# MC-DROPOUT ESTIMATOR
# ════════════════════════════════════════════════════════════════════════

class MCDropoutEstimator:
    """
    Wraps a trained DBHDSNet model and runs N stochastic forward passes
    with dropout active to estimate predictive uncertainty.

    Usage
    -----
    estimator = MCDropoutEstimator(model, cfg)
    uq = estimator.estimate(images, predicted_hazard_tiers)
    """

    def __init__(self, model: nn.Module, cfg):
        self.model         = model
        self.n_passes      = cfg.UQ.MC_N_PASSES
        self.uq_cfg        = cfg.UQ
        self.tier_thresh   = cfg.UQ.TIER_THRESHOLDS
        self.eps           = 1e-8

    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(
        self,
        images:          torch.Tensor,          # (B, 3, H, W)
        pred_tier_ids:   Optional[torch.Tensor] = None,  # (B,) predicted tier 1-4
        device:          torch.device = None,
    ) -> UQEstimate:
        """
        Run N MC-Dropout forward passes and compute uncertainty decomposition.

        Parameters
        ----------
        images        : input batch
        pred_tier_ids : predicted hazard tier per image (1-4), used for
                        tier-conditional flagging. If None, uses global thresholds.
        """
        device = device or images.device
        B      = images.shape[0]

        # ── Set model to eval (frozen BN) but activate dropout ────────
        self.model.eval()
        enable_dropout(self.model)

        all_probs = []   # list of (B, T) tensors

        for _ in range(self.n_passes):
            out    = self.model(images)
            logits = out["hazard_logits"]               # (B, T)
            probs  = torch.softmax(logits, dim=-1)      # (B, T)
            all_probs.append(probs.detach().cpu())

        disable_dropout(self.model)

        # Stack → (N, B, T)
        all_probs_t = torch.stack(all_probs, dim=0)     # (N, B, T)

        # ── Uncertainty decomposition ─────────────────────────────────
        mean_p     = all_probs_t.mean(dim=0)            # (B, T)
        variance   = all_probs_t.var(dim=0)             # (B, T)

        # Predictive entropy: H[y|x] = -Σ p̄ log p̄
        pred_entropy = self._entropy(mean_p)            # (B,)

        # Expected entropy (aleatoric): E[H[y|x,θ]] = (1/N) Σ_n H[p_n]
        per_pass_entropy = torch.stack(
            [self._entropy(all_probs_t[n]) for n in range(self.n_passes)], dim=0
        )  # (N, B)
        aleatoric = per_pass_entropy.mean(dim=0)        # (B,)

        # Epistemic = Mutual Information = Total - Aleatoric
        epistemic = (pred_entropy - aleatoric).clamp(min=0)   # (B,)

        # ── Flagging logic (hazard-conditional) ───────────────────────
        flags, reasons = self._flag_predictions(
            epistemic, pred_tier_ids, B
        )

        return UQEstimate(
            mean_probs         = mean_p,
            predictive_entropy = pred_entropy,
            aleatoric          = aleatoric,
            epistemic          = epistemic,
            variance           = variance,
            all_probs          = all_probs_t,
            flags              = flags,
            flag_reason        = reasons,
            tier_thresholds    = self.tier_thresh,
        )

    # ------------------------------------------------------------------

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H = -Σ p log p for each row."""
        return -(probs * (probs + self.eps).log()).sum(dim=-1)   # (B,)

    # ------------------------------------------------------------------

    def _flag_predictions(
        self,
        epistemic:     torch.Tensor,             # (B,)
        tier_ids:      Optional[torch.Tensor],   # (B,) or None
        B:             int,
    ) -> Tuple[torch.Tensor, list]:
        """
        Apply tier-conditional HITL flagging.

        Three zones per sample:
          • GREEN  (U < low_thresh)  → auto-accept
          • AMBER  (low ≤ U < high)  → log and monitor
          • RED    (U ≥ high_thresh) → mandatory human review flag

        Returns boolean flag tensor and reason strings.
        """
        flags   = torch.zeros(B, dtype=torch.bool)
        reasons = [""] * B

        for b in range(B):
            u = epistemic[b].item()

            if tier_ids is not None:
                tier  = int(tier_ids[b].item())
                thresh = self.tier_thresh.get(
                    tier,
                    {"low": self.uq_cfg.UNCERTAINTY_THRESH_LOW,
                     "high": self.uq_cfg.UNCERTAINTY_THRESH_HIGH}
                )
            else:
                thresh = {
                    "low":  self.uq_cfg.UNCERTAINTY_THRESH_LOW,
                    "high": self.uq_cfg.UNCERTAINTY_THRESH_HIGH,
                }

            if u < thresh["low"]:
                reasons[b] = f"GREEN  U={u:.4f} < {thresh['low']:.3f} → auto-accept"
            elif u < thresh["high"]:
                reasons[b] = f"AMBER  U={u:.4f} in [{thresh['low']:.3f}, {thresh['high']:.3f}] → monitor"
            else:
                flags[b]   = True
                reasons[b] = f"RED    U={u:.4f} ≥ {thresh['high']:.3f} → HUMAN REVIEW"

        return flags, reasons


# ════════════════════════════════════════════════════════════════════════
# DEEP ENSEMBLE COMBINER
# ════════════════════════════════════════════════════════════════════════

class DeepEnsembleEstimator:
    """
    Combines N independently trained DBHDSNet models.
    Each model produces a deterministic softmax; variance across models
    gives epistemic uncertainty without dropout.

    More reliable than MC-Dropout for out-of-distribution detection
    but requires N × training compute.

    Reference: Lakshminarayanan et al. (2017). Simple and Scalable
               Predictive Uncertainty Estimation Using Deep Ensembles.
    """

    def __init__(self, models: list, cfg):
        self.models    = models
        self.uq_cfg    = cfg.UQ
        self.eps       = 1e-8

    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, images: torch.Tensor) -> UQEstimate:
        """
        Forward pass through all ensemble members, combine predictions.
        Each model should be in eval() mode with dropout disabled.
        """
        B = images.shape[0]
        all_probs = []

        for model in self.models:
            model.eval()
            out   = model(images)
            probs = torch.softmax(out["hazard_logits"], dim=-1)
            all_probs.append(probs.detach().cpu())

        all_probs_t = torch.stack(all_probs, dim=0)     # (N, B, T)

        mean_p       = all_probs_t.mean(dim=0)
        variance     = all_probs_t.var(dim=0)
        pred_entropy = self._entropy(mean_p)
        aleatoric    = torch.stack(
            [self._entropy(all_probs_t[n]) for n in range(len(self.models))]
        ).mean(dim=0)
        epistemic    = (pred_entropy - aleatoric).clamp(min=0)

        flags  = epistemic > self.uq_cfg.UNCERTAINTY_THRESH_HIGH
        reasons = [
            f"ensemble U={epistemic[b].item():.4f}" for b in range(B)
        ]

        return UQEstimate(
            mean_probs         = mean_p,
            predictive_entropy = pred_entropy,
            aleatoric          = aleatoric,
            epistemic          = epistemic,
            variance           = variance,
            all_probs          = all_probs_t,
            flags              = flags,
            flag_reason        = reasons,
            tier_thresholds    = self.uq_cfg.TIER_THRESHOLDS,
        )

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * (probs + self.eps).log()).sum(dim=-1)
