"""
========================================================================
DBHDSNet Phase 3a — Human-in-the-Loop (HITL) Safety Protocol
Implements a three-zone safety decision engine on top of UQ estimates.

The protocol translates raw epistemic uncertainty into actionable
clinical decisions for the smart bin system:

  GREEN  (U < tier_low_thresh)   → Auto-accept, sort autonomously
  AMBER  (low ≤ U < high)        → Accept but log for quality audit
  RED    (U ≥ tier_high_thresh)  → Halt bin, alert human handler

For Tier-1 (Sharps) items, the RED threshold is tightened to 0.12
(vs 0.25 for general waste) — a sharps misclassification carries
potential bloodborne pathogen transmission risk.

Additionally produces:
  • Per-session HITL logs (JSON)
  • Alert statistics report
  • Uncertainty distribution plots
  • Confusion-weighted risk matrix
========================================================================
"""

import json
import time
import datetime
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ════════════════════════════════════════════════════════════════════════
# DECISION CONSTANTS
# ════════════════════════════════════════════════════════════════════════

ZONE_GREEN  = "GREEN"
ZONE_AMBER  = "AMBER"
ZONE_RED    = "RED"

HAZARD_NAMES = {1: "Sharps", 2: "Infectious", 3: "Pharmaceutical", 4: "General"}
RISK_COLOURS = {ZONE_GREEN: "#2ECC71", ZONE_AMBER: "#F39C12", ZONE_RED: "#E74C3C"}


# ════════════════════════════════════════════════════════════════════════
# PER-ITEM DECISION RECORD
# ════════════════════════════════════════════════════════════════════════

@dataclass
class HITLDecision:
    """Full audit record for one waste item classification decision."""
    item_id:          str
    timestamp:        str
    class_name:       str
    class_id:         int
    predicted_tier:   int           # 1–4
    tier_name:        str
    confidence:       float         # max softmax prob (mean of MC passes)
    epistemic_u:      float         # mutual information
    aleatoric_u:      float         # expected entropy
    predictive_entropy: float       # total entropy
    zone:             str           # GREEN | AMBER | RED
    action:           str           # auto_sort | log_and_sort | halt_and_alert
    low_thresh:       float
    high_thresh:      float
    n_mc_passes:      int
    top3_classes:     List[str]     # top-3 class names by mean prob
    top3_probs:       List[float]


# ════════════════════════════════════════════════════════════════════════
# HITL PROTOCOL ENGINE
# ════════════════════════════════════════════════════════════════════════

class HITLProtocol:
    """
    Evaluates each detected waste item against the three-zone safety
    policy and produces decision records for the audit trail.

    Parameters
    ----------
    cfg        : Config object with UQ tier thresholds
    class_names: ordered list of 38 class names
    log_dir    : directory for JSON audit logs
    """

    def __init__(self, cfg, class_names: List[str], log_dir: Path):
        self.cfg          = cfg
        self.uq_cfg       = cfg.UQ
        self.class_names  = class_names
        self.log_dir      = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session log accumulator
        self._session_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._decisions:  List[HITLDecision] = []
        self._zone_counts = defaultdict(int)

    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        uq_estimate,                   # UQEstimate from MCDropoutEstimator
        class_ids:    torch.Tensor,    # (B,) predicted class IDs
        item_ids:     Optional[List[str]] = None,
        n_mc_passes:  int = 30,
    ) -> List[HITLDecision]:
        """
        Evaluate a batch of predictions and return HITL decision records.

        Parameters
        ----------
        uq_estimate : UQEstimate dataclass from Phase 3a
        class_ids   : (B,) predicted class index per item
        item_ids    : optional unique identifiers per item
        """
        B         = uq_estimate.mean_probs.shape[0]
        decisions = []

        if item_ids is None:
            ts = datetime.datetime.now().strftime("%H%M%S")
            item_ids = [f"item_{ts}_{b:04d}" for b in range(B)]

        for b in range(B):
            cls_id    = int(class_ids[b].item())
            cls_name  = (self.class_names[cls_id]
                         if cls_id < len(self.class_names) else f"class_{cls_id}")

            # Infer predicted hazard tier from mean probs
            tier_pred = int(uq_estimate.mean_probs[b].argmax().item()) + 1  # 1-indexed

            # Tier-conditional thresholds
            thresh = self.uq_cfg.TIER_THRESHOLDS.get(
                tier_pred,
                {"low": self.uq_cfg.UNCERTAINTY_THRESH_LOW,
                 "high": self.uq_cfg.UNCERTAINTY_THRESH_HIGH}
            )

            u_ep = float(uq_estimate.epistemic[b].item())
            u_al = float(uq_estimate.aleatoric[b].item())
            u_pe = float(uq_estimate.predictive_entropy[b].item())
            conf = float(uq_estimate.mean_probs[b].max().item())

            # Zone determination
            if u_ep < thresh["low"]:
                zone   = ZONE_GREEN
                action = "auto_sort"
            elif u_ep < thresh["high"]:
                zone   = ZONE_AMBER
                action = "log_and_sort"
            else:
                zone   = ZONE_RED
                action = "halt_and_alert"

            # Top-3 classes by mean probability
            probs_np  = uq_estimate.mean_probs[b].numpy()
            top3_idx  = np.argsort(probs_np)[::-1][:3]
            top3_cls  = [
                (self.class_names[i] if i < len(self.class_names) else f"cls_{i}")
                for i in top3_idx
            ]
            top3_prob = [float(probs_np[i]) for i in top3_idx]

            decision = HITLDecision(
                item_id           = item_ids[b],
                timestamp         = datetime.datetime.now().isoformat(),
                class_name        = cls_name,
                class_id          = cls_id,
                predicted_tier    = tier_pred,
                tier_name         = HAZARD_NAMES.get(tier_pred, "Unknown"),
                confidence        = conf,
                epistemic_u       = u_ep,
                aleatoric_u       = u_al,
                predictive_entropy= u_pe,
                zone              = zone,
                action            = action,
                low_thresh        = thresh["low"],
                high_thresh       = thresh["high"],
                n_mc_passes       = n_mc_passes,
                top3_classes      = top3_cls,
                top3_probs        = top3_prob,
            )

            decisions.append(decision)
            self._decisions.append(decision)
            self._zone_counts[zone] += 1

        return decisions

    # ------------------------------------------------------------------

    def print_decision(self, d: HITLDecision):
        colour = {"GREEN": "\033[92m", "AMBER": "\033[93m", "RED": "\033[91m"}
        reset  = "\033[0m"
        c      = colour.get(d.zone, "")
        print(
            f"  {c}[{d.zone:5s}]{reset} "
            f"{d.class_name:30s} | "
            f"Tier {d.predicted_tier} ({d.tier_name:15s}) | "
            f"conf={d.confidence:.3f} | "
            f"U_ep={d.epistemic_u:.4f} | "
            f"→ {d.action}"
        )

    # ------------------------------------------------------------------

    def save_session_log(self) -> Path:
        """Persist all decisions from this session to JSON."""
        log_path = self.log_dir / f"hitl_session_{self._session_id}.json"
        records  = [asdict(d) for d in self._decisions]
        with open(log_path, "w") as f:
            json.dump({
                "session_id":  self._session_id,
                "n_items":     len(records),
                "zone_summary": dict(self._zone_counts),
                "decisions":   records,
            }, f, indent=2)
        return log_path

    # ------------------------------------------------------------------

    def session_summary(self) -> dict:
        """Returns aggregated statistics for this session."""
        n = len(self._decisions)
        if n == 0:
            return {"n_items": 0}

        ep_vals = [d.epistemic_u for d in self._decisions]
        return {
            "n_items":           n,
            "zone_GREEN":        self._zone_counts[ZONE_GREEN],
            "zone_AMBER":        self._zone_counts[ZONE_AMBER],
            "zone_RED":          self._zone_counts[ZONE_RED],
            "pct_flagged":       100 * self._zone_counts[ZONE_RED] / n,
            "pct_amber":         100 * self._zone_counts[ZONE_AMBER] / n,
            "mean_epistemic":    float(np.mean(ep_vals)),
            "p95_epistemic":     float(np.percentile(ep_vals, 95)),
            "max_epistemic":     float(np.max(ep_vals)),
        }

    # ------------------------------------------------------------------

    def plot_uncertainty_distribution(self, save_path: Path):
        """
        Histogram of epistemic uncertainty values split by zone,
        overlaid with the tier-1 and tier-4 threshold lines.
        """
        if not self._decisions:
            return

        ep_vals   = np.array([d.epistemic_u for d in self._decisions])
        zones     = [d.zone for d in self._decisions]
        tier_vals = [d.predicted_tier for d in self._decisions]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Panel 1: overall epistemic uncertainty histogram ──────────
        ax = axes[0]
        for zone, colour in RISK_COLOURS.items():
            mask = [z == zone for z in zones]
            if any(mask):
                ax.hist(
                    ep_vals[[i for i, m in enumerate(mask) if m]],
                    bins=30, alpha=0.65, color=colour, label=zone,
                    edgecolor="white",
                )

        # Draw threshold lines for Tier 1 (tightest)
        t1 = self.uq_cfg.TIER_THRESHOLDS.get(1, {})
        ax.axvline(t1.get("low",  0.05), color="gold",   linestyle="--",
                   linewidth=1.5, label="Tier-1 low thresh")
        ax.axvline(t1.get("high", 0.12), color="crimson", linestyle="--",
                   linewidth=1.5, label="Tier-1 high thresh")

        ax.set_xlabel("Epistemic Uncertainty (MI)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Epistemic Uncertainty Distribution by Zone", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 2: per-tier violin plot ─────────────────────────────
        ax2 = axes[1]
        tier_groups = defaultdict(list)
        for ep, t in zip(ep_vals, tier_vals):
            tier_groups[t].append(ep)

        positions = sorted(tier_groups.keys())
        data_viol = [tier_groups[t] for t in positions]
        xlabels   = [f"T{t}\n{HAZARD_NAMES.get(t,'?')}" for t in positions]

        parts = ax2.violinplot(data_viol, positions=list(range(len(positions))),
                               showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax2.set_xticks(range(len(positions)))
        ax2.set_xticklabels(xlabels, fontsize=10)
        ax2.set_ylabel("Epistemic Uncertainty (MI)", fontsize=12)
        ax2.set_title("Uncertainty by Predicted Hazard Tier", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[HITL] Uncertainty distribution plot saved → {save_path}")
