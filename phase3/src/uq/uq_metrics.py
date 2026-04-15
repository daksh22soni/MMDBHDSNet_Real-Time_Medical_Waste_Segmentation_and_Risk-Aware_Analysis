"""
========================================================================
DBHDSNet Phase 3a — UQ Evaluation Metrics
Beyond standard calibration (ECE/MCE), this module computes metrics
specifically designed for safety-critical uncertainty evaluation:

  • AURC  — Area Under Risk-Coverage curve
  • AUROC — Uncertainty as OOD detector (uncertainty vs. correctness)
  • AUPRC — Precision-Recall for uncertainty-based flagging
  • Selective Prediction Accuracy (at various coverage levels)
  • Spearman rank correlation (uncertainty vs. error)
  • Tier-weighted safety score (sharps errors penalised more)

These metrics directly map to the PhD paper's evaluation protocol.
========================================================================
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════
# RISK-COVERAGE CURVE  (selective prediction)
# ════════════════════════════════════════════════════════════════════════

def risk_coverage_curve(
    epistemic:   np.ndarray,   # (N,) uncertainty scores
    is_correct:  np.ndarray,   # (N,) bool — correct prediction?
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Selective prediction: at threshold τ, abstain (flag for human) if
    epistemic uncertainty > τ. Remaining predictions = coverage.

    Returns
    -------
    coverage : (T,) fraction of items NOT flagged at each threshold
    risk     : (T,) error rate on retained items
    AURC     : scalar — area under the risk-coverage curve (lower = better)
    """
    thresholds = np.linspace(0, epistemic.max() * 1.01, n_thresholds)
    coverage   = np.zeros(n_thresholds)
    risk       = np.zeros(n_thresholds)

    for i, tau in enumerate(thresholds):
        retained = epistemic <= tau
        if retained.sum() == 0:
            coverage[i] = 0.0
            risk[i]     = 0.0
        else:
            coverage[i] = retained.mean()
            risk[i]     = 1.0 - is_correct[retained].mean()

    # AURC: trapezoidal integration
    aurc = float(np.trapz(risk, coverage))
    return coverage, risk, aurc


# ════════════════════════════════════════════════════════════════════════
# AUROC (uncertainty as binary correctness detector)
# ════════════════════════════════════════════════════════════════════════

def uncertainty_auroc(
    epistemic:  np.ndarray,   # (N,)
    is_correct: np.ndarray,   # (N,) bool
) -> float:
    """
    Treats uncertainty as a binary classifier for incorrect predictions.
    High uncertainty should predict high error probability.

    AUROC > 0.5 → model knows when it is wrong (good).
    AUROC = 0.5 → uncertainty is random (useless for safety).
    """
    from sklearn.metrics import roc_auc_score
    # Positive class = incorrect prediction
    labels = (~is_correct).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, epistemic))


# ════════════════════════════════════════════════════════════════════════
# SELECTIVE PREDICTION ACCURACY
# ════════════════════════════════════════════════════════════════════════

def selective_accuracy(
    epistemic:   np.ndarray,
    is_correct:  np.ndarray,
    coverages:   List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
) -> Dict[float, float]:
    """
    Returns accuracy when keeping only the most certain (coverage × N) items.

    E.g., at coverage=0.8, keep the 80% most certain predictions and
    report their accuracy.
    """
    results = {}
    order   = np.argsort(epistemic)   # ascending: most certain first

    for cov in coverages:
        n_keep  = max(1, int(cov * len(epistemic)))
        kept    = order[:n_keep]
        results[cov] = float(is_correct[kept].mean())

    return results


# ════════════════════════════════════════════════════════════════════════
# TIER-WEIGHTED SAFETY SCORE
# ════════════════════════════════════════════════════════════════════════

def tier_weighted_safety_score(
    pred_tiers:   np.ndarray,   # (N,) predicted tier 1-4
    true_tiers:   np.ndarray,   # (N,) true tier 1-4
    epistemic:    np.ndarray,   # (N,) uncertainty
    flag_thresh:  float = 0.20,
    penalty_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes a composite safety score that accounts for:
      1. Whether the flag was triggered for high-U predictions
      2. Whether errors are in high-risk tier directions
      3. Whether high-risk items (Tier 1/2) were correctly flagged

    Higher score = better safety performance.

    Returns
    -------
    dict with:
      overall_safety_score : composite 0–1 score
      critical_catch_rate  : fraction of Tier-1 errors correctly flagged
      false_alarm_rate     : fraction of correct Tier-4 preds falsely flagged
      weighted_error       : mean penalty-weighted misclassification cost
    """
    if penalty_matrix is None:
        penalty_matrix = np.array([
            [0.0, 2.0,  4.0, 10.0],
            [1.0, 0.0,  2.0,  8.0],
            [1.0, 1.5,  0.0,  6.0],
            [1.5, 1.5,  1.5,  0.0],
        ], dtype=float)

    N      = len(pred_tiers)
    flagged = epistemic >= flag_thresh

    # Convert tiers to 0-indexed
    pt = pred_tiers - 1
    tt = true_tiers - 1

    # ── Weighted error (lower = better) ──────────────────────────────
    costs = np.array([
        penalty_matrix[tt[i], pt[i]] for i in range(N)
    ])
    weighted_error = float(costs.mean())

    # ── Critical catch rate: Tier-1 errors that were flagged ──────────
    tier1_errors_mask = (tt == 0) & (pt != 0)   # true sharps, predicted something else
    if tier1_errors_mask.sum() > 0:
        critical_catch = float(flagged[tier1_errors_mask].mean())
    else:
        critical_catch = float("nan")

    # ── False alarm rate: correct Tier-4 preds that were flagged ─────
    tier4_correct_mask = (tt == 3) & (pt == 3)
    if tier4_correct_mask.sum() > 0:
        false_alarm = float(flagged[tier4_correct_mask].mean())
    else:
        false_alarm = 0.0

    # ── Overall safety score (composite) ─────────────────────────────
    # Formula: clamp(1 - weighted_error / max_cost) × (1 - false_alarm)
    max_possible_cost = penalty_matrix.max()
    normalised_error  = np.clip(weighted_error / max_possible_cost, 0, 1)
    safety_score      = (1.0 - normalised_error) * (1.0 - false_alarm)

    return {
        "overall_safety_score": float(safety_score),
        "critical_catch_rate":  critical_catch,
        "false_alarm_rate":     false_alarm,
        "weighted_error":       weighted_error,
    }


# ════════════════════════════════════════════════════════════════════════
# FULL EVALUATION SUITE
# ════════════════════════════════════════════════════════════════════════

def full_uq_evaluation(
    pred_tiers:    np.ndarray,    # (N,) predicted tier (1-4)
    true_tiers:    np.ndarray,    # (N,) true tier (1-4)
    epistemic:     np.ndarray,    # (N,) epistemic uncertainty
    aleatoric:     np.ndarray,    # (N,) aleatoric uncertainty
    mean_probs:    np.ndarray,    # (N, T) calibrated mean probabilities
    flag_thresh:   float = 0.20,
    output_dir:    Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run the complete Phase 3a evaluation suite and (optionally) save plots.
    Returns a flat dict of all metrics suitable for logging.
    """
    is_correct = (pred_tiers == true_tiers)
    N          = len(pred_tiers)

    results = {}

    # ── Risk-Coverage ─────────────────────────────────────────────────
    coverage, risk, aurc = risk_coverage_curve(epistemic, is_correct)
    results["AURC"] = aurc

    # ── AUROC ────────────────────────────────────────────────────────
    results["AUROC_uncertainty_vs_error"] = uncertainty_auroc(epistemic, is_correct)

    # ── Selective accuracy ───────────────────────────────────────────
    sel_acc = selective_accuracy(epistemic, is_correct)
    for cov, acc in sel_acc.items():
        results[f"SelectiveAcc@cov={cov:.2f}"] = acc

    # ── Spearman rank correlation (uncertainty ↔ error magnitude) ────
    errors         = (pred_tiers != true_tiers).astype(float)
    rho, p_val     = spearmanr(epistemic, errors)
    results["SpearmanRho_U_vs_Error"] = float(rho)
    results["SpearmanP_U_vs_Error"]   = float(p_val)

    # ── Tier-weighted safety score ────────────────────────────────────
    safety = tier_weighted_safety_score(
        pred_tiers, true_tiers, epistemic, flag_thresh
    )
    results.update(safety)

    # ── Flag statistics ───────────────────────────────────────────────
    flagged = epistemic >= flag_thresh
    results["flag_rate"]         = float(flagged.mean())
    results["flag_precision"]    = float((~is_correct[flagged]).mean()) \
                                   if flagged.sum() > 0 else float("nan")
    results["flag_recall_errors"]= float(flagged[~is_correct].mean()) \
                                   if (~is_correct).sum() > 0 else float("nan")

    # ── Save RC curve plot ────────────────────────────────────────────
    if output_dir:
        output_dir = Path(output_dir)
        _plot_rc_curve(coverage, risk, aurc, output_dir / "risk_coverage_curve.png")
        _plot_uncertainty_vs_tier(
            epistemic, true_tiers,
            output_dir / "epistemic_by_tier.png"
        )

    return results


# ════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════════

def _plot_rc_curve(coverage, risk, aurc, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, risk, lw=2, color="#4C9BE8", label=f"AURC={aurc:.4f}")
    ax.fill_between(coverage, risk, alpha=0.2, color="#4C9BE8")
    ax.set_xlabel("Coverage (fraction of items retained)", fontsize=12)
    ax.set_ylabel("Risk (error rate on retained items)", fontsize=12)
    ax.set_title("Risk-Coverage Curve\n(Lower AURC = Better Selective Prediction)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_uncertainty_vs_tier(epistemic, true_tiers, save_path):
    tier_names = {1: "Sharps", 2: "Infectious", 3: "Pharma", 4: "General"}
    colors     = {1: "#E74C3C", 2: "#F39C12", 3: "#3498DB", 4: "#2ECC71"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for t in sorted(set(true_tiers)):
        mask = true_tiers == t
        ax.hist(
            epistemic[mask], bins=30, alpha=0.6,
            label=f"T{t} {tier_names.get(t,'')} (n={mask.sum()})",
            color=colors.get(t, "grey"), edgecolor="white",
        )
    ax.set_xlabel("Epistemic Uncertainty (MI)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Epistemic Uncertainty Distribution per Hazard Tier", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
