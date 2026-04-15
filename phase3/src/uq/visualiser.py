"""
========================================================================
DBHDSNet Phase 3a — UQ Visualisation Suite

Generates publication-quality figures:
  1. Reliability (calibration) diagrams — before/after calibration
  2. Epistemic uncertainty histograms — by hazard tier
  3. BALD score distribution — flagged vs accepted
  4. Human-in-the-loop flagging clinical dashboard
========================================================================
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TIER_NAMES  = {1: "Sharps", 2: "Infectious", 3: "Pharmaceutical", 4: "General"}
TIER_COLORS = {1: "#E74C3C", 2: "#E67E22", 3: "#F1C40F", 4: "#2ECC71"}
METHOD_COLORS = {
    "uncalibrated":   "#E74C3C",
    "temp_scaling":   "#3498DB",
    "vector_scaling": "#2ECC71",
}


# ════════════════════════════════════════════════════════════════════════
# 1 — RELIABILITY DIAGRAM
# ════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(
    diag_data:   Dict[str, Dict],    # {method: {bin_centres, bin_accs, bin_confs, ...}}
    cal_metrics: Dict[str, Dict],    # {method: {ece, ace, brier, ...}}
    save_path:   Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Reliability Diagram — Hazard Tier Classification",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.set_title("Confidence vs Accuracy", fontsize=11)
    ax.plot([0,1],[0,1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")

    for method, diag in diag_data.items():
        c      = METHOD_COLORS.get(method, "gray")
        ece    = cal_metrics.get(method, {}).get("ece", 0.0)
        label  = f"{method.replace('_',' ').title()} (ECE={ece:.3f})"
        valid  = diag["bin_counts"] > 0
        ax.plot(diag["bin_centres"][valid], diag["bin_accs"][valid],
                "o-", color=c, lw=2, markersize=5, label=label)
        ax.fill_between(diag["bin_centres"][valid],
                        diag["bin_accs"][valid], diag["bin_centres"][valid],
                        alpha=0.07, color=c)

    ax.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax.set_ylabel("Fraction Correct", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

    # Metrics bar chart
    ax2    = axes[1]
    ax2.set_title("Calibration Metrics Comparison", fontsize=11)
    show   = ["ece", "ace", "brier", "nll"]
    x      = np.arange(len(show))
    methods = list(diag_data.keys())
    width  = 0.8 / len(methods)

    for i, m in enumerate(methods):
        vals   = [cal_metrics.get(m, {}).get(s, 0) for s in show]
        col    = METHOD_COLORS.get(m, "gray")
        offset = (i - len(methods)/2 + 0.5) * width
        bars   = ax2.bar(x + offset, vals, width, label=m.replace("_"," ").title(),
                         color=col, alpha=0.82, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in show], fontsize=9)
    ax2.set_ylabel("Score (lower = better)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Reliability diagram → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 2 — PER-TIER UNCERTAINTY HISTOGRAMS
# ════════════════════════════════════════════════════════════════════════

def plot_uncertainty_histograms(
    uq_data:     Dict[str, np.ndarray],
    save_path:   Path,
    thresholds:  Optional[Dict[int, float]] = None,
):
    ep    = uq_data["epistemic"]
    tiers = uq_data["tier_predictions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Epistemic Uncertainty Distribution by Hazard Tier",
                 fontsize=13, fontweight="bold")

    for idx, tier in enumerate([1, 2, 3, 4]):
        ax     = axes[idx // 2][idx % 2]
        mask   = (tiers == tier)
        t_ep   = ep[mask]
        thresh = (thresholds or {}).get(tier, 0.20)
        n_tot  = mask.sum(); n_flag = (t_ep > thresh).sum()

        ax.hist(t_ep[t_ep <= thresh], bins=25, color=TIER_COLORS[tier],
                alpha=0.72, label=f"Accepted ({n_tot-n_flag})", edgecolor="white")
        ax.hist(t_ep[t_ep >  thresh], bins=25, color="#C0392B",
                alpha=0.72, label=f"Flagged  ({n_flag})",       edgecolor="white")
        ax.axvline(thresh, color="red", lw=2, ls="--", label=f"τ={thresh:.2f}")
        ax.set_title(f"Tier {tier} — {TIER_NAMES[tier]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epistemic Uncertainty", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        pct = 100 * n_flag / max(n_tot, 1)
        ax.text(0.97, 0.95, f"Flag rate: {pct:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Uncertainty histograms → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 3 — BALD SCORE DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════

def plot_bald_distribution(uq_data: Dict, save_path: Path):
    mi    = uq_data["mutual_info"]
    flags = uq_data["flags"].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BALD Score (Epistemic Uncertainty via Mutual Information)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(mi[~flags], bins=40, color="#3498DB", alpha=0.72,
            label=f"Accepted (n={int((~flags).sum())})", edgecolor="white")
    ax.hist(mi[flags],  bins=40, color="#E74C3C", alpha=0.72,
            label=f"Flagged  (n={int(flags.sum())})",   edgecolor="white")
    ax.set_xlabel("BALD Score", fontsize=10); ax.set_ylabel("Count", fontsize=10)
    ax.set_title("BALD: Flagged vs Accepted", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    smi  = np.sort(mi)[::-1]
    ax2.plot(range(len(smi)), smi, color="#2C3E50", lw=1.5)
    ax2.fill_between(range(len(smi)), smi, alpha=0.15, color="#2C3E50")
    ax2.set_xlabel("Sample rank (highest BALD first)", fontsize=10)
    ax2.set_ylabel("BALD Score", fontsize=10)
    ax2.set_title("Ranked BALD (Active Learning Curve)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] BALD distribution → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 4 — CLINICAL FLAGGING DASHBOARD
# ════════════════════════════════════════════════════════════════════════

def plot_flagging_dashboard(uq_summary: Dict, save_path: Path):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("DBHDSNet — Human-in-the-Loop Flagging Dashboard",
                 fontsize=14, fontweight="bold")
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

    flag_rate = uq_summary.get("flag_rate", 0)

    # Pie — overall flag rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([flag_rate, 1 - flag_rate],
            labels=["Flagged", "Accepted"],
            colors=["#E74C3C", "#2ECC71"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title(f"Overall Flag Rate\n"
                  f"({uq_summary.get('n_flagged',0)} / {uq_summary.get('n_images',0)})",
                  fontsize=10)

    # Per-tier bar
    ax2     = fig.add_subplot(gs[0, 1:])
    trates  = uq_summary.get("tier_flag_rates", {})
    tiers   = [1, 2, 3, 4]
    rates   = [trates.get(t, 0) * 100 for t in tiers]
    names   = [f"Tier {t}\n{TIER_NAMES[t]}" for t in tiers]
    colors  = [TIER_COLORS[t] for t in tiers]
    bars    = ax2.bar(names, rates, color=colors, edgecolor="white", lw=0.5)
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Flag Rate (%)", fontsize=10)
    ax2.set_title("Per-Tier Flag Rate", fontsize=10)
    ax2.set_ylim(0, 105); ax2.grid(True, axis="y", alpha=0.3)

    # Summary table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    rows = [
        ["Metric", "Value", "Interpretation"],
        ["Mean Epistemic Uncertainty",
         f"{uq_summary.get('mean_epistemic',0):.4f}",
         "Lower = globally more confident"],
        ["P90 Epistemic Uncertainty",
         f"{uq_summary.get('p90_epistemic',0):.4f}",
         "Upper tail of uncertainty distribution"],
        ["Mean BALD Score",
         f"{uq_summary.get('mean_mutual_info',0):.4f}",
         "Model disagreement across MC passes"],
        ["Mean Predictive Entropy",
         f"{uq_summary.get('mean_pred_entropy',0):.4f}",
         "Total aleatoric + epistemic uncertainty"],
        ["Images Flagged for Review",
         f"{uq_summary.get('n_flagged',0)} / {uq_summary.get('n_images',0)}",
         f"({flag_rate*100:.1f}%) → forwarded to clinical staff"],
    ]
    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="left", loc="center",
                    bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        cell.set_edgecolor("white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Flagging dashboard → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 5 — BATCH GENERATOR
# ════════════════════════════════════════════════════════════════════════

def generate_all_uq_figures(
    uq_data:         Dict,
    uq_summary:      Dict,
    diag_data:       Dict,
    cal_metrics:     Dict,
    tier_thresholds: Dict,
    output_dir:      Path,
):
    output_dir = Path(output_dir)
    print("\n" + "─"*60 + "\n  Generating Phase 3a Visualisations\n" + "─"*60)
    plot_reliability_diagram(diag_data, cal_metrics,
                             output_dir / "reliability_diagram.png")
    plot_uncertainty_histograms(uq_data, output_dir / "uncertainty_histograms.png",
                                tier_thresholds)
    plot_bald_distribution(uq_data, output_dir / "bald_distribution.png")
    plot_flagging_dashboard(uq_summary, output_dir / "flagging_dashboard.png")
    print("  All Phase 3a figures saved.\n")
