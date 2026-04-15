"""
========================================================================
DBHDSNet Phase 4 — Federated Visualisation Suite

Publication-quality figures:
  1. Federated convergence curve (global mAP vs rounds)
  2. Per-client mAP radar chart (fairness across hospitals)
  3. DP budget consumption curve (ε per round per client)
  4. Client drift score evolution
  5. Communication efficiency: mAP vs cumulative MB sent
  6. Privacy-utility trade-off curve (ε vs mAP for different σ runs)
========================================================================
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Client colour palette (one per hospital/lab)
CLIENT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
]
ROUND_COLOR   = "#2C3E50"
GLOBAL_COLOR  = "#E74C3C"


# ════════════════════════════════════════════════════════════════════════
# 1 — CONVERGENCE CURVE
# ════════════════════════════════════════════════════════════════════════

def plot_convergence(
    history_path: Path,
    save_path:    Path,
    metric_key:   str = "val_mAP_50",
):
    """
    Plots global mAP vs federated round from the training history JSON.
    Also overlays the best per-client mAP if available.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"  [VIS] History not found: {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    rounds  = [r["round"]        for r in history if metric_key in r]
    mAPs    = [r[metric_key]     for r in history if metric_key in r]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DBHDSNet Phase 4 — Federated Convergence",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(rounds, mAPs, color=GLOBAL_COLOR, lw=2.5, marker="o",
            markersize=4, label="Global model mAP@50")
    ax.fill_between(rounds, mAPs, alpha=0.10, color=GLOBAL_COLOR)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("mAP@50", fontsize=11)
    ax.set_title("Global Model Convergence", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if mAPs:
        ax.annotate(f"Best: {max(mAPs):.4f}",
                    xy=(rounds[int(np.argmax(mAPs))], max(mAPs)),
                    xytext=(rounds[int(np.argmax(mAPs))]+2, max(mAPs)-0.03),
                    fontsize=8, color=GLOBAL_COLOR,
                    arrowprops=dict(arrowstyle="->", color=GLOBAL_COLOR))

    # Per-client losses if logged
    ax2 = axes[1]
    client_ids = set()
    for r in history:
        for k in r:
            if k.startswith("loss_") and k != "loss_total":
                client_ids.add(k[5:])

    for i, cid in enumerate(sorted(client_ids)):
        rounds_c = [r["round"] for r in history if f"loss_{cid}" in r]
        losses_c = [r[f"loss_{cid}"] for r in history if f"loss_{cid}" in r]
        col = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        ax2.plot(rounds_c, losses_c, color=col, lw=1.5,
                 marker=".", markersize=3, label=cid, alpha=0.85)

    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Training Loss", fontsize=11)
    ax2.set_title("Per-Client Local Loss", fontsize=11)
    if client_ids:
        ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Convergence → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 2 — PER-CLIENT RADAR CHART (fairness)
# ════════════════════════════════════════════════════════════════════════

def plot_client_radar(
    per_client_metrics: Dict[str, Dict[str, float]],
    metric_keys:        List[str],
    save_path:          Path,
):
    """
    Radar chart showing each client's performance across multiple metrics.
    A compact model scores well across ALL clients (equitable model).
    """
    clients = list(per_client_metrics.keys())
    N       = len(metric_keys)
    if N < 3:
        return

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_title("Per-Client Performance Radar\n(Federated Model Fairness)",
                 fontsize=12, fontweight="bold", pad=20)

    for i, cid in enumerate(clients):
        vals = [per_client_metrics[cid].get(k, 0.0) for k in metric_keys]
        vals += vals[:1]
        col  = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        ax.plot(angles, vals, color=col, lw=2, label=cid)
        ax.fill(angles, vals, color=col, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Radar chart → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 3 — DP BUDGET CONSUMPTION CURVE
# ════════════════════════════════════════════════════════════════════════

def plot_dp_budget(
    history_path: Path,
    save_path:    Path,
    target_eps:   float = 8.0,
):
    """
    Plots cumulative ε per client vs federated round.
    Horizontal line at target_eps = DP budget limit.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        return

    with open(history_path) as f:
        history = json.load(f)

    # Collect per-client budget across rounds
    client_budgets: Dict[str, List] = {}
    rounds = []
    for r in history:
        if "dp_budgets" not in r:
            continue
        rounds.append(r["round"])
        for cid, eps in r["dp_budgets"].items():
            client_budgets.setdefault(cid, []).append(eps)

    if not rounds:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Cumulative DP Budget (ε) per Client vs Round",
                 fontsize=12, fontweight="bold")

    for i, (cid, budgets) in enumerate(client_budgets.items()):
        col = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        ax.plot(rounds[:len(budgets)], budgets, color=col, lw=2,
                marker=".", markersize=3, label=cid)

    ax.axhline(target_eps, color="red", lw=2, ls="--",
               label=f"Target ε = {target_eps}")
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Cumulative ε (privacy spend)", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] DP budget → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 4 — CLIENT DRIFT EVOLUTION
# ════════════════════════════════════════════════════════════════════════

def plot_drift_evolution(
    history_path: Path,
    save_path:    Path,
):
    """
    Plots per-client drift score (cosine distance to global) across rounds.
    Drift should decrease as training converges — clients align with global.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        return

    with open(history_path) as f:
        history = json.load(f)

    client_drifts: Dict[str, List] = {}
    rounds = []
    for r in history:
        if "drift_scores" not in r:
            continue
        rounds.append(r["round"])
        for cid, ds in r["drift_scores"].items():
            client_drifts.setdefault(cid, []).append(ds)

    if not rounds:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Client Drift Score vs Round\n"
                 "(Cosine distance to global model — should decrease)",
                 fontsize=11, fontweight="bold")

    for i, (cid, drifts) in enumerate(client_drifts.items()):
        col = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        ax.plot(rounds[:len(drifts)], drifts, color=col, lw=2,
                marker=".", markersize=3, label=cid)

    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Drift Score (0=identical, 1=orthogonal)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Drift evolution → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 5 — PRIVACY-UTILITY TRADE-OFF
# ════════════════════════════════════════════════════════════════════════

def plot_privacy_utility_tradeoff(
    tradeoff_data: List[Dict],   # [{"sigma": σ, "epsilon": ε, "mAP": m}, ...]
    save_path:     Path,
):
    """
    Plots mAP vs ε for different noise multiplier σ values.
    Shows the privacy-utility frontier for this dataset.
    Used in the PhD thesis to justify the chosen σ=1.1.
    """
    if not tradeoff_data:
        return

    sigmas  = [d["sigma"]   for d in tradeoff_data]
    epsilons = [d["epsilon"]  for d in tradeoff_data]
    mAPs    = [d["mAP"]     for d in tradeoff_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(epsilons, mAPs, c=sigmas, cmap="plasma_r",
                    s=80, zorder=5, edgecolors="white", lw=0.5)
    ax.plot(epsilons, mAPs, color="gray", lw=1, ls="--", alpha=0.5)

    for d in tradeoff_data:
        ax.annotate(f"σ={d['sigma']:.1f}",
                    (d["epsilon"], d["mAP"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.colorbar(sc, label="Noise multiplier σ")
    ax.set_xlabel("Privacy spend ε (lower = stronger privacy)", fontsize=11)
    ax.set_ylabel("Global mAP@50 (higher = better model)", fontsize=11)
    ax.set_title("Privacy-Utility Trade-off Curve\n"
                 "DBHDSNet Phase 4 — Federated with DP-SGD",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Privacy-utility → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 6 — BATCH GENERATOR
# ════════════════════════════════════════════════════════════════════════

def generate_all_fed_figures(
    history_path:       Path,
    per_client_metrics: Optional[Dict] = None,
    tradeoff_data:      Optional[List] = None,
    target_eps:         float = 8.0,
    output_dir:         Path  = Path("outputs/visualisations"),
):
    output_dir = Path(output_dir)
    print("\n" + "─"*60 + "\n  Generating Phase 4 Visualisations\n" + "─"*60)

    plot_convergence(history_path, output_dir / "convergence.png")
    plot_dp_budget(history_path, output_dir / "dp_budget.png", target_eps)
    plot_drift_evolution(history_path, output_dir / "drift_evolution.png")

    if per_client_metrics:
        metric_keys = ["mAP_50", "mAP_75", "hazard_acc"]
        plot_client_radar(per_client_metrics, metric_keys,
                          output_dir / "client_radar.png")

    if tradeoff_data:
        plot_privacy_utility_tradeoff(tradeoff_data,
                                      output_dir / "privacy_utility.png")

    print("  All Phase 4 figures saved.\n")
