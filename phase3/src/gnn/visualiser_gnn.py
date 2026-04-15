"""
========================================================================
DBHDSNet Phase 3b — GNN Visualisation Suite
Generates:
  1. Scene graph visualisation (NetworkX layout)
  2. Risk score distribution plots (train/val/test)
  3. Predicted vs True risk scatter plot
  4. Risk-level confusion matrix heatmap
  5. Per-tier node risk violin plot
  6. High-risk scene attention weight visualisation
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from typing import Dict, List, Optional

TIER_NAMES  = {1: "Sharps", 2: "Infectious", 3: "Pharmaceutical", 4: "General"}
TIER_COLORS = {1: "#E74C3C", 2: "#E67E22", 3: "#F1C40F", 4: "#2ECC71"}
RISK_LEVELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
RISK_COLORS = {0: "#2ECC71", 1: "#F39C12", 2: "#E74C3C"}


# ════════════════════════════════════════════════════════════════════════
# 1 — SCENE GRAPH VISUALISATION
# ════════════════════════════════════════════════════════════════════════

def plot_scene_graph(
    graph,                       # torch_geometric Data object
    pred_scene_risk: float,
    true_scene_risk: float,
    node_pred_risks: List[float],
    class_names:     List[str],
    save_path:       Path,
    title:           Optional[str] = None,
):
    """
    Draws the scene graph with:
      - Node position = (cx, cy) in image space
      - Node colour   = hazard tier colour
      - Node size     = proportional to detection confidence
      - Edge colour   = contamination risk (WHO matrix)
      - Edge width    = risk magnitude
      - Border colour = predicted per-node risk (red-green)
    """
    try:
        import networkx as nx
    except ImportError:
        print("[VIS] networkx not installed — skipping scene graph plot.")
        return

    N   = graph.num_nodes
    boxes = graph.x[:, 43:47].cpu().numpy()   # cx, cy, w, h
    tiers = graph.hazard_tiers.cpu().numpy()
    conf  = graph.x[:, 42].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_str = title or f"Scene Graph — Risk: {pred_scene_risk:.3f} (true: {true_scene_risk:.3f})"
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # ── Left: Spatial layout (positions = bbox centres) ──────────────
    ax = axes[0]
    ax.set_title("Spatial Layout (node = detected item)", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()   # image coords: y increases downward

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, tier=int(tiers[i]), conf=float(conf[i]))

    edge_idx = graph.edge_index.cpu().numpy()
    edge_risk = graph.edge_attr[:, 3].cpu().numpy()   # WHO risk
    for k in range(edge_idx.shape[1]):
        i, j = edge_idx[0, k], edge_idx[1, k]
        G.add_edge(i, j, risk=float(edge_risk[k]))

    pos = {i: (float(boxes[i, 0]), float(boxes[i, 1])) for i in range(N)}

    # Nodes
    node_colors = [TIER_COLORS.get(int(tiers[i]), "gray") for i in range(N)]
    node_sizes  = [300 + 500 * float(conf[i]) for i in range(N)]

    # Node border colour = predicted risk (0=green, 1=red)
    from matplotlib.colors import to_rgba
    node_edge_c = [plt.cm.RdYlGn_r(node_pred_risks[i]) for i in range(N)] \
                  if node_pred_risks else ["black"] * N

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, linewidths=2,
                           edgecolors=node_edge_c)

    # Edge colours from WHO risk
    edge_list  = list(G.edges())
    if edge_list:
        e_risks = [G.edges[e]["risk"] for e in edge_list]
        e_colors = [plt.cm.hot(r) for r in e_risks]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list,
                               edge_color=e_colors, width=2, alpha=0.6,
                               arrows=False, connectionstyle="arc3,rad=0.1")

    # Labels: class abbreviation
    labels = {}
    for i in range(N):
        cid = int(graph.x[i, :38].argmax().item())
        name = class_names[cid] if cid < len(class_names) else f"C{cid}"
        labels[i] = name[:6]
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)

    # Legend
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=f"T{t} {TIER_NAMES[t]}")
               for t in [1,2,3,4]]
    ax.legend(handles=patches, fontsize=7, loc="upper right")
    ax.axis("off")

    # ── Right: Per-node risk bar chart ───────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Per-Node Predicted Risk", fontsize=10)
    y_pos  = range(N)
    colors = [plt.cm.RdYlGn_r(r) for r in node_pred_risks] if node_pred_risks \
             else ["gray"] * N

    ax2.barh(y_pos, node_pred_risks or [0]*N, color=colors, edgecolor="white")
    ax2.axvline(0.7, color="red",    lw=2, ls="--", label="HIGH threshold")
    ax2.axvline(0.4, color="orange", lw=1, ls="--", label="MED threshold")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted Contamination Risk", fontsize=9)
    ax2.set_yticks(list(y_pos))
    labels_y = [f"N{i} T{int(tiers[i])}" for i in range(N)]
    ax2.set_yticklabels(labels_y, fontsize=7)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    # Risk-level annotation
    col  = "red" if pred_scene_risk >= 0.7 else "orange" if pred_scene_risk >= 0.4 else "green"
    lvl  = "HIGH" if pred_scene_risk >= 0.7 else "MEDIUM" if pred_scene_risk >= 0.4 else "LOW"
    ax2.set_title(f"Per-Node Risk  |  Scene: {pred_scene_risk:.3f} [{lvl}]",
                  fontsize=10, color=col, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# 2 — PREDICTED vs TRUE RISK SCATTER
# ════════════════════════════════════════════════════════════════════════

def plot_risk_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    metrics: Dict,
    save_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ContamRisk-GNN: Predicted vs True Scene Risk",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    hi = 0.70; md = 0.40

    colors = []
    for t in true:
        if t >= hi:   colors.append(RISK_COLORS[2])
        elif t >= md: colors.append(RISK_COLORS[1])
        else:         colors.append(RISK_COLORS[0])

    ax.scatter(true, pred, c=colors, alpha=0.6, s=25, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect prediction")
    ax.axhline(hi, color="red",    lw=1, ls=":", alpha=0.5)
    ax.axvline(hi, color="red",    lw=1, ls=":", alpha=0.5)
    ax.axhline(md, color="orange", lw=1, ls=":", alpha=0.5)
    ax.axvline(md, color="orange", lw=1, ls=":", alpha=0.5)

    ax.set_xlabel("True Contamination Risk", fontsize=10)
    ax.set_ylabel("Predicted Risk", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.text(0.05, 0.92, f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}\n"
            f"Spearman r={metrics['spearman']:.4f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    patches = [mpatches.Patch(color=RISK_COLORS[l], label=lv)
               for l, lv in RISK_LEVELS.items()]
    ax.legend(handles=patches, fontsize=8, title="True risk level")

    # ── Risk distribution ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(true, bins=30, alpha=0.6, color="#3498DB", label="True risk",  edgecolor="white")
    ax2.hist(pred, bins=30, alpha=0.6, color="#E74C3C", label="Pred risk",  edgecolor="white")
    ax2.axvline(hi, color="red",    lw=2, ls="--")
    ax2.axvline(md, color="orange", lw=2, ls="--")
    ax2.set_xlabel("Risk Score", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Risk Score Distribution", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Risk scatter → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 3 — CONFUSION MATRIX (risk levels)
# ════════════════════════════════════════════════════════════════════════

def plot_risk_confusion_matrix(
    cm:        np.ndarray,    # (3, 3)
    save_path: Path,
):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    labels  = ["LOW", "MEDIUM", "HIGH"]

    # Normalise per true class (row)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True).clip(1)
    cm_norm  = cm_norm / row_sums

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1, ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Risk Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Risk Level",      fontsize=11, fontweight="bold")
    ax.set_title("Risk-Level Confusion Matrix\n(row-normalised fractions)",
                 fontsize=11, fontweight="bold")

    # Add raw counts as secondary annotation
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.75, f"n={cm[i,j]}",
                    ha="center", va="center", fontsize=8, color="dimgray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Confusion matrix → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 4 — PER-TIER NODE RISK VIOLIN PLOT
# ════════════════════════════════════════════════════════════════════════

def plot_node_risk_by_tier(
    node_preds: np.ndarray,   # (N_total,)
    node_tiers: np.ndarray,   # (N_total,) 1-4
    save_path:  Path,
):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 5))

    data = {"Predicted Risk": node_preds, "Hazard Tier": node_tiers}
    tier_label = {1:"T1 Sharps", 2:"T2 Infect.", 3:"T3 Pharma", 4:"T4 General"}
    tl = np.array([tier_label.get(int(t), f"T{t}") for t in node_tiers])

    import pandas as pd
    df = pd.DataFrame({"Risk": node_preds, "Tier": tl})
    order = [tier_label[t] for t in [1,2,3,4] if tier_label[t] in df["Tier"].unique()]

    palette = {tier_label[t]: TIER_COLORS[t] for t in [1,2,3,4]}
    sns.violinplot(data=df, x="Tier", y="Risk", order=order,
                   palette=palette, ax=ax, inner="box", cut=0)

    ax.axhline(0.7, color="red",    lw=2, ls="--", label="HIGH threshold")
    ax.axhline(0.4, color="orange", lw=1, ls="--", label="MED threshold")
    ax.set_title("Per-Node Predicted Risk by Hazard Tier\n"
                 "(expected: Sharps > Infectious > Pharma > General)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Hazard Tier", fontsize=10)
    ax.set_ylabel("Predicted Node Risk", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Per-tier violin → {save_path}")


# ════════════════════════════════════════════════════════════════════════
# 5 — BATCH GENERATE ALL PHASE 3b FIGURES
# ════════════════════════════════════════════════════════════════════════

def generate_all_gnn_figures(
    pred_risks:   np.ndarray,
    true_risks:   np.ndarray,
    node_preds:   np.ndarray,
    node_tiers:   np.ndarray,
    metrics:      Dict,
    output_dir:   Path,
    sample_graphs: List = None,
    class_names:   List = None,
):
    output_dir = Path(output_dir)

    plot_risk_scatter(
        pred_risks, true_risks, metrics,
        output_dir / "risk_scatter.png"
    )
    plot_risk_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        output_dir / "risk_confusion_matrix.png"
    )
    plot_node_risk_by_tier(
        node_preds, node_tiers,
        output_dir / "node_risk_by_tier.png"
    )

    # Scene graph samples (first 4 high-risk scenes)
    if sample_graphs and class_names:
        for idx, (graph, pred, true, node_r) in enumerate(sample_graphs[:4]):
            plot_scene_graph(
                graph, pred, true, node_r, class_names,
                output_dir / f"scene_graph_sample_{idx+1}.png",
            )

    print("  All Phase 3b figures saved.")

from typing import List, Dict
