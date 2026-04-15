"""
========================================================================
DBHDSNet Phase 3b — Scene Graph Visualiser
Generates publication-quality figures of waste scene graphs:
  • Node-coloured by predicted risk score (green → red gradient)
  • Edge-coloured by cross-contamination risk multiplier
  • Bin-level risk score displayed as title
  • Uncertainty bars per node (optional)
Also generates a risk heatmap overlay on the original image.
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

from .scene_graph import WasteItem


# ════════════════════════════════════════════════════════════════════════
# COLOUR MAPS
# ════════════════════════════════════════════════════════════════════════

RISK_CMAP     = cm.get_cmap("RdYlGn_r")   # green=low, red=high
CONTAM_CMAP   = cm.get_cmap("Oranges")
TIER_COLOURS  = {
    1: "#E74C3C",   # Sharps    — deep red
    2: "#F39C12",   # Infectious — amber
    3: "#3498DB",   # Pharma    — blue
    4: "#2ECC71",   # General   — green
}
TIER_NAMES    = {1: "Sharps", 2: "Infectious", 3: "Pharma", 4: "General"}
RISK_LABELS   = ["Low", "Medium", "High", "Critical"]
RISK_COLOURS  = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]


# ════════════════════════════════════════════════════════════════════════
# SCENE GRAPH FIGURE
# ════════════════════════════════════════════════════════════════════════

def visualise_scene_graph(
    items:      List[WasteItem],
    item_risks: List[float],
    bin_risk:   float,
    risk_class: int,
    edge_contam_risks: Optional[List[float]] = None,
    save_path:  Optional[Path] = None,
    title:      str = "Waste Scene Graph — ContamRisk-GNN",
) -> plt.Figure:
    """
    Draws the waste scene graph with:
      • Nodes positioned at their detected (cx, cy) image coordinates
      • Node size ∝ box area (w × h)
      • Node colour = predicted item risk (green → red)
      • Node border colour = hazard tier colour
      • Edge colour = cross-contamination risk multiplier
      • Node label = class name + risk score

    Parameters
    ----------
    items           : detected waste items
    item_risks      : per-item predicted risk [0, 1]
    bin_risk        : overall bin risk score [0, 1]
    risk_class      : 0–3 (Low/Medium/High/Critical)
    edge_contam_risks: per-edge cross-contamination multipliers
    save_path       : if provided, save figure to disk
    """
    if not NX_AVAILABLE:
        print("[Visualiser] networkx not installed. pip install networkx")
        return None

    N = len(items)
    if N == 0:
        return None

    # ── Build NetworkX graph ───────────────────────────────────────────
    G = nx.DiGraph()
    for i, item in enumerate(items):
        G.add_node(i,
                   label     = f"{item.class_name}\n{item_risks[i]:.2f}",
                   risk      = item_risks[i],
                   tier      = item.hazard_tier,
                   cx        = item.box_cx,
                   cy        = 1 - item.box_cy,   # flip Y for display
                   area      = item.box_w * item.box_h,
                   )

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Add edge if items are spatially related
            dist = ((items[i].box_cx - items[j].box_cx)**2 +
                    (items[i].box_cy - items[j].box_cy)**2) ** 0.5
            if dist < 0.20:
                t_a, t_b = items[i].hazard_tier, items[j].hazard_tier
                key  = (min(t_a, t_b), max(t_a, t_b))
                contam = 1.0
                G.add_edge(i, j, contam_risk=contam)

    # ── Layout: use (cx, 1-cy) as fixed positions ─────────────────────
    pos = {i: (G.nodes[i]["cx"], G.nodes[i]["cy"]) for i in G.nodes}

    # ── Figure setup ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [2.5, 1]})

    ax = axes[0]

    # Node colours from risk scores
    node_risks  = [G.nodes[i]["risk"] for i in G.nodes]
    node_colours= [RISK_CMAP(r) for r in node_risks]
    node_sizes  = [max(500, G.nodes[i]["area"] * 8000) for i in G.nodes]
    node_borders= [TIER_COLOURS.get(G.nodes[i]["tier"], "#888") for i in G.nodes]

    # Edge colours from cross-contamination risk
    edge_contam  = [G.edges[e].get("contam_risk", 1.0) for e in G.edges]
    max_contam   = max(edge_contam) if edge_contam else 1.0
    edge_colours = [CONTAM_CMAP(c / max(max_contam, 1)) for c in edge_contam]
    edge_widths  = [0.5 + c * 2 for c in edge_contam]

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colours, node_size=node_sizes,
        edgecolors=node_borders, linewidths=2.5,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colours, width=edge_widths,
        arrows=True, arrowsize=12, alpha=0.7,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: G.nodes[i]["label"] for i in G.nodes},
        font_size=6.5, font_color="black",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(
        f"{title}\n"
        f"Bin Risk = {bin_risk:.3f}  |  "
        f"Risk Class: {RISK_LABELS[risk_class]}",
        fontsize=12, fontweight="bold",
        color=RISK_COLOURS[risk_class],
    )
    ax.axis("off")

    # ── Risk score colourbar ──────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=RISK_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Item Risk Score")

    # ── Panel 2: tier legend + bin risk gauge ─────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    # Tier legend
    tier_counts = {}
    for item in items:
        tier_counts[item.hazard_tier] = tier_counts.get(item.hazard_tier, 0) + 1

    legend_y = 0.90
    ax2.text(0.05, legend_y + 0.05, "Hazard Tiers Present:",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    for t in sorted(tier_counts.keys()):
        ax2.add_patch(plt.Rectangle(
            (0.05, legend_y - 0.04), 0.12, 0.04,
            color=TIER_COLOURS.get(t, "#888"),
            transform=ax2.transAxes,
        ))
        ax2.text(0.22, legend_y - 0.01,
                 f"Tier {t} — {TIER_NAMES.get(t, '?')} ({tier_counts[t]})",
                 fontsize=9, transform=ax2.transAxes, va="center")
        legend_y -= 0.09

    # Bin risk gauge (horizontal bar)
    gauge_y = 0.45
    ax2.text(0.05, gauge_y + 0.05, "Bin Risk Score:",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    ax2.axhline(gauge_y - 0.01, xmin=0.05, xmax=0.95,
                color="#ddd", linewidth=8, transform=ax2.transAxes)
    ax2.axhline(gauge_y - 0.01, xmin=0.05, xmax=0.05 + 0.90 * bin_risk,
                color=RISK_CMAP(bin_risk), linewidth=8,
                transform=ax2.transAxes)
    ax2.text(0.05, gauge_y - 0.08,
             f"{bin_risk:.3f}  [{RISK_LABELS[risk_class]}]",
             fontsize=12, fontweight="bold",
             color=RISK_COLOURS[risk_class],
             transform=ax2.transAxes)

    # Item risk bar chart
    bar_y_start = 0.28
    ax2.text(0.05, bar_y_start + 0.02, "Per-Item Risk:",
             fontsize=10, fontweight="bold", transform=ax2.transAxes)
    bar_height = min(0.02, 0.20 / max(N, 1))
    for i, r in enumerate(sorted(item_risks, reverse=True)[:8]):
        y = bar_y_start - 0.005 - i * (bar_height + 0.005)
        ax2.axhline(y, xmin=0.05, xmax=0.05 + 0.90 * r,
                    color=RISK_CMAP(r), linewidth=6,
                    transform=ax2.transAxes)
    ax2.text(0.05, 0.01,
             f"N items = {N}  |  Max risk = {max(item_risks, default=0):.3f}",
             fontsize=8, transform=ax2.transAxes, color="#555")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualiser] Scene graph saved → {save_path}")

    return fig


# ════════════════════════════════════════════════════════════════════════
# RISK HEATMAP OVERLAY ON ORIGINAL IMAGE
# ════════════════════════════════════════════════════════════════════════

def visualise_risk_heatmap(
    image_np:   np.ndarray,       # (H, W, 3) uint8
    items:      List[WasteItem],
    item_risks: List[float],
    bin_risk:   float,
    risk_class: int,
    save_path:  Optional[Path] = None,
) -> np.ndarray:
    """
    Overlays coloured bounding boxes (risk-coloured) on the original image.
    Returns the annotated image as numpy array.
    """
    import cv2
    img = image_np.copy()
    H, W = img.shape[:2]

    for item, risk in zip(items, item_risks):
        # Risk → colour (BGR for OpenCV)
        rgba  = RISK_CMAP(risk)
        b, g, r = int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)

        x1 = int((item.box_cx - item.box_w/2) * W)
        y1 = int((item.box_cy - item.box_h/2) * H)
        x2 = int((item.box_cx + item.box_w/2) * W)
        y2 = int((item.box_cy + item.box_h/2) * H)

        # Draw filled rectangle with alpha blending
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)
        img = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

        # Draw border
        cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 2)

        # Label
        label = f"T{item.hazard_tier} {risk:.2f}"
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Bin risk annotation
    label_str = f"Bin Risk: {bin_risk:.3f} [{RISK_LABELS[risk_class]}]"
    cv2.putText(img, label_str, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[Visualiser] Heatmap overlay saved → {save_path}")

    return img
