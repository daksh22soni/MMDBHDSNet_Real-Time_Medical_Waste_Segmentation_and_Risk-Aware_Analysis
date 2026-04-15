"""
========================================================================
DBHDSNet Phase 3b — ContamRisk-GNN
Graph Attention Network that takes a waste scene graph and outputs:
  (a) Per-item risk score          — (N,) float in [0, 1]
  (b) Bin-level risk score         — scalar float in [0, 1]
  (c) Risk category classification — (4,) logits [Low/Med/High/Crit]

Architecture
────────────
  Node features (48-dim)
        │
  ┌─────▼──────┐
  │ Input Proj  │  Linear 48 → hidden_dim (with LayerNorm + SiLU)
  └─────┬──────┘
        │
  ┌─────▼──────┐  × L layers
  │ GAT Layer   │  Multi-head Graph Attention + edge features
  │  + Residual │  + Batch/Layer Norm + Dropout
  └─────┬──────┘
        │
  ┌─────┴─────────────────────────────────────┐
  │             Dual readout                   │
  │                                           │
  │  ┌──────────────┐    ┌──────────────────┐ │
  │  │ Item Risk    │    │ Global Readout   │ │
  │  │ Head (MLP)   │    │ (sum+mean+max)   │ │
  │  │ → (N,) score │    │ → graph-level    │ │
  │  └──────────────┘    │ features         │ │
  │                      └────────┬─────────┘ │
  │                               │            │
  │                    ┌──────────▼──────────┐ │
  │                    │  Bin Risk Head      │ │
  │                    │  MLP → scalar+cls  │ │
  │                    └─────────────────────┘ │
  └───────────────────────────────────────────┘

Edge features are incorporated via edge-conditioned message passing
(ECA — Edge-Conditioned Attention): attention coefficients are
modulated by the edge feature vector projected to a scalar gate.

Reference:
  Veličković et al. (2018). Graph Attention Networks. ICLR.
  Wang et al. (2019). Dynamic Graph CNN for Learning on Point Clouds.
========================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

try:
    from torch_geometric.nn import (
        GATv2Conv, GCNConv, SAGEConv,
        global_add_pool, global_mean_pool, global_max_pool,
    )
    from torch_geometric.data import Data as PyGData
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════
# NORMALIZATION FACTORY
# ════════════════════════════════════════════════════════════════════════

def get_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    elif norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    return nn.Identity()


# ════════════════════════════════════════════════════════════════════════
# EDGE-CONDITIONED GAT LAYER
# ════════════════════════════════════════════════════════════════════════

class EdgeConditionedGATLayer(nn.Module):
    """
    GATv2 layer with edge-feature conditioning.

    Edge features are projected to a scalar gate (sigmoid) that
    modulates each message before aggregation:
        m_ij = gate(e_ij) × GATv2_message(h_i, h_j)

    This lets the model learn which spatial relationships
    (e.g., proximity, cross-contamination risk) amplify messages.
    """

    def __init__(
        self,
        in_dim:      int,
        out_dim:     int,
        edge_dim:    int,
        heads:       int   = 4,
        dropout:     float = 0.2,
        residual:    bool  = True,
        norm:        str   = "batch",
    ):
        super().__init__()
        self.residual = residual
        self.heads    = heads
        self.out_dim  = out_dim

        # GATv2 conv (supports edge features natively via edge_dim)
        self.conv = GATv2Conv(
            in_channels  = in_dim,
            out_channels = out_dim // heads,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = edge_dim,
            concat       = True,   # output = heads × (out_dim // heads)
            share_weights= False,
        )

        # Edge gate: projects edge features to per-edge scalar
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_dim * 2, 1),
            nn.Sigmoid(),
        )

        # Post-conv normalisation + activation
        self.norm    = get_norm(norm, out_dim)
        self.act     = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x:          torch.Tensor,    # (N, in_dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_attr:  torch.Tensor,    # (E, edge_dim)
    ) -> torch.Tensor:
        # Compute edge gates from edge features
        gate = self.edge_gate(edge_attr)    # (E, 1)

        # GATv2 forward (edge_attr is used inside for attention)
        out  = self.conv(x, edge_index, edge_attr)   # (N, out_dim)

        # Apply gate as global edge-level scaling
        # (edge_index[1] maps edges to their target nodes)
        # We scale the aggregated output by mean gate per node
        if edge_attr.shape[0] > 0:
            target_nodes  = edge_index[1]             # (E,)
            gate_per_node = torch.zeros(
                x.shape[0], 1, device=x.device
            ).scatter_add_(
                0, target_nodes.unsqueeze(1),
                gate
            )
            degree = torch.zeros(
                x.shape[0], 1, device=x.device
            ).scatter_add_(
                0, target_nodes.unsqueeze(1),
                torch.ones_like(gate)
            ).clamp(min=1)
            mean_gate = gate_per_node / degree        # (N, 1)
            out = out * mean_gate

        out = self.dropout(self.act(self.norm(out)))

        if self.residual:
            out = out + self.res_proj(x)

        return out


# ════════════════════════════════════════════════════════════════════════
# ITEM RISK HEAD (per-node MLP)
# ════════════════════════════════════════════════════════════════════════

class ItemRiskHead(nn.Module):
    """
    Shallow MLP that maps per-node embeddings → item risk score ∈ [0, 1].
    Risk is calibrated by tier: Tier-1 items receive a risk floor.
    """

    TIER_FLOORS = {1: 0.60, 2: 0.40, 3: 0.25, 4: 0.05}   # min risk per tier

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Tier identity projected to a floor bias
        self.tier_embed = nn.Embedding(4, 1)   # 4 tiers → 1 scalar floor
        nn.init.constant_(self.tier_embed.weight, 0.0)

    def forward(
        self,
        node_emb:   torch.Tensor,   # (N, hidden_dim)
        tier_ids:   torch.Tensor,   # (N,) 0-indexed tier
    ) -> torch.Tensor:
        base_risk = self.mlp(node_emb).squeeze(-1)    # (N,)
        tier_bias = self.tier_embed(tier_ids).squeeze(-1)  # (N,)
        return (base_risk + torch.sigmoid(tier_bias) * 0.3).clamp(0, 1)


# ════════════════════════════════════════════════════════════════════════
# BIN RISK HEAD (graph-level MLP)
# ════════════════════════════════════════════════════════════════════════

class BinRiskHead(nn.Module):
    """
    Graph-level risk scoring from multi-scale readout.

    Combines sum, mean, and max pooled node embeddings (3 × hidden_dim)
    and passes through an MLP to produce:
      • Continuous bin risk ∈ [0, 1]
      • Risk category logits (4 classes)
    """

    def __init__(self, hidden_dim: int, risk_classes: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        graph_feat_dim = hidden_dim * 3   # sum + mean + max concatenated

        self.risk_mlp = nn.Sequential(
            nn.LayerNorm(graph_feat_dim),
            nn.Linear(graph_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.risk_score  = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
        self.risk_cls    = nn.Linear(hidden_dim // 2, risk_classes)

    def forward(
        self,
        node_emb: torch.Tensor,   # (total_N, hidden_dim)
        batch:    torch.Tensor,   # (total_N,) graph assignment
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_sum  = global_add_pool(node_emb,  batch)   # (B, H)
        g_mean = global_mean_pool(node_emb, batch)   # (B, H)
        g_max  = global_max_pool(node_emb,  batch)   # (B, H)
        g      = torch.cat([g_sum, g_mean, g_max], dim=-1)  # (B, 3H)

        h = self.risk_mlp(g)
        return self.risk_score(h).squeeze(-1), self.risk_cls(h)


# ════════════════════════════════════════════════════════════════════════
# ContamRisk-GNN (FULL MODEL)
# ════════════════════════════════════════════════════════════════════════

class ContamRiskGNN(nn.Module):
    """
    Contamination Risk Graph Neural Network.

    PhD novelty claim:
      "First GNN architecture for bin-level contamination risk
      prediction from waste scene graphs, incorporating WHO/CPCB
      cross-contamination rules as structured edge features."
    """

    def __init__(self, cfg):
        super().__init__()
        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric required.\n"
                "See requirements_phase3.txt for install instructions."
            )

        gc       = cfg.GNN
        node_dim = gc.NODE_FEAT_DIM    # 48
        edge_dim = gc.EDGE_FEAT_DIM    # 7
        H        = gc.GNN_HIDDEN_DIM   # 128
        L        = gc.GNN_N_LAYERS     # 4
        heads    = gc.GNN_HEADS        # 4
        drop     = gc.GNN_DROPOUT
        norm     = gc.GNN_NORM
        residual = gc.GNN_RESIDUAL

        # ── Input projection ─────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, H),
            nn.LayerNorm(H),
            nn.SiLU(),
        )

        # ── Message-passing layers ────────────────────────────────────
        self.layers = nn.ModuleList()
        for i in range(L):
            self.layers.append(
                EdgeConditionedGATLayer(
                    in_dim   = H,
                    out_dim  = H,
                    edge_dim = edge_dim,
                    heads    = heads,
                    dropout  = drop,
                    residual = residual and (i > 0),   # no residual on first
                    norm     = norm,
                )
            )

        # ── Task heads ────────────────────────────────────────────────
        self.item_risk_head = ItemRiskHead(H, dropout=drop)
        self.bin_risk_head  = BinRiskHead(H, risk_classes=gc.RISK_CLASSES,
                                          dropout=drop)

        # ── Tier extractor (reads tier from node feature slice) ───────
        # Tiers are in positions 38–41 of node features (one-hot)
        self._tier_start = 38   # hardcoded from feature layout
        self._tier_end   = 42

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------

    def _extract_tier_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Extract tier index (0-indexed) from one-hot node features."""
        tier_oh = x[:, self._tier_start:self._tier_end]  # (N, 4)
        return tier_oh.argmax(dim=-1)                     # (N,)

    # ------------------------------------------------------------------

    def forward(self, data: "PyGData") -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        data : PyG Data or Batch object with:
               x          (N, 48)
               edge_index (2, E)
               edge_attr  (E, 7)
               batch      (N,) — graph membership indices (auto by PyG Batch)

        Returns
        -------
        dict with:
          item_risk   : (N,) per-item risk ∈ [0, 1]
          bin_risk    : (B,) bin-level risk ∈ [0, 1]
          risk_logits : (B, 4) risk category logits
        """
        x          = data.x                          # (N, 48)
        edge_index = data.edge_index                  # (2, E)
        edge_attr  = data.edge_attr                   # (E, 7)
        batch      = data.batch if hasattr(data, "batch") \
                     else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        tier_ids = self._extract_tier_ids(x)          # (N,)

        # ── Input projection ─────────────────────────────────────────
        h = self.input_proj(x)                        # (N, H)

        # ── Message passing ───────────────────────────────────────────
        for layer in self.layers:
            if edge_attr.shape[0] > 0:
                h = layer(h, edge_index, edge_attr)
            else:
                # No edges → skip attention (isolated nodes)
                h = h

        # ── Item risk ─────────────────────────────────────────────────
        item_risk = self.item_risk_head(h, tier_ids)   # (N,)

        # ── Bin risk ──────────────────────────────────────────────────
        bin_risk, risk_logits = self.bin_risk_head(h, batch)   # (B,), (B,4)

        return {
            "item_risk":   item_risk,    # (N,)
            "bin_risk":    bin_risk,     # (B,)
            "risk_logits": risk_logits,  # (B, 4)
            "node_emb":    h,            # (N, H) — for visualisation
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_scene(
        self,
        data: "PyGData",
        risk_thresholds: list = None,
    ) -> Dict[str, object]:
        """
        Inference helper: returns human-readable risk assessment.
        """
        self.eval()
        out = self.forward(data)

        if risk_thresholds is None:
            risk_thresholds = [0.25, 0.50, 0.75]

        bin_score = float(out["bin_risk"][0].item())
        risk_class = int(out["risk_logits"][0].argmax().item())
        risk_labels = ["Low", "Medium", "High", "Critical"]

        # Item-level results
        items = []
        for i in range(out["item_risk"].shape[0]):
            items.append({
                "item_idx":  i,
                "item_risk": float(out["item_risk"][i].item()),
            })

        return {
            "bin_risk_score": bin_score,
            "bin_risk_class": risk_labels[risk_class],
            "bin_risk_class_id": risk_class,
            "item_risks":    items,
        }


# ════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════

def build_contamrisk_gnn(cfg, device: torch.device) -> ContamRiskGNN:
    model = ContamRiskGNN(cfg)
    return model.to(device)
