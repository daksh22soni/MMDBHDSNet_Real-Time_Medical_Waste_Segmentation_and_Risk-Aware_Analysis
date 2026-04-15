"""
========================================================================
DBHDSNet Phase 3b — Waste Scene Graph Construction
Converts a set of per-image detections (from DBHDSNet Phase 2) into
a structured graph representation for ContamRisk-GNN.

Graph definition
────────────────
  Node  = one detected waste item
  Edge  = spatial relationship between two items
          (created when IoU > threshold OR centre distance < threshold)

Node feature vector (48-dim):
  [class_one_hot(38),           # visual identity
   hazard_tier_one_hot(4),      # clinical risk category
   cx, cy, w, h,                # spatial attributes (normalised)
   confidence,                  # detection confidence
   epistemic_uncertainty]       # UQ score from Phase 3a

Edge feature vector (7-dim):
  [Δcx, Δcy,                    # relative centre displacement
   log(w_i/w_j), log(h_i/h_j), # relative scale
   IoU(box_i, box_j),           # spatial overlap
   centre_distance,             # Euclidean distance (normalised)
   cross_contamination_risk]    # domain-knowledge risk scalar

Cross-contamination risk scalar is derived from WHO/CPCB rules:
  e.g., a Sharp item (T1) adjacent to a General item (T4) = 3.0
========================================================================
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    PyGData = object   # fallback stub


# ════════════════════════════════════════════════════════════════════════
# DETECTION ITEM (input to graph builder)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class WasteItem:
    """Represents one detected waste object."""
    class_id:        int           # 0–37
    class_name:      str
    hazard_tier:     int           # 1–4
    box_cx:          float         # normalised [0, 1]
    box_cy:          float
    box_w:           float
    box_h:           float
    confidence:      float         # detection confidence score
    epistemic_u:     float = 0.0  # uncertainty from Phase 3a
    mask_area_ratio: float = 0.0  # segmentation mask area / image area


# ════════════════════════════════════════════════════════════════════════
# GRAPH LABEL (ground truth for ContamRisk-GNN training)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SceneRiskLabel:
    """
    Ground truth contamination risk for one scene (waste bin image).
    Used only during training; inferred from WHO/CPCB rules.
    """
    item_risks:   List[float]   # per-item risk scores (0–1)
    bin_risk:     float         # overall bin-level risk (0–1)
    risk_class:   int           # 0=Low, 1=Medium, 2=High, 3=Critical
    has_cross_contamination: bool


# ════════════════════════════════════════════════════════════════════════
# SCENE GRAPH BUILDER
# ════════════════════════════════════════════════════════════════════════

class SceneGraphBuilder:
    """
    Converts a list of WasteItem detections into a PyTorch Geometric
    Data object representing the waste scene graph.

    Parameters
    ----------
    cfg                     : Config with GNN and contamination rules
    num_classes             : 38 (number of waste classes)
    num_tiers               : 4 (hazard tiers)
    """

    NUM_TIERS   = 4
    EPS         = 1e-6

    def __init__(self, cfg, num_classes: int = 38):
        self.gnn_cfg    = cfg.GNN
        self.contam_rules = cfg.DATA.CONTAMINATION_RULES
        self.num_classes  = num_classes

    # ------------------------------------------------------------------

    def build(
        self,
        items:  List[WasteItem],
        label:  Optional[SceneRiskLabel] = None,
    ) -> "PyGData":
        """
        Build a scene graph from detected items.

        Parameters
        ----------
        items : list of WasteItem (detections from Phase 2 model)
        label : optional ground-truth risk labels (training only)

        Returns
        -------
        torch_geometric.data.Data with fields:
          x         : (N, 48)  node feature matrix
          edge_index: (2, E)   COO edge indices
          edge_attr : (E, 7)   edge feature matrix
          y_item    : (N,)     per-node risk scores [training only]
          y_bin     : scalar   bin-level risk score [training only]
          y_cls     : scalar   risk class 0–3       [training only]
          n_nodes   : int      number of nodes
        """
        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric is required for Phase 3b.\n"
                "Install: pip install torch-geometric"
            )

        N = len(items)

        # ── Node features ─────────────────────────────────────────────
        node_feats = self._build_node_features(items)   # (N, 48)

        # ── Edges ─────────────────────────────────────────────────────
        edge_index, edge_attr = self._build_edges(items)

        # ── Assemble PyG Data object ──────────────────────────────────
        data = PyGData(
            x          = node_feats,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            n_nodes    = torch.tensor(N),
        )

        # ── Labels (training only) ────────────────────────────────────
        if label is not None:
            data.y_item = torch.tensor(label.item_risks,  dtype=torch.float32)
            data.y_bin  = torch.tensor([label.bin_risk],  dtype=torch.float32)
            data.y_cls  = torch.tensor([label.risk_class],dtype=torch.long)
            data.has_cross_contamination = torch.tensor(
                [int(label.has_cross_contamination)]
            )

        return data

    # ------------------------------------------------------------------

    def _build_node_features(self, items: List[WasteItem]) -> torch.Tensor:
        """
        Build (N, 48) node feature matrix.
        Feature layout:
          [0:38]  class one-hot
          [38:42] hazard tier one-hot
          [42]    box_cx
          [43]    box_cy
          [44]    box_w
          [45]    box_h
          [46]    confidence
          [47]    epistemic_uncertainty
        """
        N    = len(items)
        feat = torch.zeros(N, self.num_classes + self.NUM_TIERS + 6,
                           dtype=torch.float32)

        for i, item in enumerate(items):
            # Class one-hot
            if 0 <= item.class_id < self.num_classes:
                feat[i, item.class_id] = 1.0

            # Hazard tier one-hot (1-indexed → 0-indexed)
            t_idx = min(max(item.hazard_tier - 1, 0), self.NUM_TIERS - 1)
            feat[i, self.num_classes + t_idx] = 1.0

            # Spatial attributes
            base = self.num_classes + self.NUM_TIERS
            feat[i, base + 0] = item.box_cx
            feat[i, base + 1] = item.box_cy
            feat[i, base + 2] = item.box_w
            feat[i, base + 3] = item.box_h
            feat[i, base + 4] = item.confidence
            feat[i, base + 5] = item.epistemic_u

        return feat   # (N, 48)

    # ------------------------------------------------------------------

    def _build_edges(
        self, items: List[WasteItem]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build COO edge_index (2, E) and edge_attr (E, 7).

        Edges are created when:
          IoU(box_i, box_j) > PROXIMITY_IOU_THRESH
          OR centre_distance(i, j) < PROXIMITY_DIST_NORM
        Self-loops excluded.
        """
        N = len(items)
        iou_thresh   = self.gnn_cfg.PROXIMITY_IOU_THRESH
        dist_thresh  = self.gnn_cfg.PROXIMITY_DIST_NORM

        src_list, dst_list, feat_list = [], [], []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                iou  = self._box_iou(items[i], items[j])
                dist = self._centre_dist(items[i], items[j])

                if iou > iou_thresh or dist < dist_thresh:
                    edge_feat = self._edge_features(items[i], items[j], iou, dist)
                    src_list.append(i)
                    dst_list.append(j)
                    feat_list.append(edge_feat)

        if not src_list:
            # No edges: return empty tensors with correct shapes
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, self.gnn_cfg.EDGE_FEAT_DIM),
                                     dtype=torch.float32)
            return edge_index, edge_attr

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )   # (2, E)
        edge_attr  = torch.stack(feat_list, dim=0)   # (E, 7)
        return edge_index, edge_attr

    # ------------------------------------------------------------------

    def _edge_features(
        self,
        a: WasteItem, b: WasteItem,
        iou: float, dist: float,
    ) -> torch.Tensor:
        """
        7-dim edge feature:
          [Δcx, Δcy, log(wa/wb), log(ha/hb), IoU, dist, contam_risk]
        """
        dcx = a.box_cx - b.box_cx
        dcy = a.box_cy - b.box_cy
        log_w = np.log((a.box_w + self.EPS) / (b.box_w + self.EPS))
        log_h = np.log((a.box_h + self.EPS) / (b.box_h + self.EPS))

        contam_risk = self._cross_contamination_risk(a.hazard_tier, b.hazard_tier)

        return torch.tensor(
            [dcx, dcy, log_w, log_h, iou, dist, contam_risk],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------

    def _box_iou(self, a: WasteItem, b: WasteItem) -> float:
        """IoU between two [cx,cy,w,h] boxes."""
        ax1, ay1 = a.box_cx - a.box_w/2, a.box_cy - a.box_h/2
        ax2, ay2 = a.box_cx + a.box_w/2, a.box_cy + a.box_h/2
        bx1, by1 = b.box_cx - b.box_w/2, b.box_cy - b.box_h/2
        bx2, by2 = b.box_cx + b.box_w/2, b.box_cy + b.box_h/2

        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * \
                max(0, min(ay2, by2) - max(ay1, by1))
        union = (a.box_w * a.box_h) + (b.box_w * b.box_h) - inter
        return inter / (union + self.EPS)

    # ------------------------------------------------------------------

    def _centre_dist(self, a: WasteItem, b: WasteItem) -> float:
        """Euclidean distance between box centres (normalised coords)."""
        return float(((a.box_cx - b.box_cx)**2 + (a.box_cy - b.box_cy)**2)**0.5)

    # ------------------------------------------------------------------

    def _cross_contamination_risk(self, tier_a: int, tier_b: int) -> float:
        """
        Look up cross-contamination risk multiplier from WHO/CPCB rules.
        Uses symmetric lookup: (min_tier, max_tier).
        Returns 1.0 (baseline) if no specific rule exists.
        """
        key = (min(tier_a, tier_b), max(tier_a, tier_b))
        return float(self.contam_rules.get(key, 1.0))


# ════════════════════════════════════════════════════════════════════════
# RISK LABEL GENERATOR (for training data)
# ════════════════════════════════════════════════════════════════════════

class RiskLabelGenerator:
    """
    Generates pseudo ground-truth risk labels from WHO/CPCB rules.
    Used during ContamRisk-GNN training since real clinical risk
    annotations are unavailable.

    Risk scoring logic
    ──────────────────
    Per-item risk:
        base_tier_risk = {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.15}
        item_risk = base × (1 + max_cross_contam_multiplier_from_neighbours) / 2

    Bin-level risk (graph aggregate):
        bin_risk = 1 - Π(1 - item_risk_i)   [probability union rule]
        Clipped to [0, 1].

    Risk class:
        [0, 0.25) = Low (0)  [0.25, 0.50) = Medium (1)
        [0.50, 0.75) = High (2)  [0.75, 1.0] = Critical (3)
    """

    BASE_RISK = {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.15}
    THRESHOLDS = [0.25, 0.50, 0.75]

    def __init__(self, contam_rules: Dict[Tuple[int, int], float]):
        self.contam_rules = contam_rules

    def generate(self, items: List[WasteItem]) -> SceneRiskLabel:
        N = len(items)
        item_risks = []

        for i, item in enumerate(items):
            base = self.BASE_RISK.get(item.hazard_tier, 0.15)

            # Boost by cross-contamination risk from neighbours
            max_cross = 1.0
            for j, nbr in enumerate(items):
                if i == j:
                    continue
                key  = (min(item.hazard_tier, nbr.hazard_tier),
                        max(item.hazard_tier, nbr.hazard_tier))
                mult = self.contam_rules.get(key, 1.0)
                max_cross = max(max_cross, mult)

            # Scale by cross-contamination amplifier
            r = base * (1.0 + (max_cross - 1.0) * 0.5)
            item_risks.append(float(np.clip(r, 0, 1)))

        # Bin risk via probability union
        complements = [(1 - r) for r in item_risks]
        bin_r = 1.0 - float(np.prod(complements))
        bin_r = float(np.clip(bin_r, 0, 1))

        # Risk class
        r_cls = 0
        for thr in self.THRESHOLDS:
            if bin_r >= thr:
                r_cls += 1

        # Cross-contamination flag: any T1 adjacent to T4
        has_cross = any(
            item.hazard_tier == 1 for item in items
        ) and any(
            item.hazard_tier == 4 for item in items
        )

        return SceneRiskLabel(
            item_risks=item_risks,
            bin_risk=bin_r,
            risk_class=r_cls,
            has_cross_contamination=has_cross,
        )
