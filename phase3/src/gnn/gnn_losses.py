"""
========================================================================
DBHDSNet Phase 3b — ContamRisk-GNN Loss Functions

Updated for actual dataset classes (master_data.yaml, 38 classes).

Composite loss:
  L = λ_item × L_item_risk   (focal-MSE per-node risk — up-weights T1/T2)
    + λ_bin  × L_bin_risk    (MSE bin-level risk)
    + λ_cls  × L_risk_cls    (CrossEntropy risk category: Low/Med/High/Crit)
    + λ_cont × L_contam      (auxiliary cross-contamination BCE)
    + λ_pair × L_pair        (novel: HIGH_RISK_CLASS_PAIRS contrastive term)

Novel addition — L_pair (Class-Pair Contrastive Loss):
  Uses HIGH_RISK_CLASS_PAIRS from config to push scene embeddings of
  known-dangerous co-occurrences (e.g. needle+paperbox, syringe+bloody_objects)
  toward high risk scores, and safe co-occurrences toward low scores.
  This directly encodes WHO/CPCB pair-level clinical knowledge into training.

Tier-weighted MSE (L_item_risk):
  Nodes from T1 (blade, scalpel, needle, syringe, harris_uni_core,
                  mercury_thermometer, radioactive_objects)     → weight 4.0
  T2 (bloody_objects, mask, n95, bandage, ...)                  → weight 2.5
  T3 (oxygen_cylinder, capsule, pill, ...)                      → weight 1.5
  T4 (paperbox, cap_plastic)                                    → weight 1.0
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


# Tier → training weight (T1 items get 4× loss weight)
TIER_LOSS_WEIGHTS = {1: 4.0, 2: 2.5, 3: 1.5, 4: 1.0}

# Class name → tier (mirrors config_phase3.py HAZARD_TIER_MAP exactly)
CLASS_TIER_MAP: Dict[str, int] = {
    # T1
    "radioactive_objects": 1, "blade": 1, "harris_uni_core": 1,
    "mercury_thermometer": 1, "scalpel": 1, "syringe": 1, "needle": 1,
    # T2
    "bloody_objects": 2, "mask": 2, "n95": 2, "bandage": 2,
    "cotton_swab": 2, "covid_buffer": 2, "covid_test_case": 2, "gauze": 2,
    "iodine_swab": 2, "plastic_medical_bag": 2, "medical_gloves": 2,
    "reagent_tube": 2, "single_channel_pipette": 2,
    "transferpettor_glass": 2, "transferpettor_plastic": 2,
    "tweezer_metal": 2, "tweezer_plastic": 2, "medical_infusion_bag": 2,
    # T3
    "oxygen_cylinder": 3, "capsule": 3, "covid_buffer_box": 3,
    "glass_bottle": 3, "harris_uni_core_cap": 3, "pill": 3,
    "plastic_medical_bottle": 3, "reagent_tube_cap": 3, "unguent": 3,
    "electronic_thermometer": 3, "drug_packaging": 3,
    # T4
    "paperbox": 4, "cap_plastic": 4,
}

# Class name → index (matches master_data.yaml order)
CLASS_NAME_TO_IDX: Dict[str, int] = {
    "bloody_objects": 0, "mask": 1, "n95": 2, "oxygen_cylinder": 3,
    "radioactive_objects": 4, "bandage": 5, "blade": 6, "capsule": 7,
    "cotton_swab": 8, "covid_buffer": 9, "covid_buffer_box": 10,
    "covid_test_case": 11, "gauze": 12, "glass_bottle": 13,
    "harris_uni_core": 14, "harris_uni_core_cap": 15, "iodine_swab": 16,
    "mercury_thermometer": 17, "paperbox": 18, "pill": 19,
    "plastic_medical_bag": 20, "plastic_medical_bottle": 21,
    "medical_gloves": 22, "reagent_tube": 23, "reagent_tube_cap": 24,
    "scalpel": 25, "single_channel_pipette": 26, "syringe": 27,
    "transferpettor_glass": 28, "transferpettor_plastic": 29,
    "tweezer_metal": 30, "tweezer_plastic": 31, "unguent": 32,
    "electronic_thermometer": 33, "cap_plastic": 34, "drug_packaging": 35,
    "medical_infusion_bag": 36, "needle": 37,
}


class ContamRiskLoss(nn.Module):
    """
    Composite ContamRisk-GNN loss incorporating dataset-specific
    class knowledge from master_data.yaml.

    Labels expected in each PyG Data batch:
      data.y_item : (total_N,)  float  — per-node risk score [0,1]
      data.y_bin  : (B,)        float  — bin risk score [0,1]
      data.y_cls  : (B,)        long   — risk class {0=Low,1=Med,2=High,3=Crit}
      data.has_cross_contamination : (B,) int {0,1} — auxiliary flag
      data.x      : (total_N, 48)  — node features (class one-hot in [0:38])
      data.batch  : (total_N,) — graph assignment
    """

    def __init__(self, cfg):
        super().__init__()
        gc = cfg.GNN
        self.lambda_item   = gc.LAMBDA_ITEM_RISK
        self.lambda_bin    = gc.LAMBDA_BIN_RISK
        self.lambda_cls    = gc.LAMBDA_RISK_CLS
        self.lambda_contam = gc.LAMBDA_CONTAM
        self.lambda_pair   = getattr(gc, "LAMBDA_PAIR", 0.5)

        # Pre-build pair index sets from HIGH_RISK_CLASS_PAIRS
        self._high_risk_pairs = self._build_pair_tensors(
            getattr(cfg.DATA, "HIGH_RISK_PAIRS", [])
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_pair_tensors(pairs: list) -> List[Tuple[int, int, float]]:
        """Convert (name_a, name_b, weight) → (idx_a, idx_b, weight)."""
        result = []
        for name_a, name_b, w in pairs:
            ia = CLASS_NAME_TO_IDX.get(name_a)
            ib = CLASS_NAME_TO_IDX.get(name_b)
            if ia is not None and ib is not None:
                result.append((ia, ib, float(w)))
        return result

    # ------------------------------------------------------------------

    def _tier_weights(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns per-node loss weights based on hazard tier.
        Class one-hot is in x[:, 0:38]; tier is inferred from argmax.
        """
        cls_ids = x[:, :38].argmax(dim=-1)          # (N,)
        # Build id → weight tensor
        w_map = torch.ones(38, device=x.device)
        for name, idx in CLASS_NAME_TO_IDX.items():
            tier = CLASS_TIER_MAP.get(name, 4)
            w_map[idx] = TIER_LOSS_WEIGHTS.get(tier, 1.0)
        return w_map[cls_ids]                        # (N,)

    # ------------------------------------------------------------------

    def _pair_contrastive_loss(
        self,
        bin_risk: torch.Tensor,    # (B,) predicted scene risk
        x:        torch.Tensor,    # (total_N, 48) node features
        batch:    torch.Tensor,    # (total_N,) graph assignment
    ) -> torch.Tensor:
        """
        For each scene containing a HIGH_RISK_CLASS_PAIR, penalise
        predicted bin_risk that is too LOW (below 0.65 for T1+T4 pairs,
        below 0.45 for others).

        This directly encodes clinical pair knowledge into training:
        a bin containing (needle + paperbox) must score ≥ 0.65 or be penalised.
        """
        if not self._high_risk_pairs:
            return torch.tensor(0., device=bin_risk.device)

        cls_ids = x[:, :38].argmax(dim=-1)          # (N,)
        B       = bin_risk.shape[0]
        loss    = torch.tensor(0., device=bin_risk.device)
        count   = 0

        for b in range(B):
            mask     = (batch == b)
            scene_cls = cls_ids[mask].tolist()
            scene_set = set(scene_cls)

            for ia, ib, w in self._high_risk_pairs:
                if ia in scene_set and ib in scene_set:
                    # Minimum expected risk for this pair
                    min_risk = 0.65 if w >= 3.0 else (0.50 if w >= 2.0 else 0.40)
                    # Hinge: penalise if predicted risk < min_risk
                    shortfall = F.relu(min_risk - bin_risk[b])
                    loss  = loss + w * shortfall
                    count += 1

        return loss / max(count, 1)

    # ------------------------------------------------------------------

    def forward(
        self,
        out:  Dict[str, torch.Tensor],
        data,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = out["item_risk"].device

        if not (hasattr(data, "y_item") and data.y_item is not None):
            dummy = torch.tensor(0.0, device=device)
            return dummy, {"loss_total": 0.0}

        y_item = data.y_item.to(device)
        y_bin  = data.y_bin.to(device).view(-1)
        y_cls  = data.y_cls.to(device).long().view(-1)
        y_cc   = data.has_cross_contamination.to(device).float()

        # ── 1. Tier-weighted per-item risk MSE ───────────────────────
        node_w    = self._tier_weights(data.x.to(device), data.batch.to(device))
        pred_item = out["item_risk"]
        l_item    = (node_w * (pred_item - y_item).pow(2)).mean()

        # ── 2. Bin-level risk MSE ────────────────────────────────────
        pred_bin  = out["bin_risk"]
        l_bin     = F.mse_loss(pred_bin, y_bin)

        # ── 3. Risk category CE ──────────────────────────────────────
        l_cls     = F.cross_entropy(out["risk_logits"], y_cls)

        # ── 4. Cross-contamination auxiliary BCE ─────────────────────
        l_contam  = F.binary_cross_entropy(
            pred_bin.clamp(1e-6, 1 - 1e-6), y_cc
        )

        # ── 5. Class-pair contrastive (novel — dataset-specific) ──────
        l_pair = self._pair_contrastive_loss(
            pred_bin, data.x.to(device), data.batch.to(device)
        )

        total = (
            self.lambda_item   * l_item   +
            self.lambda_bin    * l_bin    +
            self.lambda_cls    * l_cls    +
            self.lambda_contam * l_contam +
            self.lambda_pair   * l_pair
        )

        return total, {
            "loss_item":   l_item.item(),
            "loss_bin":    l_bin.item(),
            "loss_cls":    l_cls.item(),
            "loss_contam": l_contam.item(),
            "loss_pair":   l_pair.item(),
            "loss_total":  total.item(),
        }


# ════════════════════════════════════════════════════════════════════════
# GNN EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════

class GNNMetrics:
    """Tracks evaluation metrics across a validation / test epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._bin_preds    = []
        self._bin_targets  = []
        self._cls_preds    = []
        self._cls_targets  = []
        self._item_preds   = []
        self._item_targets = []

    def update(self, out: Dict[str, torch.Tensor], data):
        if not hasattr(data, "y_bin"):
            return
        self._bin_preds.append(out["bin_risk"].detach().cpu())
        self._bin_targets.append(data.y_bin.view(-1).cpu())
        self._cls_preds.append(out["risk_logits"].detach().cpu())
        self._cls_targets.append(data.y_cls.view(-1).cpu())
        self._item_preds.append(out["item_risk"].detach().cpu())
        self._item_targets.append(data.y_item.cpu())

    def compute(self) -> Dict[str, float]:
        import numpy as np

        bp = torch.cat(self._bin_preds).numpy()
        bt = torch.cat(self._bin_targets).numpy()
        cp = torch.cat(self._cls_preds)
        ct = torch.cat(self._cls_targets).numpy().astype(int)
        ip = torch.cat(self._item_preds).numpy()
        it = torch.cat(self._item_targets).numpy()

        bin_mae     = float(np.abs(bp - bt).mean())
        bin_rmse    = float(np.sqrt(((bp - bt) ** 2).mean()))
        bin_pearson = float(np.corrcoef(bp, bt)[0, 1]) \
                      if bp.std() > 0 and bt.std() > 0 else 0.0

        cls_pred = cp.argmax(dim=-1).numpy()
        cls_acc  = float((cls_pred == ct).mean())
        item_mae = float(np.abs(ip - it).mean())

        # High-risk sensitivity: fraction of Crit/High scenes (cls 2,3)
        # correctly identified — clinically the most important metric
        hi_mask  = (ct >= 2)
        hi_sens  = float((cls_pred[hi_mask] >= 2).mean()) if hi_mask.any() else 0.0

        from sklearn.metrics import f1_score
        try:
            macro_f1 = float(f1_score(ct, cls_pred, average="macro",
                                       zero_division=0))
        except Exception:
            macro_f1 = 0.0

        return {
            "bin_MAE":         bin_mae,
            "bin_RMSE":        bin_rmse,
            "bin_Pearson":     bin_pearson,
            "cls_accuracy":    cls_acc,
            "cls_macro_F1":    macro_f1,
            "cls_hi_sensitivity": hi_sens,   # clinical key metric
            "item_MAE":        item_mae,
        }
