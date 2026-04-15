"""
========================================================================
DBHDSNet — Loss Functions
Novel composite loss:
  L_total = λ_box·L_ciou + λ_obj·L_obj + λ_cls·L_focal
           + λ_seg·L_mask + λ_hazard·L_hazard + λ_hier·L_hierarchy

L_hierarchy is the PhD-novel contribution:
  A cost-sensitive cross-entropy where the penalty matrix P[i,j]
  reflects the clinical risk of confusing hazard tier i with tier j.
  Misclassifying sharps (Tier 1) as general waste (Tier 4) incurs
  a 10× higher penalty than a same-tier confusion.
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# ════════════════════════════════════════════════════════════════════════
# 1 — UTILITY: IoU VARIANTS
# ════════════════════════════════════════════════════════════════════════

def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU loss for bounding boxes.
    pred / target : (N, 4) in [cx, cy, w, h] normalised format.
    """
    # Convert to x1y1x2y2
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    # Intersection
    ix1 = torch.max(p_x1, t_x1)
    iy1 = torch.max(p_y1, t_y1)
    ix2 = torch.min(p_x2, t_x2)
    iy2 = torch.min(p_y2, t_y2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    # Areas
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union  = p_area + t_area - inter + eps
    iou    = inter / union

    # Enclosing box
    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)
    c2     = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    # Centre distance
    rho2 = (pred[:, 0] - target[:, 0]).pow(2) + \
           (pred[:, 1] - target[:, 1]).pow(2)

    # Aspect ratio consistency
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(target[:, 2] / (target[:, 3] + eps)) -
        torch.atan(pred[:,   2] / (pred[:,   3] + eps))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()


# ════════════════════════════════════════════════════════════════════════
# 2 — FOCAL LOSS (classification)
# ════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for dense detection with optional label smoothing.
    pred  : (N, C) raw logits
    target: (N,) integer class indices OR (N, C) soft targets
    """
    def __init__(self, gamma: float = 1.5, alpha: float = 0.25,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ls    = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_cls = pred.shape[-1]

        # Build soft targets with label smoothing
        if target.dim() == 1:
            t_one_hot = torch.zeros_like(pred)
            t_one_hot.scatter_(-1, target.unsqueeze(-1).long(), 1.0)
        else:
            t_one_hot = target.float()

        if self.ls > 0:
            t_one_hot = t_one_hot * (1 - self.ls) + self.ls / n_cls

        p      = torch.sigmoid(pred)
        ce     = F.binary_cross_entropy_with_logits(pred, t_one_hot, reduction="none")
        p_t    = p * t_one_hot + (1 - p) * (1 - t_one_hot)
        weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * t_one_hot + (1 - self.alpha) * (1 - t_one_hot)
        loss    = alpha_t * weight * ce
        return loss.mean()


# ════════════════════════════════════════════════════════════════════════
# 3 — MASK LOSS (Dice + BCE)
# ════════════════════════════════════════════════════════════════════════

def dice_bce_mask_loss(
    pred_masks:   torch.Tensor,   # (N, Hm, Wm) float, sigmoid already applied
    target_masks: torch.Tensor,   # (N, Hm, Wm) binary float {0, 1}
    eps:          float = 1.0,
) -> torch.Tensor:
    """Combined Dice + BCE loss for instance segmentation masks."""
    if pred_masks.numel() == 0:
        return pred_masks.sum() * 0

    pred_flat = pred_masks.flatten(1)    # (N, H*W)
    tgt_flat  = target_masks.flatten(1)  # (N, H*W)

    # BCE (on logits — pass raw; sigmoid done inside)
    bce = F.binary_cross_entropy(pred_flat, tgt_flat, reduction="mean")

    # Dice
    inter = (pred_flat * tgt_flat).sum(1)
    union = pred_flat.sum(1) + tgt_flat.sum(1)
    dice  = 1.0 - (2 * inter + eps) / (union + eps)

    return (bce + dice.mean()) / 2.0


# ════════════════════════════════════════════════════════════════════════
# 4 — HAZARD HIERARCHY LOSS  (PhD novel contribution)
# ════════════════════════════════════════════════════════════════════════

class HazardHierarchyLoss(nn.Module):
    """
    Cost-sensitive cross-entropy for hazard tier classification.

    The penalty matrix P (num_tiers × num_tiers) encodes clinical risk:
        P[i, j] = cost of predicting tier j when the true tier is i.

    A standard cross-entropy treats all wrong predictions equally.
    This loss amplifies the gradient for high-risk confusions
    (e.g. sharps → general waste) and reduces it for low-risk ones.

    Mathematical form:
        L_hier = mean_i (sum_j P[true_i, j] * softmax(logits)[j])
                × cross_entropy(logits, true_tier)
    """

    def __init__(self, penalty_matrix: List[List[float]]):
        super().__init__()
        P = torch.tensor(penalty_matrix, dtype=torch.float32)
        self.register_buffer("P", P)           # (num_tiers, num_tiers)

    def forward(
        self,
        logits:     torch.Tensor,   # (B, num_tiers) or (N, num_tiers)
        true_tiers: torch.Tensor,   # (B,) or (N,)  integer tier indices (0-based)
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0

        probs = torch.softmax(logits, dim=-1)               # (N, T)

        # Expected penalty for each sample: sum_j P[true_i, j] * p_j
        true_idx  = true_tiers.long()                       # (N,)
        P_row     = self.P[true_idx]                        # (N, T)
        exp_cost  = (P_row * probs).sum(dim=-1)             # (N,)

        # Standard cross-entropy
        ce = F.cross_entropy(logits, true_idx, reduction="none")  # (N,)

        # Scale CE loss by expected penalty (higher penalty → stronger gradient)
        weighted  = exp_cost * ce
        return weighted.mean()


# ════════════════════════════════════════════════════════════════════════
# 5 — TARGET BUILDER (matches predictions to ground truth)
# ════════════════════════════════════════════════════════════════════════

class TargetBuilder:
    """
    Assigns ground-truth boxes to grid cells across FPN scales.
    Uses scale-based assignment: GT box area determines which scale
    is responsible, then the cell containing the GT centre is assigned.

    Assignment thresholds (normalised box max-side length):
        P3 (stride 8)  : max_side ∈ [0,    0.15)
        P4 (stride 16) : max_side ∈ [0.10, 0.35)
        P5 (stride 32) : max_side ∈ [0.25, 1.00)
    Overlapping ranges allow each GT to be assigned to ≤2 scales.
    """

    SCALE_RANGES = {
        "P3": (0.00, 0.15),
        "P4": (0.10, 0.35),
        "P5": (0.25, 1.00),
    }
    STRIDES = {"P3": 8, "P4": 16, "P5": 32}

    def __init__(self, img_size: int = 640, num_classes: int = 38, num_protos: int = 32):
        self.img_size    = img_size
        self.num_classes = num_classes
        self.num_protos  = num_protos

    def build(
        self,
        gt_boxes:  List[torch.Tensor],   # per-image list of (Ni, 5) [cls,cx,cy,w,h]
        pred_dict: Dict[str, dict],      # raw prediction dict (for shape info)
        device:    torch.device,
    ) -> Dict[str, dict]:
        """
        Returns per-scale target tensors for loss computation.
        """
        B = len(gt_boxes)
        targets = {}

        for scale, preds in pred_dict.items():
            H, W  = preds["reg"].shape[-2:]
            stride = self.STRIDES[scale]
            lo, hi = self.SCALE_RANGES[scale]

            # Initialise targets for this scale
            tgt_reg  = torch.zeros(B, H, W, 4,                device=device)
            tgt_obj  = torch.zeros(B, H, W, 1,                device=device)
            tgt_cls  = torch.zeros(B, H, W, self.num_classes, device=device)
            tgt_mask = torch.zeros(B, H, W, self.num_protos,  device=device)
            pos_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

            for b in range(B):
                boxes = gt_boxes[b]          # (Ni, 5) [cls_id, cx, cy, w, h]
                if boxes.numel() == 0:
                    continue

                cls_ids = boxes[:, 0].long()
                cx  = boxes[:, 1]
                cy  = boxes[:, 2]
                bw  = boxes[:, 3]
                bh  = boxes[:, 4]
                max_side = torch.max(bw, bh)

                for i in range(len(boxes)):
                    ms = max_side[i].item()
                    if not (lo <= ms < hi):
                        continue

                    # Grid cell responsible for this GT
                    gx = int(cx[i].item() * W)
                    gy = int(cy[i].item() * H)
                    gx = min(gx, W - 1)
                    gy = min(gy, H - 1)

                    # Fill targets at (b, gy, gx)
                    tgt_reg[b, gy, gx]  = boxes[i, 1:]     # [cx,cy,w,h]
                    tgt_obj[b, gy, gx]  = 1.0
                    tgt_cls[b, gy, gx, cls_ids[i]] = 1.0
                    pos_mask[b, gy, gx] = True

            targets[scale] = {
                "reg":     tgt_reg,    # (B,H,W,4)
                "obj":     tgt_obj,    # (B,H,W,1)
                "cls":     tgt_cls,    # (B,H,W,nc)
                "pos_mask": pos_mask,  # (B,H,W) bool
            }

        return targets


# ════════════════════════════════════════════════════════════════════════
# 6 — COMPOSITE LOSS (full DBHDSNet loss)
# ════════════════════════════════════════════════════════════════════════

class DBHDSNetLoss(nn.Module):
    """
    Full composite loss:
        L = λ_box·L_ciou  + λ_obj·L_bce_obj  + λ_cls·L_focal
          + λ_seg·L_mask   + λ_hazard·L_ce_hazard
          + λ_hier·L_hierarchy
    """

    def __init__(self, cfg):
        super().__init__()
        lc = cfg.LOSS
        mc = cfg.MODEL

        self.lambda_box   = lc.LAMBDA_BOX
        self.lambda_obj   = lc.LAMBDA_OBJ
        self.lambda_cls   = lc.LAMBDA_CLS
        self.lambda_seg   = lc.LAMBDA_SEG
        self.lambda_haz   = lc.LAMBDA_HAZARD
        self.lambda_hier  = lc.LAMBDA_HIERARCHY

        self.focal        = FocalLoss(
            gamma=lc.FOCAL_GAMMA, alpha=lc.FOCAL_ALPHA,
            label_smoothing=cfg.TRAIN.LABEL_SMOOTHING,
        )
        self.hier_loss    = HazardHierarchyLoss(lc.HAZARD_PENALTY_MATRIX)
        self.target_builder = TargetBuilder(
            img_size    = mc.IMG_SIZE,
            num_classes = mc.NUM_CLASSES,
            num_protos  = mc.NUM_PROTO_MASKS,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        model_out:     dict,
        gt_boxes:      List[torch.Tensor],   # per-image (Ni, 5)
        gt_masks:      List[torch.Tensor],   # per-image (Ni, mH, mW)
        gt_hazard:     List[torch.Tensor],   # per-image (Ni,) hazard tiers 1-4
        device:        torch.device,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor
        loss_dict  : dict with individual loss components for logging
        """
        preds        = model_out["predictions"]
        hazard_logits= model_out["hazard_logits"]      # (B, 4)
        proto_masks  = model_out["proto_masks"]        # (B, K, Hm, Wm)
        B            = hazard_logits.shape[0]

        # ── Build targets ─────────────────────────────────────────────
        targets = self.target_builder.build(gt_boxes, preds, device)

        # ── Per-scale detection losses ────────────────────────────────
        total_box = torch.tensor(0., device=device)
        total_obj = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)
        n_scales  = len(preds)

        for scale, pred in preds.items():
            tgt    = targets[scale]
            pm     = tgt["pos_mask"]                  # (B, H, W)
            H, W   = pm.shape[1], pm.shape[2]

            # Reshape preds to (B, H, W, C)
            reg_p  = pred["reg"].permute(0,2,3,1)     # (B,H,W,4)
            obj_p  = pred["obj"].permute(0,2,3,1)     # (B,H,W,1)
            cls_p  = pred["cls"].permute(0,2,3,1)     # (B,H,W,nc)

            # Objectness loss (all cells)
            total_obj = total_obj + F.binary_cross_entropy_with_logits(
                obj_p, tgt["obj"], reduction="mean"
            )

            if pm.any():
                # Box loss (positive cells only)
                reg_pos = reg_p[pm]                   # (N_pos, 4)
                tgt_reg = tgt["reg"][pm]              # (N_pos, 4)
                total_box = total_box + ciou_loss(
                    self._decode_reg(reg_pos, scale, H, W, device),
                    tgt_reg,
                )

                # Classification loss (positive cells only)
                cls_pos = cls_p[pm]                   # (N_pos, nc)
                tgt_cls = tgt["cls"][pm]              # (N_pos, nc)
                total_cls = total_cls + self.focal(cls_pos, tgt_cls)

        total_box = total_box / n_scales
        total_obj = total_obj / n_scales
        total_cls = total_cls / n_scales

        # ── Mask segmentation loss ────────────────────────────────────
        total_seg = self._seg_loss(proto_masks, gt_boxes, gt_masks, targets, preds, device)

        # ── Hazard tier loss ──────────────────────────────────────────
        total_haz, total_hier = self._hazard_loss(hazard_logits, gt_hazard, device)

        # ── Composite total ───────────────────────────────────────────
        total = (
            self.lambda_box  * total_box  +
            self.lambda_obj  * total_obj  +
            self.lambda_cls  * total_cls  +
            self.lambda_seg  * total_seg  +
            self.lambda_haz  * total_haz  +
            self.lambda_hier * total_hier
        )

        loss_dict = {
            "loss_box":      total_box.item(),
            "loss_obj":      total_obj.item(),
            "loss_cls":      total_cls.item(),
            "loss_seg":      total_seg.item(),
            "loss_hazard":   total_haz.item(),
            "loss_hierarchy": total_hier.item(),
            "loss_total":    total.item(),
        }

        return total, loss_dict

    # ------------------------------------------------------------------

    def _decode_reg(self, reg, scale, H, W, device):
        """Quick decode for loss computation (no grids needed)."""
        return torch.sigmoid(reg)   # simplified for loss; full decode in heads.py

    # ------------------------------------------------------------------

    def _seg_loss(self, proto, gt_boxes, gt_masks, targets, preds, device):
        """Compute mask loss for matched positive predictions."""
        B, K, Hm, Wm = proto.shape
        total = torch.tensor(0., device=device)
        count = 0

        for b in range(B):
            if gt_masks[b].numel() == 0:
                continue

            # Use P3 (finest scale) for mask coefficient
            pred_coeff = preds["P3"]["mask"]     # (B, K, H3, W3)
            tgt_pos = targets["P3"]["pos_mask"][b]  # (H3, W3)

            if not tgt_pos.any():
                continue

            # Mask coefficients at positive locations
            coeff_flat = pred_coeff[b].permute(1,2,0)[tgt_pos]  # (N_pos, K)
            n_pos = coeff_flat.shape[0]
            n_gt  = gt_masks[b].shape[0]

            if n_pos == 0 or n_gt == 0:
                continue

            # Assemble predicted masks
            proto_b   = proto[b]                      # (K, Hm, Wm)
            pred_masks = torch.sigmoid(
                torch.mm(coeff_flat, proto_b.view(K, -1))
            ).view(n_pos, Hm, Wm)                    # (N_pos, Hm, Wm)

            # Resize GT masks to proto resolution
            gt_m = gt_masks[b].float().unsqueeze(1)  # (N_gt, 1, mH, mW)
            gt_m = F.interpolate(gt_m, size=(Hm, Wm), mode="nearest").squeeze(1)

            # Match pred → GT by index (simplified: take min(n_pos, n_gt) pairs)
            n_match = min(n_pos, n_gt)
            total += dice_bce_mask_loss(
                pred_masks[:n_match], gt_m[:n_match]
            )
            count += 1

        return total / max(count, 1)

    # ------------------------------------------------------------------

    def _hazard_loss(self, hazard_logits, gt_hazard, device):
        """
        Compute hazard tier CE + hierarchy loss.
        gt_hazard: list of (Ni,) int tensors with tiers 1–4 (1-indexed).
        """
        all_logits, all_tiers = [], []

        for b, tiers in enumerate(gt_hazard):
            if tiers.numel() == 0:
                continue
            # Image-level hazard = most dangerous tier in this image
            worst_tier = tiers.min()                  # tier 1 = most dangerous
            all_logits.append(hazard_logits[b:b+1])   # (1, 4)
            all_tiers.append(worst_tier.unsqueeze(0))

        if not all_logits:
            return torch.tensor(0., device=device), torch.tensor(0., device=device)

        logits_cat = torch.cat(all_logits, dim=0)     # (N, 4)
        tiers_cat  = torch.cat(all_tiers,  dim=0) - 1 # 0-indexed

        ce_loss   = F.cross_entropy(logits_cat, tiers_cat)
        hier_loss = self.hier_loss(logits_cat, tiers_cat)

        return ce_loss, hier_loss
