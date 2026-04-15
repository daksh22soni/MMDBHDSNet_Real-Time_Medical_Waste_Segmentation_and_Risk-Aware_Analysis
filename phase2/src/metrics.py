"""
========================================================================
DBHDSNet — Evaluation Metrics
Computes mAP@0.5, mAP@0.75, per-class AP, hazard tier accuracy,
confusion matrix, and calibration error (ECE) for the UQ head.
========================================================================
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ════════════════════════════════════════════════════════════════════════
# 1 — IoU COMPUTATION
# ════════════════════════════════════════════════════════════════════════

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    Both in [cx, cy, w, h] normalised format.
    Returns (N, M) IoU matrix.
    """
    def to_xyxy(b):
        return torch.stack([
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ], dim=1)

    b1 = to_xyxy(boxes1)   # (N, 4)
    b2 = to_xyxy(boxes2)   # (M, 4)

    N, M = b1.shape[0], b2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros(N, M, device=boxes1.device)

    x1 = torch.max(b1[:, 0:1], b2[:, 0].unsqueeze(0))   # (N, M)
    y1 = torch.max(b1[:, 1:2], b2[:, 1].unsqueeze(0))
    x2 = torch.min(b1[:, 2:3], b2[:, 2].unsqueeze(0))
    y2 = torch.min(b1[:, 3:4], b2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)     # (N, M)
    a1    = ((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])).unsqueeze(1)
    a2    = ((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])).unsqueeze(0)
    union = a1 + a2 - inter + 1e-7

    return inter / union


# ════════════════════════════════════════════════════════════════════════
# 2 — PER-CLASS AVERAGE PRECISION
# ════════════════════════════════════════════════════════════════════════

def compute_ap(
    recall:    np.ndarray,
    precision: np.ndarray,
) -> float:
    """
    Compute AP using the 101-point interpolation (COCO-style).
    """
    recall    = np.concatenate([[0.0], recall,    [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Integrate
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in thresholds:
        p = precision[recall >= t]
        ap += (p.max() if p.size > 0 else 0.0)
    return ap / 101.0


def compute_class_ap(
    preds_by_class:   Dict[int, List[Tuple]],   # cls → [(score, tp, n_gt)]
    n_gt_by_class:    Dict[int, int],
    iou_thresh:       float = 0.5,
) -> Dict[int, float]:
    """Compute AP for each class."""
    class_aps = {}

    for cls_id, entries in preds_by_class.items():
        n_gt = n_gt_by_class.get(cls_id, 0)
        if n_gt == 0:
            class_aps[cls_id] = 0.0
            continue

        # Sort by score descending
        entries.sort(key=lambda x: -x[0])
        scores = np.array([e[0] for e in entries])
        tps    = np.array([e[1] for e in entries], dtype=float)
        fps    = 1.0 - tps

        tp_cum  = np.cumsum(tps)
        fp_cum  = np.cumsum(fps)
        recall    = tp_cum / (n_gt + 1e-7)
        precision = tp_cum / (tp_cum + fp_cum + 1e-7)

        class_aps[cls_id] = compute_ap(recall, precision)

    return class_aps


# ════════════════════════════════════════════════════════════════════════
# 3 — mAP COMPUTATION
# ════════════════════════════════════════════════════════════════════════

def compute_map(
    preds:      List[dict],   # list of {"boxes":(N,4), "scores":(N,), "labels":(N,)}
    targets:    List[dict],   # list of {"boxes":(M,4), "labels":(M,)}
    iou_thresh: float = 0.5,
    num_classes: Optional[int] = None,
) -> float:
    """
    Compute mean Average Precision at a given IoU threshold.

    Parameters
    ----------
    preds   : per-image prediction dicts
    targets : per-image ground-truth dicts
    iou_thresh : IoU threshold for TP/FP assignment

    Returns
    -------
    mAP : float
    """
    preds_by_class  = defaultdict(list)   # cls → [(score, is_tp)]
    n_gt_by_class   = defaultdict(int)

    for pred, tgt in zip(preds, targets):
        p_boxes  = pred["boxes"]       # (N, 4)
        p_scores = pred["scores"]      # (N,)
        p_labels = pred["labels"]      # (N,)
        t_boxes  = tgt["boxes"]        # (M, 4)
        t_labels = tgt["labels"]       # (M,)

        if t_boxes.numel() == 0:
            # All predictions are FP
            for i in range(len(p_boxes)):
                cls = int(p_labels[i].item()) if p_labels.numel() else 0
                preds_by_class[cls].append((p_scores[i].item(), 0))
            continue

        # Count GTs per class
        for lbl in t_labels:
            n_gt_by_class[int(lbl.item())] += 1

        if p_boxes.numel() == 0:
            continue

        # Move to CPU numpy for matching
        pb = p_boxes.cpu()
        tb = t_boxes.cpu()
        pl = p_labels.cpu().long()
        tl = t_labels.cpu().long()
        ps = p_scores.cpu()

        iou_mat = box_iou(pb, tb)   # (N, M)
        matched_gt = torch.full((len(tb),), False)

        # Sort predictions by score
        order = ps.argsort(descending=True)

        for idx in order:
            cls      = int(pl[idx].item())
            score    = float(ps[idx].item())

            if iou_mat.shape[1] == 0:
                preds_by_class[cls].append((score, 0))
                continue

            iou_row  = iou_mat[idx]
            # Only consider GTs of the same class
            same_cls = (tl == cls)
            iou_row  = iou_row * same_cls.float()

            best_iou, best_gt = iou_row.max(0)
            is_tp = (best_iou >= iou_thresh) and not matched_gt[best_gt]

            if is_tp:
                matched_gt[best_gt] = True

            preds_by_class[cls].append((score, int(is_tp)))

    class_aps = compute_class_ap(preds_by_class, n_gt_by_class, iou_thresh)
    if not class_aps:
        return 0.0
    return float(np.mean(list(class_aps.values())))


# ════════════════════════════════════════════════════════════════════════
# 4 — HAZARD TIER ACCURACY
# ════════════════════════════════════════════════════════════════════════

def hazard_accuracy(
    pred_logits: torch.Tensor,   # (N, 4)
    true_tiers:  torch.Tensor,   # (N,) 0-indexed
) -> dict:
    """
    Returns accuracy, macro-F1, and per-tier precision/recall
    for the hazard classification head.
    """
    pred_cls = pred_logits.argmax(dim=-1).cpu()
    true_cls = true_tiers.cpu()

    correct = (pred_cls == true_cls).float()
    acc     = correct.mean().item()

    n_tiers = pred_logits.shape[1]
    tier_names = ["Sharps", "Infectious", "Pharmaceutical", "General"]

    per_tier = {}
    f1s      = []

    for t in range(n_tiers):
        tp = ((pred_cls == t) & (true_cls == t)).sum().float()
        fp = ((pred_cls == t) & (true_cls != t)).sum().float()
        fn = ((pred_cls != t) & (true_cls == t)).sum().float()

        prec = (tp / (tp + fp + 1e-7)).item()
        rec  = (tp / (tp + fn + 1e-7)).item()
        f1   = 2 * prec * rec / (prec + rec + 1e-7)

        name = tier_names[t] if t < len(tier_names) else f"Tier{t+1}"
        per_tier[name] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)

    return {
        "accuracy":    acc,
        "macro_f1":    float(np.mean(f1s)),
        "per_tier":    per_tier,
    }


# ════════════════════════════════════════════════════════════════════════
# 5 — EXPECTED CALIBRATION ERROR (for UQ evaluation)
# ════════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    confidences: np.ndarray,   # (N,) max predicted probability
    accuracies:  np.ndarray,   # (N,) 1 if correct, 0 if wrong
    n_bins:      int = 15,
) -> float:
    """
    ECE measures the alignment between predicted confidence and
    actual accuracy across confidence bins.
    Lower ECE = better calibrated model.
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    N       = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / N) * abs(bin_acc - bin_conf)

    return float(ece)


# ════════════════════════════════════════════════════════════════════════
# 6 — CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════

def build_confusion_matrix(
    pred_labels: torch.Tensor,   # (N,) predicted class IDs
    true_labels: torch.Tensor,   # (N,) true class IDs
    num_classes: int,
) -> np.ndarray:
    """
    Returns (num_classes, num_classes) confusion matrix.
    Rows = true, Cols = predicted.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred = pred_labels.cpu().numpy()
    true = true_labels.cpu().numpy()
    for t, p in zip(true, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm
