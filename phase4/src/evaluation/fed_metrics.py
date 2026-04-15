"""
========================================================================
DBHDSNet Phase 4 — Federated Evaluation Metrics

Computes:
  • Global model evaluation (mAP@50, mAP@75, per-class AP, hazard accuracy)
  • Per-client local evaluation (measures per-site performance)
  • Convergence gap: global_mAP / max(per_client_mAP) — measures federation benefit
  • Fairness metric: std(per_client_mAP) — lower = fairer model across hospitals
  • Communication efficiency: mAP per MB communicated
========================================================================
"""

from __future__ import annotations
import copy
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from ..metrics import MeanAveragePrecision


# ════════════════════════════════════════════════════════════════════════
# 1 — GLOBAL MODEL EVALUATOR
# ════════════════════════════════════════════════════════════════════════

class FedEvaluator:
    """
    Evaluates the global (or EMA) model on the federated val/test set.
    Wraps Phase 2's MeanAveragePrecision metric.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.ec     = cfg.EVAL

    @torch.no_grad()
    def evaluate(
        self,
        model:     nn.Module,
        loader,
        desc:      str = "Evaluating",
    ) -> Dict[str, float]:
        """
        Returns dict with mAP_50, mAP_75, hazard_acc, and per-tier AP.
        """
        model.eval()
        map_metric   = MeanAveragePrecision(
            iou_thresholds = [self.ec.MAP_IOU_50, self.ec.MAP_IOU_75],
            num_classes    = self.cfg.MODEL.NUM_CLASSES,
        )
        hazard_correct = 0
        hazard_total   = 0

        pbar = tqdm(loader, desc=desc, ncols=100, leave=False)
        for batch in pbar:
            images    = batch["images"].to(self.device, non_blocking=True)
            gt_hazard = [h.to(self.device) for h in batch["hazard_tiers"]]

            out = model(images)

            # mAP update
            if "boxes" in out and "scores" in out and "class_ids" in out:
                map_metric.update(
                    preds      = out,
                    targets    = batch,
                    conf_thresh = self.ec.CONF_THRESH,
                    nms_thresh  = self.ec.NMS_IOU_THRESH,
                )

            # Hazard accuracy
            if "hazard_logits" in out:
                preds  = out["hazard_logits"].argmax(dim=-1)   # 0-indexed
                for b, tiers in enumerate(gt_hazard):
                    if tiers.numel() > 0:
                        true_tier = int(tiers.min().item()) - 1   # 0-indexed
                        hazard_correct += int(preds[b].item() == true_tier)
                        hazard_total   += 1

        metrics = map_metric.compute()
        metrics["hazard_acc"] = hazard_correct / max(hazard_total, 1)
        return metrics


# ════════════════════════════════════════════════════════════════════════
# 2 — PER-CLIENT EVALUATOR
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_per_client(
    global_model: nn.Module,
    clients:      list,
    val_dataset,
    cfg,
    device:       torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the global model on each client's local validation shard.
    Returns {client_id: {mAP_50, mAP_75, hazard_acc}}.

    This measures: how well does the global model perform for each
    individual hospital? A large variance indicates unfairness.
    """
    from ..federation.client import ClientDataShard
    from torch.utils.data import DataLoader

    evaluator = FedEvaluator(cfg, device)
    per_client = {}

    for client in clients:
        shard = ClientDataShard(val_dataset, prefix=client.prefix)
        if len(shard) == 0:
            per_client[client.client_id] = {"mAP_50": 0.0, "hazard_acc": 0.0}
            continue

        loader = DataLoader(
            torch.utils.data.Subset(val_dataset, shard.indices),
            batch_size  = 4,
            shuffle     = False,
            num_workers = 2,
            collate_fn  = val_dataset.collate_fn,
        )
        metrics = evaluator.evaluate(
            global_model, loader,
            desc = f"  Eval [{client.client_id}]",
        )
        per_client[client.client_id] = metrics

    return per_client


# ════════════════════════════════════════════════════════════════════════
# 3 — FEDERATION BENEFIT METRICS
# ════════════════════════════════════════════════════════════════════════

def compute_federation_benefit(
    global_metrics:    Dict[str, float],
    per_client_metrics: Dict[str, Dict[str, float]],
    metric_key:        str = "mAP_50",
) -> Dict[str, float]:
    """
    Computes federated learning benefit indicators.

    Convergence gap:
      Δ_conv = global_mAP - mean(per_client_mAP_local_only)
      > 0 means federation improved performance over local training alone.

    Fairness (coefficient of variation):
      CV = std(per_client_mAP) / mean(per_client_mAP)
      Lower is fairer. FL should produce a more equitable model than
      any single client training locally.

    Best-client ratio:
      global_mAP / max(per_client_mAP)
      If close to 1.0, global model matches the best client's performance.
    """
    import numpy as np

    per_vals = [m.get(metric_key, 0.0) for m in per_client_metrics.values()]
    global_v = global_metrics.get(metric_key, 0.0)

    if not per_vals:
        return {}

    mean_local = float(np.mean(per_vals))
    std_local  = float(np.std(per_vals))
    max_local  = float(max(per_vals))
    cv         = std_local / (mean_local + 1e-9)

    return {
        "global_mAP":          global_v,
        "mean_client_mAP":     mean_local,
        "std_client_mAP":      std_local,
        "max_client_mAP":      max_local,
        "convergence_gap":     global_v - mean_local,   # > 0 = FL benefit
        "fairness_cv":         cv,                       # lower = fairer
        "best_client_ratio":   global_v / (max_local + 1e-9),
        "worst_client_mAP":    float(min(per_vals)),
    }
