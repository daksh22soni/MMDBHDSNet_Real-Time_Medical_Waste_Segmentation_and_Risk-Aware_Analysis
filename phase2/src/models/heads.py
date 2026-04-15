"""
========================================================================
DBHDSNet — Detection & Segmentation Heads
Anchor-free detection head (one per FPN scale) that predicts:
  [cx, cy, w, h] + objectness + 38-class scores + 32 mask coefficients
Mask assembly: coefficients @ proto_masks → instance binary mask
Uncertainty: variance across N MC-Dropout forward passes
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ConvBnAct


# ════════════════════════════════════════════════════════════════════════
# SHARED HEAD STEM (Conv stack before final prediction)
# ════════════════════════════════════════════════════════════════════════

class HeadStem(nn.Module):
    """3-layer conv stack shared between cls and reg sub-heads."""
    def __init__(self, in_ch: int, out_ch: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(ConvBnAct(in_ch if i == 0 else out_ch, out_ch, 3, p=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# ANCHOR-FREE DETECTION HEAD (per FPN scale)
# ════════════════════════════════════════════════════════════════════════

class DetectionHead(nn.Module):
    """
    Anchor-free head for one FPN level.

    For each grid cell predicts:
      reg   : (4,)  — [tx, ty, tw, th]
      obj   : (1,)  — objectness logit
      cls   : (num_classes,)
      mask  : (num_protos,) — mask coefficients for this detection

    Total output channels = 5 + num_classes + num_protos
    """

    STRIDE_MAP = {0: 8, 1: 16, 2: 32}   # scale_idx → stride

    def __init__(
        self,
        in_ch:       int,
        num_classes: int = 38,
        num_protos:  int = 32,
        stem_depth:  int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_protos  = num_protos

        # Shared stems (separate for cls and reg for better gradients)
        self.cls_stem  = HeadStem(in_ch, 256, stem_depth)
        self.reg_stem  = HeadStem(in_ch, 256, stem_depth)

        # Final prediction layers
        self.cls_pred  = nn.Conv2d(256, num_classes, 1)
        self.obj_pred  = nn.Conv2d(256, 1,           1)
        self.reg_pred  = nn.Conv2d(256, 4,           1)
        self.mask_pred = nn.Conv2d(256, num_protos,  1)

        self._init_weights()

    def _init_weights(self):
        """Initialise bias for objectness head with prior probability."""
        nn.init.constant_(self.obj_pred.bias, -math.log((1 - 0.01) / 0.01))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        reg_pred   : (B, 4,           H, W)
        obj_pred   : (B, 1,           H, W)
        cls_pred   : (B, num_classes, H, W)
        mask_coeff : (B, num_protos,  H, W)
        """
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)

        return (
            self.reg_pred(reg_feat),
            self.obj_pred(reg_feat),
            self.cls_pred(cls_feat),
            self.mask_pred(cls_feat),
        )


import math


# ════════════════════════════════════════════════════════════════════════
# MULTI-SCALE DETECTION HEAD WRAPPER
# ════════════════════════════════════════════════════════════════════════

class MultiScaleHead(nn.Module):
    """
    Applies one DetectionHead per FPN level (P3, P4, P5).
    Note: in DBHDSNet, P5 is replaced by the fusion-enhanced P5.

    Returns a dict of raw predictions keyed by scale name.
    """

    STRIDES = {"P3": 8, "P4": 16, "P5": 32}

    def __init__(self, fpn_ch: int = 256, num_classes: int = 38, num_protos: int = 32):
        super().__init__()
        # Shared-weight heads across scales (more parameter-efficient)
        self.head = DetectionHead(fpn_ch, num_classes, num_protos)

    def forward(
        self, fpn_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        fpn_feats : {"P3": tensor, "P4": tensor, "P5": tensor}
        Returns   : {"P3": {"reg", "obj", "cls", "mask"}, ...}
        """
        preds = {}
        for scale, feat in fpn_feats.items():
            reg, obj, cls, mask = self.head(feat)
            preds[scale] = {
                "reg":  reg,    # (B, 4,  H, W)
                "obj":  obj,    # (B, 1,  H, W)
                "cls":  cls,    # (B, nc, H, W)
                "mask": mask,   # (B, K,  H, W)
                "stride": self.STRIDES[scale],
            }
        return preds


# ════════════════════════════════════════════════════════════════════════
# BOX DECODING (anchor-free)
# ════════════════════════════════════════════════════════════════════════

def decode_boxes(
    reg:    torch.Tensor,   # (B, 4, H, W)  — raw tx,ty,tw,th
    stride: int,
    img_size: int = 640,
) -> torch.Tensor:
    """
    Converts raw regression output to normalised [cx, cy, w, h] boxes.

    Decoding:
        cx = (sigmoid(tx) + grid_x) * stride / img_size
        cy = (sigmoid(ty) + grid_y) * stride / img_size
        w  = exp(tw) * stride / img_size        (clamped to avoid explosion)
        h  = exp(th) * stride / img_size
    """
    B, _, H, W = reg.shape
    device = reg.device

    grid_y = torch.arange(H, device=device).float()
    grid_x = torch.arange(W, device=device).float()
    gx, gy = torch.meshgrid(grid_x, grid_y, indexing="xy")   # (H, W)
    gx = gx.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    gy = gy.unsqueeze(0).unsqueeze(0)

    tx, ty, tw, th = reg[:, 0:1], reg[:, 1:2], reg[:, 2:3], reg[:, 3:4]

    cx = (torch.sigmoid(tx) + gx) * stride / img_size
    cy = (torch.sigmoid(ty) + gy) * stride / img_size
    w  = torch.exp(tw.clamp(-4, 4)) * stride / img_size
    h  = torch.exp(th.clamp(-4, 4)) * stride / img_size

    return torch.cat([cx, cy, w, h], dim=1)   # (B, 4, H, W)


# ════════════════════════════════════════════════════════════════════════
# MASK ASSEMBLY
# ════════════════════════════════════════════════════════════════════════

def assemble_masks(
    proto:  torch.Tensor,       # (B, K, Hm, Wm)
    coeffs: torch.Tensor,       # (N, K)  — per-detection coefficients
    boxes:  torch.Tensor,       # (N, 4)  — [cx,cy,w,h] normalised
    img_size: int = 640,
) -> torch.Tensor:
    """
    Assemble per-instance binary masks from proto masks and coefficients.

    Returns (N, Hm, Wm) float32 masks (sigmoid applied, threshold later).
    """
    B, K, Hm, Wm = proto.shape
    N = coeffs.shape[0]

    if N == 0:
        return torch.zeros((0, Hm, Wm), device=proto.device)

    # (N, K) @ (K, Hm*Wm) → (N, Hm*Wm)
    proto_flat  = proto[0].view(K, -1)            # use batch[0] (single image)
    mask_logits = torch.mm(coeffs, proto_flat)     # (N, Hm*Wm)
    masks       = torch.sigmoid(mask_logits).view(N, Hm, Wm)

    # Crop mask to predicted bounding box
    masks = _crop_masks(masks, boxes, Hm, Wm)

    return masks


def _crop_masks(
    masks:  torch.Tensor,   # (N, Hm, Wm)
    boxes:  torch.Tensor,   # (N, 4)  [cx, cy, w, h] normalised
    Hm: int, Wm: int,
) -> torch.Tensor:
    """Zero out mask pixels outside each predicted bounding box."""
    N = masks.shape[0]
    x1 = ((boxes[:, 0] - boxes[:, 2] / 2) * Wm).long().clamp(0, Wm)
    y1 = ((boxes[:, 1] - boxes[:, 3] / 2) * Hm).long().clamp(0, Hm)
    x2 = ((boxes[:, 0] + boxes[:, 2] / 2) * Wm).long().clamp(0, Wm)
    y2 = ((boxes[:, 1] + boxes[:, 3] / 2) * Hm).long().clamp(0, Hm)

    for i in range(N):
        crop_mask = torch.zeros_like(masks[i])
        crop_mask[y1[i]:y2[i], x1[i]:x2[i]] = 1.0
        masks[i] = masks[i] * crop_mask

    return masks


# ════════════════════════════════════════════════════════════════════════
# UNCERTAINTY ESTIMATION (MC-Dropout)
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mc_dropout_uncertainty(
    model,
    images: torch.Tensor,
    n_passes: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs N stochastic forward passes with dropout enabled (MC-Dropout).
    Returns:
        mean_hazard_probs : (B, num_hazard_tiers)
        epistemic_uncertainty : (B,)  — mean predictive variance across tiers
    """
    model.branch_b.enable_mc_dropout()

    all_probs = []
    for _ in range(n_passes):
        out = model(images)
        hazard_logits = out["hazard_logits"]              # (B, num_tiers)
        probs = torch.softmax(hazard_logits, dim=-1)
        all_probs.append(probs.unsqueeze(0))              # (1, B, num_tiers)

    model.branch_b.disable_mc_dropout()

    all_probs = torch.cat(all_probs, dim=0)              # (N, B, num_tiers)
    mean_probs = all_probs.mean(dim=0)                   # (B, num_tiers)
    variance   = all_probs.var(dim=0).mean(dim=-1)       # (B,)

    return mean_probs, variance
