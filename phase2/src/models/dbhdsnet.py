"""
========================================================================
DBHDSNet — Dual-Branch Hazard-Discriminative Segmentation Network
Full model integrating:
  • Shared ResNet-50 backbone (LoRA-compatible)
  • Branch A: CNN-FPN segmentation branch  (visual features)
  • Branch B: Transformer hazard branch    (semantic context)
  • Bidirectional cross-attention fusion   (F_v ↔ F_h)
  • Multi-scale anchor-free detection heads
  • Proto-mask assembly for instance segmentation
  • Hazard tier classification head
  • MC-Dropout uncertainty estimation
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .backbone  import ResNetBackbone
from .branch_a  import BranchA
from .branch_b  import BranchB
from .fusion    import FusionAdapter
from .heads     import MultiScaleHead, decode_boxes, assemble_masks, mc_dropout_uncertainty


# ════════════════════════════════════════════════════════════════════════
# DBHDSNet
# ════════════════════════════════════════════════════════════════════════

class DBHDSNet(nn.Module):
    """
    Dual-Branch Hazard-Discriminative Segmentation Network.

    Forward pass returns a dict with keys:
        predictions  : dict of raw per-scale predictions (for loss)
        hazard_logits: (B, num_hazard_tiers)
        proto_masks  : (B, K, Hm, Wm)

    During inference (not training), additionally returns:
        boxes_decoded: list of (Ni, 4) decoded boxes per image
        scores       : list of (Ni,) objectness*cls_score
        class_ids    : list of (Ni,) predicted class indices
        masks        : list of (Ni, Hm, Wm) assembled instance masks
        uncertainty  : (B,) epistemic uncertainty from MC-Dropout
    """

    def __init__(self, cfg):
        super().__init__()
        mc   = cfg.MODEL
        self.cfg      = mc
        self.img_size = mc.IMG_SIZE

        # ── Shared backbone ───────────────────────────────────────────
        self.backbone = ResNetBackbone(
            pretrained = mc.BACKBONE_PRETRAINED,
            freeze_bn  = mc.FREEZE_BN,
        )

        # ── Branch A: CNN segmentation (FPN + proto masks) ────────────
        self.branch_a = BranchA(
            fpn_out_ch  = mc.FPN_OUT_CHANNELS,
            num_protos  = mc.NUM_PROTO_MASKS,
        )

        # ── Branch B: Transformer hazard branch ───────────────────────
        self.branch_b = BranchB(
            in_channels      = 1024,          # C4 channel count
            embed_dim        = mc.VIT_EMBED_DIM,
            depth            = 6,
            heads            = mc.FUSION_HEADS,
            lora_rank        = mc.LORA_RANK,
            lora_alpha       = mc.LORA_ALPHA,
            num_hazard_tiers = mc.NUM_HAZARD_TIERS,
            mc_dropout       = mc.MC_DROPOUT_RATE,
        )

        # ── Cross-attention fusion (P5 ↔ Branch B tokens) ─────────────
        self.fusion = FusionAdapter(
            fpn_ch     = mc.FPN_OUT_CHANNELS,
            embed_dim  = mc.VIT_EMBED_DIM,
            fusion_dim = mc.FUSION_DIM,
            heads      = mc.FUSION_HEADS,
            dropout    = mc.FUSION_DROPOUT,
            depth      = 2,
        )

        # ── Multi-scale detection + segmentation heads ─────────────────
        self.det_heads = MultiScaleHead(
            fpn_ch      = mc.FPN_OUT_CHANNELS,
            num_classes = mc.NUM_CLASSES,
            num_protos  = mc.NUM_PROTO_MASKS,
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        """Xavier / He initialisation for newly added modules."""
        for m in [self.fusion, self.det_heads]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        images:         torch.Tensor,
        targets:        Optional[List[dict]] = None,
        return_decoded: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        images  : (B, 3, H, W)
        targets : list of target dicts per image (used for matching in loss)
        return_decoded : if True, run NMS and return final detections

        Returns
        -------
        dict with "predictions", "hazard_logits", "proto_masks",
        and optionally decoded detections.
        """

        # ── 1. Shared backbone ────────────────────────────────────────
        backbone_feats = self.backbone(images)
        # {"C3": (B,512,80,80), "C4": (B,1024,40,40), "C5": (B,2048,20,20)}

        # ── 2. Branch A: FPN + proto masks ────────────────────────────
        fpn_feats, proto_masks = self.branch_a(backbone_feats)
        # fpn_feats: {"P3","P4","P5"}, proto_masks: (B,K,160,160)

        # ── 3. Branch B: Transformer hazard branch ────────────────────
        h_tokens, cls_token, hazard_logits = self.branch_b(backbone_feats["C4"])
        # h_tokens: (B,1600,embed_dim), hazard_logits: (B,4)

        # ── 4. Cross-attention fusion on P5 ───────────────────────────
        p5_fused, h_fused = self.fusion(fpn_feats["P5"], h_tokens)
        fpn_feats["P5"]   = p5_fused               # replace P5 with fused

        # ── 5. Multi-scale detection heads ────────────────────────────
        predictions = self.det_heads(fpn_feats)
        # {"P3": {"reg","obj","cls","mask","stride"}, "P4": ..., "P5": ...}

        output = {
            "predictions":   predictions,     # raw, for loss computation
            "hazard_logits": hazard_logits,    # (B, 4)
            "proto_masks":   proto_masks,      # (B, K, 160, 160)
            "h_tokens":      h_fused,          # (B, N_h, embed_dim) for aux losses
        }

        # ── 6. (Optional) decode boxes → final detections ─────────────
        if return_decoded:
            decoded = self._decode_all_scales(predictions, proto_masks)
            output.update(decoded)

        return output

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_all_scales(
        self,
        predictions: Dict[str, dict],
        proto_masks: torch.Tensor,
    ) -> dict:
        """
        Decode raw predictions from all scales into boxes, scores, masks.
        Returns lists (one entry per image in batch).
        """
        B = proto_masks.shape[0]
        all_boxes, all_scores, all_cls, all_coeffs = [], [], [], []

        for scale, preds in predictions.items():
            reg    = preds["reg"]       # (B, 4,  H, W)
            obj    = preds["obj"]       # (B, 1,  H, W)
            cls    = preds["cls"]       # (B, nc, H, W)
            coeff  = preds["mask"]      # (B, K,  H, W)
            stride = preds["stride"]

            # Decode boxes
            boxes_dec = decode_boxes(reg, stride, self.img_size)  # (B,4,H,W)

            # Compute scores
            obj_score = torch.sigmoid(obj)   # (B,1,H,W)
            cls_prob  = torch.sigmoid(cls)   # (B,nc,H,W)

            B_, _, H, W = boxes_dec.shape

            # Flatten spatial dims
            boxes_flat  = boxes_dec.permute(0,2,3,1).reshape(B_,-1,4)    # (B,HW,4)
            obj_flat    = obj_score.permute(0,2,3,1).reshape(B_,-1,1)    # (B,HW,1)
            cls_flat    = cls_prob.permute(0,2,3,1).reshape(B_,-1,self.cfg.NUM_CLASSES)
            coeff_flat  = coeff.permute(0,2,3,1).reshape(B_,-1,self.cfg.NUM_PROTO_MASKS)

            scores_flat = obj_flat * cls_flat.max(dim=-1, keepdim=True)[0]   # (B,HW,1)
            cls_ids     = cls_flat.argmax(dim=-1, keepdim=True)               # (B,HW,1)

            all_boxes.append(boxes_flat)
            all_scores.append(scores_flat)
            all_cls.append(cls_ids)
            all_coeffs.append(coeff_flat)

        # Concatenate across scales: (B, total_anchors, ?)
        boxes_all  = torch.cat(all_boxes,  dim=1)
        scores_all = torch.cat(all_scores, dim=1).squeeze(-1)
        cls_all    = torch.cat(all_cls,    dim=1).squeeze(-1)
        coeffs_all = torch.cat(all_coeffs, dim=1)

        # Per-image NMS
        from .heads import _nms
        batch_boxes, batch_scores, batch_cls, batch_masks = [], [], [], []

        for b in range(B):
            keeps = _nms(boxes_all[b], scores_all[b],
                         iou_thresh=0.45, conf_thresh=0.25,
                         max_det=300)
            if keeps.numel() > 0:
                det_boxes  = boxes_all[b][keeps]
                det_scores = scores_all[b][keeps]
                det_cls    = cls_all[b][keeps]
                det_coeffs = coeffs_all[b][keeps]
                det_masks  = assemble_masks(
                    proto_masks[b:b+1], det_coeffs, det_boxes, self.img_size
                )
            else:
                det_boxes  = boxes_all[b, :0]
                det_scores = scores_all[b, :0]
                det_cls    = cls_all[b, :0]
                det_masks  = proto_masks.new_zeros(0, *proto_masks.shape[-2:])

            batch_boxes.append(det_boxes)
            batch_scores.append(det_scores)
            batch_cls.append(det_cls)
            batch_masks.append(det_masks)

        return {
            "boxes":    batch_boxes,
            "scores":   batch_scores,
            "class_ids": batch_cls,
            "masks":    batch_masks,
        }

    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self, images: torch.Tensor, n_passes: int = 20
    ) -> Tuple[dict, torch.Tensor]:
        """Inference with MC-Dropout uncertainty estimation."""
        # Standard forward for detections
        self.eval()
        out = self(images, return_decoded=True)

        # MC-Dropout for hazard uncertainty
        mean_probs, uncertainty = mc_dropout_uncertainty(self, images, n_passes)
        out["hazard_mean_probs"] = mean_probs    # (B, 4)
        out["uncertainty"]       = uncertainty   # (B,)
        return out

    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze backbone — use during early warm-up epochs."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True


# ════════════════════════════════════════════════════════════════════════
# NMS (placed here to avoid circular import from heads.py)
# ════════════════════════════════════════════════════════════════════════

def _nms(
    boxes:      torch.Tensor,   # (N, 4) cx,cy,w,h normalised
    scores:     torch.Tensor,   # (N,)
    iou_thresh: float = 0.45,
    conf_thresh: float = 0.25,
    max_det:    int   = 300,
) -> torch.Tensor:
    """Batched-class NMS. Returns indices of kept detections."""
    mask  = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    try:
        from torchvision.ops import nms as tv_nms
        keep = tv_nms(boxes_xyxy, scores, iou_threshold=iou_thresh)
    except Exception:
        # Fallback: greedy NMS
        order  = scores.argsort(descending=True)
        keep   = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            ious = _iou(boxes_xyxy[i:i+1], boxes_xyxy[order[1:]])[0]
            order = order[1:][ious <= iou_thresh]
        keep = torch.tensor(keep, device=boxes.device)

    # Map back through conf_mask (reindex)
    conf_idx = mask.nonzero(as_tuple=False).squeeze(1)
    keep_orig = conf_idx[keep[:max_det]]
    return keep_orig


def _iou(box1, boxes):
    x1 = torch.max(box1[:, 0], boxes[:, 0])
    y1 = torch.max(box1[:, 1], boxes[:, 1])
    x2 = torch.min(box1[:, 2], boxes[:, 2])
    y2 = torch.min(box1[:, 3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])
    a2 = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


# ════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════

def build_model(cfg, device: torch.device) -> DBHDSNet:
    model = DBHDSNet(cfg)
    model = model.to(device)
    return model
