"""
========================================================================
DBHDSNet — Branch A: CNN Segmentation Branch
Feature Pyramid Network (FPN) on C3/C4/C5 → multi-scale feature maps
P3, P4, P5 for anchor-free detection + instance segmentation.
Proto-mask network generates K=32 learnable prototype masks.
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .backbone import ConvBnAct


# ════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════════════

class BottleneckCSP(nn.Module):
    """Cross-Stage Partial bottleneck (YOLOv5 style)."""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        mid = out_ch // 2
        self.cv1 = ConvBnAct(in_ch,  mid, 1)
        self.cv2 = ConvBnAct(in_ch,  mid, 1)
        self.cv3 = ConvBnAct(mid * 2, out_ch, 1)
        self.m   = nn.Sequential(*[
            _ResBlock(mid, mid, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True):
        super().__init__()
        self.cv1 = ConvBnAct(in_ch, out_ch, 3, p=1)
        self.cv2 = ConvBnAct(out_ch, out_ch, 3, p=1)
        self.use_residual = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.use_residual else out


# ════════════════════════════════════════════════════════════════════════
# FEATURE PYRAMID NETWORK
# ════════════════════════════════════════════════════════════════════════

class FPN(nn.Module):
    """
    Top-down FPN: fuses C3, C4, C5 → P3, P4, P5.
    All output feature maps have `out_ch` channels.

    P3 → stride 8  (80×80 for 640-input)
    P4 → stride 16 (40×40)
    P5 → stride 32 (20×20)
    """

    BACKBONE_CHS = {"C3": 512, "C4": 1024, "C5": 2048}

    def __init__(self, out_ch: int = 256):
        super().__init__()
        # Lateral projections (1×1)
        self.lat5 = ConvBnAct(2048, out_ch, 1)
        self.lat4 = ConvBnAct(1024, out_ch, 1)
        self.lat3 = ConvBnAct(512,  out_ch, 1)

        # Output convolutions (3×3)
        self.out5 = BottleneckCSP(out_ch,         out_ch, n=3)
        self.out4 = BottleneckCSP(out_ch * 2,     out_ch, n=3)
        self.out3 = BottleneckCSP(out_ch * 2,     out_ch, n=3)

        # PAN bottom-up path
        self.down4 = ConvBnAct(out_ch, out_ch, 3, s=2, p=1)
        self.down3 = ConvBnAct(out_ch, out_ch, 3, s=2, p=1)
        self.pan4  = BottleneckCSP(out_ch * 2, out_ch, n=3)
        self.pan5  = BottleneckCSP(out_ch * 2, out_ch, n=3)

    def forward(self, features: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = features["C3"], features["C4"], features["C5"]

        # ── Top-down path ─────────────────────────────────────────────
        p5 = self.lat5(c5)                                       # 20×20
        p4_td = self.lat4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="nearest")              # 40×40
        p3_td = self.lat3(c3) + F.interpolate(
            p4_td, size=c3.shape[-2:], mode="nearest")           # 80×80

        p5 = self.out5(p5)
        p4 = self.out4(torch.cat([p4_td, F.interpolate(
            p5, size=p4_td.shape[-2:], mode="nearest")], dim=1))
        p3 = self.out3(torch.cat([p3_td, F.interpolate(
            p4, size=p3_td.shape[-2:], mode="nearest")], dim=1))

        # ── Bottom-up path (PAN) ──────────────────────────────────────
        p4 = self.pan4(torch.cat([p4, self.down3(p3)], dim=1))  # 40×40
        p5 = self.pan5(torch.cat([p5, self.down4(p4)], dim=1))  # 20×20

        return {"P3": p3, "P4": p4, "P5": p5}


# ════════════════════════════════════════════════════════════════════════
# PROTO-MASK NETWORK
# ════════════════════════════════════════════════════════════════════════

class ProtoMaskNet(nn.Module):
    """
    Generates K=32 prototype masks at 2× the P3 resolution.
    For 640-input: P3 is 80×80 → proto masks are 160×160.

    Instance masks are assembled as:
        mask = sigmoid(coefficients[i] @ proto_masks.view(K, -1))
                      .view(H_mask, W_mask)
    """

    def __init__(self, in_ch: int = 256, num_protos: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnAct(in_ch, 256, 3, p=1),
            nn.Upsample(scale_factor=2, mode="nearest"),   # 160×160
            ConvBnAct(256, 256, 3, p=1),
            ConvBnAct(256, num_protos, 1, act=False),      # (B, K, 160, 160)
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        return self.net(p3)   # (B, K, H_mask, W_mask)


# ════════════════════════════════════════════════════════════════════════
# BRANCH A (full)
# ════════════════════════════════════════════════════════════════════════

class BranchA(nn.Module):
    """
    CNN Segmentation Branch.

    Inputs : backbone feature dict {"C3", "C4", "C5"}
    Outputs:
        fpn_features : dict {"P3", "P4", "P5"}  — spatial feature maps
        proto_masks  : (B, K, H_mask, W_mask)   — prototype masks
    """

    def __init__(self, fpn_out_ch: int = 256, num_protos: int = 32):
        super().__init__()
        self.fpn   = FPN(out_ch=fpn_out_ch)
        self.proto = ProtoMaskNet(in_ch=fpn_out_ch, num_protos=num_protos)

    def forward(self, backbone_feats: Dict[str, torch.Tensor]
                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        fpn_feats   = self.fpn(backbone_feats)
        proto_masks = self.proto(fpn_feats["P3"])
        return fpn_feats, proto_masks
