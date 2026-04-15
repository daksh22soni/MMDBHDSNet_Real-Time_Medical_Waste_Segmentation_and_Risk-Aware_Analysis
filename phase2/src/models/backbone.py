"""
========================================================================
DBHDSNet — Shared ResNet-50 Backbone
Extracts multi-scale features (C3, C4, C5) for both branches.
BatchNorm layers are frozen to preserve ImageNet statistics.
========================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

import torchvision.models as tvm
from torchvision.models import ResNet50_Weights


# ════════════════════════════════════════════════════════════════════════
# BACKBONE
# ════════════════════════════════════════════════════════════════════════

class ResNetBackbone(nn.Module):
    """
    Shared feature extractor built on ResNet-50.

    Outputs three feature maps at 1/8, 1/16, and 1/32 of the input:
        C3 → (B, 512,  H/8,  W/8)   [layer2 output]
        C4 → (B, 1024, H/16, W/16)  [layer3 output]
        C5 → (B, 2048, H/32, W/32)  [layer4 output]

    The stem (conv1+bn1+relu+maxpool) and layer1 are always frozen.
    BatchNorm statistics are frozen throughout.
    """

    OUT_CHANNELS = {"C3": 512, "C4": 1024, "C5": 2048}

    def __init__(self, pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base    = tvm.resnet50(weights=weights)

        # ── Stem ─────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        # C2: 256 ch, 1/4 resolution (layer1)
        self.layer1 = base.layer1   # stride 1, output: (B, 256, H/4, W/4)
        # C3: 512 ch, 1/8 resolution
        self.layer2 = base.layer2   # stride 2, output: (B, 512, H/8, W/8)
        # C4: 1024 ch, 1/16 resolution
        self.layer3 = base.layer3   # stride 2, output: (B, 1024, H/16, W/16)
        # C5: 2048 ch, 1/32 resolution
        self.layer4 = base.layer4   # stride 2, output: (B, 2048, H/32, W/32)

        # ── Freeze stem & layer1 always ───────────────────────────────
        for m in [self.stem, self.layer1]:
            for p in m.parameters():
                p.requires_grad = False

        # ── Optionally freeze BN throughout ──────────────────────────
        if freeze_bn:
            self._freeze_bn()

    # ------------------------------------------------------------------

    def _freeze_bn(self):
        """Set all BN layers to eval mode permanently."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override to keep BN in eval even during model.train()."""
        super().train(mode)
        self._freeze_bn()
        return self

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  — normalised input image

        Returns
        -------
        dict with keys "C3", "C4", "C5" mapping to feature tensors
        """
        x  = self.stem(x)     # (B,  64, H/4,  W/4)
        c2 = self.layer1(x)   # (B, 256, H/4,  W/4)
        c3 = self.layer2(c2)  # (B, 512, H/8,  W/8)
        c4 = self.layer3(c3)  # (B,1024, H/16, W/16)
        c5 = self.layer4(c4)  # (B,2048, H/32, W/32)
        return {"C3": c3, "C4": c4, "C5": c5}


# ════════════════════════════════════════════════════════════════════════
# CHANNEL PROJECTION (1×1 conv — shared utility)
# ════════════════════════════════════════════════════════════════════════

class ConvBnAct(nn.Module):
    """Conv → BN → activation block."""
    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        k:      int  = 1,
        s:      int  = 1,
        p:      int  = 0,
        act:    bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
