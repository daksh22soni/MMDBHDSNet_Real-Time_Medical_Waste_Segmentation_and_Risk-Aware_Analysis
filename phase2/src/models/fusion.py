"""
========================================================================
DBHDSNet — Cross-Attention Fusion Module
Bidirectional cross-attention between:
  • F_v (visual)  : P5 from FPN  — spatial texture/shape features
  • F_h (hazard)  : spatial tokens from Branch B — semantic context

Two attention directions:
  A→B : Q = F_v,  K = V = F_h  →  visual tokens attend to hazard context
  B→A : Q = F_h,  K = V = F_v  →  hazard tokens attend to visual detail

A learned gating network controls how much each branch contributes.
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ════════════════════════════════════════════════════════════════════════
# PROJECTION (aligns dimensions before cross-attention)
# ════════════════════════════════════════════════════════════════════════

class DimAlign(nn.Module):
    """Projects two tensors to a common dimension."""
    def __init__(self, dim_v: int, dim_h: int, out_dim: int):
        super().__init__()
        self.proj_v = nn.Linear(dim_v, out_dim, bias=False)
        self.proj_h = nn.Linear(dim_h, out_dim, bias=False)

    def forward(self, f_v, f_h):
        return self.proj_v(f_v), self.proj_h(f_h)


# ════════════════════════════════════════════════════════════════════════
# GATED CROSS-ATTENTION UNIT (one direction)
# ════════════════════════════════════════════════════════════════════════

class CrossAttentionUnit(nn.Module):
    """
    Computes cross-attention: Query=source, Key/Value=context.
    Output = LN(source + dropout(cross_attn(source, context))).
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm  = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self,
                query:   torch.Tensor,   # (B, N_q, dim)
                context: torch.Tensor,   # (B, N_k, dim)
                ) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context)
        return self.norm(query + self.drop(attn_out))


# ════════════════════════════════════════════════════════════════════════
# MAIN FUSION MODULE
# ════════════════════════════════════════════════════════════════════════

class BidirectionalFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.

    Inputs (after DimAlign projection to `fusion_dim`):
        f_v  : (B, N_v, fusion_dim)   N_v = 20×20 = 400  for P5 at 640px
        f_h  : (B, N_h, fusion_dim)   N_h = 40×40 = 1600 for C4-derived tokens

    Outputs:
        fused_v  : (B, N_v, fusion_dim)  — enhanced P5 tokens
        fused_h  : (B, N_h, fusion_dim)  — enhanced hazard tokens

    The gate_alpha scalar per-element weight controls the blend:
        out = gate * cross_attended + (1 - gate) * original
    """

    def __init__(
        self,
        dim_v:      int   = 256,    # FPN P5 channel count
        dim_h:      int   = 256,    # Branch B embed_dim
        fusion_dim: int   = 256,    # common attention dimension
        heads:      int   = 8,
        dropout:    float = 0.1,
        depth:      int   = 2,      # number of fusion rounds
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Project both branches to the same dimension
        self.align = DimAlign(dim_v, dim_h, fusion_dim)

        # Bidirectional cross-attention (repeated `depth` times)
        self.attn_v2h = nn.ModuleList([
            CrossAttentionUnit(fusion_dim, heads, dropout) for _ in range(depth)
        ])
        self.attn_h2v = nn.ModuleList([
            CrossAttentionUnit(fusion_dim, heads, dropout) for _ in range(depth)
        ])

        # Learned gating scalars (per-dimension, sigmoid-normalised)
        self.gate_v = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.gate_h = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # MLP to refine the fused visual tokens before reshaping back
        self.refine_v = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        p5: torch.Tensor,              # (B, C, H5, W5)  — FPN P5 feature map
        h_tokens: torch.Tensor,        # (B, N_h, embed_dim) — Branch B tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        p5_fused    : (B, C_out, H5, W5)  — enhanced P5, same spatial size
        h_fused     : (B, N_h, fusion_dim) — updated hazard tokens
        """
        B, C, H5, W5 = p5.shape
        N_v = H5 * W5

        # Flatten P5 spatial dims → token sequence
        f_v = p5.flatten(2).transpose(1, 2)    # (B, N_v, C)

        # Align both to fusion_dim
        f_v, f_h = self.align(f_v, h_tokens)   # each: (B, N_?, fusion_dim)

        # Bidirectional cross-attention rounds
        for ca_v2h, ca_h2v in zip(self.attn_v2h, self.attn_h2v):
            # Visual tokens attend to hazard context
            attn_v = ca_v2h(f_v, f_h)          # (B, N_v, fusion_dim)
            # Hazard tokens attend to visual detail
            attn_h = ca_h2v(f_h, f_v)          # (B, N_h, fusion_dim)
            # Gated residual update
            f_v = self.gate_v(f_v) * attn_v + (1 - self.gate_v(f_v)) * f_v
            f_h = self.gate_h(f_h) * attn_h + (1 - self.gate_h(f_h)) * f_h

        # Refine visual tokens
        f_v = self.refine_v(f_v)               # (B, N_v, fusion_dim)

        # Reshape visual tokens back to spatial map
        p5_fused = f_v.transpose(1, 2).view(B, self.fusion_dim, H5, W5)

        return p5_fused, f_h


# ════════════════════════════════════════════════════════════════════════
# CHANNEL ADAPTER (maps fusion_dim back to fpn_out_ch if different)
# ════════════════════════════════════════════════════════════════════════

class FusionAdapter(nn.Module):
    """
    Wraps BidirectionalFusion and adapts the output channel count of
    the enhanced P5 back to the FPN output channel count expected by
    the detection heads.
    """
    def __init__(
        self,
        fpn_ch:     int = 256,
        embed_dim:  int = 256,
        fusion_dim: int = 256,
        heads:      int = 8,
        dropout:    float = 0.1,
        depth:      int  = 2,
    ):
        super().__init__()
        self.fusion = BidirectionalFusion(
            dim_v=fpn_ch, dim_h=embed_dim,
            fusion_dim=fusion_dim, heads=heads,
            dropout=dropout, depth=depth,
        )
        # Project fused dim back to fpn_ch (no-op if equal)
        self.adapt = (
            nn.Conv2d(fusion_dim, fpn_ch, 1)
            if fusion_dim != fpn_ch else nn.Identity()
        )

    def forward(self, p5, h_tokens):
        p5_fused, h_fused = self.fusion(p5, h_tokens)
        p5_fused = self.adapt(p5_fused)        # (B, fpn_ch, H5, W5)
        return p5_fused, h_fused
