"""
========================================================================
DBHDSNet — Branch B: Transformer Hazard Branch
Takes C4 backbone features, applies ViT-style self-attention blocks
(LoRA-adapted) to capture global semantic context, and produces:
  - Hazard spatial tokens  : (B, seq_len, embed_dim) for fusion
  - Hazard class logits    : (B, 4)  —  4 hazard tiers
========================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import timm

from .lora import inject_lora, freeze_non_lora


# ════════════════════════════════════════════════════════════════════════
# POSITIONAL EMBEDDING (learnable 2-D sin-cos)
# ════════════════════════════════════════════════════════════════════════

def sincos_pos_embed(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Fixed 2-D sinusoidal positional embedding for square spatial grids.
    seq_len must be a perfect square.
    Returns (1, seq_len, dim) float tensor.
    """
    grid_size = int(seq_len ** 0.5)
    assert grid_size * grid_size == seq_len, "seq_len must be a perfect square"

    pos_y = torch.arange(grid_size, device=device).float()
    pos_x = torch.arange(grid_size, device=device).float()
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)   # (seq_len, 2)

    omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
    omega = 1.0 / (10000 ** omega)   # (dim/4,)

    out = torch.einsum("nd,d->nd", grid, omega.repeat(2))  # (seq_len, dim/2)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # (seq_len, dim)
    return emb.unsqueeze(0)   # (1, seq_len, dim)


# ════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT ViT-STYLE TRANSFORMER BLOCKS
# ════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Standard Pre-LN transformer block (attention + MLP)."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = nn.Identity()   # can replace with StochasticDepth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        nx  = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx)
        x   = x + attn_out
        # MLP with residual
        x   = x + self.mlp(self.norm2(x))
        return x


# ════════════════════════════════════════════════════════════════════════
# BRANCH B
# ════════════════════════════════════════════════════════════════════════

class BranchB(nn.Module):
    """
    Transformer Hazard Branch.

    Architecture:
        C4  (B, 1024, 40, 40)
            │  project to embed_dim via 1×1 conv
            ▼
        tokens  (B, 40×40=1600, embed_dim)
            │  + sincos positional embedding
            │  [class token prepended]
            ▼
        N × TransformerBlock   (with LoRA on attn projections)
            │
            ├── spatial tokens  (B, 1600, embed_dim) → fusion
            └── class token     (B, embed_dim) → hazard MLP → (B, 4)

    Parameters
    ----------
    in_channels  : C4 channel count (1024 for ResNet-50)
    embed_dim    : transformer hidden dimension (default 256 for efficiency)
    depth        : number of transformer blocks
    heads        : number of attention heads
    lora_rank    : LoRA rank for attention layers
    lora_alpha   : LoRA scaling alpha
    num_hazard_tiers : number of output hazard classes (4)
    mc_dropout   : dropout rate for MC-Dropout at inference (uncertainty)
    """

    def __init__(
        self,
        in_channels:     int   = 1024,
        embed_dim:       int   = 256,
        depth:           int   = 6,
        heads:           int   = 8,
        mlp_ratio:       float = 4.0,
        lora_rank:       int   = 16,
        lora_alpha:      float = 32.0,
        num_hazard_tiers: int  = 4,
        mc_dropout:      float = 0.3,
        attn_dropout:    float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth     = depth

        # ── Patch projection: C4 spatial pixels → tokens ──────────────
        # Each spatial location in C4 becomes one token
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        # ── Learnable [CLS] token ─────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, heads=heads,
                mlp_ratio=mlp_ratio, dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        # ── Apply LoRA to all attention projections ────────────────────
        inject_lora(
            self.blocks,
            target_keys = ("in_proj_weight", "out_proj", "q_proj", "k_proj", "v_proj"),
            rank   = lora_rank,
            alpha  = lora_alpha,
            verbose = True,
        )

        # ── Layer normalisation at output ─────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ── Hazard head (MLP on CLS token) ────────────────────────────
        # MC-Dropout is applied here during inference for uncertainty
        self.hazard_head = nn.Sequential(
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim // 2, num_hazard_tiers),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, c4: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        c4 : (B, 1024, H4, W4)  — C4 backbone feature map

        Returns
        -------
        spatial_tokens : (B, H4*W4, embed_dim)  — for cross-attention fusion
        cls_token      : (B, embed_dim)          — for hazard classification
        hazard_logits  : (B, num_hazard_tiers)
        """
        B, _, H4, W4 = c4.shape
        seq_len = H4 * W4

        # Project C4 features to embed_dim tokens
        x = self.proj(c4)                         # (B, embed_dim, H4, W4)
        x = x.flatten(2).transpose(1, 2)          # (B, seq_len, embed_dim)

        # Add positional embedding
        pos = sincos_pos_embed(seq_len, self.embed_dim, c4.device)
        x   = x + pos

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)          # (B, 1+seq_len, embed_dim)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)                           # (B, 1+seq_len, embed_dim)

        cls_out      = x[:, 0]                    # (B, embed_dim)
        spatial_out  = x[:, 1:]                   # (B, seq_len, embed_dim)

        hazard_logits = self.hazard_head(cls_out)  # (B, num_hazard_tiers)

        return spatial_out, cls_out, hazard_logits

    # ------------------------------------------------------------------

    def enable_mc_dropout(self):
        """Enable dropout at inference time for MC-Dropout uncertainty."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        """Standard eval mode — dropout off."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
