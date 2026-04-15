"""
========================================================================
DBHDSNet — LoRA (Low-Rank Adaptation)
Wraps linear layers in transformer attention blocks so only the low-rank
delta matrices are trainable, keeping original pretrained weights frozen.
Reference: Hu et al., 2021 — LoRA: Low-Rank Adaptation of LLMs
========================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# CORE LoRA LINEAR LAYER
# ════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Replaces a pretrained nn.Linear with:
        output = x @ W^T  +  (x @ A^T) @ B^T  *  (alpha / rank)

    W is frozen; A and B are the trainable LoRA matrices.
    Bias (if any) is kept and remains trainable.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 16,
        alpha:        float = 32.0,
        dropout:      float = 0.05,
        bias:         bool  = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.alpha        = alpha
        self.scaling      = alpha / rank

        # ── Frozen pretrained weight ──────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Trainable LoRA matrices ───────────────────────────────────
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B initialised to zero so LoRA starts as identity

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base computation
        out = F.linear(x, self.weight, self.bias)
        # LoRA delta
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return out + self.scaling * lora_out

    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank:   int   = 16,
        alpha:  float = 32.0,
        dropout: float = 0.05,
    ) -> "LoRALinear":
        """
        Convert an existing nn.Linear into a LoRALinear,
        copying its pretrained weights.
        """
        has_bias = linear.bias is not None
        lora = cls(
            in_features  = linear.in_features,
            out_features = linear.out_features,
            rank         = rank,
            alpha        = alpha,
            dropout      = dropout,
            bias         = has_bias,
        )
        # Copy pretrained weights (no grad)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if has_bias:
                lora.bias.copy_(linear.bias)
        return lora

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


# ════════════════════════════════════════════════════════════════════════
# INJECT LoRA INTO A MODULE (in-place)
# ════════════════════════════════════════════════════════════════════════

def inject_lora(
    module:      nn.Module,
    target_keys: tuple = ("qkv", "proj", "fc1", "fc2", "attn"),
    rank:        int   = 16,
    alpha:       float = 32.0,
    dropout:     float = 0.05,
    verbose:     bool  = True,
) -> nn.Module:
    """
    Walk the module tree and replace every nn.Linear whose name
    contains any of `target_keys` with a LoRALinear.

    Returns the modified module (in-place).
    """
    replaced = []

    def _replace(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # Check if this layer should receive LoRA
                if any(k in name for k in target_keys):
                    lora_layer = LoRALinear.from_linear(
                        child, rank=rank, alpha=alpha, dropout=dropout
                    )
                    setattr(parent, name, lora_layer)
                    replaced.append(full_name)
            else:
                _replace(child, full_name)

    _replace(module)

    if verbose and replaced:
        print(f"[LoRA] Injected into {len(replaced)} layers: "
              + ", ".join(replaced[:5])
              + ("…" if len(replaced) > 5 else ""))

    return module


# ════════════════════════════════════════════════════════════════════════
# FREEZE / UNFREEZE HELPERS
# ════════════════════════════════════════════════════════════════════════

def freeze_non_lora(module: nn.Module):
    """Freeze every parameter that is NOT a LoRA matrix (A or B)."""
    for name, param in module.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        is_bias  = name.endswith(".bias")
        param.requires_grad = is_lora or is_bias


def unfreeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def count_lora_params(module: nn.Module) -> dict:
    """Return counts of LoRA-trainable vs total parameters."""
    lora_params  = sum(
        p.numel() for name, p in module.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and p.requires_grad
    )
    total_params = sum(p.numel() for p in module.parameters())
    return {
        "lora_trainable": lora_params,
        "total":          total_params,
        "ratio_pct":      100 * lora_params / max(total_params, 1),
    }
