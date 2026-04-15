"""
========================================================================
DBHDSNet Phase 4 — Hospital Client (Local Trainer)

Each federated client represents one hospital/lab site.
The client:
  1. Receives global model parameters (LoRA + hazard head + fusion gates)
  2. Filters its local data shard (by dataset prefix: v1_, v5_, etc.)
  3. Runs LOCAL_EPOCHS of local SGD with:
       - FedProx proximal penalty (if enabled)
       - DP-SGD gradient clipping + noise (if ENABLE_DP=True)
       - Tier-weighted loss (T1 sharps → 3× weight)
  4. Returns a ClientUpdate to the server

The backbone, FPN, and BN are ALWAYS FROZEN on the client.
Only the LoRA matrices, hazard head, and fusion gating are updated.
This enforces the LoRA-only communication protocol.

Non-IID data handling:
  Each client has a different class distribution. Hospital_B (v5)
  is COVID-dominant (high T2). Hospital_C (v6) is sharps-dominant (T1).
  The FedProx proximal term prevents any single client from pulling
  the global model too far toward its local distribution.
========================================================================
"""

import copy
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..utils import (
    AverageMeter, MetricTracker, format_time, gpu_info,
)
from ..privacy.dp_engine import DPEngine
from .aggregators import FedProxAggregator, ClientUpdate, _comm_params


# ════════════════════════════════════════════════════════════════════════
# 1 — CLIENT DATA SHARD BUILDER
# ════════════════════════════════════════════════════════════════════════

class ClientDataShard:
    """
    Filters the full master_dataset to the subset belonging to
    one client (identified by its dataset prefix).

    Each image filename was prefixed during Phase 1 Part 2:
        v1_originalname.jpg    → hospital_A shard
        v5_originalname.jpg    → hospital_B shard
        nv2_originalname.jpg   → lab_A shard
        ...
    """

    def __init__(self, full_dataset, prefix: str, min_samples: int = 50):
        self.prefix      = prefix
        self.full_ds     = full_dataset
        self.min_samples = min_samples
        self.indices     = self._filter_indices()

    def _filter_indices(self) -> List[int]:
        """Return indices of samples whose filename starts with self.prefix."""
        indices = []
        for i, sample in enumerate(self.full_ds.samples):
            img_path = Path(sample["img_path"])
            if img_path.stem.startswith(self.prefix + "_"):
                indices.append(i)
        if len(indices) < self.min_samples:
            import warnings
            warnings.warn(
                f"[Client {self.prefix}] Only {len(indices)} samples found "
                f"(min={self.min_samples}). Consider merging small clients."
            )
        return indices

    def build_loader(self, batch_size: int, num_workers: int = 2) -> DataLoader:
        from ..dataset import MedWasteDataset
        shard = Subset(self.full_ds, self.indices)
        return DataLoader(
            shard,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
            collate_fn  = self.full_ds.collate_fn,
        )

    def __len__(self) -> int:
        return len(self.indices)


# ════════════════════════════════════════════════════════════════════════
# 2 — TIER-WEIGHTED LOCAL LOSS WRAPPER
# ════════════════════════════════════════════════════════════════════════

# Tier loss weights (used in local training, same as Phase 2 + amplified)
TIER_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0}


def tier_weighted_loss(
    base_loss:   torch.Tensor,       # scalar (unreduced) or per-sample
    hazard_tiers: List[torch.Tensor], # list of (Ni,) per-image tier tensors
    device:      torch.device,
) -> torch.Tensor:
    """
    Applies tier-based loss weight: images containing T1 items get 3×
    weight in the local objective, encouraging the model to pay extra
    attention to the most dangerous items in each client's shard.
    """
    B       = len(hazard_tiers)
    weights = []
    for tiers in hazard_tiers:
        if tiers.numel() == 0:
            weights.append(1.0)
        else:
            worst_tier = int(tiers.min().item())   # 1 = most dangerous
            weights.append(TIER_WEIGHTS.get(worst_tier, 1.0))

    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return (base_loss * w).mean()


# ════════════════════════════════════════════════════════════════════════
# 3 — FEDPROX LOCAL OBJECTIVE
# ════════════════════════════════════════════════════════════════════════

def fedprox_penalty(
    model:      nn.Module,
    global_sd:  Dict[str, torch.Tensor],
    mu:         float,
    patterns:   List[str],
) -> torch.Tensor:
    """
    Proximal term: μ/2 · ‖w_local - w_global‖²_F
    Applied over communicable parameters only (LoRA + hazard + gates).
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if any(p in name for p in patterns) and name in global_sd:
            w_g = global_sd[name].to(param.device).float()
            penalty = penalty + (param.float() - w_g).pow(2).sum()
    return (mu / 2.0) * penalty


# ════════════════════════════════════════════════════════════════════════
# 4 — CLIENT TRAINER
# ════════════════════════════════════════════════════════════════════════

class ClientTrainer:
    """
    Runs local training for one hospital client in one federated round.

    Receives the global model, trains for LOCAL_EPOCHS, returns
    a ClientUpdate containing the updated communicable parameters.
    """

    def __init__(
        self,
        client_info: dict,       # entry from CLIENTS list in config
        cfg,
        device:      torch.device,
    ):
        self.client_id  = client_info["client_id"]
        self.prefix     = client_info["prefix"]
        self.cfg        = cfg
        self.fc         = cfg.FED
        self.dc         = cfg.DP
        self.device     = device
        self.patterns   = self.fc.COMM_PARAM_PATTERNS

        self._shard:    Optional[ClientDataShard] = None
        self._n_train:  int = 0

    # ------------------------------------------------------------------

    def set_data_shard(self, shard: ClientDataShard):
        self._shard   = shard
        self._n_train = len(shard)

    # ------------------------------------------------------------------

    def _freeze_non_comm_params(self, model: nn.Module):
        """
        Enforce LoRA-only training: freeze backbone, BN, FPN, Branch A.
        Only LoRA matrices + hazard head + fusion gates are trainable.
        """
        for name, param in model.named_parameters():
            if any(p in name for p in self.patterns):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    # ------------------------------------------------------------------

    def train_one_round(
        self,
        global_model:   nn.Module,
        global_sd:      Dict[str, torch.Tensor],
        criterion,                           # Phase 2 composite loss
        fed_round:      int,
    ) -> ClientUpdate:
        """
        Runs LOCAL_EPOCHS of local training and returns a ClientUpdate.
        """
        if self._shard is None:
            raise RuntimeError(f"[{self.client_id}] set_data_shard() not called.")

        # ── Deep-copy global model to local model ─────────────────────
        local_model = copy.deepcopy(global_model).to(self.device)
        self._freeze_non_comm_params(local_model)

        n_trainable = sum(p.numel() for p in local_model.parameters()
                         if p.requires_grad)

        # ── Build local dataloader ────────────────────────────────────
        loader = self._shard.build_loader(
            batch_size  = self.fc.LOCAL_BATCH_SIZE,
            num_workers = self.cfg.LOCAL.NUM_WORKERS if hasattr(self.cfg, "LOCAL") else 2,
        )

        # ── DP engine ────────────────────────────────────────────────
        dp_engine = DPEngine(self.cfg, self._n_train, self.client_id)

        # ── Optimizer ────────────────────────────────────────────────
        trainable = [p for p in local_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr           = self.fc.LOCAL_LR,
            weight_decay = self.fc.LOCAL_WEIGHT_DECAY,
            betas        = (0.9, 0.999),
        )

        scaler       = GradScaler(enabled=True)
        tracker      = MetricTracker()
        total_steps  = 0
        t0           = time.time()

        mu       = self.fc.FEDPROX_MU if self.fc.AGGREGATION == "fedprox" else 0.0
        use_prox = mu > 0.0

        # ── Local training epochs ─────────────────────────────────────
        for epoch in range(self.fc.LOCAL_EPOCHS):
            local_model.train()

            pbar = tqdm(
                loader,
                desc  = (f"  [{self.client_id}] Rnd {fed_round:04d} "
                         f"Ep {epoch+1}/{self.fc.LOCAL_EPOCHS}"),
                ncols = 120,
                leave = False,
            )

            for batch in pbar:
                images     = batch["images"].to(self.device, non_blocking=True)
                gt_hazard  = [h.to(self.device) for h in batch["hazard_tiers"]]
                B          = images.shape[0]

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    out        = local_model(images)
                    base_loss, loss_dict = criterion(out, batch, self.device)

                    # ── Tier-weighted loss ─────────────────────────────
                    tw_loss = tier_weighted_loss(base_loss, gt_hazard, self.device)

                    # ── FedProx proximal term ──────────────────────────
                    prox = fedprox_penalty(
                        local_model, global_sd, mu, self.patterns
                    ) if use_prox else torch.tensor(0.0, device=self.device)

                    total_loss = tw_loss + prox

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # ── DP clipping + noise ───────────────────────────────
                if dp_engine.enabled:
                    eps = dp_engine.step(local_model, B, self.device)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, self.fc.LOCAL_GRAD_CLIP)
                    eps = 0.0

                scaler.step(optimizer)
                scaler.update()
                total_steps += 1

                loss_dict["loss_prox"] = prox.item()
                loss_dict["epsilon"]   = eps
                tracker.update(loss_dict, n=B)

                avgs = tracker.averages()
                pbar.set_postfix(OrderedDict([
                    ("loss",  f"{avgs.get('loss_total', avgs.get('loss_uq_total', 0)):.4f}"),
                    ("prox",  f"{avgs.get('loss_prox', 0):.4f}"),
                    ("ε",     f"{eps:.3f}"),
                ]))

                if dp_engine.budget_exhausted:
                    tqdm.write(
                        f"  [{self.client_id}] DP budget exhausted "
                        f"(ε={dp_engine.spent_epsilon:.2f} > target "
                        f"{self.dc.TARGET_EPSILON}). Stopping local training."
                    )
                    break

            if dp_engine.budget_exhausted:
                break

        # ── Compute drift score ───────────────────────────────────────
        drift_score = self._compute_drift(global_sd, local_model)

        elapsed = format_time(time.time() - t0)
        tqdm.write(
            f"  [{self.client_id}] Round {fed_round:04d} done  "
            f"│ steps={total_steps}  ε={dp_engine.spent_epsilon:.3f}  "
            f"drift={drift_score:.4f}  ⏱ {elapsed}"
        )

        return ClientUpdate(
            client_id   = self.client_id,
            state_dict  = local_model.state_dict(),
            n_samples   = self._n_train,
            local_steps = total_steps,
            metrics     = tracker.averages(),
            dp_epsilon  = dp_engine.spent_epsilon,
            drift_score = drift_score,
        )

    # ------------------------------------------------------------------

    def _compute_drift(
        self,
        global_sd:   Dict[str, torch.Tensor],
        local_model: nn.Module,
    ) -> float:
        """Cosine distance between global and local communicable params."""
        local_sd    = local_model.state_dict()
        g_parts, l_parts = [], []

        for name in global_sd:
            if any(p in name for p in self.patterns):
                if name in local_sd:
                    g_parts.append(global_sd[name].float().flatten())
                    l_parts.append(local_sd[name].cpu().float().flatten())

        if not g_parts:
            return 0.0

        g_vec = torch.cat(g_parts)
        l_vec = torch.cat(l_parts)

        if g_vec.norm() < 1e-9 or l_vec.norm() < 1e-9:
            return 0.0

        cos_sim = F.cosine_similarity(g_vec.unsqueeze(0), l_vec.unsqueeze(0)).item()
        return max(1.0 - cos_sim, 0.0)


from collections import OrderedDict
