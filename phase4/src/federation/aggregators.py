"""
========================================================================
DBHDSNet Phase 4 — Federated Aggregation Algorithms

Implements four server-side aggregation strategies:

  FedAvg   (McMahan et al., 2017)
    — Weighted average of client model parameters.
    — Weight = n_samples_i / Σ n_samples.

  FedProx  (Li et al., 2020)
    — Adds a proximal term μ/2 ‖w - w_global‖² to each client's
      local objective, limiting how far clients drift from the
      global model. The server-side aggregation is identical to
      FedAvg; the proximal term lives in the client trainer.

  FedNova  (Wang et al., 2021)
    — Corrects for gradient heterogeneity from unequal local steps.
    — Normalises each client's update by τ_eff (effective local steps)
      before aggregating, then applies the combined gradient via
      a server learning rate ηₛ.

  FedAdam  (Reddi et al., 2021)
    — Applies server-side Adam to the aggregated pseudo-gradient.
    — Most powerful for heterogeneous non-IID data but requires
      tuning the server LR and τ adaptivity parameter.

All aggregators operate on the COMMUNICATION SUBSET of parameters
(LoRA matrices + hazard head + fusion gates) when LORA_ONLY_COMM=True.
The frozen backbone / BN parameters are never touched.

Reference:
  McMahan et al. (2017) — Communication-Efficient Learning (AISTATS)
  Li et al.     (2020) — Federated Optimisation in Heterogeneous Networks
  Wang et al.   (2021) — Tackling the Objective Inconsistency Problem (NeurIPS)
  Reddi et al.  (2021) — Adaptive Federated Optimisation (ICLR)
========================================================================
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════════
# 0 — HELPERS
# ════════════════════════════════════════════════════════════════════════

def _comm_params(
    state_dict: Dict[str, torch.Tensor],
    patterns:   List[str],
) -> Dict[str, torch.Tensor]:
    """Extract the communicable subset of a state_dict."""
    if not patterns:
        return state_dict
    return {
        k: v for k, v in state_dict.items()
        if any(p in k for p in patterns)
    }


def _merge_state(
    global_sd: Dict[str, torch.Tensor],
    comm_sd:   Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Merge communicated parameters back into a full state dict."""
    merged = copy.deepcopy(global_sd)
    for k, v in comm_sd.items():
        if k in merged:
            merged[k] = v.clone()
    return merged


# ════════════════════════════════════════════════════════════════════════
# 1 — CLIENT UPDATE CONTAINER
# ════════════════════════════════════════════════════════════════════════

class ClientUpdate:
    """
    Container holding everything the server needs from one client
    after a local training round.

    Attributes
    ----------
    client_id    : identifier string
    state_dict   : full or comm-subset model state after local training
    n_samples    : number of local training samples (for weighting)
    local_steps  : number of local gradient steps taken (for FedNova)
    metrics      : dict of local training metrics (loss, mAP, etc.)
    dp_epsilon   : per-round epsilon spent (for privacy accounting)
    drift_score  : cosine distance to global model (for drift-aware weighting)
    """
    def __init__(
        self,
        client_id:   str,
        state_dict:  Dict[str, torch.Tensor],
        n_samples:   int,
        local_steps: int,
        metrics:     Dict[str, float],
        dp_epsilon:  float = 0.0,
        drift_score: float = 0.0,
    ):
        self.client_id   = client_id
        self.state_dict  = {k: v.cpu() for k, v in state_dict.items()}
        self.n_samples   = n_samples
        self.local_steps = local_steps
        self.metrics     = metrics
        self.dp_epsilon  = dp_epsilon
        self.drift_score = drift_score


# ════════════════════════════════════════════════════════════════════════
# 2 — BASE AGGREGATOR
# ════════════════════════════════════════════════════════════════════════

class BaseAggregator:
    """
    Abstract base class for federated aggregation.

    Subclasses implement `aggregate()` which takes a list of
    ClientUpdate objects and the current global state_dict, and
    returns the updated global state_dict.
    """
    def __init__(self, cfg):
        self.cfg       = cfg
        self.fc        = cfg.FED
        self.patterns  = self.fc.COMM_PARAM_PATTERNS if self.fc.LORA_ONLY_COMM else []

    def aggregate(
        self,
        global_sd: Dict[str, torch.Tensor],
        updates:   List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _compute_weights(self, updates: List[ClientUpdate]) -> List[float]:
        """Compute per-client aggregation weights."""
        strategy = self.fc.CLIENT_WEIGHTING

        if strategy == "uniform":
            w = [1.0 / len(updates)] * len(updates)

        elif strategy == "n_samples":
            total = sum(u.n_samples for u in updates) + 1e-9
            w = [u.n_samples / total for u in updates]

        elif strategy == "drift_aware":
            # Down-weight clients with high drift (far from global model)
            # drift_score ∈ [0, 1]: 0=identical, 1=orthogonal
            # weight ∝ n_samples × (1 - drift_score)
            total_n = sum(u.n_samples for u in updates) + 1e-9
            raw = [
                (u.n_samples / total_n) * (1.0 - u.drift_score)
                for u in updates
            ]
            total_w = sum(raw) + 1e-9
            w = [r / total_w for r in raw]

        else:
            raise ValueError(f"Unknown CLIENT_WEIGHTING: {strategy}")

        return w

    def _compute_drift(
        self,
        global_sd: Dict[str, torch.Tensor],
        update:    ClientUpdate,
    ) -> float:
        """
        Cosine distance between global parameters and client update.
        Only computed over the communicable parameter subset.
        """
        g_vec = _to_vector(_comm_params(global_sd, self.patterns))
        c_vec = _to_vector(_comm_params(update.state_dict, self.patterns))

        if g_vec.norm() < 1e-9 or c_vec.norm() < 1e-9:
            return 0.0

        cos_sim = torch.nn.functional.cosine_similarity(
            g_vec.unsqueeze(0), c_vec.unsqueeze(0)
        ).item()
        return max(1.0 - cos_sim, 0.0)


def _to_vector(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict to a single 1D tensor."""
    parts = [v.float().flatten() for v in sd.values()
             if v.dtype.is_floating_point]
    return torch.cat(parts) if parts else torch.tensor([0.0])


# ════════════════════════════════════════════════════════════════════════
# 3 — FEDAVG
# ════════════════════════════════════════════════════════════════════════

class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging (McMahan et al., 2017).
    w_global ← Σ_i (n_i / N) · w_i
    """

    def aggregate(
        self,
        global_sd: Dict[str, torch.Tensor],
        updates:   List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_sd

        weights = self._compute_weights(updates)
        new_comm: Dict[str, torch.Tensor] = {}

        for i, update in enumerate(updates):
            client_comm = _comm_params(update.state_dict, self.patterns)
            for k, v in client_comm.items():
                if k not in new_comm:
                    new_comm[k] = weights[i] * v.float().clone()
                else:
                    new_comm[k].add_(weights[i] * v.float())

        return _merge_state(global_sd, new_comm)


# ════════════════════════════════════════════════════════════════════════
# 4 — FEDPROX
# ════════════════════════════════════════════════════════════════════════

class FedProxAggregator(FedAvgAggregator):
    """
    FedProx (Li et al., 2020).

    Server-side aggregation is identical to FedAvg.
    The proximal term μ/2 ‖w - w_global‖² is enforced in the
    CLIENT trainer (see ClientTrainer._fedprox_loss()).

    This aggregator stores the current global model state so the
    client trainer can access it for the proximal penalty.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mu         = cfg.FED.FEDPROX_MU
        self._global_sd: Optional[Dict[str, torch.Tensor]] = None

    def set_global(self, global_sd: Dict[str, torch.Tensor]):
        """Called before distributing the global model to clients."""
        self._global_sd = {k: v.clone() for k, v in global_sd.items()}

    def proximal_penalty(
        self,
        model:     nn.Module,
        global_sd: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes μ/2 ‖w_local - w_global‖²_F over communicable params.
        Called inside the client training loop.
        """
        ref = global_sd or self._global_sd
        if ref is None:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if any(p in name for p in self.patterns) and name in ref:
                w_g = ref[name].to(param.device).float()
                penalty = penalty + (param.float() - w_g).pow(2).sum()

        return (self.mu / 2.0) * penalty


# ════════════════════════════════════════════════════════════════════════
# 5 — FEDNOVA
# ════════════════════════════════════════════════════════════════════════

class FedNovaAggregator(BaseAggregator):
    """
    FedNova — Normalised Federated Averaging (Wang et al., 2021).

    Corrects client-drift from heterogeneous local steps:
      Δ_i    = w_i^local - w_global       (client i update)
      τ_eff  = local_steps + ρ/(1-ρ)     (effective steps with momentum)
      ĝ_i    = Δ_i / τ_eff_i             (normalised gradient)
      Δ_agg  = Σ_i p_i · ĝ_i            (aggregated normalised gradient)
      τ_avg  = Σ_i p_i · τ_eff_i
      w_new  = w_global - ηₛ · τ_avg · Δ_agg
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rho    = cfg.FED.FEDNOVA_RHO
        self.eta_s  = cfg.FED.LOCAL_LR    # server LR = local LR for simplicity

    def _tau_eff(self, local_steps: int) -> float:
        """Effective number of steps accounting for momentum."""
        if self.rho == 0.0:
            return float(local_steps)
        return (self.rho / (1.0 - self.rho)) * (1.0 - self.rho ** local_steps) \
               / (1.0 - self.rho)

    def aggregate(
        self,
        global_sd: Dict[str, torch.Tensor],
        updates:   List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_sd

        weights  = self._compute_weights(updates)
        tau_effs = [self._tau_eff(u.local_steps) for u in updates]
        tau_avg  = sum(p * t for p, t in zip(weights, tau_effs))

        # Aggregate normalised gradients
        agg_grad: Dict[str, torch.Tensor] = {}
        global_comm = _comm_params(global_sd, self.patterns)

        for i, update in enumerate(updates):
            client_comm = _comm_params(update.state_dict, self.patterns)
            tau_i       = tau_effs[i]

            for k in global_comm:
                if k not in client_comm:
                    continue
                delta_i  = (client_comm[k].float() - global_comm[k].float())
                norm_grad = delta_i / (tau_i + 1e-9)

                if k not in agg_grad:
                    agg_grad[k] = weights[i] * norm_grad
                else:
                    agg_grad[k] += weights[i] * norm_grad

        # Apply aggregated gradient to global model
        new_comm: Dict[str, torch.Tensor] = {}
        for k, g_comm in global_comm.items():
            if k in agg_grad:
                new_comm[k] = g_comm.float() - self.eta_s * tau_avg * agg_grad[k]
            else:
                new_comm[k] = g_comm.float().clone()

        return _merge_state(global_sd, new_comm)


# ════════════════════════════════════════════════════════════════════════
# 6 — FEDADAM (server-side Adam on pseudo-gradient)
# ════════════════════════════════════════════════════════════════════════

class FedAdamAggregator(BaseAggregator):
    """
    FedAdam — Adaptive Federated Optimisation (Reddi et al., 2021).

    Computes a pseudo-gradient Δ = w_global - Σ_i p_i w_i^local
    and applies server-side Adam to update the global model.

    m_t  ← β1·m_{t-1} + (1-β1)·Δ
    v_t  ← β2·v_{t-1} + (1-β2)·Δ²
    w    ← w - ηₛ · m_t / (√v_t + τ)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        fc       = cfg.FED
        self.lr  = fc.FEDADAM_LR
        self.b1  = fc.FEDADAM_BETA1
        self.b2  = fc.FEDADAM_BETA2
        self.eps = fc.FEDADAM_EPS
        self.tau = fc.FEDADAM_TAU
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}
        self._t = 0

    def aggregate(
        self,
        global_sd: Dict[str, torch.Tensor],
        updates:   List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            return global_sd

        self._t += 1
        weights     = self._compute_weights(updates)
        global_comm = _comm_params(global_sd, self.patterns)

        # ── Compute FedAvg pseudo-gradient ───────────────────────────
        fedavg_comm: Dict[str, torch.Tensor] = {}
        for i, update in enumerate(updates):
            client_comm = _comm_params(update.state_dict, self.patterns)
            for k, v in client_comm.items():
                if k not in fedavg_comm:
                    fedavg_comm[k] = weights[i] * v.float().clone()
                else:
                    fedavg_comm[k] += weights[i] * v.float()

        # Pseudo-gradient: global - fedavg (direction of improvement)
        pseudo_grad: Dict[str, torch.Tensor] = {}
        for k in global_comm:
            if k in fedavg_comm:
                pseudo_grad[k] = global_comm[k].float() - fedavg_comm[k]

        # ── Server Adam step ─────────────────────────────────────────
        new_comm: Dict[str, torch.Tensor] = {}
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t

        for k, g_val in global_comm.items():
            if k not in pseudo_grad:
                new_comm[k] = g_val.float().clone()
                continue

            g = pseudo_grad[k]
            if k not in self._m:
                self._m[k] = torch.zeros_like(g)
                self._v[k] = torch.zeros_like(g)

            self._m[k] = self.b1 * self._m[k] + (1 - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1 - self.b2) * (g ** 2)

            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2

            new_comm[k] = g_val.float() - self.lr * m_hat / (v_hat.sqrt() + self.tau)

        return _merge_state(global_sd, new_comm)


# ════════════════════════════════════════════════════════════════════════
# 7 — AGGREGATOR FACTORY
# ════════════════════════════════════════════════════════════════════════

def build_aggregator(cfg) -> BaseAggregator:
    """Returns the configured aggregator instance."""
    name = cfg.FED.AGGREGATION.lower()
    if name == "fedavg":
        return FedAvgAggregator(cfg)
    if name == "fedprox":
        return FedProxAggregator(cfg)
    if name == "fednova":
        return FedNovaAggregator(cfg)
    if name == "fedadam":
        return FedAdamAggregator(cfg)
    raise ValueError(f"Unknown aggregation: {name!r}. "
                     "Choose: fedavg | fedprox | fednova | fedadam")
