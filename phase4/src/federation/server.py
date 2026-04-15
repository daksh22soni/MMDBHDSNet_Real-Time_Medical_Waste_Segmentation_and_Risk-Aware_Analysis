"""
========================================================================
DBHDSNet Phase 4 — Federated Server

The server orchestrates the full federated training loop:

  for round in 1..NUM_ROUNDS:
    1. SELECT active clients (random subset or all)
    2. BROADCAST global model (comm subset only) to selected clients
    3. LOCAL TRAINING — each client trains for LOCAL_EPOCHS
    4. AGGREGATE updates → new global model via chosen aggregator
    5. UPDATE per-client drift scores (EMA smoothed)
    6. EVALUATE global model every EVAL_EVERY rounds
    7. CHECKPOINT if best performance
    8. LOG per-round metrics (convergence, DP budgets, per-client stats)

Novel PhD contributions tracked here:
  • Per-client DP budget consumption across rounds
  • Client drift score evolution (for drift-aware re-weighting)
  • Communication payload per round (bytes for LoRA-only vs full)
  • Convergence gap: global mAP vs best per-client mAP (measures federation benefit)
========================================================================
"""

import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import (
    AverageMeter, MetricTracker, ModelEMA, EarlyStopping,
    CheckpointManager, TrainingHistory, gpu_info,
    format_time, print_banner, make_run_name, set_seed,
)
from .aggregators import BaseAggregator, ClientUpdate, build_aggregator
from .client import ClientTrainer, ClientDataShard
from ..evaluation.fed_metrics import FedEvaluator
from ..privacy.dp_audit import DPAuditReport


# ════════════════════════════════════════════════════════════════════════
# 1 — COMMUNICATION PAYLOAD ESTIMATOR
# ════════════════════════════════════════════════════════════════════════

def estimate_payload_bytes(
    model:    nn.Module,
    patterns: List[str],
) -> Tuple[int, int]:
    """
    Returns (comm_bytes, full_bytes).
    comm_bytes = bytes in LoRA-only communication subset.
    full_bytes = bytes if entire model were communicated.
    """
    comm_bytes, full_bytes = 0, 0
    for name, p in model.named_parameters():
        b = p.numel() * p.element_size()
        full_bytes += b
        if not patterns or any(pat in name for pat in patterns):
            comm_bytes += b
    return comm_bytes, full_bytes


# ════════════════════════════════════════════════════════════════════════
# 2 — CLIENT SELECTION STRATEGIES
# ════════════════════════════════════════════════════════════════════════

def select_clients(
    all_clients:       List[ClientTrainer],
    n_select:          int,
    strategy:          str = "random",
    drift_scores:      Optional[Dict[str, float]] = None,
    rng:               Optional[random.Random] = None,
) -> List[ClientTrainer]:
    """
    Client selection strategies:
      'random'     — uniform random subset (standard FL)
      'all'        — all clients every round
      'low_drift'  — prefer clients with low drift (more aligned with global)
    """
    rng   = rng or random
    n_sel = min(n_select, len(all_clients))

    if strategy == "all":
        return all_clients

    if strategy == "random":
        return rng.sample(all_clients, n_sel)

    if strategy == "low_drift":
        if drift_scores:
            ranked = sorted(all_clients,
                            key=lambda c: drift_scores.get(c.client_id, 0.5))
            return ranked[:n_sel]
        return rng.sample(all_clients, n_sel)

    return rng.sample(all_clients, n_sel)


# ════════════════════════════════════════════════════════════════════════
# 3 — FEDERATED SERVER
# ════════════════════════════════════════════════════════════════════════

class FederatedServer:
    """
    Orchestrates the full federated learning experiment.

    Usage:
        server = FederatedServer(global_model, clients, cfg, device)
        server.fit(train_dataset, val_loader, test_loader)
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients:      List[ClientTrainer],
        cfg,
        device:       torch.device,
    ):
        self.model    = global_model.to(device)
        self.clients  = clients
        self.cfg      = cfg
        self.fc       = cfg.FED
        self.device   = device
        self.run_name = make_run_name("phase4_fed")

        ckpt_dir = Path(cfg.DATA.CHECKPOINT_DIR) / "global"
        log_dir  = Path(cfg.DATA.LOG_DIR)
        out_dir  = Path(cfg.DATA.OUTPUT_DIR)

        self.aggregator  = build_aggregator(cfg)
        self.ema         = ModelEMA(global_model, decay=0.9999)
        self.early_stop  = EarlyStopping(
            patience  = cfg.FED.PATIENCE,
            min_delta = cfg.FED.MIN_DELTA,
        )
        self.ckpt_mgr    = CheckpointManager(ckpt_dir, keep_last_n=cfg.FED.KEEP_LAST_N)
        self.history     = TrainingHistory(log_dir / f"{self.run_name}_history.json")
        self.evaluator   = FedEvaluator(cfg, device)
        self.dp_audit    = DPAuditReport(cfg, out_dir / "privacy_audit")

        # Per-client EMA drift scores (initialised to 0)
        self.drift_scores: Dict[str, float] = {
            c.client_id: 0.0 for c in clients
        }
        # Per-client accumulated ε
        self.dp_budget:    Dict[str, float] = {
            c.client_id: 0.0 for c in clients
        }

        self.best_map   = 0.0
        self.start_round = 1

        # Estimate communication savings
        comm_b, full_b = estimate_payload_bytes(
            global_model, self.fc.COMM_PARAM_PATTERNS
        )
        self._comm_ratio = comm_b / max(full_b, 1)
        print(f"\n  LoRA-only communication: {comm_b/1e6:.2f} MB / round  "
              f"({100*self._comm_ratio:.1f}% of full model)\n")

    # ------------------------------------------------------------------

    def resume(self, ckpt_path: Optional[Path] = None) -> bool:
        state = self.ckpt_mgr.load(ckpt_path)
        if state is None:
            return False
        self.model.load_state_dict(state["model_state"])
        if "ema_state"     in state: self.ema.load_state_dict(state["ema_state"])
        if "early_stop"    in state: self.early_stop.load_state_dict(state["early_stop"])
        if "drift_scores"  in state: self.drift_scores  = state["drift_scores"]
        if "dp_budget"     in state: self.dp_budget      = state["dp_budget"]
        self.start_round = state.get("round", 0) + 1
        self.best_map    = state.get("best_map", 0.0)
        tqdm.write(f"[Server] Resumed from round {state['round']:04d} | "
                   f"best mAP={self.best_map:.4f}")
        return True

    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset,
        val_loader,
        test_loader,
        criterion,
    ):
        print_banner(f"Phase 4 — Federated Training  |  Run: {self.run_name}")
        print(gpu_info())
        print(f"  Clients        : {len(self.clients)}")
        print(f"  Rounds         : {self.fc.NUM_ROUNDS}")
        print(f"  Aggregation    : {self.fc.AGGREGATION.upper()}")
        print(f"  LoRA-only comm : {self.fc.LORA_ONLY_COMM}  "
              f"({100*self._comm_ratio:.1f}% of model per round)")
        print(f"  DP enabled     : {self.cfg.DP.ENABLE_DP}  "
              f"(target ε={self.cfg.DP.TARGET_EPSILON}, "
              f"δ={self.cfg.DP.TARGET_DELTA})")
        print()

        # Build per-client data shards
        self._build_shards(train_dataset)

        t0 = time.time()

        for fed_round in range(self.start_round, self.fc.NUM_ROUNDS + 1):
            round_t = time.time()

            # ── 1. Select clients ─────────────────────────────────────
            selected = select_clients(
                self.clients,
                n_select     = self.fc.CLIENTS_PER_ROUND,
                strategy     = "random",
                drift_scores = self.drift_scores,
            )

            if len(selected) < self.fc.MIN_CLIENTS:
                tqdm.write(f"  Round {fed_round:04d}: only {len(selected)} clients "
                           f"(< min={self.fc.MIN_CLIENTS}). Skipping.")
                continue

            # ── 2. Broadcast global model ─────────────────────────────
            global_sd = copy.deepcopy(self.model.state_dict())

            # ── 3. Local training (sequential simulation) ─────────────
            updates: List[ClientUpdate] = []
            for client in selected:
                upd = client.train_one_round(
                    global_model = self.model,
                    global_sd    = global_sd,
                    criterion    = criterion,
                    fed_round    = fed_round,
                )
                updates.append(upd)
                self.dp_budget[upd.client_id] += upd.dp_epsilon

            # ── 4. Aggregate ──────────────────────────────────────────
            new_sd = self.aggregator.aggregate(global_sd, updates)
            self.model.load_state_dict(new_sd)
            self.ema.update(self.model)

            # ── 5. Update drift scores (EMA) ──────────────────────────
            alpha = self.fc.DRIFT_EMA_ALPHA
            for upd in updates:
                old  = self.drift_scores[upd.client_id]
                self.drift_scores[upd.client_id] = (
                    (1 - alpha) * old + alpha * upd.drift_score
                )

            # ── 6. Evaluate ───────────────────────────────────────────
            val_metrics = {}
            is_best     = False

            if fed_round % self.fc.EVAL_EVERY == 0 or fed_round == self.fc.NUM_ROUNDS:
                val_metrics = self.evaluator.evaluate(
                    self.ema.ema, val_loader, desc=f"Val Rnd {fed_round:04d}"
                )
                val_map = val_metrics.get("mAP_50", 0.0)
                is_best = val_map > self.best_map
                if is_best:
                    self.best_map = val_map

            # ── 7. Checkpoint ─────────────────────────────────────────
            periodic = (fed_round % self.fc.SAVE_EVERY == 0)
            self.ckpt_mgr.save({
                "round":        fed_round,
                "model_state":  self.model.state_dict(),
                "ema_state":    self.ema.state_dict(),
                "early_stop":   self.early_stop.state_dict(),
                "drift_scores": self.drift_scores,
                "dp_budget":    self.dp_budget,
                "best_map":     self.best_map,
                "run_name":     self.run_name,
            }, is_best, fed_round, periodic)

            # ── 8. Log ────────────────────────────────────────────────
            round_time = format_time(time.time() - round_t)
            flag       = "★ " if is_best else "  "
            map_str    = f"mAP={val_metrics.get('mAP_50', 0):.4f}" \
                         if val_metrics else "(no eval)"

            tqdm.write(
                f"\n  {flag}Round {fed_round:04d}/{self.fc.NUM_ROUNDS:04d}  "
                f"│ {map_str}  "
                f"│ clients={len(selected)}  "
                f"│ avg_drift={sum(self.drift_scores.values())/len(self.drift_scores):.4f}  "
                f"│ ⏱ {round_time}"
            )
            self._log_round(fed_round, updates, val_metrics, round_time)

            # ── 9. Early stopping ─────────────────────────────────────
            if val_metrics and self.early_stop(val_metrics.get("mAP_50", 0)):
                tqdm.write("[Server] Early stopping triggered.")
                break

        # ── Final evaluation on test set ─────────────────────────────
        print_banner("Phase 4 — Final Test Evaluation")
        best_state = self.ckpt_mgr.load_best()
        if best_state:
            self.model.load_state_dict(best_state["model_state"])

        test_metrics = self.evaluator.evaluate(
            self.ema.ema, test_loader, desc="Final Test"
        )
        self._print_final_metrics(test_metrics)

        # ── Generate DP audit report ──────────────────────────────────
        self.dp_audit.generate(self.dp_budget, self.clients, test_metrics)

        print(f"\n  Total time: {format_time(time.time() - t0)}")
        print_banner("Phase 4 Complete!")

        return test_metrics

    # ------------------------------------------------------------------

    def _build_shards(self, train_dataset):
        """Assign data shards to each client."""
        for client in self.clients:
            shard = ClientDataShard(train_dataset, prefix=client.prefix)
            client.set_data_shard(shard)
            tqdm.write(f"  [{client.client_id}] shard: {len(shard)} images")

    # ------------------------------------------------------------------

    def _log_round(
        self,
        fed_round:   int,
        updates:     List[ClientUpdate],
        val_metrics: dict,
        round_time:  str,
    ):
        record = {
            "round":       fed_round,
            "n_clients":   len(updates),
            "round_time":  round_time,
            "drift_scores": copy.deepcopy(self.drift_scores),
            "dp_budgets":  copy.deepcopy(self.dp_budget),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        for upd in updates:
            record[f"loss_{upd.client_id}"] = upd.metrics.get("loss_total", 0)
        self.history.append(record)

    # ------------------------------------------------------------------

    def _print_final_metrics(self, metrics: dict):
        print("\n  Global model — Final test metrics:")
        for k, v in metrics.items():
            print(f"    {k:30s}: {v:.4f}" if isinstance(v, float) else
                  f"    {k:30s}: {v}")
        print(f"\n  Best val mAP@50: {self.best_map:.4f}")
        print(f"\n  Per-client DP budget consumed (ε):")
        for cid, eps in self.dp_budget.items():
            status = "OK" if eps <= self.cfg.DP.TARGET_EPSILON else "EXCEEDED"
            print(f"    {cid:20s}: ε={eps:.3f}  [{status}]")
