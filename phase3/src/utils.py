"""
========================================================================
DBHDSNet — Utilities
Covers: dual terminal+file logging · checkpoint manager ·
        EarlyStopping · ModelEMA · AverageMeter · GPU info
========================================================================
"""

import os
import sys
import json
import math
import time
import copy
import shutil
import logging
import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import psutil


# ════════════════════════════════════════════════════════════════════════
# 1 — DUAL OUTPUT (terminal + file simultaneously)
# ════════════════════════════════════════════════════════════════════════

class DualOutput:
    """
    Redirects sys.stdout so every print() goes to both the terminal
    and a persistent log file.  tqdm progress bars still render
    correctly on the terminal because tqdm writes directly to the
    underlying file descriptor when it detects a DualOutput wrapper.
    """
    def __init__(self, log_path: Path):
        self.terminal = sys.__stdout__
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        # Strip ANSI escape codes before writing to file
        import re
        clean = re.sub(r"\x1b\[[0-9;]*m", "", message)
        self.log.write(clean)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        self.log.close()


def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """
    Sets up Python logging + DualOutput for stdout.
    Returns the root logger for structured log messages.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{run_name}.log"

    # Redirect stdout (captures print() and tqdm summaries)
    sys.stdout = DualOutput(log_file)

    # Python logging for structured messages
    logger = logging.getLogger("DBHDSNet")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler (full log)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        # Stream handler (terminal only)
        sh = logging.StreamHandler(sys.__stdout__)
        sh.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


# ════════════════════════════════════════════════════════════════════════
# 2 — CHECKPOINT MANAGER
# ════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves / loads training checkpoints.
    Keeps the best model (by val_mAP) and the last N periodic checkpoints.
    """
    def __init__(self, ckpt_dir: Path, keep_last_n: int = 3):
        self.ckpt_dir   = Path(ckpt_dir)
        self.keep_last_n = keep_last_n
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._periodic_queue: list[Path] = []

    # ------------------------------------------------------------------
    def save(
        self,
        state: Dict[str, Any],
        is_best: bool,
        epoch: int,
        periodic: bool = False,
    ) -> Path:
        """
        state  – dict with keys: epoch, model_state, optimizer_state,
                                  scheduler_state, ema_state, metrics
        """
        # Always save "last.pt"
        last_path = self.ckpt_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            shutil.copy2(last_path, best_path)

        if periodic:
            epoch_path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
            shutil.copy2(last_path, epoch_path)
            self._periodic_queue.append(epoch_path)
            # Prune old periodic checkpoints
            while len(self._periodic_queue) > self.keep_last_n:
                old = self._periodic_queue.pop(0)
                if old.exists():
                    old.unlink()

        return last_path

    # ------------------------------------------------------------------
    def load(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load checkpoint; returns None if no checkpoint exists."""
        if path is None:
            path = self.ckpt_dir / "last.pt"
        path = Path(path)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    def load_best(self) -> Optional[Dict[str, Any]]:
        return self.load(self.ckpt_dir / "best.pt")


# ════════════════════════════════════════════════════════════════════════
# 3 — EARLY STOPPING
# ════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training if val_mAP does not improve by min_delta for
    `patience` consecutive epochs.
    """
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score: float = -float("inf")
        self.counter    = 0
        self.stop       = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def state_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "counter":    self.counter,
            "stop":       self.stop,
        }

    def load_state_dict(self, d: dict):
        self.best_score = d["best_score"]
        self.counter    = d["counter"]
        self.stop       = d["stop"]


# ════════════════════════════════════════════════════════════════════════
# 4 — EXPONENTIAL MOVING AVERAGE OF MODEL WEIGHTS
# ════════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Maintains an EMA copy of the model weights.
    EMA weights are used for validation / inference.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        self.ema     = copy.deepcopy(model).eval()
        self.decay   = decay
        self.tau     = tau
        self.updates = 0
        # Disable gradients on EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_((1 - d) * msd[k].detach())

    def state_dict(self) -> dict:
        return {
            "ema_state":  self.ema.state_dict(),
            "updates":    self.updates,
            "decay":      self.decay,
        }

    def load_state_dict(self, d: dict):
        self.ema.load_state_dict(d["ema_state"])
        self.updates = d["updates"]
        self.decay   = d["decay"]


# ════════════════════════════════════════════════════════════════════════
# 5 — AVERAGE METER (running statistics)
# ════════════════════════════════════════════════════════════════════════

class AverageMeter:
    """Tracks running average for scalar metrics."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Tracks multiple AverageMeters by name."""
    def __init__(self):
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if k not in self._meters:
                self._meters[k] = AverageMeter(k)
            self._meters[k].update(v, n)

    def reset(self):
        for m in self._meters.values():
            m.reset()

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def summary(self) -> str:
        parts = [f"{k}={v:.4f}" for k, v in self.averages().items()]
        return "  ".join(parts)


# ════════════════════════════════════════════════════════════════════════
# 6 — GPU UTILITIES
# ════════════════════════════════════════════════════════════════════════

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Returns the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_info() -> str:
    """Returns a formatted string of GPU memory usage."""
    if not torch.cuda.is_available():
        return "GPU: Not available"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total  = torch.cuda.get_device_properties(idx).total_memory / 1e9
    alloc  = torch.cuda.memory_allocated(idx) / 1e9
    reserv = torch.cuda.memory_reserved(idx) / 1e9
    return (
        f"GPU[{idx}] {name} | "
        f"Allocated {alloc:.2f}GB / Reserved {reserv:.2f}GB / Total {total:.2f}GB"
    )


def count_parameters(model: nn.Module) -> dict:
    """Returns total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def set_seed(seed: int = 42):
    """Full reproducibility seed."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════
# 7 — TRAINING HISTORY PERSISTENCE
# ════════════════════════════════════════════════════════════════════════

class TrainingHistory:
    """Saves per-epoch metrics to JSON for later analysis / plotting."""
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.history: list[dict] = []
        if self.save_path.exists():
            with open(self.save_path, "r") as f:
                self.history = json.load(f)

    def append(self, epoch_record: dict):
        self.history.append(epoch_record)
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def __len__(self):
        return len(self.history)


# ════════════════════════════════════════════════════════════════════════
# 8 — MISC HELPERS
# ════════════════════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))


def make_run_name(prefix: str = "dbhdsnet") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def print_banner(text: str, width: int = 70, char: str = "═"):
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")
