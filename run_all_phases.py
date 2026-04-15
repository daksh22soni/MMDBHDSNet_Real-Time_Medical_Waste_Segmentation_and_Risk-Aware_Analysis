"""
========================================================================
DBHDSNet — Master Pipeline Runner
Executes all 5 phases sequentially with a SINGLE continuous progress
bar showing real-time overall project completion and ETA.

Folder layout expected (one level up from each phase folder):

    dbhdsnet_project/               ← place this file HERE
    ├── run_all_phases.py           ← this file
    ├── master_dataset/             ← your Phase 1 dataset output
    │   ├── train/
    │   ├── valid/
    │   ├── test/
    │   └── master_data.yaml
    ├── phase2/
    │   ├── config.py
    │   └── train.py
    ├── phase3/
    │   ├── config_phase3.py
    │   ├── run_phase3a.py
    │   └── run_phase3b.py
    └── phase4/
        ├── config_phase4.py
        └── train_phase4.py

Usage:
    # Full pipeline (all phases)
    python run_all_phases.py

    # Start from a specific phase (skip earlier ones)
    python run_all_phases.py --start-phase 3a

    # Run only specific phases
    python run_all_phases.py --only 2 3a

    # Dry run (smoke test every phase — fast)
    python run_all_phases.py --dry-run

    # Resume each phase from its last checkpoint
    python run_all_phases.py --resume

    # Skip DP in Phase 4 (ablation)
    python run_all_phases.py --no-dp

How the progress bar works:
    A single tqdm bar lives at the BOTTOM of the terminal for the
    entire run. It NEVER reprints — it only updates in-place using
    tqdm's \r mechanism. All phase output is printed ABOVE it via
    tqdm.write(). The bar advances by the estimated % weight of each
    phase step as it completes.
========================================================================
"""

import sys
import os
import time
import signal
import argparse
import datetime
import subprocess
import threading
import re
from pathlib import Path
from typing import List, Optional

# ── Require tqdm ─────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required.  pip install tqdm")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to match your system
# ════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent      # dbhdsnet_project/

# Phase directories (relative to ROOT)
PHASE2_DIR = ROOT / "phase2"
PHASE3_DIR = ROOT / "phase3"
PHASE4_DIR = ROOT / "phase4"

# Python executable (same env as phases)
PYTHON = sys.executable

# ── Estimated duration for each step (minutes) ───────────────────────
# Used to compute the continuous ETA before actual times are known.
# Adjust for your GPU — these are for a single RTX 3090 / A100.
STEP_ESTIMATES: dict[str, float] = {
    "p1_verify":     2,      # Dataset verification (instant)
    "p2_inspect":    1,      # inspect_dataset.py
    "p2_train":      600,    # Phase 2 training (150 epochs on 14k images)
    "p3a_finetune":  45,     # Phase 3a UQ fine-tuning (20 epochs)
    "p3a_calibrate": 10,     # Post-hoc calibration + MC-Dropout report
    "p3b_graphs":    30,     # Build scene graphs (inference on 14k images)
    "p3b_train":     120,    # Phase 3b GNN training (100 epochs)
    "p4_inspect":    1,      # inspect_clients.py
    "p4_train":      480,    # Phase 4 federated training (100 rounds)
}


# ════════════════════════════════════════════════════════════════════════
# PHASE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════

# Each step: (step_id, phase_label, description, cwd, cmd_args, estimate_key)
# cmd_args is a list — first element is the script path relative to cwd.

def build_steps(args) -> List[dict]:
    dry   = ["--dry-run"]    if args.dry_run else []
    res   = ["--resume"]     if args.resume  else []
    no_dp = ["--no-dp"]      if args.no_dp   else []

    steps = [
        # ── Phase 1 — Dataset verification ───────────────────────────
        {
            "id":    "p1_verify",
            "phase": "1",
            "label": "Phase 1 — Dataset Verification",
            "desc":  "Verifying master_dataset structure and class names",
            "cwd":   PHASE2_DIR,
            "cmd":   [PYTHON, "scripts/inspect_dataset.py"],
            "est":   "p1_verify",
            "skip_on_dry": False,
        },
        # ── Phase 2 — DBHDSNet Training ───────────────────────────────
        {
            "id":    "p2_inspect",
            "phase": "2",
            "label": "Phase 2 — Dataset Inspection",
            "desc":  "Verifying 38-class HAZARD_TIER_MAP and dataset stats",
            "cwd":   PHASE2_DIR,
            "cmd":   [PYTHON, "scripts/inspect_dataset.py"],
            "est":   "p2_inspect",
            "skip_on_dry": False,
        },
        {
            "id":    "p2_train",
            "phase": "2",
            "label": "Phase 2 — DBHDSNet Training",
            "desc":  "Training dual-branch segmentation model (150 epochs)",
            "cwd":   PHASE2_DIR,
            "cmd":   [PYTHON, "train.py"] + dry + res,
            "est":   "p2_train",
            "skip_on_dry": False,
        },
        # ── Phase 3a — Uncertainty Quantification ─────────────────────
        {
            "id":    "p3a_finetune",
            "phase": "3a",
            "label": "Phase 3a — UQ Fine-Tuning",
            "desc":  "MC-Dropout + uncertainty-aware loss fine-tuning",
            "cwd":   PHASE3_DIR,
            "cmd":   [PYTHON, "run_phase3a_fixed.py"] + dry + res,
            "est":   "p3a_finetune",
            "skip_on_dry": False,
        },
        {
            "id":    "p3a_calibrate",
            "phase": "3a",
            "label": "Phase 3a — Calibration + UQ Report",
            "desc":  "Temperature scaling, ECE evaluation, flagging dashboard",
            "cwd":   PHASE3_DIR,
            "cmd":   [PYTHON, "run_phase3a_fixed.py", "--calibrate-only"],
            "est":   "p3a_calibrate",
            "skip_on_dry": True,    # dry-run skips this step
        },
        # ── Phase 3b — ContamRisk-GNN ─────────────────────────────────
        {
            "id":    "p3b_graphs",
            "phase": "3b",
            "label": "Phase 3b — Scene Graph Construction",
            "desc":  "Building waste-item scene graphs from detections",
            "cwd":   PHASE3_DIR,
            "cmd":   [PYTHON, "run_phase3b_fixed.py", "--rebuild-graphs"] + dry,
            "est":   "p3b_graphs",
            "skip_on_dry": False,
        },
        {
            "id":    "p3b_train",
            "phase": "3b",
            "label": "Phase 3b — ContamRisk-GNN Training",
            "desc":  "Training GATv2 contamination risk network (100 epochs)",
            "cwd":   PHASE3_DIR,
            "cmd":   [PYTHON, "run_phase3b_fixed.py"] + dry + res,
            "est":   "p3b_train",
            "skip_on_dry": False,
        },
        # ── Phase 4 — Federated Learning ──────────────────────────────
        {
            "id":    "p4_inspect",
            "phase": "4",
            "label": "Phase 4 — Client Shard Inspection",
            "desc":  "Verifying 6 hospital/lab client data shards",
            "cwd":   PHASE4_DIR,
            "cmd":   [PYTHON, "scripts/inspect_clients.py"],
            "est":   "p4_inspect",
            "skip_on_dry": False,
        },
        {
            "id":    "p4_train",
            "phase": "4",
            "label": "Phase 4 — Federated Training",
            "desc":  "FedProx + DP-SGD across 6 clients (100 rounds)",
            "cwd":   PHASE4_DIR,
            "cmd":   [PYTHON, "train_phase4_fixed.py"] + dry + res + no_dp,
            "est":   "p4_train",
            "skip_on_dry": False,
        },
    ]
    return steps


# ════════════════════════════════════════════════════════════════════════
# OVERALL PROGRESS BAR (continuous, never reprints)
# ════════════════════════════════════════════════════════════════════════

class OverallProgressBar:
    """
    A SINGLE tqdm bar that sits at the bottom of the terminal for
    the entire run — never reprints, only updates in-place.

    Architecture:
      • The bar itself tracks 'total_weight' units (sum of all step
        estimates in minutes). Each completed step advances the bar
        by that step's estimated weight.
      • A background ticker thread updates the bar every second so
        the ETA countdown is live — even during long phase runs.
      • All phase output goes through self.write() which calls
        tqdm.write() so it appears ABOVE the bar cleanly.

    Position=0 keeps the bar anchored at the bottom. The
    dynamic_miniters=0 + mininterval=0 ensure the tick thread
    updates visually every second without tqdm rate-limiting it.
    """

    def __init__(self, steps: List[dict], total_steps: int):
        # Total weight = sum of estimated minutes across all steps
        self.total_weight = sum(
            STEP_ESTIMATES.get(s["est"], 5) for s in steps
        )
        self.total_steps  = total_steps
        self.completed_weight = 0.0
        self.current_step_desc = "Initialising…"
        self.step_idx     = 0
        self.start_time   = time.time()
        self._lock        = threading.Lock()
        self._done        = False
        self._failed      = False

        # Estimated total minutes → seconds
        total_est_secs = self.total_weight * 60

        self._bar = tqdm(
            total        = int(total_est_secs),
            desc         = "Overall",
            unit         = "s",
            dynamic_ncols= True,
            position     = 0,
            leave        = True,
            bar_format   = (
                "{desc}: {percentage:5.1f}%"
                " |{bar}|"
                " {elapsed} elapsed"
                " · ETA {remaining}"
                " · {postfix}"
            ),
            mininterval  = 0.1,
            miniters     = 0,
        )
        self._bar.set_postfix_str(self.current_step_desc, refresh=False)

        # Background ticker: advances bar by real elapsed seconds each tick
        self._last_tick   = time.time()
        self._ticker      = threading.Thread(
            target = self._tick_loop, daemon=True
        )
        self._ticker.start()

    # ------------------------------------------------------------------

    def _tick_loop(self):
        """Advances the bar by 1 real second every second so ETA is live."""
        while not self._done:
            time.sleep(1.0)
            now  = time.time()
            with self._lock:
                if self._done:
                    break
                delta = now - self._last_tick
                self._last_tick = now
                # Advance by actual elapsed seconds (never exceed total)
                advance = min(delta, self._bar.total - self._bar.n)
                if advance > 0:
                    self._bar.update(int(advance))
            self._bar.refresh()

    # ------------------------------------------------------------------

    def step_start(self, step: dict, step_number: int):
        """Called just before a step begins."""
        with self._lock:
            self.step_idx         = step_number
            self.current_step_desc = (
                f"[{step_number}/{self.total_steps}] "
                f"Phase {step['phase']} — {step['label'].split('—')[-1].strip()}"
            )
            self._bar.set_postfix_str(self.current_step_desc, refresh=True)

        self.write(
            f"\n{'─'*70}\n"
            f"  ► STEP {step_number}/{self.total_steps}  │  {step['label']}\n"
            f"    {step['desc']}\n"
            f"{'─'*70}"
        )

    # ------------------------------------------------------------------

    def step_done(self, step: dict, elapsed_sec: float, success: bool):
        """Called after a step finishes (success or failure)."""
        est_min  = STEP_ESTIMATES.get(step["est"], 5)
        est_sec  = est_min * 60

        with self._lock:
            # Jump bar forward to the correct cumulative position
            self.completed_weight += est_min
            target_n = int(
                (self.completed_weight / self.total_weight) * self._bar.total
            )
            advance = max(0, target_n - self._bar.n)
            if advance > 0:
                self._bar.update(advance)
            self._last_tick = time.time()

        status = "✓  DONE" if success else "✗  FAILED"
        self.write(
            f"  {status}  │  {step['label']}  "
            f"│  actual {_fmt_sec(elapsed_sec)}  "
            f"│  est {_fmt_sec(est_sec)}"
        )

    # ------------------------------------------------------------------

    def write(self, msg: str):
        """Print a message ABOVE the progress bar (never disrupts it)."""
        tqdm.write(msg)

    # ------------------------------------------------------------------

    def finish(self, success: bool):
        self._done = True
        elapsed    = time.time() - self.start_time
        with self._lock:
            # Fill bar to 100%
            remaining = self._bar.total - self._bar.n
            if remaining > 0:
                self._bar.update(remaining)
            status = "COMPLETE" if success else "FAILED"
            self._bar.set_postfix_str(
                f"Pipeline {status} — total {_fmt_sec(elapsed)}",
                refresh=True,
            )
        self._bar.close()


# ════════════════════════════════════════════════════════════════════════
# STEP RUNNER
# ════════════════════════════════════════════════════════════════════════

def run_step(
    step:     dict,
    bar:      OverallProgressBar,
    dry_run:  bool,
) -> bool:
    """
    Executes one pipeline step as a subprocess.

    Streams stdout/stderr from the subprocess ABOVE the progress bar
    via tqdm.write() so the bar stays anchored at the bottom.

    Returns True on success, False on failure.
    """
    if dry_run and step.get("skip_on_dry", False):
        bar.write(f"  [DRY-RUN] Skipping step: {step['label']}")
        bar.step_done(step, elapsed_sec=0.0, success=True)
        return True

    cwd = Path(step["cwd"])
    cmd = [str(c) for c in step["cmd"]]

    if not cwd.exists():
        bar.write(f"  ⚠  Directory not found: {cwd}")
        bar.write(f"     Skipping: {step['label']}")
        bar.step_done(step, elapsed_sec=0.0, success=False)
        return False

    t_start = time.time()

    # ANSI stripper for clean log output
    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd      = str(cwd),
            stdout   = subprocess.PIPE,
            stderr   = subprocess.STDOUT,
            text     = True,
            bufsize  = 1,
            encoding = "utf-8",
            errors   = "replace",
            env      = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )

        # Stream output line-by-line above the progress bar
        for line in proc.stdout:
            clean = _ansi_re.sub("", line).rstrip()
            if clean:
                bar.write(f"  │  {clean}")

        proc.wait()
        elapsed = time.time() - t_start
        success = proc.returncode == 0
        bar.step_done(step, elapsed, success)
        return success

    except FileNotFoundError as e:
        bar.write(f"  ✗  Command not found: {e}")
        bar.write(f"     cmd = {' '.join(cmd)}")
        bar.step_done(step, time.time() - t_start, success=False)
        return False

    except KeyboardInterrupt:
        bar.write("\n  ⚠  Interrupted by user (Ctrl+C)")
        if proc and proc.poll() is None:
            proc.terminate()
        bar.step_done(step, time.time() - t_start, success=False)
        raise


# ════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DBHDSNet — Run all phases sequentially",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--start-phase", type=str, default="1",
        metavar="PHASE",
        help=(
            "Skip all phases before this one and start here.\n"
            "Values: 1  2  3a  3b  4\n"
            "Example: --start-phase 3b  (skips 1, 2, 3a)"
        ),
    )
    p.add_argument(
        "--only", nargs="+", type=str, default=None,
        metavar="PHASE",
        help=(
            "Run ONLY the specified phase(s).\n"
            "Example: --only 3a 3b"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to every phase (fast smoke test)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Pass --resume to every phase (continue from last checkpoint)",
    )
    p.add_argument(
        "--no-dp", action="store_true",
        help="Disable differential privacy in Phase 4 (ablation)",
    )
    p.add_argument(
        "--skip-on-fail", action="store_true",
        help=(
            "Continue to next phase even if current phase fails.\n"
            "Default: stop pipeline on first failure."
        ),
    )
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# PHASE FILTER
# ════════════════════════════════════════════════════════════════════════

PHASE_ORDER = ["1", "2", "3a", "3b", "4"]


def filter_steps(steps: List[dict], args) -> List[dict]:
    """Return the subset of steps to actually run."""
    if args.only:
        target = set(args.only)
        return [s for s in steps if s["phase"] in target]

    if args.start_phase != "1":
        start_idx = PHASE_ORDER.index(args.start_phase) \
                    if args.start_phase in PHASE_ORDER else 0
        allowed   = set(PHASE_ORDER[start_idx:])
        return [s for s in steps if s["phase"] in allowed]

    return steps


# ════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ════════════════════════════════════════════════════════════════════════

def preflight_checks(bar: OverallProgressBar):
    """Quick sanity checks before the pipeline starts."""
    ok = True

    for name, path in [
        ("phase2/", PHASE2_DIR),
        ("phase3/", PHASE3_DIR),
        ("phase4/", PHASE4_DIR),
    ]:
        if not path.exists():
            bar.write(f"  ⚠  Missing: {path}")
            bar.write(f"     Place {name} at {ROOT}/")
            ok = False
        else:
            bar.write(f"  ✓  Found: {path.name}/")

    # Check config files have been updated (naive: look for placeholder)
    configs = [
        PHASE2_DIR / "config.py",
        PHASE3_DIR / "config_phase3.py",
        PHASE4_DIR / "config_phase4.py",
    ]
    for cfg_path in configs:
        if cfg_path.exists():
            content = cfg_path.read_text(encoding="utf-8")
            if "/path/to/your/master_dataset" in content:
                bar.write(
                    f"  ⚠  DATASET_ROOT not set in {cfg_path.name}\n"
                    f"     Edit DATASET_ROOT in {cfg_path}"
                )
                ok = False
            else:
                bar.write(f"  ✓  {cfg_path.name} — DATASET_ROOT configured")

    return ok


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args  = parse_args()
    steps = build_steps(args)
    steps = filter_steps(steps, args)

    if not steps:
        print("No steps selected. Check --start-phase / --only arguments.")
        sys.exit(1)

    total_est_min = sum(STEP_ESTIMATES.get(s["est"], 5) for s in steps)

    # ── Print startup banner (before bar opens) ───────────────────────
    print()
    print("═" * 70)
    print("  DBHDSNet — Full Pipeline Runner")
    print("  Phases: 1 → 2 → 3a → 3b → 4")
    print(f"  Steps selected   : {len(steps)}")
    print(f"  Estimated total  : {_fmt_min(total_est_min)}")
    print(f"  Mode             : {'DRY-RUN  ' if args.dry_run else 'FULL TRAIN'}"
          + ("  (resume)"  if args.resume  else "")
          + ("  (no-dp)"   if args.no_dp   else ""))
    print("═" * 70)
    print()

    # ── Open the single continuous progress bar ───────────────────────
    bar = OverallProgressBar(steps, total_steps=len(steps))

    # ── Pre-flight checks ─────────────────────────────────────────────
    bar.write("\n  PRE-FLIGHT CHECKS")
    bar.write("  " + "─" * 40)
    ok = preflight_checks(bar)
    if not ok:
        bar.write(
            "\n  ✗  Pre-flight failed. Fix the issues above, then re-run.\n"
        )
        bar.finish(success=False)
        sys.exit(1)
    bar.write("  ✓  All pre-flight checks passed.\n")

    # ── Print execution plan ──────────────────────────────────────────
    bar.write("  EXECUTION PLAN")
    bar.write("  " + "─" * 40)
    for i, step in enumerate(steps, 1):
        est = STEP_ESTIMATES.get(step["est"], 5)
        bar.write(
            f"  {i:2d}. [{step['phase']:2s}] {step['label']:45s} "
            f"~{_fmt_min(est)}"
        )
    bar.write("")

    # ── Run steps ────────────────────────────────────────────────────
    pipeline_ok = True
    failed_steps = []

    for i, step in enumerate(steps, 1):
        bar.step_start(step, i)

        try:
            success = run_step(step, bar, args.dry_run)
        except KeyboardInterrupt:
            bar.write("\n  Pipeline interrupted by user.")
            bar.finish(success=False)
            sys.exit(130)

        if not success:
            failed_steps.append(step["label"])
            pipeline_ok = False
            if not args.skip_on_fail:
                bar.write(
                    f"\n  ✗  Step failed: {step['label']}\n"
                    f"     Stopping pipeline. Use --skip-on-fail to continue.\n"
                    f"     Or use --start-phase {step['phase']} to resume here.\n"
                )
                bar.finish(success=False)
                sys.exit(1)
            else:
                bar.write(
                    f"  ⚠  Step failed but --skip-on-fail is set — continuing.\n"
                )

    # ── Final summary ─────────────────────────────────────────────────
    bar.write("\n" + "═" * 70)
    if pipeline_ok:
        total_elapsed = time.time() - bar.start_time
        bar.write("  ✓  ALL PHASES COMPLETE")
        bar.write(f"  Total time: {_fmt_sec(total_elapsed)}")
        bar.write("")
        bar.write("  Output locations:")
        bar.write(f"    Phase 2 best model  : {PHASE2_DIR}/checkpoints/best.pt")
        bar.write(f"    Phase 3a UQ report  : {PHASE3_DIR}/outputs/calibration/")
        bar.write(f"    Phase 3b GNN model  : {PHASE3_DIR}/checkpoints/gnn/best.pt")
        bar.write(f"    Phase 4 fed model   : {PHASE4_DIR}/checkpoints/global/best.pt")
        bar.write(f"    DP audit report     : {PHASE4_DIR}/outputs/privacy_audit/")
        bar.write(f"    All visualisations  : */outputs/visualisations/")
        bar.write("═" * 70 + "\n")
    else:
        bar.write(f"  ✗  PIPELINE FINISHED WITH FAILURES")
        bar.write(f"  Failed steps:")
        for fs in failed_steps:
            bar.write(f"    • {fs}")
        bar.write("═" * 70 + "\n")

    bar.finish(success=pipeline_ok)
    sys.exit(0 if pipeline_ok else 1)


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def _fmt_min(minutes: float) -> str:
    """Format minutes as HHh MMm or MMm SSs."""
    if minutes >= 60:
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"{h}h {m:02d}m"
    if minutes >= 1:
        return f"{int(minutes)}m"
    return f"{int(minutes*60)}s"


def _fmt_sec(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
