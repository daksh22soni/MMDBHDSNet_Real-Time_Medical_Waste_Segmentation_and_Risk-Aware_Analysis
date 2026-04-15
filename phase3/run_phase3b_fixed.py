"""
========================================================================
DBHDSNet — Phase 3b: ContamRisk-GNN Entry Point

Usage:
    # Full pipeline: build graphs → train GNN → evaluate → visualise
    python run_phase3b.py

    # Resume GNN training from checkpoint
    python run_phase3b.py --resume

    # Only rebuild scene graphs (re-run DBHDSNet inference)
    python run_phase3b.py --rebuild-graphs

    # Only evaluate (load best checkpoint)
    python run_phase3b.py --eval-only

    # Dry run (sanity check)
    python run_phase3b.py --dry-run
========================================================================
"""

import sys
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "phase2"))

import torch
import numpy as np

from config_phase3 import CFG3
from src.utils import (
    setup_logging, set_seed, get_device, gpu_info,
    print_banner, count_parameters, make_run_name
)

try:
    from src.dataset import build_dataloaders
    from src.models  import build_model
except ImportError:
    from src.dataset import build_dataloaders
    from src.models  import build_model

from src.gnn.scene_graph    import SceneGraphDataset
from src.gnn.contamrisk_gnn import ContamRiskGNN
from src.gnn.gnn_trainer    import Phase3bTrainer
from src.gnn.visualiser_gnn import generate_all_gnn_figures


# ════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3b — ContamRisk-GNN")
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--rebuild-graphs", action="store_true",
                   help="Force re-run DBHDSNet inference to rebuild scene graphs")
    p.add_argument("--eval-only",      action="store_true")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--ckpt",           type=str,  default=None)
    p.add_argument("--device",         type=str,  default=None)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# SCENE GRAPH CACHE (save/load to avoid re-running inference)
# ════════════════════════════════════════════════════════════════════════

GRAPH_CACHE = {
    "train": Path(CFG3.DATA.OUTPUT_DIR) / "contamrisk" / "graphs_train.pkl",
    "valid": Path(CFG3.DATA.OUTPUT_DIR) / "contamrisk" / "graphs_valid.pkl",
    "test":  Path(CFG3.DATA.OUTPUT_DIR) / "contamrisk" / "graphs_test.pkl",
}


def _save_graphs(graphs, split: str):
    with open(GRAPH_CACHE[split], "wb") as f:
        pickle.dump(graphs, f)
    print(f"  [Cache] Saved {len(graphs)} {split} graphs → {GRAPH_CACHE[split]}")


def _load_graphs(split: str):
    path = GRAPH_CACHE[split]
    if path.exists():
        with open(path, "rb") as f:
            graphs = pickle.load(f)
        print(f"  [Cache] Loaded {len(graphs)} {split} graphs from {path}")
        return graphs
    return None


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    CFG3.make_dirs()

    logger = setup_logging(CFG3.DATA.LOG_DIR, "phase3b")
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    logger.info(f"Phase 3b | Device: {device}")
    logger.info(gpu_info())

    # ── Load Phase 2 (or Phase 3a) detection model ────────────────────
    p2_ckpt = Path(CFG3.DATA.PHASE2_CKPT)
    if not p2_ckpt.exists():
        logger.error(f"Checkpoint not found: {p2_ckpt}")
        sys.exit(1)

    logger.info("Loading detection model…")
    det_model = build_model(CFG3, device)
    state     = torch.load(p2_ckpt, map_location=device, weights_only=False)
    det_model.load_state_dict(state["model_state"], strict=False)
    det_model.eval()
    logger.info("  Detection model loaded.")

    # ── Data loaders ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(CFG3)

    # ── Build or load scene graphs ────────────────────────────────────
    graph_builder = SceneGraphDataset(det_model, CFG3, device)

    def get_graphs(split: str, loader) -> list:
        if not args.rebuild_graphs:
            cached = _load_graphs(split)
            if cached is not None:
                return cached
        graphs = graph_builder.build(loader, split=split,
                                     desc=f"Building {split} graphs")
        _save_graphs(graphs, split)
        return graphs

    logger.info("Building scene graphs (or loading from cache)…")
    train_graphs = get_graphs("train", train_loader)
    val_graphs   = get_graphs("valid", val_loader)
    test_graphs  = get_graphs("test",  test_loader)

    logger.info(f"  Graphs — Train:{len(train_graphs)}  "
                f"Val:{len(val_graphs)}  Test:{len(test_graphs)}")

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print_banner("Phase 3b — DRY RUN")
        try:
            from torch_geometric.data import Batch
            g1, g2 = train_graphs[0], train_graphs[1]
            batch  = Batch.from_data_list([g1, g2]).to(device)
            gnn    = ContamRiskGNN(CFG3).to(device)
            out    = gnn(batch)
            print(f"  scene_risk shape : {out['scene_risk'].shape}")
            print(f"  node_risk  shape : {out['node_risk'].shape}")
            print(f"  ✓ GNN forward pass OK. Dry run complete.")
        except Exception as e:
            print(f"  ✗ Dry run failed: {e}")
        return

    # ── Build GNN ─────────────────────────────────────────────────────
    logger.info("Building ContamRiskGNN…")
    gnn = ContamRiskGNN(CFG3).to(device)
    params = count_parameters(gnn)
    logger.info(f"  GNN params: {params['total']:,} total / {params['trainable']:,} trainable")

    # ── Phase 3b Trainer ─────────────────────────────────────────────
    trainer = Phase3bTrainer(gnn, CFG3, device)

    if args.resume or args.eval_only:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        trainer.resume(ckpt_path)

    # ── Train ─────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("Starting ContamRisk-GNN training…")
        trainer.fit(train_graphs, val_graphs)

    # ── Evaluate ──────────────────────────────────────────────────────
    logger.info("Evaluating on test set…")
    test_metrics = trainer.evaluate(test_graphs, save_report=True)

    # ── Collect predictions for visualisation ─────────────────────────
    logger.info("Collecting predictions for visualisation…")
    try:
        from torch_geometric.loader import DataLoader as GeoDataLoader
        from torch_geometric.data   import Batch

        test_loader_geo = GeoDataLoader(
            test_graphs, batch_size=CFG3.GNN.GNN_BATCH_SIZE, shuffle=False
        )

        all_pred, all_true = [], []
        all_np, all_nt     = [], []

        gnn.eval()
        with torch.no_grad():
            for data in test_loader_geo:
                data = data.to(device)
                out  = gnn(data)
                all_pred.extend(out["scene_risk"].cpu().tolist())
                all_true.extend(data.y.view(-1).cpu().tolist())
                all_np.extend(out["node_risk"].cpu().tolist())
                all_nt.extend(data.hazard_tiers.cpu().tolist())

        from src.dataset import get_class_names
        class_names = get_class_names(CFG3.DATA.YAML)

        generate_all_gnn_figures(
            pred_risks   = np.array(all_pred),
            true_risks   = np.array(all_true),
            node_preds   = np.array(all_np),
            node_tiers   = np.array(all_nt),
            metrics      = test_metrics,
            output_dir   = Path(CFG3.DATA.OUTPUT_DIR) / "contamrisk",
            class_names  = class_names,
        )
    except Exception as e:
        logger.warning(f"Visualisation step failed (non-fatal): {e}")

    print_banner(
        f"Phase 3b Complete!  "
        f"MAE={test_metrics.get('mae',0):.4f}  "
        f"HIGH-Sensitivity={test_metrics.get('high_sensitivity',0)*100:.1f}%"
    )


if __name__ == "__main__":
    main()
