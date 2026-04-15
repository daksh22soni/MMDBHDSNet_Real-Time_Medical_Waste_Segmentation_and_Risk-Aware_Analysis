"""
========================================================================
DBHDSNet — Phase 3b Training Entry Point
ContamRisk-GNN: Scene Graph Construction + Training + Evaluation

Workflow
────────
  1. Load Phase 2 DBHDSNet (best.pt) + Phase 3a UQ estimator
  2. Run inference over all train/val/test images
  3. Build scene graphs with cross-contamination edge features
  4. Generate pseudo risk labels from WHO/CPCB rules
  5. (Optionally) augment with synthetic scene copies
  6. Train ContamRisk-GNN on scene graph dataset
  7. Evaluate: bin_MAE, bin_RMSE, bin_Pearson, cls_acc, cls_F1
  8. Generate scene graph visualisations for sample images

Run
───
    # Full pipeline (build graphs then train)
    python train_phase3b_gnn.py

    # Only build scene graphs (if graphs already built, skip this)
    python train_phase3b_gnn.py --build-graphs-only

    # Only train (assumes graphs already in outputs/scene_graphs/)
    python train_phase3b_gnn.py --skip-graph-build

    # Resume GNN training from checkpoint
    python train_phase3b_gnn.py --resume --skip-graph-build

    # Dry run
    python train_phase3b_gnn.py --dry-run
========================================================================
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
PHASE2_ROOT = ROOT.parent / "phase2_dbhdsnet"
sys.path.insert(0, str(PHASE2_ROOT))

import torch
from tqdm import tqdm

from config_phase3 import CFG
from src.utils import (
    setup_logging, set_seed, get_device, gpu_info,
    count_parameters, print_banner, make_run_name,
)
from src.uq.mc_dropout   import MCDropoutEstimator
from src.uq.calibration  import CalibrationTrainer
from src.gnn.contamrisk_gnn import build_contamrisk_gnn
from src.gnn.gnn_dataset import SceneGraphPipeline, build_gnn_dataloaders
from src.gnn.gnn_trainer import GNNTrainer
from src.gnn.gnn_visualiser import visualise_scene_graph
from src.gnn.gnn_losses  import GNNMetrics

try:
    from src.models.dbhdsnet import build_model as build_phase2_model
except ImportError:
    from phase2_dbhdsnet.src.models.dbhdsnet import build_model as build_phase2_model
try:
    from src.dataset import build_dataloaders, get_class_names
except ImportError:
    from phase2_dbhdsnet.src.dataset import build_dataloaders, get_class_names


# ════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3b — ContamRisk-GNN")
    p.add_argument("--build-graphs-only", action="store_true")
    p.add_argument("--skip-graph-build",  action="store_true")
    p.add_argument("--resume",            action="store_true")
    p.add_argument("--ckpt",             type=str, default=None)
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--device",           type=str, default=None)
    p.add_argument("--synthetic-copies", type=int, default=None)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
# GRAPH BUILD PHASE
# ════════════════════════════════════════════════════════════════════════

def build_scene_graphs(
    model, uq_estimator, cfg, class_names, device,
    graph_dir, synthetic_copies=3, dry_run=False
):
    """
    Runs Phase 2 inference + UQ over all splits and saves scene graphs.
    """
    from src.dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    pipeline = SceneGraphPipeline(
        phase2_model = model,
        uq_estimator = uq_estimator,
        cfg          = cfg,
        class_names  = class_names,
        device       = device,
        output_dir   = graph_dir,
    )

    split_loaders = [
        ("train", train_loader, synthetic_copies),
        ("valid", val_loader,   0),    # no augmentation for valid/test
        ("test",  test_loader,  0),
    ]

    for split, loader, n_copies in split_loaders:
        if dry_run:
            print(f"  [DRY RUN] Would build {split} scene graphs")
            continue
        n = pipeline.build_all(loader, split=split, synthetic_copies=n_copies)
        print(f"  {split}: {n} scene graphs saved.")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE SAMPLES
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualise_samples(model, loader, class_names, cfg, device, output_dir, n=5):
    """Generate scene graph visualisations for n sample images."""
    from src.uq.mc_dropout import MCDropoutEstimator
    from src.gnn.scene_graph import SceneGraphBuilder, RiskLabelGenerator, WasteItem

    model.eval()
    sg_builder = SceneGraphBuilder(cfg, num_classes=len(class_names))
    lbl_gen    = RiskLabelGenerator(cfg.DATA.CONTAMINATION_RULES)
    uq_est     = MCDropoutEstimator(model, cfg)

    count = 0
    for batch in loader:
        images = batch["images"].to(device)
        out    = model(images, return_decoded=True)
        uq     = uq_est.estimate(images[:1], device=device)
        B      = images.shape[0]

        for b in range(min(B, n - count)):
            det_boxes = out.get("boxes",    [None]*B)[b]
            det_scores= out.get("scores",   [None]*B)[b]
            det_cls   = out.get("class_ids",[None]*B)[b]

            if det_boxes is None or len(det_boxes) == 0:
                continue

            items = []
            for i in range(min(len(det_boxes), cfg.GNN.MAX_ITEMS_PER_SCENE)):
                cls_id   = int(det_cls[i].item())
                cls_name = (class_names[cls_id]
                            if cls_id < len(class_names) else f"class_{cls_id}")
                tier     = cfg.DATA.HAZARD_MAP.get(cls_name, 4)
                items.append(WasteItem(
                    class_id=cls_id, class_name=cls_name,
                    hazard_tier=tier,
                    box_cx=float(det_boxes[i,0]), box_cy=float(det_boxes[i,1]),
                    box_w =float(det_boxes[i,2]), box_h =float(det_boxes[i,3]),
                    confidence=float(det_scores[i]),
                    epistemic_u=float(uq.epistemic[0]) if b == 0 else 0.0,
                ))

            if len(items) < 2:
                continue

            label = lbl_gen.generate(items)
            fake_risks = label.item_risks

            save_path = output_dir / f"scene_graph_sample_{count:03d}.png"
            visualise_scene_graph(
                items      = items,
                item_risks = fake_risks,
                bin_risk   = label.bin_risk,
                risk_class = label.risk_class,
                save_path  = save_path,
            )
            count += 1

        if count >= n:
            break

    print(f"[Visualiser] {count} sample scene graph figures saved → {output_dir}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    CFG.make_dirs()

    run_name = make_run_name("phase3b_gnn")
    logger   = setup_logging(CFG.DATA.LOG_DIR, run_name)

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Device: {device}")
    logger.info(gpu_info())

    # ── Load Phase 2 model ────────────────────────────────────────────
    print_banner("Phase 3b — ContamRisk-GNN")
    logger.info("Loading Phase 2 model…")
    model = build_phase2_model(CFG, device)

    if Path(CFG.DATA.PHASE2_CKPT).exists():
        ckpt = torch.load(
            CFG.DATA.PHASE2_CKPT, map_location="cpu", weights_only=False
        )
        model_state = ckpt.get(
            "model_state",
            ckpt.get("ema_state", {}).get("ema_state", ckpt)
        )
        model.load_state_dict(model_state, strict=False)
        logger.info("  Phase 2 weights loaded.")
    else:
        logger.warning("  Phase 2 checkpoint not found — using random weights.")

    model.eval()
    class_names = get_class_names(CFG.DATA.YAML)

    # ── UQ estimator (for uncertainty node features) ──────────────────
    uq_estimator = MCDropoutEstimator(model, CFG)

    # ── Scene graph directory ─────────────────────────────────────────
    graph_dir = Path(CFG.DATA.OUTPUT_DIR) / "scene_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build scene graphs ────────────────────────────────────
    if not args.skip_graph_build:
        synthetic_copies = (
            args.synthetic_copies
            if args.synthetic_copies is not None
            else CFG.GNN.SYNTHETIC_SCENES_PER_IMAGE
        )
        if not args.dry_run:
            build_scene_graphs(
                model, uq_estimator, CFG, class_names, device,
                graph_dir, synthetic_copies, dry_run=False,
            )
        else:
            print("  [DRY RUN] Skipping scene graph build.")

    if args.build_graphs_only:
        print_banner("Graph build complete. Run with --skip-graph-build to train GNN.")
        return

    # ── Step 2: Load GNN dataloaders ─────────────────────────────────
    logger.info("Loading GNN dataloaders…")
    try:
        train_gnn, val_gnn, test_gnn = build_gnn_dataloaders(graph_dir, CFG)
        logger.info(
            f"  Train: {len(train_gnn.dataset)} graphs | "
            f"Val: {len(val_gnn.dataset)} | "
            f"Test: {len(test_gnn.dataset)}"
        )
    except Exception as e:
        logger.error(
            f"Failed to load GNN dataloaders: {e}\n"
            "  Run without --skip-graph-build first to build scene graphs."
        )
        return

    # ── Step 3: Build ContamRisk-GNN ─────────────────────────────────
    logger.info("Building ContamRisk-GNN…")
    gnn_model = build_contamrisk_gnn(CFG, device)

    params = count_parameters(gnn_model)
    logger.info(f"  Total params  : {params['total']:,}")
    logger.info(f"  Trainable     : {params['trainable']:,}")

    # ── Step 4: Visualise pre-training samples ────────────────────────
    vis_dir = Path(CFG.DATA.OUTPUT_DIR) / f"{run_name}_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        _, val_loader_p2, _ = build_dataloaders(CFG)
        visualise_samples(
            model, val_loader_p2, class_names, CFG, device, vis_dir, n=8
        )

    # ── Step 5: Train ContamRisk-GNN ─────────────────────────────────
    trainer = GNNTrainer(gnn_model, CFG, device)

    if args.resume:
        ckpt_path = Path(args.ckpt) if args.ckpt else None
        resumed   = trainer.resume(ckpt_path)
        if not resumed:
            logger.warning("No GNN checkpoint found — training from scratch.")

    if not args.dry_run:
        trainer.fit(train_gnn, val_gnn)
    else:
        print("  [DRY RUN] Skipping GNN training.")

    # ── Step 6: Test evaluation ───────────────────────────────────────
    logger.info("\nFinal test evaluation…")
    gnn_model.eval()
    test_metrics_obj = GNNMetrics()

    with torch.no_grad():
        for batch in tqdm(test_gnn, desc="  Test eval", ncols=100):
            batch = batch.to(device)
            out   = gnn_model(batch)
            test_metrics_obj.update(out, batch)

    test_m = test_metrics_obj.compute()
    logger.info("  Test results:")
    for k, v in test_m.items():
        logger.info(f"    {k:25s}: {v:.5f}")

    print_banner(
        f"Phase 3b Complete | Test bin_MAE={test_m.get('bin_MAE',0):.5f} | "
        f"cls_F1={test_m.get('cls_macro_F1',0):.4f}"
    )


if __name__ == "__main__":
    main()
