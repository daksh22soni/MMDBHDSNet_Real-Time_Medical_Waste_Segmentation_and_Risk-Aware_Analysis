"""
========================================================================
DBHDSNet Phase 3b — Scene Graph Dataset
Builds a graph dataset from Phase 2 model predictions over the
training images. Each sample = one waste bin scene graph.

Two sources of training data:
  1. REAL scenes: run Phase 2 model over all train/valid images,
     collect detections, build scene graph per image.
  2. SYNTHETIC augmentation: randomise item positions within each
     scene (N copies per real scene) to expand the graph dataset.

Each graph is stored as a PyG Data object and saved to disk.
The GNNDataset class loads these graphs on-the-fly during training.
========================================================================
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

try:
    from torch_geometric.data import Data as PyGData, Dataset as PyGDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    PyGDataset = object

from .scene_graph import SceneGraphBuilder, RiskLabelGenerator, WasteItem


# ════════════════════════════════════════════════════════════════════════
# GRAPH DATASET (in-memory from pre-built .pt files)
# ════════════════════════════════════════════════════════════════════════

class GNNDataset(PyGDataset if PYGEOMETRIC_AVAILABLE else object):
    """
    Loads pre-built scene graph .pt files from a directory.
    Compatible with PyG DataLoader for mini-batching graphs.
    """

    def __init__(self, graph_dir: Path, split: str = "train"):
        self.graph_dir = Path(graph_dir) / split
        self.graph_files = sorted(self.graph_dir.glob("*.pt"))
        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric required for GNNDataset.")
        super().__init__(root=str(graph_dir))
        print(f"[GNNDataset] {split}: {len(self.graph_files)} scene graphs loaded.")

    def len(self) -> int:
        return len(self.graph_files)

    def get(self, idx: int) -> PyGData:
        return torch.load(self.graph_files[idx], weights_only=False)


# ════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER PIPELINE (runs Phase 2 model → collects detections)
# ════════════════════════════════════════════════════════════════════════

class SceneGraphPipeline:
    """
    Runs the Phase 2 DBHDSNet model over a dataset split,
    collects per-image detections, builds scene graphs with
    pseudo risk labels, and saves them to disk as .pt files.

    Call `build_all()` once before GNN training.
    """

    def __init__(
        self,
        phase2_model,           # trained DBHDSNet (eval mode)
        uq_estimator,           # MCDropoutEstimator from Phase 3a
        cfg,
        class_names:  List[str],
        device:       torch.device,
        output_dir:   Path,
    ):
        self.model        = phase2_model
        self.uq_estimator = uq_estimator
        self.cfg          = cfg
        self.class_names  = class_names
        self.device       = device
        self.output_dir   = Path(output_dir)
        self.gnn_cfg      = cfg.GNN

        self.graph_builder = SceneGraphBuilder(cfg, num_classes=len(class_names))
        self.label_gen     = RiskLabelGenerator(cfg.DATA.CONTAMINATION_RULES)
        self.hazard_map    = cfg.DATA.HAZARD_MAP

    # ------------------------------------------------------------------

    def build_all(
        self,
        data_loader,           # Phase 2-compatible DataLoader
        split: str = "train",
        synthetic_copies: int = 3,
    ):
        """
        Full pipeline: inference → scene graph → save.
        """
        save_dir = self.output_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        graph_idx = 0

        print(f"\n[SceneGraphPipeline] Building {split} scene graphs…")

        pbar = tqdm(
            data_loader,
            desc   = f"  [{split}] Building scene graphs",
            ncols  = 120,
            leave  = True,
        )

        for batch in pbar:
            images    = batch["images"].to(self.device, non_blocking=True)
            gt_boxes  = batch["boxes"]
            img_paths = batch["img_paths"]
            B         = images.shape[0]

            with torch.no_grad():
                # Phase 2 inference
                out     = self.model(images, return_decoded=True)
                # Phase 3a UQ
                uq_est  = self.uq_estimator.estimate(images, device=self.device)

            for b in range(B):
                det_boxes   = out.get("boxes",    [None] * B)[b]
                det_scores  = out.get("scores",   [None] * B)[b]
                det_cls     = out.get("class_ids",[None] * B)[b]

                if det_boxes is None or len(det_boxes) == 0:
                    continue

                # ── Build WasteItem list ──────────────────────────────
                items = self._detections_to_items(
                    det_boxes, det_scores, det_cls, uq_est, b
                )

                if len(items) < self.gnn_cfg.MIN_ITEMS_PER_SCENE:
                    continue
                if len(items) > self.gnn_cfg.MAX_ITEMS_PER_SCENE:
                    items = items[:self.gnn_cfg.MAX_ITEMS_PER_SCENE]

                # ── Build and save real scene graph ───────────────────
                label = self.label_gen.generate(items)
                graph = self.graph_builder.build(items, label)
                torch.save(graph, save_dir / f"graph_{graph_idx:06d}.pt")
                graph_idx += 1

                # ── Synthetic augmented copies ────────────────────────
                if split == "train":
                    for _ in range(synthetic_copies):
                        aug_items = self._augment_positions(items)
                        aug_label = self.label_gen.generate(aug_items)
                        aug_graph = self.graph_builder.build(aug_items, aug_label)
                        torch.save(
                            aug_graph,
                            save_dir / f"graph_{graph_idx:06d}.pt"
                        )
                        graph_idx += 1

            pbar.set_postfix(n_graphs=graph_idx)

        print(f"[SceneGraphPipeline] {split}: {graph_idx} graphs saved → {save_dir}")
        return graph_idx

    # ------------------------------------------------------------------

    def _detections_to_items(
        self,
        boxes:     torch.Tensor,    # (N, 4) cx cy w h normalised
        scores:    torch.Tensor,    # (N,)
        cls_ids:   torch.Tensor,    # (N,)
        uq_est,                     # UQEstimate
        batch_idx: int,
    ) -> List[WasteItem]:
        """Convert raw detections into WasteItem objects."""
        items = []
        N = min(len(boxes), self.gnn_cfg.MAX_ITEMS_PER_SCENE)

        # Per-image MC uncertainty: use mean epistemic as proxy per-item
        # (item-level UQ requires per-detection heads; image-level is an approximation)
        img_epistemic = float(uq_est.epistemic[batch_idx].item()) \
                        if uq_est.epistemic.shape[0] > batch_idx else 0.0

        for i in range(N):
            cls_id   = int(cls_ids[i].item())
            cls_name = (self.class_names[cls_id]
                        if cls_id < len(self.class_names) else f"class_{cls_id}")
            tier     = self.hazard_map.get(cls_name, 4)

            items.append(WasteItem(
                class_id     = cls_id,
                class_name   = cls_name,
                hazard_tier  = tier,
                box_cx       = float(boxes[i, 0].item()),
                box_cy       = float(boxes[i, 1].item()),
                box_w        = float(boxes[i, 2].item()),
                box_h        = float(boxes[i, 3].item()),
                confidence   = float(scores[i].item()),
                epistemic_u  = img_epistemic,
            ))
        return items

    # ------------------------------------------------------------------

    def _augment_positions(self, items: List[WasteItem]) -> List[WasteItem]:
        """
        Synthetic augmentation: randomly jitter item positions
        while keeping tiers and classes fixed.
        """
        aug = []
        for item in items:
            jitter = 0.08
            new_cx = float(np.clip(item.box_cx + random.uniform(-jitter, jitter), 0.05, 0.95))
            new_cy = float(np.clip(item.box_cy + random.uniform(-jitter, jitter), 0.05, 0.95))
            scale  = random.uniform(0.85, 1.15)
            new_w  = float(np.clip(item.box_w * scale, 0.01, 0.8))
            new_h  = float(np.clip(item.box_h * scale, 0.01, 0.8))

            aug.append(WasteItem(
                class_id    = item.class_id,
                class_name  = item.class_name,
                hazard_tier = item.hazard_tier,
                box_cx      = new_cx,
                box_cy      = new_cy,
                box_w       = new_w,
                box_h       = new_h,
                confidence  = item.confidence,
                epistemic_u = item.epistemic_u,
            ))
        return aug


# ════════════════════════════════════════════════════════════════════════
# DATALOADER FACTORIES
# ════════════════════════════════════════════════════════════════════════

def build_gnn_dataloaders(graph_dir: Path, cfg):
    """
    Returns (train_loader, val_loader, test_loader) for GNN training.
    """
    if not PYGEOMETRIC_AVAILABLE:
        raise ImportError("torch-geometric required.")

    train_ds = GNNDataset(graph_dir, "train")
    valid_ds = GNNDataset(graph_dir, "valid")
    test_ds  = GNNDataset(graph_dir, "test")

    train_loader = PyGDataLoader(
        train_ds,
        batch_size  = cfg.GNN.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.GNN.NUM_WORKERS,
        drop_last   = True,
        pin_memory  = True,
    )
    valid_loader = PyGDataLoader(
        valid_ds,
        batch_size  = cfg.GNN.BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.GNN.NUM_WORKERS,
    )
    test_loader = PyGDataLoader(
        test_ds,
        batch_size  = cfg.GNN.BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.GNN.NUM_WORKERS,
    )

    return train_loader, valid_loader, test_loader
