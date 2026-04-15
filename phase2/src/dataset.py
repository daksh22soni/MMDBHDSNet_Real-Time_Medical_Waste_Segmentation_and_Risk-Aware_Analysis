"""
========================================================================
DBHDSNet — YOLO Segmentation Dataset Loader
Reads YOLO polygon labels (class_id x1 y1 x2 y2 … xN yN),
converts them to bounding boxes + binary instance masks,
and returns batches ready for DBHDSNet training.
========================================================================
"""

import os
import cv2
import yaml
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ════════════════════════════════════════════════════════════════════════
# 1 — LABEL PARSING
# ════════════════════════════════════════════════════════════════════════

def parse_yolo_seg_label(label_path: Path, img_w: int, img_h: int
                          ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Parse a YOLO segmentation label file.

    Returns
    -------
    boxes   : (N, 5) float32 — [class_id, cx, cy, w, h] normalised 0-1
    polygons: list of (K_i, 2) float32 arrays — normalised polygon vertices
    """
    boxes, polygons = [], []

    if not Path(label_path).exists():
        return np.zeros((0, 5), dtype=np.float32), []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:          # need at least class + 2 points
                continue
            # YOLO seg format: class_id x1 y1 x2 y2 … xN yN
            # Number of coordinate values must be even (pairs)
            cls_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32)

            # Validate: must have even count of coords and ≥ 3 points
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue

            poly = coords.reshape(-1, 2)   # (K, 2)

            # Derive bounding box from polygon extent
            x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
            x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            bw = x_max - x_min
            bh = y_max - y_min

            boxes.append([cls_id, cx, cy, bw, bh])
            polygons.append(poly)

    if boxes:
        return np.array(boxes, dtype=np.float32), polygons
    return np.zeros((0, 5), dtype=np.float32), []


def polygon_to_mask(polygon: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Rasterise a normalised polygon into a binary mask of shape (img_h, img_w).
    """
    pts = (polygon * np.array([img_w, img_h])).astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ════════════════════════════════════════════════════════════════════════
# 2 — MOSAIC AUGMENTATION (4-image)
# ════════════════════════════════════════════════════════════════════════

def mosaic4(imgs, boxes_list, polys_list, img_size: int):
    """
    Combines 4 images into a mosaic.
    Returns mosaic image (img_size, img_size, 3), merged boxes & polygons.
    """
    s  = img_size
    s2 = img_size // 2

    # Mosaic centre offset (random jitter ±25%)
    cx = int(random.uniform(s2 * 0.5, s2 * 1.5))
    cy = int(random.uniform(s2 * 0.5, s2 * 1.5))

    mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
    merged_boxes, merged_polys = [], []

    positions = [
        (0,    0,    cx,   cy),    # top-left
        (cx,   0,    s,    cy),    # top-right
        (0,    cy,   cx,   s),     # bottom-left
        (cx,   cy,   s,    s),     # bottom-right
    ]

    for idx, (x1, y1, x2, y2) in enumerate(positions):
        img  = imgs[idx]
        boxes = boxes_list[idx]
        polys = polys_list[idx]

        h_patch, w_patch = y2 - y1, x2 - x1

        # Resize image patch
        patch = cv2.resize(img, (w_patch, h_patch))
        mosaic_img[y1:y2, x1:x2] = patch

        if len(boxes) == 0:
            continue

        # Scale and translate boxes/polygons into mosaic coordinates
        for bi, box in enumerate(boxes):
            cls_id, cx_n, cy_n, bw_n, bh_n = box
            # Convert normalised → mosaic pixel, then re-normalise
            # Original image coords normalised → patch pixel
            new_cx = (cx_n * w_patch + x1) / s
            new_cy = (cy_n * h_patch + y1) / s
            new_bw = bw_n * w_patch / s
            new_bh = bh_n * h_patch / s

            # Clip to mosaic bounds
            new_cx = np.clip(new_cx, 0, 1)
            new_cy = np.clip(new_cy, 0, 1)
            new_bw = min(new_bw, 2 * min(new_cx, 1 - new_cx))
            new_bh = min(new_bh, 2 * min(new_cy, 1 - new_cy))

            if new_bw > 0.005 and new_bh > 0.005:
                merged_boxes.append([cls_id, new_cx, new_cy, new_bw, new_bh])
                if polys and bi < len(polys):
                    p = polys[bi].copy()
                    p[:, 0] = (p[:, 0] * w_patch + x1) / s
                    p[:, 1] = (p[:, 1] * h_patch + y1) / s
                    p = np.clip(p, 0, 1)
                    merged_polys.append(p)

    merged_boxes = np.array(merged_boxes, dtype=np.float32) if merged_boxes \
                   else np.zeros((0, 5), dtype=np.float32)
    return mosaic_img, merged_boxes, merged_polys


# ════════════════════════════════════════════════════════════════════════
# 3 — AUGMENTATION PIPELINE
# ════════════════════════════════════════════════════════════════════════

def get_train_transforms(img_size: int, cfg) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=cfg.AUG_FLIP_H),
        A.VerticalFlip(p=cfg.AUG_FLIP_V),
        A.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.7, hue=0.015, p=0.5,
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.ToGray(p=0.01),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ════════════════════════════════════════════════════════════════════════
# 4 — DATASET CLASS
# ════════════════════════════════════════════════════════════════════════

class MedWasteDataset(Dataset):
    """
    YOLO segmentation dataset for DBHDSNet.

    Each sample returns:
        image      : (3, img_size, img_size) float tensor, normalised
        boxes      : (N, 5) tensor — [cls_id, cx, cy, w, h] normalised
        masks      : (N, mask_h, mask_w) uint8 tensor  — binary instance masks
        hazard_tiers: (N,) int tensor — hazard tier (1-4) per instance
        img_path   : str
    """

    def __init__(
        self,
        img_dir:      Path,
        lbl_dir:      Path,
        img_size:     int,
        class_names:  List[str],
        hazard_map:   Dict[str, int],
        train_cfg,                       # TrainConfig instance
        split:        str = "train",     # "train" | "valid" | "test"
        use_mosaic:   bool = True,
    ):
        self.img_dir     = Path(img_dir)
        self.lbl_dir     = Path(lbl_dir)
        self.img_size    = img_size
        self.mask_size   = img_size // 4  # proto masks at 1/4 resolution
        self.class_names = class_names
        self.n_classes   = len(class_names)
        self.split       = split
        self.use_mosaic  = use_mosaic and (split == "train")
        self.cfg         = train_cfg

        # Build class_id → hazard_tier lookup
        self.id2tier = self._build_id2tier(hazard_map)

        # Collect valid image-label pairs
        self.samples = self._collect_samples()
        print(f"[Dataset] {split}: {len(self.samples)} valid samples found.")

        # Augmentation transforms
        if split == "train":
            self.transforms = get_train_transforms(img_size, train_cfg)
        else:
            self.transforms = get_val_transforms(img_size)

    # ------------------------------------------------------------------

    def _build_id2tier(self, hazard_map: Dict[str, int]) -> Dict[int, int]:
        """Maps class_id integer → hazard tier (1-4)."""
        mapping = {}
        for idx, name in enumerate(self.class_names):
            mapping[idx] = hazard_map.get(name, 4)  # default: general
        return mapping

    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[dict]:
        valid = []
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            valid.append({"img": img_path, "lbl": lbl_path})

        return valid

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------

    def _load_single(self, index: int):
        """Load and preprocess a single sample (without mosaic)."""
        s = self.samples[index]
        img = cv2.imread(str(s["img"]))
        if img is None:
            # Fallback: blank image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))

        boxes, polygons = parse_yolo_seg_label(s["lbl"], orig_w, orig_h)
        return img, boxes, polygons

    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        # ── Mosaic augmentation (training only) ──────────────────────
        if self.use_mosaic and random.random() < self.cfg.AUG_MOSAIC:
            indices = [index] + random.choices(range(len(self)), k=3)
            imgs, boxes_list, polys_list = [], [], []
            for i in indices:
                img, boxes, polys = self._load_single(i)
                imgs.append(img)
                boxes_list.append(boxes)
                polys_list.append(polys)
            img, boxes, polygons = mosaic4(
                imgs, boxes_list, polys_list, self.img_size
            )
        else:
            img, boxes, polygons = self._load_single(index)

        # ── Apply pixel-level transforms ─────────────────────────────
        transformed = self.transforms(image=img)
        img_tensor  = transformed["image"]          # (3, H, W) float

        # ── Build instance masks at mask resolution ──────────────────
        n          = len(boxes)
        mask_h = mask_w = self.mask_size
        masks_np   = np.zeros((n, mask_h, mask_w), dtype=np.uint8)

        for i, poly in enumerate(polygons):
            m = polygon_to_mask(poly, self.img_size, self.img_size)
            masks_np[i] = cv2.resize(
                m, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST
            )

        # ── Hazard tiers ─────────────────────────────────────────────
        tiers = np.array(
            [self.id2tier.get(int(b[0]), 4) for b in boxes],
            dtype=np.int64
        )

        return {
            "image":       img_tensor,                              # (3,H,W)
            "boxes":       torch.from_numpy(boxes),                 # (N,5)
            "masks":       torch.from_numpy(masks_np),              # (N,mH,mW)
            "hazard_tiers": torch.from_numpy(tiers),                # (N,)
            "img_path":    str(self.samples[index]["img"]),
        }


# ════════════════════════════════════════════════════════════════════════
# 5 — COLLATE FUNCTION (variable-length annotations)
# ════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate that handles variable numbers of instances per image.
    boxes and masks are kept as lists-of-tensors; image is stacked.
    """
    images        = torch.stack([b["image"] for b in batch], dim=0)
    boxes         = [b["boxes"]        for b in batch]
    masks         = [b["masks"]        for b in batch]
    hazard_tiers  = [b["hazard_tiers"] for b in batch]
    img_paths     = [b["img_path"]     for b in batch]

    return {
        "images":      images,       # (B, 3, H, W)
        "boxes":       boxes,        # list of (Ni, 5)
        "masks":       masks,        # list of (Ni, mH, mW)
        "hazard_tiers": hazard_tiers, # list of (Ni,)
        "img_paths":   img_paths,
    }


# ════════════════════════════════════════════════════════════════════════
# 6 — YAML LOADER
# ════════════════════════════════════════════════════════════════════════

def load_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_class_names(yaml_path: Path) -> List[str]:
    """Extracts ordered class name list from YOLO YAML."""
    data = load_yaml(yaml_path)
    names = data.get("names", {})
    if isinstance(names, list):
        return names
    # dict format: {0: 'cat', 1: 'dog', ...}
    return [names[i] for i in sorted(names.keys())]


# ════════════════════════════════════════════════════════════════════════
# 7 — DATALOADER FACTORY
# ════════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train / valid / test DataLoaders from config.
    Returns (train_loader, valid_loader, test_loader).
    """
    class_names = get_class_names(cfg.DATA.YAML)
    hazard_map  = cfg.DATA.HAZARD_MAP

    train_ds = MedWasteDataset(
        img_dir     = cfg.DATA.TRAIN_IMG,
        lbl_dir     = cfg.DATA.TRAIN_LBL,
        img_size    = cfg.MODEL.IMG_SIZE,
        class_names = class_names,
        hazard_map  = hazard_map,
        train_cfg   = cfg.TRAIN,
        split       = "train",
        use_mosaic  = True,
    )
    valid_ds = MedWasteDataset(
        img_dir     = cfg.DATA.VALID_IMG,
        lbl_dir     = cfg.DATA.VALID_LBL,
        img_size    = cfg.MODEL.IMG_SIZE,
        class_names = class_names,
        hazard_map  = hazard_map,
        train_cfg   = cfg.TRAIN,
        split       = "valid",
        use_mosaic  = False,
    )
    test_ds = MedWasteDataset(
        img_dir     = cfg.DATA.TEST_IMG,
        lbl_dir     = cfg.DATA.TEST_LBL,
        img_size    = cfg.MODEL.IMG_SIZE,
        class_names = class_names,
        hazard_map  = hazard_map,
        train_cfg   = cfg.TRAIN,
        split       = "test",
        use_mosaic  = False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.TRAIN.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.TRAIN.NUM_WORKERS,
        pin_memory  = cfg.TRAIN.PIN_MEMORY,
        collate_fn  = collate_fn,
        drop_last   = True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size  = cfg.TRAIN.VAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.TRAIN.NUM_WORKERS,
        pin_memory  = cfg.TRAIN.PIN_MEMORY,
        collate_fn  = collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.TRAIN.VAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.TRAIN.NUM_WORKERS,
        pin_memory  = cfg.TRAIN.PIN_MEMORY,
        collate_fn  = collate_fn,
    )

    return train_loader, valid_loader, test_loader
