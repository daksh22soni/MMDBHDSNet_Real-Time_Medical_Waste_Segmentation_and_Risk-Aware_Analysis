"""
========================================================================
DBHDSNet Phase 4 — Client Shard Inspector

Run BEFORE training to verify:
  1. Each client prefix finds the correct images in your dataset
  2. Per-client class distribution (non-IID characterisation)
  3. Per-client hazard tier distribution
  4. Minimum shard sizes (warn if < 50 images)

Usage:
    python scripts/inspect_clients.py

Update DATASET_ROOT in config_phase4.py first.
========================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from collections import Counter, defaultdict

try:
    from config_phase4 import CFG4, CLIENTS, HAZARD_TIER_MAP
except ImportError as e:
    print(f"ERROR: {e}\nRun from the phase4_dbhdsnet/ directory.")
    sys.exit(1)


def sep(char="─", w=70):
    print(char * w)


def inspect_clients():
    dataset_root = CFG4.DATA.ROOT
    train_lbl    = CFG4.DATA.TRAIN_LBL
    train_img    = CFG4.DATA.TRAIN_IMG

    if not dataset_root.exists():
        print(f"ERROR: DATASET_ROOT not found: {dataset_root}")
        print("Update DATASET_ROOT in config_phase4.py Section A.")
        sys.exit(1)

    # Load class names
    with open(CFG4.DATA.YAML) as f:
        yaml_data = yaml.safe_load(f)
    class_names = [yaml_data["names"][i] for i in sorted(yaml_data["names"])]
    nc          = len(class_names)

    print("\n" + "═" * 70)
    print("  DBHDSNet Phase 4 — Client Shard Inspector")
    print("═" * 70)
    print(f"\n  Dataset root : {dataset_root}")
    print(f"  Classes      : {nc}  (nc={yaml_data['nc']})")
    print(f"  Clients      : {len(CLIENTS)}\n")

    # Scan label files per client
    label_files = sorted(train_lbl.glob("*.txt"))
    total_imgs  = len(list(train_img.glob("*")))

    # Per-client stats
    client_stats = {}
    for info in CLIENTS:
        prefix = info["prefix"]
        cid    = info["client_id"]

        # Count images by prefix
        imgs   = [f for f in train_img.iterdir()
                  if f.stem.startswith(prefix + "_")]
        lbls   = [f for f in label_files
                  if f.stem.startswith(prefix + "_")]

        # Class distribution
        class_counts  = Counter()
        tier_counts   = Counter()
        n_annotations = 0

        for lbl_file in lbls:
            try:
                lines = lbl_file.read_text().strip().split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        name = class_names[cls_id] if cls_id < nc else f"cls_{cls_id}"
                        tier = HAZARD_TIER_MAP.get(name, 4)
                        tier_counts[tier] += 1
                        n_annotations += 1
            except Exception:
                pass

        client_stats[cid] = {
            "prefix":       prefix,
            "n_images":     len(imgs),
            "n_labels":     len(lbls),
            "n_annotations": n_annotations,
            "class_counts": class_counts,
            "tier_counts":  tier_counts,
        }

    # ── Print per-client summary ─────────────────────────────────────
    for info in CLIENTS:
        cid  = info["client_id"]
        st   = client_stats[cid]
        n    = st["n_images"]
        pct  = 100 * n / max(total_imgs, 1)
        warn = "  ⚠  < 50 images!" if n < 50 else ""

        sep()
        print(f"  [{cid}]  prefix={st['prefix']}  "
              f"→  {n} images ({pct:.1f}% of train set){warn}")
        print(f"  Description: {info.get('description', '')}")
        print(f"  Annotations: {st['n_annotations']:,}")

        # Tier distribution
        tc     = st["tier_counts"]
        total_a = max(sum(tc.values()), 1)
        tier_names = {1:"Sharps/Hazardous",2:"Infectious",3:"Pharma",4:"General"}
        print("  Hazard tier distribution:")
        for t in [1, 2, 3, 4]:
            bar_len = int(30 * tc.get(t, 0) / total_a)
            bar     = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    T{t} {tier_names[t]:18s} {bar}  "
                  f"{tc.get(t,0):5d}  ({100*tc.get(t,0)/total_a:.1f}%)")

        # Top-5 classes
        top5 = st["class_counts"].most_common(5)
        print("  Top 5 classes:")
        for cls_id, cnt in top5:
            name = class_names[cls_id] if cls_id < nc else f"cls_{cls_id}"
            tier = HAZARD_TIER_MAP.get(name, 4)
            print(f"    T{tier} {name:<30s} {cnt:5d}")

    sep("═")
    print("\n  NON-IID CHARACTERISATION")
    sep()
    print("  Client                 | Images | T1%  | T2%  | T3%  | T4%")
    sep("-")
    for info in CLIENTS:
        cid  = info["client_id"]
        st   = client_stats[cid]
        tc   = st["tier_counts"]
        tot  = max(sum(tc.values()), 1)
        t1p  = 100 * tc.get(1, 0) / tot
        t2p  = 100 * tc.get(2, 0) / tot
        t3p  = 100 * tc.get(3, 0) / tot
        t4p  = 100 * tc.get(4, 0) / tot
        print(f"  {cid:22s} | {st['n_images']:6d} | "
              f"{t1p:4.1f} | {t2p:4.1f} | {t3p:4.1f} | {t4p:4.1f}")

    print()
    sep("═")
    print("\n  ✓ Inspection complete.")
    print("  If any client has 0 images, check that your master_dataset")
    print("  labels use the correct prefix (v1_, v5_, v6_, nv2_, nv3_, nv4_).")
    print("  Update DATASET_ROOT in config_phase4.py, then run:")
    print("  python train_phase4.py --dry-run")
    sep("═")


if __name__ == "__main__":
    inspect_clients()
