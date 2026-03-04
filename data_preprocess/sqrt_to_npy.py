# data_preprocess/sqrt_to_npy.py
import argparse
import json
import random
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Build paths/labels NPY using sqrt-based per-class sampling."
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="/root/project/dataset/whole_relabeled_dataset",
    help="Dataset root containing split/class folders (train/val/test).",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="/root/project/dataset/cache_npy_sqrt",
    help="Output directory for *_paths.npy and *_labels.npy.",
)
parser.add_argument("--splits", type=str, default="train")
parser.add_argument(
    "--classes",
    type=str,
    default="",
    help="Comma-separated class names. If empty, auto-detect from train split.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--total",
    type=int,
    default=0,
    help="Optional total samples per split. 0 means auto (sqrt by min class).",
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMG_EXTS]


def detect_classes(dataset_root: Path):
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise RuntimeError(f"Missing train dir for class detection: {train_dir}")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def sqrt_targets(counts, total_override=0):
    # counts: dict[class] -> n
    if not counts:
        return {}
    min_n = min(counts.values())
    sqrt_scores = {k: np.sqrt(v) for k, v in counts.items()}

    if total_override and total_override > 0:
        total_sqrt = sum(sqrt_scores.values())
        targets = {
            k: int(np.floor(sqrt_scores[k] / total_sqrt * total_override))
            for k in counts
        }
        # Ensure at least 1 for non-empty class
        for k, n in counts.items():
            if n > 0 and targets[k] == 0:
                targets[k] = 1
    else:
        # Default: scale by sqrt(min class) to keep min class intact and soften imbalance.
        targets = {k: int(np.floor(np.sqrt(counts[k] * min_n))) for k in counts}

    # Cap by available counts
    targets = {k: min(targets[k], counts[k]) for k in counts}
    return targets


def main():
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    random.seed(args.seed)

    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root not found: {dataset_root}")

    if args.classes.strip():
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = detect_classes(dataset_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] Missing split: {split_dir}")
            continue

        per_class_paths = {}
        counts = {}
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                per_class_paths[cls] = []
                counts[cls] = 0
                continue
            paths = list_images(cls_dir)
            per_class_paths[cls] = paths
            counts[cls] = len(paths)

        targets = sqrt_targets(counts, total_override=args.total)
        sampled_paths = []
        sampled_labels = []

        for cls_idx, cls in enumerate(classes):
            paths = per_class_paths[cls]
            k = targets.get(cls, 0)
            if k <= 0:
                continue
            if k >= len(paths):
                chosen = paths
            else:
                chosen = random.sample(paths, k)
            sampled_paths.extend([str(p) for p in chosen])
            sampled_labels.extend([cls_idx] * len(chosen))

        sampled_paths = np.array(sampled_paths, dtype=np.str_)
        sampled_labels = np.array(sampled_labels, dtype=np.int64)

        np.save(out_dir / f"{split}_paths.npy", sampled_paths)
        np.save(out_dir / f"{split}_labels.npy", sampled_labels)

        meta = {
            "split": split,
            "seed": args.seed,
            "total_override": args.total,
            "counts": counts,
            "targets": targets,
            "saved": len(sampled_paths),
        }
        (out_dir / f"{split}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] {split}: saved {len(sampled_paths)} samples to {out_dir}")


if __name__ == "__main__":
    main()
