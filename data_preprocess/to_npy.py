import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [p for p in root.rglob("*") if p.suffix in exts and p.is_file()]

def build_class_map(dataset_root: Path, split: str):
    split_dir = dataset_root / split
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class2idx = {c: i for i, c in enumerate(classes)}
    return classes, class2idx

def process_image(img_path: Path, img_size):
    """
    Load -> Resize (Force Shape) -> Array
    Assumes inputs are already cropped/clean.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        # Resize mandatory for memmap fixed shape
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), resample=Image.BICUBIC)
        return np.asarray(img)

def save_paths_cache(dataset_root: Path, out_dir: Path, splits=("train", "val", "test")):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes, class2idx = build_class_map(dataset_root, "train")
    (out_dir / "classes.json").write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        paths = []
        labels = []
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in list_images(cls_dir):
                paths.append(str(p))
                labels.append(class2idx[cls])

        paths = np.array(paths, dtype=np.str_)
        labels = np.array(labels, dtype=np.int64)

        np.save(out_dir / f"{split}_paths.npy", paths)
        np.save(out_dir / f"{split}_labels.npy", labels)
        print(f"[OK] {split}: {len(paths)} samples -> saved paths/labels to {out_dir}")

def save_images_cache(
    dataset_root: Path,
    out_dir: Path,
    img_size: int = 224,
    dtype: str = "uint8",
    splits=("train", "val", "test")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes, class2idx = build_class_map(dataset_root, "train")
    (out_dir / "classes.json").write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8")

    np_dtype = np.uint8 if dtype == "uint8" else np.float32

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] skip missing split: {split_dir}")
            continue

        # 0. Collect Candidates
        candidates = []
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in list_images(cls_dir):
                candidates.append((p, class2idx[cls]))

        if not candidates:
            print(f"[WARN] {split}: no images found")
            continue

        # 1. Pass 1: Simple Validation (Load Check)
        valid_items = [] # (img_path, label)
        stats = {
            "total": len(candidates),
            "valid": 0,
            "dropped": 0,
            "errors": []
        }
        
        print(f"[{split}] Pass 1: Validating {len(candidates)} samples...")
        for img_path, label in tqdm(candidates, desc=f"Validate {split}"):
            try:
                # Actual Load Check (Safety)
                _ = process_image(img_path, img_size)
                valid_items.append((img_path, label))
                stats["valid"] += 1
            except Exception as e:
                stats["dropped"] += 1
                if len(stats["errors"]) < 5:
                    stats["errors"].append(f"{img_path.name}: {str(e)}")

        N_valid = len(valid_items)
        if N_valid == 0:
            print(f"[ERR] {split}: No valid images found after validation.")
            continue

        print(f"[{split}] Pass 2: Writing {N_valid} samples...")
        
        # 2. Pass 2: Writing to Memmap
        H = W = img_size
        dat_path = out_dir / f"{split}_images.dat"
        X = np.memmap(dat_path, mode="w+", dtype=np_dtype, shape=(N_valid, H, W, 3))
        y = np.zeros((N_valid,), dtype=np.int64)
        paths = np.empty((N_valid,), dtype=np.str_)
        
        for i, (img_path, label) in enumerate(tqdm(valid_items, desc=f"Write {split}")):
            try:
                arr = process_image(img_path, img_size)
                
                # Type Conversion
                if np_dtype == np.float32:
                    arr = arr.astype(np.float32) / 255.0
                else:
                    arr = arr.astype(np.uint8)
                
                X[i] = arr
                y[i] = label
                paths[i] = str(img_path)
            except Exception as e:
                print(f"\n[FATAL] Write failed for {img_path.name}: {e}")
                sys.exit(1)

        X.flush()

        # Save Metadata
        np.save(out_dir / f"{split}_labels.npy", y)
        np.save(out_dir / f"{split}_paths.npy", paths)

        meta = {
            "split": split,
            "shape": [int(N_valid), int(H), int(W), 3],
            "dtype": dtype,
            "count": N_valid,
            "stats": stats,
            "data_file": f"{split}_images.dat",
            "labels_file": f"{split}_labels.npy",
            "paths_file": f"{split}_paths.npy",
        }
        (out_dir / f"{split}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"[OK] {split}: Saved {N_valid} items.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/root/project/dataset/dataset_cropped", type=str, help="e.g. /root/project/dataset/dataset_cropped")
    ap.add_argument("--out_dir", default="/root/project/dataset/cache_npy", type=str, help="e.g. /root/project/dataset/cache_npy")
    ap.add_argument("--mode", type=str, default="paths", choices=["paths", "images"], help="paths: save paths+labels, images: save resized images too")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "float32"])
    ap.add_argument("--splits", type=str, default="train,val,test")

    args = ap.parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    if not (dataset_root / "train").exists():
        print(f"[ERR] Train folder not found at: {dataset_root / 'train'}")
        print("Please check --dataset_root path.")
        sys.exit(1)

    if args.mode == "paths":
        save_paths_cache(dataset_root, out_dir, splits=splits)
    else:
        save_images_cache(dataset_root, out_dir, img_size=args.imgsz, dtype=args.dtype, splits=splits)

if __name__ == "__main__":
    main()