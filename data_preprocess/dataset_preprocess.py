import os
import shutil
import json
import random
import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessor (split with optional balancing)")
    parser.add_argument("--src", type=str, default="/root/project/dataset/Relabeled", help="Source directory containing class folders (A1~A8)")
    parser.add_argument("--dst", type=str, default="/root/project/dataset/whole_relabeled_dataset", help="Destination directory for train/val/test split")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move", "symlink"], help="File transfer mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/Val/Test split ratios")
    parser.add_argument("--balanced", default=False, action="store_true",
                        help="If set, downsample each class to min valid count before split")
    
    # New Options
    parser.add_argument("--strict_classes", default=True, action="store_true", help="Fail if any class has 0 valid samples")
    parser.add_argument("--skip_image_verify", default=True, action="store_true", help="Skip detailed image verification (PIL.verify)")
    parser.add_argument("--skip_json_parse", default=True, action="store_true", help="Skip JSON parsing check")
    
    return parser.parse_args()

def find_json_pair(img_path: Path):
    """
    Try to find corresponding JSON file with various naming conventions.
    1. stemming suffix: image.jpg -> image.json / image.JSON
    """
    candidates = [
        img_path.with_suffix(".json"),
        img_path.with_suffix(".JSON")
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def is_valid_pair(img_path: Path, json_path: Path, skip_img_verify=False, skip_json_parse=False):
    """
    Checks if image and json pair is valid.
    Returns (True, None) or (False, reason_string)
    """
    if not json_path or not json_path.exists():
        return False, "no_json"
    
    if not skip_json_parse:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception:
            return False, "bad_json"

    if not skip_img_verify:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            return False, "bad_image"

    return True, None

def transfer_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        if dst.exists():
            os.remove(dst)
        os.symlink(src.absolute(), dst)

def main():
    args = parse_args()
    
    # 0. Path Validation
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    print(f"[*] Source: {src_root}")
    print(f"[*] Dest:   {dst_root}")
    print(f"[*] Mode:   {args.mode}")
    print(f"[*] Seed:   {args.seed}")
    
    # Check A1 existence as a basic sanity check
    if not (src_root / "A1").exists():
        print(f"[ERR] Source root or 'A1' folder not found at: {src_root}")
        print("Please check the path. Exiting.")
        sys.exit(1)

    random.seed(args.seed)
    classes = [f"A{i}" for i in range(1, 9)]
    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    
    # 1. Scanning & Validation
    print("\n[1] Scanning and Validating...")
    valid_samples = defaultdict(list) # cls -> list of (img_path, json_path)
    # stats uses nested dict compatible with json dump later
    stats = {cls: {"total": 0, "valid": 0, "no_json": 0, "bad_json": 0, "bad_image": 0} for cls in classes}
    
    for cls in classes:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            print(f"  [WARN] Class dir not found: {cls}")
            if args.strict_classes:
                print(f"[ERR] Strict mode: Missing class {cls}. Exiting.")
                sys.exit(1)
            continue
            
        imgs = []
        for ext in extensions:
            imgs.extend(list(cls_dir.glob(ext)))
        imgs = sorted(imgs)
        stats[cls]["total"] = len(imgs)
        
        for img_path in tqdm(imgs, desc=f"  CHECK {cls}", leave=False):
            json_path = find_json_pair(img_path)
            
            valid, reason = is_valid_pair(img_path, json_path, args.skip_image_verify, args.skip_json_parse)
            
            if valid:
                valid_samples[cls].append((img_path, json_path))
                stats[cls]["valid"] += 1
            else:
                stats[cls][reason] += 1

    # 2. Sample Selection (optional balanced downsampling)
    print("\n[2] Sample Selection...")
    available_classes = [cls for cls in classes if len(valid_samples[cls]) > 0]
    
    if not available_classes:
        print("[ERR] No valid classes found. Exiting.")
        sys.exit(1)

    # Check strict mode for 0 valid sample classes
    empty_classes = set(classes) - set(available_classes)
    if args.strict_classes and empty_classes:
        print(f"[ERR] Strict mode: Classes {empty_classes} have 0 valid samples. Exiting.")
        sys.exit(1)

    selected_samples = {}
    if args.balanced:
        min_cnt = min(len(valid_samples[cls]) for cls in available_classes)
        print(f"  => Balanced mode ON")
        print(f"  => Minimum Class Count K = {min_cnt}")
        print(f"  => Downsampling all classes to {min_cnt} samples.")
        for cls in available_classes:
            samples = valid_samples[cls]
            selected_samples[cls] = random.sample(samples, min_cnt)
    else:
        print("  => Balanced mode OFF (using all valid samples per class).")
        for cls in available_classes:
            selected_samples[cls] = list(valid_samples[cls])

    # 3. Splitting
    print("\n[3] Splitting Train/Val/Test...")
    r_train, r_val, r_test = args.split
    total_ratio = r_train + r_val + r_test
    r_train /= total_ratio
    r_val /= total_ratio
    
    # Init Record
    record = {
        "metadata": {
            "seed": args.seed,
            "balanced": args.balanced,
            "split_ratios": [r_train, r_val, r_test / total_ratio],
        },
        "stats": stats,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "class_split_counts": {},
        "files": {"train": [], "val": [], "test": []}
    }
    
    # 4. Processing & Saving
    print("\n[4] Processing Files...")
    
    for cls in available_classes:
        samples = selected_samples[cls]
        random.shuffle(samples) 

        n_total = len(samples)
        n_train = int(n_total * r_train)
        n_val = int(n_total * r_val)
        n_test = n_total - n_train - n_val

        record["class_split_counts"][cls] = {
            "total": n_total,
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }
        record["split_counts"]["train"] += n_train
        record["split_counts"]["val"] += n_val
        record["split_counts"]["test"] += n_test
        
        splits = {
            "train": samples[:n_train],
            "val":   samples[n_train:n_train+n_val],
            "test":  samples[n_train+n_val:]
        }
        
        for split_name, items in splits.items():
            split_dir = dst_root / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path, json_path in tqdm(items, desc=f"  {cls} -> {split_name}", leave=False):
                # Unique Filename: {cls}_{stem}
                # User Feedback: Avoid "jpg.jpg" redundancy.
                # New Strategy: A1_foo.jpg / A1_foo.json
                
                new_stem = f"{cls}_{img_path.stem}"
                
                new_img_name = f"{new_stem}{img_path.suffix}" 
                new_json_name = f"{new_stem}.json"
                
                dst_img = split_dir / new_img_name
                dst_json = split_dir / new_json_name
                
                transfer_file(img_path, dst_img, args.mode)
                transfer_file(json_path, dst_json, args.mode)
                
                record["files"][split_name].append({
                    "cls": cls,
                    "src_img": str(img_path),
                    "dst_img": str(dst_img)
                })

    # 5. Reporting
    print("\n[5] Summary")
    total_processed = (
        record["split_counts"]["train"]
        + record["split_counts"]["val"]
        + record["split_counts"]["test"]
    )
    print(f"  Balanced mode: {args.balanced}")
    print(f"  Total Processed Images: {total_processed}")
    print(f"  Split Counts: train={record['split_counts']['train']}, "
          f"val={record['split_counts']['val']}, test={record['split_counts']['test']}")
    print("  Class Stats:")
    for cls in classes:
        s = stats[cls]
        print(f"    {cls}: Total={s['total']}, Valid={s['valid']}, Drops[NoJSON={s['no_json']}, BadJSON={s['bad_json']}, BadImg={s['bad_image']}]")

    log_file = dst_root / f"splits_seed{args.seed}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Split log saved to: {log_file}")
    print("[*] DONE.")

if __name__ == "__main__":
    main()
