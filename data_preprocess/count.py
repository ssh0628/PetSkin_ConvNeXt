import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count image samples per split/class (train/val/test x A1~A8)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/root/project/dataset/dataset_aware_cropped",
        help="Dataset root containing split folders (train/val/test).",
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--classes", type=str, default="A1,A2,A3,A4,A5,A6,A7,A8")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any split/class directory is missing.",
    )
    return parser.parse_args()


def count_images(dir_path: Path) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix in IMG_EXTS)


def main():
    args = parse_args()
    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    print(f"[*] Dataset root: {root}")
    print(f"[*] Splits: {splits}")
    print(f"[*] Classes: {classes}")
    print("")

    grand_total = 0
    class_totals = {cls: 0 for cls in classes}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            msg = f"[WARN] Missing split dir: {split_dir}"
            if args.strict:
                raise RuntimeError(msg)
            print(msg)
            continue

        print(f"[{split}]")
        split_total = 0
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                msg = f"  - {cls}: MISSING ({cls_dir})"
                if args.strict:
                    raise RuntimeError(msg)
                print(msg)
                continue

            n = count_images(cls_dir)
            print(f"  - {cls}: {n}")
            split_total += n
            class_totals[cls] += n

        print(f"  => {split} total: {split_total}")
        print("")
        grand_total += split_total

    print("[ALL SPLITS] class totals")
    for cls in classes:
        print(f"  - {cls}: {class_totals[cls]}")
    print(f"[*] Grand total: {grand_total}")


if __name__ == "__main__":
    main()
