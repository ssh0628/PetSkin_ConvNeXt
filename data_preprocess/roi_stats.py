import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Read JSON annotations and report class-wise ROI size statistics."
    )
    p.add_argument(
        "--root",
        type=str,
        default="/root/project/dataset/dataset",
        help="Dataset root containing train/val/test/A1..A8 folders.",
    )
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--classes", type=str, default="A1,A2,A3,A4,A5,A6,A7,A8")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search JSON files under each class folder.",
    )
    return p.parse_args()


def extract_first_valid_roi(data):
    """
    Expected format:
      labelingInfo -> [ ... ] -> box -> location[0] -> width, height
    Returns (w, h) or None.
    """
    infos = data.get("labelingInfo")
    if not isinstance(infos, list):
        return None

    for item in infos:
        box = item.get("box") if isinstance(item, dict) else None
        if not isinstance(box, dict):
            continue
        locations = box.get("location")
        if not isinstance(locations, list) or not locations:
            continue

        for loc in locations:
            if not isinstance(loc, dict):
                continue
            try:
                w = int(loc.get("width"))
                h = int(loc.get("height"))
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                return w, h
    return None


def iter_json_files(class_dir: Path, recursive: bool):
    if recursive:
        yield from class_dir.rglob("*.json")
        yield from class_dir.rglob("*.JSON")
    else:
        yield from class_dir.glob("*.json")
        yield from class_dir.glob("*.JSON")


def new_stats():
    return {
        "json_total": 0,
        "roi_valid": 0,
        "roi_invalid": 0,
        "w_sum": 0.0,
        "h_sum": 0.0,
        "area_sum": 0.0,
        "min_side_sum": 0.0,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    print(f"[*] Root: {root}")
    print(f"[*] Splits: {splits}")
    print(f"[*] Classes: {classes}")
    print(f"[*] Recursive JSON search: {args.recursive}")
    print("")

    class_stats = {cls: new_stats() for cls in classes}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[WARN] Missing split: {split_dir}")
            continue

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue

            for jp in iter_json_files(cls_dir, args.recursive):
                st = class_stats[cls]
                st["json_total"] += 1

                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    st["roi_invalid"] += 1
                    continue

                roi = extract_first_valid_roi(data)
                if roi is None:
                    st["roi_invalid"] += 1
                    continue

                w, h = roi
                st["roi_valid"] += 1
                st["w_sum"] += float(w)
                st["h_sum"] += float(h)
                st["area_sum"] += float(w * h)
                st["min_side_sum"] += float(min(w, h))

    print("[Class-wise ROI stats]")
    for cls in classes:
        st = class_stats[cls]
        n = st["roi_valid"]
        if n == 0:
            print(
                f"- {cls}: valid_roi=0 json_total={st['json_total']} roi_invalid={st['roi_invalid']}"
            )
            continue

        avg_w = st["w_sum"] / n
        avg_h = st["h_sum"] / n
        avg_area = st["area_sum"] / n
        avg_min_side = st["min_side_sum"] / n
        print(
            f"- {cls}: valid_roi={n} json_total={st['json_total']} "
            f"avg_w={avg_w:.2f} avg_h={avg_h:.2f} avg_area={avg_area:.2f} "
            f"avg_min_side={avg_min_side:.2f} roi_invalid={st['roi_invalid']}"
        )


if __name__ == "__main__":
    main()
