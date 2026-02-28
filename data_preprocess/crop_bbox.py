import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    p = argparse.ArgumentParser("Crop by bbox (no padding) and resize")
    p.add_argument("--src", type=str, default="/root/project/dataset/dataset",
                   help="Input dataset root containing train/val/test folders")
    p.add_argument("--dst", type=str, default="/root/project/dataset/dataset_cropped",
                   help="Output dataset root (same structure)")
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--classes", type=str, default="A1,A2,A3,A4,A5,A6,A7,A8")

    # resize option: 0 disables resize
    p.add_argument("--resize", type=int, default=224,
                   help="If >0, resize cropped image to (resize, resize). 0 disables resize.")

    # Policy for missing/invalid bbox
    p.add_argument("--fallback", type=str, default="keep", choices=["keep", "drop"],
                   help="If bbox missing/invalid: keep original image or drop sample")

    # Copy JSON?
    p.add_argument("--copy_json", action="store_true", help="Copy JSON alongside cropped images")

    return p.parse_args()


def find_json_for_image(img_path: Path) -> Path | None:
    # Standard pairing: xxx.jpg -> xxx.json / xxx.JSON
    c1 = img_path.with_suffix(".json")
    if c1.exists():
        return c1
    c2 = img_path.with_suffix(".JSON")
    if c2.exists():
        return c2

    # Double extension: xxx.jpg.json
    c3 = img_path.with_name(img_path.name + ".json")
    if c3.exists():
        return c3
    c4 = img_path.with_name(img_path.name + ".JSON")
    if c4.exists():
        return c4

    return None


def parse_bbox(json_path: Path):
    """
    Extract bbox from JSON.
    Expected format:
      labelingInfo -> [ ... ] -> box -> location[0] -> x, y, width, height
    Returns: (x, y, w, h) or None
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        if "labelingInfo" in data:
            for item in data["labelingInfo"]:
                box = item.get("box")
                if not box:
                    continue
                locs = box.get("location") or []
                if not locs:
                    continue
                loc = locs[0]
                x = int(loc.get("x"))
                y = int(loc.get("y"))
                w = int(loc.get("width"))
                h = int(loc.get("height"))
                if w > 0 and h > 0:
                    return (x, y, w, h)
    except Exception:
        return None

    return None


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def crop_by_bbox(im: Image.Image, bbox):
    """
    bbox: (x, y, w, h)
    No padding. Crop exactly to bbox.
    Clamp if bbox exceeds image boundaries.
    """
    if bbox is None:
        return None

    x, y, w, h = bbox
    W, H = im.size

    x1 = clamp(x, 0, W)
    y1 = clamp(y, 0, H)
    x2 = clamp(x + w, 0, W)
    y2 = clamp(y + h, 0, H)

    # Check if valid crop area
    if x2 <= x1 or y2 <= y1:
        return None

    return im.crop((x1, y1, x2, y2))


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # Check basic structure
    for sp in splits:
        if not (src_root / sp).exists():
            raise RuntimeError(f"Missing split folder: {src_root/sp}")

    dst_root.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)

    for sp in splits:
        for cls in classes:
            src_dir = src_root / sp / cls
            if not src_dir.exists():
                stats["missing_class_dir"] += 1
                continue

            imgs = [p for p in src_dir.iterdir() if p.is_file() and p.suffix in IMG_EXTS]
            if not imgs:
                stats["empty_class_dir"] += 1
                continue

            out_dir = dst_root / sp / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(imgs, desc=f"{sp}/{cls}", leave=False):
                json_path = find_json_for_image(img_path)
                if json_path is None:
                    stats["no_json"] += 1
                    if args.fallback == "drop":
                        continue
                    # keep original
                    try:
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")
                            if args.resize and args.resize > 0:
                                im = im.resize((args.resize, args.resize), resample=Image.BILINEAR)
                            im.save(out_dir / img_path.name, quality=95)
                        stats["kept_original"] += 1
                    except Exception:
                        stats["bad_image"] += 1
                    continue

                bbox = parse_bbox(json_path)

                try:
                    with Image.open(img_path) as im:
                        im = im.convert("RGB")
                        cropped = crop_by_bbox(im, bbox)

                        if cropped is None:
                            stats["no_or_bad_bbox"] += 1
                            if args.fallback == "drop":
                                continue
                            cropped = im
                            stats["kept_original"] += 1
                        else:
                            stats["cropped"] += 1

                        if args.resize and args.resize > 0:
                            cropped = cropped.resize((args.resize, args.resize), resample=Image.BILINEAR)

                        out_img = out_dir / img_path.name
                        cropped.save(out_img, quality=95)

                        if args.copy_json:
                            out_json = out_dir / (img_path.stem + ".json")
                            shutil.copy2(json_path, out_json)

                except Exception:
                    stats["bad_image"] += 1
                    continue

    print("\n[DONE] Crop finished.")
    for k in sorted(stats.keys()):
        print(f"- {k}: {stats[k]}")
    print(f"\nOutput dataset root: {dst_root}")


if __name__ == "__main__":
    main()