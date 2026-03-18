import argparse
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from timm import create_model
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_json_for_image(img_path: Path) -> Path | None:
    c1 = img_path.with_suffix(".json")
    if c1.exists():
        return c1
    c2 = img_path.with_suffix(".JSON")
    if c2.exists():
        return c2
    c3 = img_path.with_name(img_path.name + ".json")
    if c3.exists():
        return c3
    c4 = img_path.with_name(img_path.name + ".JSON")
    if c4.exists():
        return c4
    return None


def extract_roi_box(json_path: Path | None, img_w: int, img_h: int):
    if not json_path or not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "labelingInfo" not in data:
            return None
        for info_item in data["labelingInfo"]:
            box_info = info_item.get("box")
            if not box_info:
                continue
            locations = box_info.get("location") or []
            if not locations:
                continue
            loc = locations[0]
            x = int(loc.get("x"))
            y = int(loc.get("y"))
            w = int(loc.get("width"))
            h = int(loc.get("height"))
            if w <= 0 or h <= 0:
                continue
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
    except Exception:
        return None
    return None


def square_crop_clamp(img: Image.Image, cx: float, cy: float, window_size: int):
    img_w, img_h = img.size
    half = window_size / 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + window_size
    y2 = y1 + window_size

    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 -= shift
    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 -= shift

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return img.crop((x1, y1, x2, y2))


def clamp_center_for_bbox(off_c: float, box_lo: int, box_hi: int, crop_size: int, img_size: int):
    half = crop_size / 2.0
    lo_img = half
    hi_img = img_size - half
    lo_box = box_hi - half
    hi_box = box_lo + half
    lo = max(lo_img, lo_box)
    hi = min(hi_img, hi_box)
    if lo <= hi:
        return max(lo, min(off_c, hi))
    if lo_img <= hi_img:
        return max(lo_img, min(off_c, hi_img))
    return off_c


def deterministic_hash_offset(path: str, base_size: float, k_offset: float):
    digest = hashlib.md5(path.encode("utf-8")).digest()
    a = int.from_bytes(digest[:8], "big")
    b = int.from_bytes(digest[8:], "big")
    rx = (a / ((1 << 64) - 1)) * 2.0 - 1.0
    ry = (b / ((1 << 64) - 1)) * 2.0 - 1.0
    return rx * k_offset * base_size, ry * k_offset * base_size


def crop_view(img: Image.Image, path: str, roi_box, view_type: int, img_size: int, n_scale: float = 1.2, k_offset: float = 0.1):
    """
    Existing 3-view crop policy reused as-is.
    view_type:
      0 -> tight ROI crop
      1 -> extended square crop
      2 -> offset square crop
    """
    img_w, img_h = img.size
    if roi_box is None:
        return None

    x1, y1, x2, y2 = roi_box
    w0 = x2 - x1
    h0 = y2 - y1
    if min(w0, h0) < 20:
        return None

    cx = x1 + w0 / 2
    cy = y1 + h0 / 2

    if view_type == 0:
        crop = img.crop((x1, y1, x2, y2))
        return crop.resize((img_size, img_size), resample=Image.BICUBIC)

    scale = max(1, int(round(n_scale * max(w0, h0))))
    if view_type == 2:
        dx, dy = deterministic_hash_offset(path, scale, k_offset)
        cx = clamp_center_for_bbox(cx + dx, x1, x2, scale, img_w)
        cy = clamp_center_for_bbox(cy + dy, y1, y2, scale, img_h)

    crop = square_crop_clamp(img, cx, cy, scale)
    return crop.resize((img_size, img_size), resample=Image.BICUBIC)


def filter_small_roi_rows(rows, min_side: int = 20):
    kept = []
    dropped = 0
    for row in tqdm(rows, desc="Filter ROI rows"):
        try:
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                roi_box = extract_roi_box(row["json_path"], img.width, img.height)
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                if min(x2 - x1, y2 - y1) < min_side:
                    dropped += 1
                    continue
            else:
                dropped += 1
                continue
        except Exception:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def strip_module_prefix(state_dict: dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_split_rows(npy_dir: Path, split: str):
    paths_file = npy_dir / f"{split}_paths.npy"
    labels_file = npy_dir / f"{split}_labels.npy"
    if not paths_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Missing: {paths_file} or {labels_file}")

    paths = np.load(paths_file, allow_pickle=False)
    labels = np.load(labels_file, allow_pickle=False).astype(np.int64)
    if len(paths) != len(labels):
        raise ValueError(f"paths/labels length mismatch: {len(paths)} vs {len(labels)}")

    rows = []
    for p, y in zip(paths, labels):
        image_path = Path(p.item() if isinstance(p, np.generic) else str(p))
        rows.append(
            {
                "sample_id": image_path.as_posix(),
                "image_path": image_path,
                "json_path": find_json_for_image(image_path),
                "label": int(y),
            }
        )
    return rows


class ThreeViewCSVDataset(Dataset):
    """
    Returns one sample with three image tensors:
      roi  -> view_type=0
      ext  -> view_type=1
      off  -> view_type=2
    This produces the decision profile DP(x) with shape [3, 8].
    """
    def __init__(self, rows, transform, img_size: int):
        self.rows = rows
        self.transform = transform
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        json_path = row["json_path"]
        label = int(row["label"])

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                roi_box = extract_roi_box(json_path, img.width, img.height)
                views = [crop_view(img, image_path.as_posix(), roi_box, v, self.img_size) for v in range(3)]
                if any(v is None for v in views):
                    raise ValueError("roi too small")
        except Exception:
            raise RuntimeError(f"Failed to load valid sample: {image_path}")

        views = [self.transform(v) for v in views]
        return torch.stack(views, dim=0), label, row["sample_id"]


def main():
    ap = argparse.ArgumentParser(description="Build Decision Templates from a single 3-view ConvNeXt model.")
    ap.add_argument("--ckpt_path", default="/root/project/Result/multi_roi_trained/3view(independent)+WCE_ver3/best_model.pth", type=str, help="best checkpoint path")
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy_sqrt", type=str, help="directory containing train_paths.npy/train_labels.npy")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="split to build DT from")
    ap.add_argument("--save_dt_path", default="/root/project/Result/DT/WCE_ver3_DT", type=str, help="output .npy path for DT, shape [8,3,8]")
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float)
    ap.add_argument("--num_classes", default=8, type=int)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ckpt_path = Path(args.ckpt_path)
    npy_dir = Path(args.npy_dir)
    save_dt_path = Path(args.save_dt_path)

    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"
    assert npy_dir.exists(), f"npy_dir not found: {npy_dir}"

    classes_path = npy_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json not found in {npy_dir}")
    meta = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    if len(classes) != args.num_classes:
        raise ValueError(f"--num_classes={args.num_classes} but classes.json has {len(classes)} classes")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model = create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
    ).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    data_config = resolve_data_config({}, model=model)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])

    rows = load_split_rows(npy_dir, args.split)
    rows, dropped_small = filter_small_roi_rows(rows, min_side=20)
    print(f"[INFO] dropped_small_roi(<20px)={dropped_small}")
    ds = ThreeViewCSVDataset(rows, tfm, args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dt_sum = np.zeros((args.num_classes, 3, args.num_classes), dtype=np.float64)
    counts = np.zeros(args.num_classes, dtype=np.int64)

    with torch.no_grad():
        for views, y, _sample_ids in tqdm(loader, desc="Build DP"):
            # views: [B, 3, C_in, H, W]
            batch_probs = []
            for v in range(3):
                x = views[:, v].to(device, non_blocking=True)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_probs.append(probs)

            # DP batch shape: [B, 3, 8]
            dp_batch = np.stack(batch_probs, axis=1)
            y_np = y.numpy()
            for i in range(dp_batch.shape[0]):
                cls = int(y_np[i])
                dt_sum[cls] += dp_batch[i]
                counts[cls] += 1

    for cls_idx, cls_name in enumerate(classes):
        print(f"[COUNT] {cls_name}: {int(counts[cls_idx])}")
        if counts[cls_idx] == 0:
            raise ValueError(f"class '{cls_name}' has zero samples in split '{args.split}'")

    dt = dt_sum / counts[:, None, None]
    save_dt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_dt_path, dt.astype(np.float32))
    print(f"[SAVED] DT -> {save_dt_path}")


if __name__ == "__main__":
    main()