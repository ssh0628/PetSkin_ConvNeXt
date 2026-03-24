import argparse
import csv
import json
import random
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
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


def normalize_rows(cm: np.ndarray):
    denom = cm.sum(axis=1, keepdims=True).astype(np.float32)
    denom[denom == 0] = 1.0
    return cm.astype(np.float32) / denom


def plot_cm(cm, classes, save_path: Path, title: str):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)

    is_float = np.issubdtype(cm.dtype, np.floating)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            s = f"{v:.2f}" if is_float else str(v)
            plt.text(j, i, s, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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


class ThreeCropDataset(Dataset):
    """
    DP(x) is [8, 8] in this order:
      0: roi_a on ROI
      1: roi_b on ROI
      2: ext_a on EXT
      3: ext_b on EXT
      4: off_a on OFF
      5: off_b on OFF
      6: 3view_a on mean logits of ROI/EXT/OFF
      7: 3view_b on mean logits of ROI/EXT/OFF
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
                roi_img = crop_view(img, image_path.as_posix(), roi_box, 0, self.img_size)
                ext_img = crop_view(img, image_path.as_posix(), roi_box, 1, self.img_size)
                off_img = crop_view(img, image_path.as_posix(), roi_box, 2, self.img_size)
                if roi_img is None or ext_img is None or off_img is None:
                    raise ValueError("roi too small")
        except Exception:
            raise RuntimeError(f"Failed to load valid sample: {image_path}")

        return (
            self.transform(roi_img),
            self.transform(ext_img),
            self.transform(off_img),
            label,
            row["sample_id"],
        )


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / float(temperature), dim=1)


def load_model(ckpt_path: Path, model_name: str, num_classes: int, drop_path: float, device: torch.device):
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    ).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def forward_3view_mean(model: torch.nn.Module, roi_img: torch.Tensor, ext_img: torch.Tensor, off_img: torch.Tensor):
    logits_roi = model(roi_img)
    logits_ext = model(ext_img)
    logits_off = model(off_img)
    return (logits_roi + logits_ext + logits_off) / 3.0


def compute_metric(dt_i: np.ndarray, dp_x: np.ndarray, metric: str) -> float:
    if metric == "s2":
        return float(1.0 - np.mean(np.abs(dt_i - dp_x)))
    if metric == "n":
        return float(1.0 - np.mean((dt_i - dp_x) ** 2))
    if metric == "s1":
        denom = np.sum(np.maximum(dt_i, dp_x))
        return float(np.sum(np.minimum(dt_i, dp_x)) / max(denom, 1e-12))
    if metric == "s3":
        term = np.maximum(np.minimum(dt_i, 1.0 - dp_x), np.minimum(1.0 - dt_i, dp_x))
        return float(1.0 - np.mean(term))
    if metric == "i1":
        denom = np.sum(dt_i)
        return float(np.sum(np.minimum(dt_i, dp_x)) / max(denom, 1e-12))
    if metric == "i2":
        return float(1.0 - np.mean(np.maximum(0.0, dt_i - dp_x)))
    if metric == "i3":
        return float(np.mean(np.maximum(1.0 - dt_i, dp_x)))
    raise ValueError(f"Unsupported metric: {metric}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate DT fusion from 8 models: roi/ext/off/3view x 2 seeds.")
    ap.add_argument("--ckpt_roi_a", default="/root/project/Result/single_roi_trained/WCE/best_model.pth", type=str)
    ap.add_argument("--ckpt_roi_b", default="/root/project/Result/single_roi_trained/WCE_2/best_model.pth", type=str)
    ap.add_argument("--ckpt_ext_a", default="/root/project/Result/single_roi_trained/extended_WCE/best_model.pth", type=str)
    ap.add_argument("--ckpt_ext_b", default="/root/project/Result/single_roi_trained/extended_WCE_2/best_model.pth", type=str)
    ap.add_argument("--ckpt_off_a", default="/root/project/Result/single_roi_trained/offset_WCE/best_model.pth", type=str)
    ap.add_argument("--ckpt_off_b", default="/root/project/Result/single_roi_trained/offset_WCE_2/best_model.pth", type=str)
    ap.add_argument("--ckpt_3view_a", default="/root/project/Result/multi_roi_trained/3view(independent)+WCE_ver3/best_model.pth", type=str)
    ap.add_argument("--ckpt_3view_b", default="/root/project/Result/multi_roi_trained/3view(independent)+WCE_ver3_seed_2/best_model.pth", type=str)
    ap.add_argument("--dt_path", default="/root/project/Result/DT/8models_WCE_DT.npy", type=str)
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy_sqrt", type=str)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--metric", default="i1", choices=["s2", "n", "s1", "s3", "i1", "i2", "i3"])
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--out_dir", default="/root/project/Result/multi_roi_inference/DT/8models_WCE_DT/i1_test", type=str)
    ap.add_argument("--save_pred_csv", default="/root/project/Result/multi_roi_inference/DT/8models_WCE_DT/i1_test/pred.csv", type=str)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float)
    ap.add_argument("--num_classes", default=8, type=int)
    ap.add_argument("--temperature_roi_a", default=1.0, type=float)
    ap.add_argument("--temperature_roi_b", default=1.0, type=float)
    ap.add_argument("--temperature_ext_a", default=1.0, type=float)
    ap.add_argument("--temperature_ext_b", default=1.0, type=float)
    ap.add_argument("--temperature_off_a", default=1.0, type=float)
    ap.add_argument("--temperature_off_b", default=1.0, type=float)
    ap.add_argument("--temperature_3view_a", default=1.0, type=float)
    ap.add_argument("--temperature_3view_b", default=1.0, type=float)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ckpt_paths = {
        "roi_a": Path(args.ckpt_roi_a),
        "roi_b": Path(args.ckpt_roi_b),
        "ext_a": Path(args.ckpt_ext_a),
        "ext_b": Path(args.ckpt_ext_b),
        "off_a": Path(args.ckpt_off_a),
        "off_b": Path(args.ckpt_off_b),
        "3view_a": Path(args.ckpt_3view_a),
        "3view_b": Path(args.ckpt_3view_b),
    }
    dt_path = Path(args.dt_path)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path_obj in ckpt_paths.items():
        assert path_obj.exists(), f"{name} not found: {path_obj}"
    assert dt_path.exists(), f"dt_path not found: {dt_path}"
    assert npy_dir.exists(), f"npy_dir not found: {npy_dir}"

    classes_path = npy_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json not found in {npy_dir}")
    meta = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    if len(classes) != args.num_classes:
        raise ValueError(f"--num_classes={args.num_classes} but classes.json has {len(classes)} classes")

    dt = np.load(dt_path)
    if dt.shape != (args.num_classes, 8, args.num_classes):
        raise ValueError(f"DT shape must be ({args.num_classes}, 8, {args.num_classes}), got {dt.shape}")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model_roi_a = load_model(ckpt_paths["roi_a"], args.model_name, args.num_classes, args.drop_path, device)
    model_roi_b = load_model(ckpt_paths["roi_b"], args.model_name, args.num_classes, args.drop_path, device)
    model_ext_a = load_model(ckpt_paths["ext_a"], args.model_name, args.num_classes, args.drop_path, device)
    model_ext_b = load_model(ckpt_paths["ext_b"], args.model_name, args.num_classes, args.drop_path, device)
    model_off_a = load_model(ckpt_paths["off_a"], args.model_name, args.num_classes, args.drop_path, device)
    model_off_b = load_model(ckpt_paths["off_b"], args.model_name, args.num_classes, args.drop_path, device)
    model_3view_a = load_model(ckpt_paths["3view_a"], args.model_name, args.num_classes, args.drop_path, device)
    model_3view_b = load_model(ckpt_paths["3view_b"], args.model_name, args.num_classes, args.drop_path, device)

    data_config = resolve_data_config({}, model=model_roi_a)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])

    rows = load_split_rows(npy_dir, args.split)
    rows, dropped_small = filter_small_roi_rows(rows, min_side=20)
    print(f"[INFO] dropped_small_roi(<20px)={dropped_small}")
    ds = ThreeCropDataset(rows, tfm, args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true_all = []
    y_pred_all = []
    pred_rows = []

    with torch.no_grad():
        for roi_img, ext_img, off_img, y, sample_ids in tqdm(loader, desc="DT Eval"):
            roi_img = roi_img.to(device, non_blocking=True)
            ext_img = ext_img.to(device, non_blocking=True)
            off_img = off_img.to(device, non_blocking=True)

            prob_roi_a = softmax_with_temperature(model_roi_a(roi_img), args.temperature_roi_a).cpu().numpy()
            prob_roi_b = softmax_with_temperature(model_roi_b(roi_img), args.temperature_roi_b).cpu().numpy()
            prob_ext_a = softmax_with_temperature(model_ext_a(ext_img), args.temperature_ext_a).cpu().numpy()
            prob_ext_b = softmax_with_temperature(model_ext_b(ext_img), args.temperature_ext_b).cpu().numpy()
            prob_off_a = softmax_with_temperature(model_off_a(off_img), args.temperature_off_a).cpu().numpy()
            prob_off_b = softmax_with_temperature(model_off_b(off_img), args.temperature_off_b).cpu().numpy()

            logits_3view_a = forward_3view_mean(model_3view_a, roi_img, ext_img, off_img)
            logits_3view_b = forward_3view_mean(model_3view_b, roi_img, ext_img, off_img)
            prob_3view_a = softmax_with_temperature(logits_3view_a, args.temperature_3view_a).cpu().numpy()
            prob_3view_b = softmax_with_temperature(logits_3view_b, args.temperature_3view_b).cpu().numpy()

            dp_batch = np.stack(
                [
                    prob_roi_a,
                    prob_roi_b,
                    prob_ext_a,
                    prob_ext_b,
                    prob_off_a,
                    prob_off_b,
                    prob_3view_a,
                    prob_3view_b,
                ],
                axis=1,
            )
            y_np = y.numpy()

            for i in range(dp_batch.shape[0]):
                dp_x = dp_batch[i]
                scores = np.asarray([compute_metric(dt[c], dp_x, args.metric) for c in range(args.num_classes)], dtype=np.float32)
                y_pred = int(np.argmax(scores))
                y_true = int(y_np[i])

                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

                if args.save_pred_csv:
                    row = {
                        "sample_id": sample_ids[i],
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                    for c in range(args.num_classes):
                        row[f"score_{c}"] = float(scores[c])
                    pred_rows.append(row)

    y_true_arr = np.asarray(y_true_all, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred_all, dtype=np.int64)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_p, macro_r, _macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(args.num_classes)))
    cm_norm = normalize_rows(cm)
    cls_report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(args.num_classes)),
        target_names=classes,
        zero_division=0,
        digits=6,
    )

    print(f"[EVAL] metric={args.metric}")
    print(f"[EVAL] Accuracy={acc:.6f}")
    print(f"[EVAL] MacroF1={macro_f1:.6f}")
    print("[CONFUSION_MATRIX]")
    print(cm)

    (out_dir / "eval_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "eval_results.txt").write_text(
        f"Metric:   {args.metric}\n"
        f"Accuracy: {acc:.6f}\n"
        f"Macro P:  {macro_p:.6f}\n"
        f"Macro R:  {macro_r:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n"
        f"\n"
        f"[Classification Report]\n"
        f"{cls_report}\n",
        encoding="utf-8",
    )
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(out_dir / "cm_raw.npy", cm)
    np.save(out_dir / "cm_row_norm.npy", cm_norm)
    plot_cm(cm, classes, out_dir / "cm_raw.png", "Confusion Matrix (Raw)")
    plot_cm(cm_norm, classes, out_dir / "cm_row_norm.png", "Confusion Matrix (Row-Normalized)")

    if args.save_pred_csv:
        save_pred_csv = Path(args.save_pred_csv)
        save_pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_pred_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["sample_id", "y_true", "y_pred"] + [f"score_{c}" for c in range(args.num_classes)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pred_rows)
        print(f"[SAVED] predictions -> {save_pred_csv}")

    print(f"[SAVED] Results saved to {out_dir}")


if __name__ == "__main__":
    main()