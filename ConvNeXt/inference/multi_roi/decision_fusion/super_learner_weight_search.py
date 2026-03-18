# inference/multi_roi/super_learner_weight_search.py
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from timm import create_model
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
    return None


def extract_roi_box(json_path: Path, img_w: int, img_h: int):
    if not json_path or not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "labelingInfo" in data:
            for info_item in data["labelingInfo"]:
                if "box" not in info_item:
                    continue
                box_info = info_item["box"]
                if "location" not in box_info or len(box_info["location"]) == 0:
                    continue
                loc = box_info["location"][0]
                x = int(loc.get("x"))
                y = int(loc.get("y"))
                w = int(loc.get("width"))
                h = int(loc.get("height"))
                if w <= 0 or h <= 0:
                    continue
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
    except Exception:
        return None
    return None


def rect_crop_clamp(img: Image.Image, cx: float, cy: float, crop_w: int, crop_h: int):
    img_w, img_h = img.size
    crop_w = max(1, int(crop_w))
    crop_h = max(1, int(crop_h))
    half_w = crop_w / 2
    half_h = crop_h / 2
    x1 = int(round(cx - half_w))
    y1 = int(round(cy - half_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

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

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
        x1 = max(0, x2 - 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
        y1 = max(0, y2 - 1)

    return img.crop((x1, y1, x2, y2))


def deterministic_hash_offset(path: str, base_w: float, base_h: float, k_offset: float):
    digest = hashlib.md5(path.encode("utf-8")).digest()
    a = int.from_bytes(digest[:8], byteorder="big", signed=False)
    b = int.from_bytes(digest[8:16], byteorder="big", signed=False)
    max_shift_x = k_offset * base_w
    max_shift_y = k_offset * base_h
    rx = (a / ((1 << 64) - 1)) * 2.0 - 1.0
    ry = (b / ((1 << 64) - 1)) * 2.0 - 1.0
    return rx * max_shift_x, ry * max_shift_y


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


def crop_three_views(
    img: Image.Image,
    path: str,
    roi_box,
    imgsz: int,
    drop_min_side: int,
    ext_ratio: float,
    k_offset: float,
    offset_mode: str,
):
    if roi_box is None:
        return None, None, None

    x1, y1, x2, y2 = roi_box
    w0 = x2 - x1
    h0 = y2 - y1
    if min(w0, h0) < drop_min_side:
        return None, None, None

    cx = x1 + w0 / 2
    cy = y1 + h0 / 2

    roi_crop = img.crop((x1, y1, x2, y2)).resize((imgsz, imgsz), resample=Image.BICUBIC)

    w_ext = max(1, int(round(ext_ratio * w0)))
    h_ext = max(1, int(round(ext_ratio * h0)))
    ext_crop = rect_crop_clamp(img, cx, cy, w_ext, h_ext).resize((imgsz, imgsz), resample=Image.BICUBIC)

    if offset_mode == "hash":
        dx, dy = deterministic_hash_offset(path, w_ext, h_ext, k_offset)
    else:
        dx = k_offset * w_ext
        dy = k_offset * h_ext

    # Shift only within feasible range so lesion bbox stays in view.
    off_cx = clamp_center_for_bbox(cx + dx, x1, x2, w_ext, img.width)
    off_cy = clamp_center_for_bbox(cy + dy, y1, y2, h_ext, img.height)
    off_crop = rect_crop_clamp(img, off_cx, off_cy, w_ext, h_ext).resize((imgsz, imgsz), resample=Image.BICUBIC)

    return roi_crop, ext_crop, off_crop


class NPYPath3ViewDataset(Dataset):
    def __init__(
        self,
        npy_dir: Path,
        split: str,
        transform=None,
        imgsz: int = 224,
        drop_min_side: int = 20,
        ext_ratio: float = 1.2,
        k_offset: float = 0.1,
        offset_mode: str = "hash",
    ):
        self.npy_dir = Path(npy_dir)
        self.transform = transform
        self.imgsz = int(imgsz)
        self.drop_min_side = int(drop_min_side)
        self.ext_ratio = float(ext_ratio)
        self.k_offset = float(k_offset)
        self.offset_mode = offset_mode
        paths_file = self.npy_dir / f"{split}_paths.npy"
        labels_file = self.npy_dir / f"{split}_labels.npy"
        if not paths_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"Missing: {paths_file} or {labels_file}")

        raw_paths = np.load(paths_file, allow_pickle=False)
        raw_labels = np.load(labels_file, allow_pickle=False).astype(np.int64)
        valid_paths = []
        valid_labels = []
        dropped = 0
        for p, y in zip(raw_paths, raw_labels):
            path = p.item() if isinstance(p, np.generic) else str(p)
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    roi_box = extract_roi_box(find_json_for_image(Path(path)), img.width, img.height)
                if roi_box is None:
                    dropped += 1
                    continue
                x1, y1, x2, y2 = roi_box
                if min(x2 - x1, y2 - y1) < self.drop_min_side:
                    dropped += 1
                    continue
                valid_paths.append(path)
                valid_labels.append(int(y))
            except Exception:
                dropped += 1
        self.paths = np.asarray(valid_paths, dtype=object)
        self.labels = np.asarray(valid_labels, dtype=np.int64)
        print(f"[{split.upper()}] valid={len(self.paths)} dropped={dropped}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        path = p.item() if isinstance(p, np.generic) else str(p)
        y = int(self.labels[idx])
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                roi_box = extract_roi_box(find_json_for_image(Path(path)), img.width, img.height)
                roi_img, ext_img, off_img = crop_three_views(
                    img=img,
                    path=path,
                    roi_box=roi_box,
                    imgsz=self.imgsz,
                    drop_min_side=self.drop_min_side,
                    ext_ratio=self.ext_ratio,
                    k_offset=self.k_offset,
                    offset_mode=self.offset_mode,
                )
                if roi_img is None or ext_img is None or off_img is None:
                    raise ValueError(f"invalid roi for {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load valid sample: {path}") from e

        if self.transform:
            roi_img = self.transform(roi_img)
            ext_img = self.transform(ext_img)
            off_img = self.transform(off_img)
        return roi_img, ext_img, off_img, y


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


def score_from_logits(z: np.ndarray, y: np.ndarray, metric: str) -> float:
    pred = z.argmax(axis=1)
    if metric == "macro_f1":
        _, _, macro_f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        return float(macro_f1)
    return float((pred == y).mean())


def make_weight_grid(step: float):
    n = int(round(1.0 / step))
    if abs(n * step - 1.0) > 1e-8:
        raise ValueError(f"grid_step must divide 1 exactly. got step={step}")
    weights = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            weights.append((i * step, j * step, k * step))
    return weights


def search_best_weights(z_roi: np.ndarray, z_ext: np.ndarray, z_off: np.ndarray, y: np.ndarray, metric: str, step: float):
    best_w = (1 / 3, 1 / 3, 1 / 3)
    best_v = -1e18
    for w_roi, w_ext, w_off in make_weight_grid(step):
        z = (w_roi * z_roi) + (w_ext * z_ext) + (w_off * z_off)
        v = score_from_logits(z, y, metric)
        if v > best_v:
            best_v = v
            best_w = (w_roi, w_ext, w_off)
    return best_w, float(best_v)


def run_forward(model, loader, device, n_views: int):
    z_roi_all, z_ext_all, z_off_all, y_all = [], [], [], []
    with torch.no_grad():
        for x_roi, x_ext, x_off, y in loader:
            x_roi = x_roi.to(device, non_blocking=True)
            z_roi = model(x_roi).detach().cpu().numpy()
            if n_views == 1:
                z_ext = z_roi
                z_off = z_roi
            else:
                x_ext = x_ext.to(device, non_blocking=True)
                x_off = x_off.to(device, non_blocking=True)
                z_ext = model(x_ext).detach().cpu().numpy()
                z_off = model(x_off).detach().cpu().numpy()
            z_roi_all.append(z_roi)
            z_ext_all.append(z_ext)
            z_off_all.append(z_off)
            y_all.append(y.numpy())

    return (
        np.concatenate(z_roi_all, axis=0),
        np.concatenate(z_ext_all, axis=0),
        np.concatenate(z_off_all, axis=0),
        np.concatenate(y_all, axis=0),
    )


def evaluate_and_save(z: np.ndarray, y_true: np.ndarray, classes, out_dir: Path, args):
    y_pred = z.argmax(axis=1)
    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)

    print(f"[EVAL] Accuracy={acc:.4f} (N={len(y_true)})")
    print(f"[EVAL] Macro P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print("\n[Classification Report]")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = normalize_rows(cm)
    (out_dir / "eval_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "eval_results.txt").write_text(
        f"Accuracy: {acc:.6f}\n"
        f"Macro P:  {macro_p:.6f}\n"
        f"Macro R:  {macro_r:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n\n"
        f"{report}",
        encoding="utf-8",
    )
    np.save(out_dir / "cm_raw.npy", cm)
    np.save(out_dir / "cm_row_norm.npy", cm_norm)
    plot_cm(cm, classes, out_dir / "cm_raw.png", "Confusion Matrix (Raw)")
    plot_cm(cm_norm, classes, out_dir / "cm_row_norm.png", "Confusion Matrix (Row-Normalized)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/project/Result/multi_roi_trained/3view(independent)+CE/best_model.pth", type=str)
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy_sqrt", type=str)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float)
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--out_dir", default="/root/project/Result/multi_roi_inference/decision_fusion/super_learner/3view+CE", type=str)

    ap.add_argument("--n_views", default=3, choices=[1, 3], type=int)
    ap.add_argument("--fusion", default="weighted", choices=["logit_mean", "prob_mean", "weighted"])
    ap.add_argument("--offset_mode", default="hash", choices=["fixed", "hash"])
    ap.add_argument("--drop_min_side", default=20, type=int)
    ap.add_argument("--ext_ratio", default=1.2, type=float)
    ap.add_argument("--k_offset", default=0.1, type=float)
    ap.add_argument("--search_weights", default="search")
    ap.add_argument("--grid_step", default=0.01, type=float)
    ap.add_argument("--metric", default="macro_f1", choices=["acc", "macro_f1"])
    ap.add_argument("--cache_logits", action="store_true")
    ap.add_argument("--save_logits_path", default="/root/project/Result/multi_roi_inference/decision_fusion/super_learner/3view+CE", type=str)
    ap.add_argument("--load_logits_path", default="", type=str)
    ap.add_argument("--save_weights", default="/root/project/Result/multi_roi_inference/decision_fusion/super_learner/3view+CE/weights.json", type=str)
    ap.add_argument("--use_weights", default="", type=str)

    args = ap.parse_args()
    if (args.search_weights or args.use_weights) and args.fusion != "weighted":
        print(f"[INFO] fusion changed to 'weighted' (search/use weights requested, was '{args.fusion}')")
        args.fusion = "weighted"

    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes_path = npy_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json not found in {npy_dir}")
    classes = json.loads(classes_path.read_text(encoding="utf-8")).get("classes", [])
    if not classes:
        raise ValueError("classes.json exists but 'classes' is empty")
    num_classes = len(classes)

    z_roi = z_ext = z_off = y_true = None
    if args.load_logits_path:
        lp = Path(args.load_logits_path)
        pack = np.load(lp)
        z_roi = pack["z_roi"]
        z_ext = pack["z_ext"]
        z_off = pack["z_off"]
        y_true = pack["y"]
        print(f"[INFO] loaded logits from {lp} (N={len(y_true)})")
    else:
        if args.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")
        print(f"[INFO] device={device}, split={args.split}, n_views={args.n_views}, fusion={args.fusion}")

        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt}")

        model = create_model(
            args.model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=args.drop_path,
        ).to(device)
        state = torch.load(str(ckpt), map_location="cpu")
        state = strip_module_prefix(state)
        model.load_state_dict(state, strict=True)
        model.eval()

        data_config = resolve_data_config({}, model=model)
        tfm = transforms.Compose(
            [
                transforms.Resize((args.imgsz, args.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(data_config["mean"], data_config["std"]),
            ]
        )

        ds = NPYPath3ViewDataset(
            npy_dir=npy_dir,
            split=args.split,
            transform=tfm,
            imgsz=args.imgsz,
            drop_min_side=args.drop_min_side,
            ext_ratio=args.ext_ratio,
            k_offset=args.k_offset,
            offset_mode=args.offset_mode,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        z_roi, z_ext, z_off, y_true = run_forward(model, loader, device, args.n_views)
        print(f"[INFO] forward done: N={len(y_true)}")

    if args.cache_logits or args.save_logits_path:
        save_path = Path(args.save_logits_path) if args.save_logits_path else (out_dir / f"{args.split}_logits.npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, z_roi=z_roi, z_ext=z_ext, z_off=z_off, y=y_true)
        print(f"[SAVED] logits cache -> {save_path}")

    # default weights
    w_roi, w_ext, w_off = (1.0, 0.0, 0.0) if args.n_views == 1 else (1 / 3, 1 / 3, 1 / 3)

    if args.use_weights:
        wp = Path(args.use_weights)
        info = json.loads(wp.read_text(encoding="utf-8"))
        w_roi = float(info["w_roi"])
        w_ext = float(info["w_ext"])
        w_off = float(info["w_off"])
        s = w_roi + w_ext + w_off
        if s <= 0:
            raise ValueError("Invalid weights sum <= 0")
        w_roi, w_ext, w_off = w_roi / s, w_ext / s, w_off / s
        print(f"[INFO] loaded weights from {wp}: ({w_roi:.4f}, {w_ext:.4f}, {w_off:.4f})")

    if args.search_weights:
        if args.n_views != 3:
            raise ValueError("--search_weights is only valid when --n_views=3")
        best_w, best_v = search_best_weights(z_roi, z_ext, z_off, y_true, args.metric, args.grid_step)
        w_roi, w_ext, w_off = best_w
        print(f"[BEST] metric={args.metric}, value={best_v:.6f}")
        print(f"[BEST] weights: w_roi={w_roi:.4f}, w_ext={w_ext:.4f}, w_off={w_off:.4f}")
        if args.save_weights:
            sp = Path(args.save_weights)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(
                json.dumps(
                    {
                        "w_roi": w_roi,
                        "w_ext": w_ext,
                        "w_off": w_off,
                        "metric": args.metric,
                        "value": best_v,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[SAVED] best weights -> {sp}")

    if args.n_views == 1:
        z_final = z_roi
    else:
        if args.fusion == "prob_mean":
            p = (
                torch.softmax(torch.from_numpy(z_roi), dim=1)
                + torch.softmax(torch.from_numpy(z_ext), dim=1)
                + torch.softmax(torch.from_numpy(z_off), dim=1)
            ) / 3.0
            z_final = p.numpy()
        elif args.fusion == "weighted":
            z_final = (w_roi * z_roi) + (w_ext * z_ext) + (w_off * z_off)
            print(f"[INFO] weighted fusion: w_roi={w_roi:.4f}, w_ext={w_ext:.4f}, w_off={w_off:.4f}")
        else:
            z_final = (z_roi + z_ext + z_off) / 3.0

    evaluate_and_save(z_final, y_true, classes, out_dir, args)
    print(f"[SAVED] results -> {out_dir}")


if __name__ == "__main__":
    main()
"""
python /Users/sonseunghyeon/Desktop/creamoff/workspace/Custom-Co-Correcting/models/ConvNeXt/inference/super_learner_weight_search.py \
  --split val --n_views 3 --search_weights --grid_step 0.1 --metric acc \
  --cache_logits \
  --save_logits_path /Users/sonseunghyeon/Desktop/creamoff/workspace/Custom-Co-Correcting/models/ConvNeXt/inference/val_logits.npz \
  --save_weights /Users/sonseunghyeon/Desktop/creamoff/workspace/Custom-Co-Correcting/models/ConvNeXt/inference/best_weights.json

python /Users/sonseunghyeon/Desktop/creamoff/workspace/Custom-Co-Correcting/models/ConvNeXt/inference/super_learner_weight_search.py \
  --split test --n_views 3 \
  --use_weights /Users/sonseunghyeon/Desktop/creamoff/workspace/Custom-Co-Correcting/models/ConvNeXt/inference/best_weights.json
"""
