# inference/multi_roi/majority_voting.py
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
    # clamp/shift 후에도 최소 1픽셀 이상 보장
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
    # Map to [-max_shift, max_shift] (deterministic per path)
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
    min_side = min(w0, h0)
    if min_side < drop_min_side:
        return None, None, None

    cx = x1 + w0 / 2
    cy = y1 + h0 / 2

    # View 1: ROI = bbox 그대로 crop
    roi_crop = img.crop((x1, y1, x2, y2)).resize((imgsz, imgsz), resample=Image.BICUBIC)

    # View 2: Extended = bbox 중심 고정 + (w,h)만 ext_ratio 배 확장
    w_ext = max(1, int(round(ext_ratio * w0)))
    h_ext = max(1, int(round(ext_ratio * h0)))
    ext_crop = rect_crop_clamp(img, cx, cy, w_ext, h_ext).resize((imgsz, imgsz), resample=Image.BICUBIC)

    # View 3: Offset = extended 크기 기준 deterministic 이동
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


def strip_module_prefix(state_dict: dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


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
    ap.add_argument("--out_dir", default="/root/project/Result/multi_roi_inference/decision_fusion/majority_voting/3view+CE", type=str)

    # 1-view(단일 ROI) / 3-view(ROI+Extended+Offset) 추론 모드
    ap.add_argument("--n_views", default=3, choices=[1, 3], type=int)
    ap.add_argument("--fusion", default="majority_vote", choices=["logit_mean", "prob_mean", "majority_vote"])
    ap.add_argument("--tie_break", default="logit_sum", choices=["roi", "logit_sum"])
    # bbox가 너무 작은 샘플을 드롭/대체하는 최소 기준
    ap.add_argument("--drop_min_side", default=20, type=int)
    # extended view 배율: bbox (w,h)에 곱해 확장 (예: 2.0이면 가로/세로 2배)
    ap.add_argument("--ext_ratio", default=1.2, type=float)
    # extended crop 최소 크기(px): ROI가 작아도 ext가 너무 작아지지 않게 보정, 0이면 비활성화
    ap.add_argument("--k_offset", default=0.1, type=float)
    # offset 생성 방식: fixed(고정 이동) 또는 hash(샘플별 고정 pseudo-random)
    ap.add_argument("--offset_mode", default="hash", choices=["fixed", "hash"])
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert npy_dir.exists(), f"npy_dir not found: {npy_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes_path = npy_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json not found in {npy_dir}")
    meta = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    if not classes:
        raise ValueError("classes.json exists but 'classes' is empty")
    num_classes = len(classes)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"[INFO] device={device}")
    print(f"[INFO] split={args.split}, n_views={args.n_views}, fusion={args.fusion}, offset_mode={args.offset_mode}")
    print(f"[INFO] num_classes={num_classes}")

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

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_roi, x_ext, x_off, y in loader:
            x_roi = x_roi.to(device, non_blocking=True)
            x_ext = x_ext.to(device, non_blocking=True)
            x_off = x_off.to(device, non_blocking=True)

            z_roi = model(x_roi)
            if args.n_views == 1:
                z = z_roi
                pred = z.argmax(1)
            else:
                z_ext = model(x_ext)
                z_off = model(x_off)
                if args.fusion == "prob_mean":
                    p = (torch.softmax(z_roi, dim=1) + torch.softmax(z_ext, dim=1) + torch.softmax(z_off, dim=1)) / 3.0
                    z = p
                    pred = z.argmax(1)
                elif args.fusion == "majority_vote":
                    pred_roi = z_roi.argmax(1)
                    pred_ext = z_ext.argmax(1)
                    pred_off = z_off.argmax(1)
                    view_preds = torch.stack([pred_roi, pred_ext, pred_off], dim=1)  # [B, 3]

                    # Vote counts per class for each sample: [B, C]
                    vote_counts = torch.nn.functional.one_hot(view_preds, num_classes=num_classes).sum(dim=1)
                    max_counts, maj_pred = vote_counts.max(dim=1)

                    # Tie for 3-view means all three votes are different (max count == 1)
                    tie_mask = max_counts.eq(1)
                    if args.tie_break == "roi":
                        tie_pred = pred_roi
                    else:
                        tie_pred = (z_roi + z_ext + z_off).argmax(1)
                    pred = torch.where(tie_mask, tie_pred, maj_pred)
                else:
                    z = (z_roi + z_ext + z_off) / 3.0
                    pred = z.argmax(1)
            pred = pred.detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)

    print(f"[EVAL] Accuracy={acc:.4f} (N={len(y_true)})")
    print(f"[EVAL] Macro P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print("\n[Classification Report]")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = normalize_rows(cm)

    (out_dir / "eval_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "eval_results.txt").write_text(
        f"Accuracy: {acc:.6f}\n"
        f"Macro P:  {macro_p:.6f}\n"
        f"Macro R:  {macro_r:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n\n"
        f"{report}",
        encoding="utf-8",
    )
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(out_dir / "cm_raw.npy", cm)
    np.save(out_dir / "cm_row_norm.npy", cm_norm)
    plot_cm(cm, classes, out_dir / "cm_raw.png", "Confusion Matrix (Raw)")
    plot_cm(cm_norm, classes, out_dir / "cm_row_norm.png", "Confusion Matrix (Row-Normalized)")

    print(f"[SAVED] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
