import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from timm import create_model
from timm.data import resolve_data_config

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True


class NPYPathDataset(Dataset):
    def __init__(self, npy_dir: Path, split: str, transform=None, fallback_size: int = 224):
        self.npy_dir = Path(npy_dir)
        self.transform = transform
        self.fallback_size = int(fallback_size)

        paths_file = self.npy_dir / f"{split}_paths.npy"
        labels_file = self.npy_dir / f"{split}_labels.npy"
        if not paths_file.exists() or not labels_file.exists():
            raise FileNotFoundError(f"Missing: {paths_file} or {labels_file}")

        # allow_pickle=False 권장 (보안/일관성)
        self.paths = np.load(paths_file, allow_pickle=False)
        self.labels = np.load(labels_file, allow_pickle=False).astype(np.int64)

        if len(self.paths) != len(self.labels):
            raise ValueError(f"paths/labels length mismatch: {len(self.paths)} vs {len(self.labels)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        path = p.item() if isinstance(p, np.generic) else str(p)
        y = int(self.labels[idx])

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # 깨진 이미지면 검은 이미지로 대체 (원하면 raise로 바꿔도 됨)
            img = Image.new("RGB", (self.fallback_size, self.fallback_size))

        if self.transform:
            img = self.transform(img)

        return img, y


def normalize_rows(cm: np.ndarray):
    denom = cm.sum(axis=1, keepdims=True).astype(np.float32)
    denom[denom == 0] = 1.0
    return cm.astype(np.float32) / denom


def plot_cm(cm, classes, save_path: Path, title: str):
    plt.figure()
    plt.imshow(cm)  # 색 지정 안 함
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
    """Handle nn.DataParallel 'module.' prefix if present."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/project/convnext/convnext_relabeled_tiny/best_model.pth",
                    type=str, help="path to best_model.pth")
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy", type=str,
                    help="dir containing {split}_paths.npy/{split}_labels.npy and classes.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float, help="should match training")
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--workers", default=16, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--out_dir", default="/root/project/convnext/convnext_relabeled_tiny", type=str,
                    help="output directory")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)

    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert npy_dir.exists(), f"npy_dir not found: {npy_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # classes.json으로 NUM_CLASSES + 클래스명 고정
    classes_path = npy_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json not found in {npy_dir}")

    meta = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = meta.get("classes", [])
    if not classes:
        raise ValueError("classes.json exists but 'classes' is empty")

    num_classes = len(classes)

    # device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"[INFO] device={device}")
    print(f"[INFO] num_classes={num_classes}, classes={classes}")
    print(f"[INFO] evaluating split='{args.split}' from {npy_dir}")
    print(f"[INFO] ckpt={ckpt}")

    # model
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

    # timm mean/std
    data_config = resolve_data_config({}, model=model)
    tfm = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])

    ds = NPYPathDataset(npy_dir, args.split, transform=tfm, fallback_size=args.imgsz)
    pin = (device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            pred = out.argmax(1).detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Metrics
    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )

    print(f"[TEST] Accuracy={acc:.4f} (N={len(y_true)})")
    print(f"[TEST] Macro P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print("\n[Classification Report]")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = normalize_rows(cm)

    # Save
    (out_dir / "eval_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "test_results.txt").write_text(
        f"Accuracy: {acc:.6f}\n"
        f"Macro P:  {macro_p:.6f}\n"
        f"Macro R:  {macro_r:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n\n"
        f"{report}",
        encoding="utf-8"
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