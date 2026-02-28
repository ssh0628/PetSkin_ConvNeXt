import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

from timm import create_model
from timm.data import resolve_data_config

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
            img = Image.new("RGB", (self.fallback_size, self.fallback_size))

        if self.transform:
            img = self.transform(img)

        return img, y, path


def strip_module_prefix(state_dict: dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/project/convnext/convnext_relabeled_tiny5/best_model.pth", type=str, help="best_model.pth path")
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy", type=str, help="dir with test_paths.npy/test_labels.npy/classes.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.2, type=float, help="must match training")
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)

    # pairs example: "A1->A3,A3->A1,A2->A3,A5->A6"
    ap.add_argument("--pairs", default="A1->A3,A3->A1,A2->A3,A5->A6,A6->A5,A4->A8", type=str, help="comma-separated pairs like A1->A3,A3->A1")
    ap.add_argument("--per_pair", default=2, type=int, help="max samples per pair to export")

    # Sample selection method
    # margin: descending order of (p_pred - p_true) = 'confidently wrong' samples
    # p_pred: descending order of p_pred
    ap.add_argument("--rank_by", default="margin", choices=["margin", "p_pred"], type=str)

    ap.add_argument("--out_dir", default="/root/project/convnext/convnext_relabeled_tiny5/confusion", type=str, help="output folder to copy images into")
    ap.add_argument("--copy_json", action="store_true", help="also copy paired json if exists")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)

    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert npy_dir.exists(), f"npy_dir not found: {npy_dir}"

    classes_path = npy_dir / "classes.json"
    assert classes_path.exists(), f"classes.json not found in {npy_dir}"
    classes = json.loads(classes_path.read_text(encoding="utf-8")).get("classes", [])
    assert classes, "classes.json has empty classes"
    class2idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # parse pairs
    pairs = []
    if args.pairs.strip():
        for token in args.pairs.split(","):
            token = token.strip()
            if not token:
                continue
            if "->" not in token:
                raise ValueError(f"bad pair format: {token} (use A1->A3)")
            a, b = token.split("->")
            a, b = a.strip(), b.strip()
            if a not in class2idx or b not in class2idx:
                raise ValueError(f"unknown class in pair: {token} (classes={classes})")
            pairs.append((a, b))
    else:
        # Default: Major confusion pairs mentioned by user
        pairs = [("A1", "A3"), ("A3", "A1"), ("A2", "A3"), ("A5", "A6"), ("A6", "A5"), ("A4", "A8"), ("A8", "A4")]

    # device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

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

    data_config = resolve_data_config({}, model=model)
    tfm = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])

    ds = NPYPathDataset(npy_dir, args.split, transform=tfm, fallback_size=args.imgsz)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # bucket: (true, pred) -> list of dicts
    buckets = {(a, b): [] for (a, b) in pairs}

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            y = y.numpy()

            logits = model(x)
            probs = softmax(logits).detach().cpu().numpy()
            pred = probs.argmax(axis=1)

            for i in range(len(y)):
                t = int(y[i])
                p = int(pred[i])

                true_name = classes[t]
                pred_name = classes[p]

                key = (true_name, pred_name)
                if key not in buckets:
                    continue
                p_pred = float(probs[i, p])
                p_true = float(probs[i, t])
                margin = p_pred - p_true

                buckets[key].append({
                    "path": paths[i],
                    "true": true_name,
                    "pred": pred_name,
                    "p_pred": p_pred,
                    "p_true": p_true,
                    "margin": margin,
                })

    ensure_dir(out_dir)

    summary = {}
    for (a, b), items in buckets.items():
        # sort
        if args.rank_by == "margin":
            items.sort(key=lambda d: d["margin"], reverse=True)
        else:
            items.sort(key=lambda d: d["p_pred"], reverse=True)

        picked = items[: args.per_pair]
        pair_dir = ensure_dir(out_dir / f"true_{a}__pred_{b}")
        meta_lines = []

        for k, d in enumerate(picked):
            src = Path(d["path"])
            # Prevent filename collision
            dst_name = f"{k:04d}__p_pred{d['p_pred']:.3f}__p_true{d['p_true']:.3f}__margin{d['margin']:.3f}__{src.name}"
            dst = pair_dir / dst_name
            safe_copy(src, dst)

            if args.copy_json:
                # json pairing: same stem .json or .JSON or name+".json"
                candidates = [
                    src.with_suffix(".json"),
                    src.with_suffix(".JSON"),
                    src.with_name(src.name + ".json"),
                    src.with_name(src.name + ".JSON"),
                ]
                for jp in candidates:
                    if jp.exists():
                        safe_copy(jp, pair_dir / (dst.stem + ".json"))
                        break

            meta_lines.append(json.dumps(d, ensure_ascii=False))

        (pair_dir / "meta.jsonl").write_text("\n".join(meta_lines), encoding="utf-8")
        summary[f"{a}->{b}"] = {"total_misclassified": len(items), "exported": len(picked)}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] exported to: {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()