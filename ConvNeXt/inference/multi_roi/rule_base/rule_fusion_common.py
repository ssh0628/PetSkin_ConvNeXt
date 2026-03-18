import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from timm import create_model
from timm.data import resolve_data_config
from torch.utils.data import DataLoader
from torchvision import transforms

from ConvNeXt.inference.multi_roi.decision_fusion.majority_voting import (
    NPYPath3ViewDataset,
    normalize_rows,
    plot_cm,
    strip_module_prefix,
)


def build_parser(default_out_dir: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/project/convnext/convnext_relabeled_tiny/best_model.pth", type=str)
    ap.add_argument("--npy_dir", default="/root/project/dataset/cache_npy", type=str)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float)
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--out_dir", default=default_out_dir, type=str)

    ap.add_argument("--n_views", default=3, choices=[1, 3], type=int)
    ap.add_argument("--drop_min_side", default=20, type=int)
    ap.add_argument("--ext_ratio", default=1.2, type=float)
    ap.add_argument("--k_offset", default=0.1, type=float)
    ap.add_argument("--offset_mode", default="hash", choices=["fixed", "hash"])
    ap.add_argument("--probability_threshold", "--probablity_threshold", dest="probability_threshold", default=0.8, type=float)
    ap.add_argument("--gap_threshold", default=0.2, type=float)
    return ap


def run_rule_fusion(args, rule_name: str, selector: Callable):
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
    print(f"[INFO] split={args.split}, n_views={args.n_views}, fusion={rule_name}, offset_mode={args.offset_mode}")
    print(f"[INFO] thresholds: p={args.probability_threshold}, gap={args.gap_threshold}")
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
    stage_counts = {}

    with torch.no_grad():
        for x_roi, x_ext, x_off, y in loader:
            x_roi = x_roi.to(device, non_blocking=True)
            z_roi = model(x_roi)

            if args.n_views == 1:
                pred_np = z_roi.argmax(1).detach().cpu().numpy()
                for _ in range(pred_np.shape[0]):
                    stage_counts["ROI_ONLY"] = stage_counts.get("ROI_ONLY", 0) + 1
            else:
                x_ext = x_ext.to(device, non_blocking=True)
                x_off = x_off.to(device, non_blocking=True)
                z_ext = model(x_ext)
                z_off = model(x_off)

                preds = []
                for i in range(z_roi.size(0)):
                    scores = torch.stack([z_roi[i], z_ext[i], z_off[i]], dim=0)
                    pred_i, info = selector(
                        scores=scores,
                        probability_threshold=args.probability_threshold,
                        gap_threshold=args.gap_threshold,
                        from_logits=True,
                        return_details=True,
                    )
                    preds.append(pred_i)
                    stage = info.get("stage", info.get("mode", "UNKNOWN"))
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                pred_np = np.asarray(preds, dtype=np.int64)

            y_true.append(y.numpy())
            y_pred.append(pred_np)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)

    print(f"[EVAL] Accuracy={acc:.4f} (N={len(y_true)})")
    print(f"[EVAL] Macro P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f1:.4f}")
    print(f"[EVAL] Stage counts={stage_counts}")
    print("\n[Classification Report]")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = normalize_rows(cm)

    args_dump = vars(args).copy()
    args_dump["fusion"] = rule_name

    (out_dir / "eval_args.json").write_text(
        json.dumps(args_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "eval_results.txt").write_text(
        f"Accuracy: {acc:.6f}\n"
        f"Macro P:  {macro_p:.6f}\n"
        f"Macro R:  {macro_r:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n\n"
        f"Stage counts: {json.dumps(stage_counts, ensure_ascii=False)}\n\n"
        f"{report}",
        encoding="utf-8",
    )
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": classes}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "stage_counts.json").write_text(
        json.dumps(stage_counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    np.save(out_dir / "cm_raw.npy", cm)
    np.save(out_dir / "cm_row_norm.npy", cm_norm)
    plot_cm(cm, classes, out_dir / "cm_raw.png", "Confusion Matrix (Raw)")
    plot_cm(cm_norm, classes, out_dir / "cm_row_norm.png", "Confusion Matrix (Row-Normalized)")

    print(f"[SAVED] Results saved to {out_dir}")
