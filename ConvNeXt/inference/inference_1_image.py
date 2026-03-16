import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFile
from timm import create_model
from timm.data import resolve_data_config
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
    c4 = img_path.with_name(img_path.name + ".JSON")
    if c4.exists():
        return c4
    return None


def extract_roi_box(json_path: Path | None, img_w: int, img_h: int):
    if not json_path or not json_path.exists():
        return None
    try:
        import json
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


def strip_module_prefix(state_dict: dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/root/project/dataset/image1.png", type=str, help="path to input image")
    ap.add_argument("--ckpt", default="/root/project/Result/single_roi_trained/WCE/best_model.pth", type=str, help="path to model checkpoint")
    ap.add_argument("--classes", default="A1,A2,A3,A4,A5,A6,A7,A8", type=str, help="comma-separated class names")
    ap.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k", type=str)
    ap.add_argument("--drop_path", default=0.1, type=float)
    ap.add_argument("--imgsz", default=224, type=int)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--topk", default=3, type=int)
    ap.add_argument("--no_bbox", action="store_true", help="disable bbox crop even if paired json exists")
    args = ap.parse_args()

    image_path = Path(args.image)
    ckpt = Path(args.ckpt)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    assert image_path.exists(), f"image not found: {image_path}"
    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    if not classes:
        raise ValueError("--classes is empty")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model = create_model(
        args.model_name,
        pretrained=False,
        num_classes=len(classes),
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

    try:
        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"failed to open image: {image_path} ({e})") from e

    roi_box = None
    if not args.no_bbox:
        roi_box = extract_roi_box(find_json_for_image(image_path), img.width, img.height)
        if roi_box is not None:
            img = img.crop(roi_box)

    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = min(max(1, args.topk), len(classes))
        top_probs, top_idxs = torch.topk(probs, k=topk)

    pred_idx = int(top_idxs[0].item())
    pred_class = classes[pred_idx]

    print(f"[INFO] image={image_path}")
    print(f"[INFO] device={device}")
    print(f"[INFO] bbox_used={roi_box is not None}")
    if roi_box is not None:
        print(f"[INFO] bbox={roi_box}")
    print(f"[PRED] class={pred_class} (idx={pred_idx}) prob={float(top_probs[0].item()):.6f}")
    print("[TOPK]")
    for rank, (idx_t, prob_t) in enumerate(zip(top_idxs.tolist(), top_probs.tolist()), start=1):
        print(f"{rank}. {classes[idx_t]} (idx={idx_t}) prob={float(prob_t):.6f}")


if __name__ == "__main__":
    main()
