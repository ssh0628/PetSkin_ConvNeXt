
import argparse
import json
import shutil
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

class ImageProcessor:
    def __init__(self, interpolation=Image.BICUBIC, debug=False):
        self.interpolation = interpolation
        self.debug = debug

    def pad_reflection(self, img, pad_l, pad_t, pad_r, pad_b):
        """
        Applies reflection padding.
        """
        w, h = img.size
        
        def get_reflection(src_img, size, axis):
            src_w, src_h = src_img.size
            dim = src_w if axis == 0 else src_h
            
            if size > dim:
                reps = (size // dim) + 2
                if axis == 0:
                    tiled = Image.new(src_img.mode, (src_w * reps, src_h))
                    for i in range(reps):
                        tiled.paste(src_img, (i * src_w, 0))
                else:
                    tiled = Image.new(src_img.mode, (src_w, src_h * reps))
                    for i in range(reps):
                        tiled.paste(src_img, (0, i * src_h))
                src_img = tiled
                src_w, src_h = src_img.size

            if axis == 0: # Horizontal
                strip = src_img.crop((0, 0, size, src_h))
                reflected = ImageOps.mirror(strip)
                return reflected
            else: # Vertical
                strip = src_img.crop((0, 0, src_w, size))
                reflected = ImageOps.flip(strip)
                return reflected

        # 1. Horizontal Padding
        if pad_l > 0 or pad_r > 0:
            new_w = w + pad_l + pad_r
            h_img = Image.new(img.mode, (new_w, h))
            h_img.paste(img, (pad_l, 0))
            
            if pad_l > 0:
                left_pad = get_reflection(img, pad_l, axis=0)
                if left_pad.width > pad_l:
                    left_pad = left_pad.crop((left_pad.width - pad_l, 0, left_pad.width, h))
                h_img.paste(left_pad, (0, 0))
                
            if pad_r > 0:
                right_src = img
                if pad_r > w:
                     reps = (pad_r // w) + 2
                     tiled = Image.new(img.mode, (w * reps, h))
                     for i in range(reps):
                        tiled.paste(img, (i * w, 0))
                     right_src = tiled
                
                rw, rh = right_src.size
                right_strip = right_src.crop((rw - pad_r, 0, rw, rh))
                right_pad = ImageOps.mirror(right_strip)
                h_img.paste(right_pad, (pad_l + w, 0))
            
            img = h_img
            w = new_w

        # 2. Vertical Padding
        if pad_t > 0 or pad_b > 0:
            new_h = h + pad_t + pad_b
            v_img = Image.new(img.mode, (w, new_h))
            v_img.paste(img, (0, pad_t))
            
            if pad_t > 0:
                top_pad = get_reflection(img, pad_t, axis=1)
                if top_pad.height > pad_t:
                    top_pad = top_pad.crop((0, top_pad.height - pad_t, w, top_pad.height))
                v_img.paste(top_pad, (0, 0))
                
            if pad_b > 0:
                bottom_src = img
                if pad_b > h:
                    reps = (pad_b // h) + 2
                    tiled = Image.new(img.mode, (w, h * reps))
                    for i in range(reps):
                        tiled.paste(img, (0, i * h))
                    bottom_src = tiled
                
                bh_w, bh_h = bottom_src.size
                bottom_strip = bottom_src.crop((0, bh_h - pad_b, bh_w, bh_h))
                bottom_pad = ImageOps.flip(bottom_strip)
                v_img.paste(bottom_pad, (0, pad_t + h))
                
            img = v_img

        return img

    def _square_crop_clamp(self, img, cx, cy, window_size):
        img_w, img_h = img.size
        half = window_size / 2
        
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + window_size
        y2 = y1 + window_size
        
        # Shift Clamp
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

    def context_aware_crop_resize(self, img, roi_box, target_size=(224, 224), small_roi_min_side=112):
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        target_w, target_h = target_size
        target_min_side = min(target_w, target_h)

        center_x = roi_x1 + roi_w / 2
        center_y = roi_y1 + roi_h / 2

        # If ROI is small, use fixed-size center window crop (salvage path).
        if min(roi_w, roi_h) < small_roi_min_side:
            crop_img = self._square_crop_clamp(img, center_x, center_y, target_min_side)
            if crop_img.size != target_size:
                crop_img = crop_img.resize(target_size, self.interpolation)
            return crop_img

        # ROI is already large enough -> simple ROI crop + downscale.
        else:
            crop_img = img.crop(roi_box)
            return crop_img.resize(target_size, self.interpolation)


def parse_args():
    p = argparse.ArgumentParser("Crop using Context-Aware logic (like new_PetSkin.py)")
    p.add_argument("--src", type=str, default="/root/project/dataset/dataset",
                   help="Input dataset root")
    p.add_argument("--dst", type=str, default="/root/project/dataset/dataset_aware_cropped_min_drop",
                   help="Output dataset root")
    p.add_argument("--splits", type=str, default="train,val")
    p.add_argument("--classes", type=str, default="A1,A2,A3,A4,A5,A6,A7,A8")
    p.add_argument("--resize", type=int, default=224, help="Target size (both width and height)")
    p.add_argument("--drop_min_side", type=int, default=40,
                   help="Drop sample if ROI min(width, height) is smaller than this value")
    p.add_argument("--small_roi_min_side", type=int, default=80,
                   help="If ROI min-side is below this value, use fixed 224 center crop salvage")
    p.add_argument("--copy_json", action="store_true", help="Copy JSON alongside cropped images")
    return p.parse_args()


def find_json_for_image(img_path: Path) -> Path | None:
    # 1. Same name .json
    c1 = img_path.with_suffix(".json")
    if c1.exists(): return c1
    # 2. Same name .JSON
    c2 = img_path.with_suffix(".JSON")
    if c2.exists(): return c2
    # 3. .jpg.json
    c3 = img_path.with_name(img_path.name + ".json")
    if c3.exists(): return c3
    return None

def extract_roi_box(json_path: Path, img_w: int, img_h: int):
    """
    Extracts valid ROI box from JSON, intersecting with image bounds.
    Returns: (x1, y1, x2, y2) or None
    """
    if not json_path: return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'labelingInfo' in data:
            for info_item in data['labelingInfo']:
                if 'box' in info_item:
                    box_info = info_item
                    if 'location' in box_info['box'] and len(box_info['box']['location']) > 0:
                        loc = box_info['box']['location'][0]
                        try:
                            x = int(loc.get('x'))
                            y = int(loc.get('y'))
                            w = int(loc.get('width'))
                            h = int(loc.get('height'))
                            
                            if w > 0 and h > 0:
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h
                                
                                # Intersect
                                inter_x1 = max(0, x1)
                                inter_y1 = max(0, y1)
                                inter_x2 = min(img_w, x2)
                                inter_y2 = min(img_h, y2)
                                
                                final_w = inter_x2 - inter_x1
                                final_h = inter_y2 - inter_y1
                                
                                if final_w > 1 and final_h > 1:
                                    return (inter_x1, inter_y1, inter_x2, inter_y2)
                        except (ValueError, TypeError):
                            continue
    except Exception:
        pass
    return None

def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    
    target_size = (args.resize, args.resize)
    processor = ImageProcessor()

    print(f"[*] Source: {src_root}")
    print(f"[*] Dest:   {dst_root}")
    print(f"[*] Resize: {target_size}")
    print(f"[*] Drop min side: {args.drop_min_side}")
    print(f"[*] Small ROI threshold: {args.small_roi_min_side}")

    stats = defaultdict(int)

    for sp in splits:
        for cls in classes:
            src_dir = src_root / sp / cls
            if not src_dir.exists():
                stats["missing_dir"] += 1
                continue

            imgs = [p for p in src_dir.iterdir() if p.is_file() and p.suffix in IMG_EXTS]
            if not imgs:
                stats["empty_dir"] += 1
                continue
            
            out_dir = dst_root / sp / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(imgs, desc=f"{sp}/{cls}", leave=False):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        
                        # ROI Extraction
                        json_path = find_json_for_image(img_path)
                        roi_box = extract_roi_box(json_path, img.width, img.height)
                        
                        if roi_box:
                            roi_w = roi_box[2] - roi_box[0]
                            roi_h = roi_box[3] - roi_box[1]
                            roi_min_side = min(roi_w, roi_h)
                            if roi_min_side < args.drop_min_side:
                                stats["dropped_small_roi"] += 1
                                continue

                            # Apply context-aware logic with small-ROI salvage.
                            final_img = processor.context_aware_crop_resize(
                                img,
                                roi_box,
                                target_size,
                                small_roi_min_side=args.small_roi_min_side,
                            )
                            stats["processed_with_roi"] += 1
                        else:
                            # Fallback Logic (same as new_PetSkin fallback)
                            if img.width < target_size[0] or img.height < target_size[1]:
                                # Treat whole image as ROI for context expansion
                                roi_box = (0, 0, img.width, img.height)
                                final_img = processor.context_aware_crop_resize(img, roi_box, target_size)
                                stats["fallback_small_img"] += 1
                            else:
                                final_img = img.resize(target_size, Image.BICUBIC)
                                stats["fallback_resize"] += 1

                        out_img_path = out_dir / img_path.name
                        final_img.save(out_img_path, quality=95)
                        
                        if args.copy_json and json_path:
                            out_json = out_dir / json_path.name
                            shutil.copy2(json_path, out_json)
                            
                except Exception as e:
                    stats["errors"] += 1
                    # print(f"Error {img_path}: {e}")

    print("\n[DONE] Processing finished.")
    for k in sorted(stats.keys()):
        print(f"- {k}: {stats[k]}")

if __name__ == "__main__":
    main()
