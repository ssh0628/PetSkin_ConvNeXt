# multi-roi_train/convnext_view_independent_WCE.py
import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# -------------------------
# ROI view config (constants)
# -------------------------
DROP_MIN_SIDE = 20
SMALL_ROI_MIN_SIDE = 80
N_SCALE = 2.0
K_OFFSET = 0.1
USE_OFF_VIEW = True
NUM_VIEWS = 3  # roi, extended, offset (when USE_OFF_VIEW)

# -------------------------
# ROI helpers (JSON -> bbox)
# -------------------------
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
                if "box" in info_item:
                    box_info = info_item["box"]
                    if "location" in box_info and len(box_info["location"]) > 0:
                        loc = box_info["location"][0]
                        x = int(loc.get("x"))
                        y = int(loc.get("y"))
                        w = int(loc.get("width"))
                        h = int(loc.get("height"))
                        if w > 0 and h > 0:
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


def crop_view(img: Image.Image, roi_box, view_type: int, img_size: int):
    img_w, img_h = img.size
    if roi_box is None:
        return img.resize((img_size, img_size), resample=Image.BICUBIC)

    x1, y1, x2, y2 = roi_box
    w0 = x2 - x1
    h0 = y2 - y1
    min_side = min(w0, h0)
    if min_side < DROP_MIN_SIDE:
        return None

    cx = x1 + w0 / 2
    cy = y1 + h0 / 2

    # Small ROI: use scale-based extended window instead of tight ROI.
    if min_side < SMALL_ROI_MIN_SIDE:
        view_type = 1  # treat as extended

    if view_type == 0:
        crop = img.crop((x1, y1, x2, y2))
        return crop.resize((img_size, img_size), resample=Image.BICUBIC)

    # extended / offset: scale-based square window
    scale = max(img_size, int(round(N_SCALE * max(w0, h0))))
    if view_type == 2:
        dx = random.uniform(-K_OFFSET * scale, K_OFFSET * scale)
        dy = random.uniform(-K_OFFSET * scale, K_OFFSET * scale)
        cx += dx
        cy += dy

    crop = square_crop_clamp(img, cx, cy, scale)
    return crop.resize((img_size, img_size), resample=Image.BICUBIC)
# =========================
# 0. PIL / CUDA / AMP 설정
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Safe AMP setup
amp_device = "cuda" if device.type == "cuda" else "cpu"
amp_dtype = torch.float16 # Default safe choice
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16

print(f"AMP Config: device={amp_device}, dtype={amp_dtype}")

# =========================
# 1. 설정
# =========================
NPY_DIR = "/root/project/dataset/cache_npy_sqrt"
SAVE_DIR = "/root/project/convnext/convnext_sqrt_roi_1_WCE"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

NUM_CLASSES = 8
BATCH_SIZE = 256
NUM_EPOCHS = 200
IMG_SIZE = 224
RANDOM_SEED = 0
PATIENCE = 20
Freeze = 5
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_CHANNELS_LAST = True
USE_COMPILE = False

LOSS_MODE = "wce"  # "ce", "wce", "focal"
FOCAL_GAMMA = 2.0
FOCAL_USE_ALPHA = True
FOCAL_ALPHA_MODE = "inv_sqrt"  # "inv", "inv_sqrt"

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

WEIGHT_DECAY = 0.1
LR1 = 1e-3
LR2 = 3e-5

pretrained = True
model_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
# model_name = "convnextv2_femto.fcmae_ft_in1k"
drop_path_rate = 0.2

print(f"[Info] Reading Data from: {NPY_DIR}")

# -------------------------
# ROI (JSON -> bbox)
# -------------------------
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
                if "box" in info_item:
                    box_info = info_item["box"]
                    if "location" in box_info and len(box_info["location"]) > 0:
                        loc = box_info["location"][0]
                        x = int(loc.get("x"))
                        y = int(loc.get("y"))
                        w = int(loc.get("width"))
                        h = int(loc.get("height"))
                        if w > 0 and h > 0:
                            x1, y1 = x, y
                            x2, y2 = x + w, y + h
                            # clamp to image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img_w, x2)
                            y2 = min(img_h, y2)
                            if x2 > x1 and y2 > y1:
                                return (x1, y1, x2, y2)
    except Exception:
        return None
    return None


def build_class_weights(labels, num_classes, mode="inv_sqrt"):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    if mode == "inv":
        weights = 1.0 / counts
    else:
        weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        return loss.mean()
    
# =========================
# 1.1 Class Check
# =========================
cls_json_path = os.path.join(NPY_DIR, "classes.json")
if os.path.exists(cls_json_path):
    with open(cls_json_path, 'r') as f:
        meta = json.load(f)
        classes = meta.get("classes", [])
        if len(classes) != NUM_CLASSES:
            print(f"[WARN] classes.json count ({len(classes)}) != NUM_CLASSES ({NUM_CLASSES})")
            print(f"      Overwriting NUM_CLASSES to {len(classes)}")
            NUM_CLASSES = len(classes)
        else:
            print(f"[OK] Verified NUM_CLASSES={NUM_CLASSES}")
else:
    print(f"[WARN] classes.json not found in {NPY_DIR}. Assuming NUM_CLASSES={NUM_CLASSES}")


# =========================
# 2. Dataset Definition (NPY)
# =========================
class NPYPathDataset(Dataset):
    def __init__(self, npy_dir, split, transform=None):
        super().__init__()
        self.transform = transform
        self.npy_dir = Path(npy_dir)
        self.split = split
        
        paths_file = self.npy_dir / f"{split}_paths.npy"
        labels_file = self.npy_dir / f"{split}_labels.npy"
        
        if not paths_file.exists() or not labels_file.exists():
            raise RuntimeError(f"[ERR] Missing NPY files for split '{split}' in {npy_dir}")
            
        self.paths = np.load(paths_file, allow_pickle=True)
        self.labels = np.load(labels_file, allow_pickle=True).astype(np.int64)
        
        print(f"[{split.upper()}] Loaded {len(self.paths)} samples.")

    def __len__(self):
        return len(self.paths) * (3 if USE_OFF_VIEW else 2)

    def __getitem__(self, idx):
        views_per_sample = 3 if USE_OFF_VIEW else 2
        base_idx = idx // views_per_sample
        view_type = idx % views_per_sample
        path = str(self.paths[base_idx])
        label = int(self.labels[base_idx])
        
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                roi_box = extract_roi_box(find_json_for_image(Path(path)), img.width, img.height)
                cropped = crop_view(img, roi_box, view_type, IMG_SIZE)
                if cropped is None:
                    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
                else:
                    img = cropped
        except Exception as e:
            # Fallback for corrupt images (return black or error)
            # Here we print and return a black image to avoid crash
            print(f"Error loading {path}: {e}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, label, idx


# =========================
# 3. Model
# =========================
model = create_model(
    model_name,
    pretrained=pretrained,
    num_classes=NUM_CLASSES,
    drop_path_rate=drop_path_rate
)
if USE_CHANNELS_LAST:
    model = model.to(device, memory_format=torch.channels_last)
else:
    model = model.to(device)

if USE_COMPILE and device.type == "cuda":
    model = torch.compile(model)

# Backbone freeze (Init)
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

data_config = resolve_data_config({}, model=model)

# =========================
# 4. Transform
# =========================
"""
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])
"""
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

# =========================
# 5. Dataset & DataLoader
# =========================
seed_everything = True
if seed_everything:
    # Basic seed setting if needed, though DataLoader handles shuffle
    torch.manual_seed(RANDOM_SEED)

train_dataset = NPYPathDataset(NPY_DIR, "train", transform=train_transform)
val_dataset = NPYPathDataset(NPY_DIR, "val", transform=eval_transform)
test_dataset = NPYPathDataset(NPY_DIR, "test", transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
)

# =========================
# 6. Optimizer / Scheduler
# =========================
class_weights = build_class_weights(train_dataset.labels, NUM_CLASSES, mode=FOCAL_ALPHA_MODE).to(device)
print(f"[Info] LOSS_MODE={LOSS_MODE}, class_weights={class_weights.detach().cpu().tolist()}")

if LOSS_MODE == "ce":
    criterion = nn.CrossEntropyLoss()
elif LOSS_MODE == "wce":
    criterion = nn.CrossEntropyLoss(weight=class_weights)
elif LOSS_MODE == "focal":
    focal_alpha = class_weights if FOCAL_USE_ALPHA else None
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=focal_alpha)
else:
    raise ValueError(f"Unsupported LOSS_MODE: {LOSS_MODE}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR1,
    weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# =========================
# 7. Train Loop
# =========================
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

best_val = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

    # Backbone unfreeze logic
    if epoch == Freeze:
        print(">>> Unfreezing backbone")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-epoch, eta_min=1e-6)

    # ---- Train ----
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y, idx in tqdm(train_loader, desc="Train"):
        if USE_CHANNELS_LAST:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = model(x)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (out.argmax(1) == y).sum().item()
        total += bs

    train_loss = loss_sum / total
    train_acc = correct / total

    scheduler.step()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for x, y, idx in tqdm(val_loader, desc="Val"):
            if USE_CHANNELS_LAST:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
                pred = out.argmax(1)

            bs = y.size(0)
            val_loss_sum += loss.item() * bs
            correct += (pred == y).sum().item()
            total += bs

    val_acc = correct / total
    val_loss = val_loss_sum / total
    lr = optimizer.param_groups[0]["lr"]

    print(f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} LR={lr:.2e}")

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, lr, train_loss, train_acc, val_loss, val_acc])

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        patience_counter = 0
        print(f"New Best Validation Accuracy: {best_val:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early Stopping triggered at Epoch {epoch+1}")
            break

# =========================
# 8. Test
# =========================
print(f"\n[Test] Loading best model from {SAVE_DIR}/best_model.pth")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location="cpu"))
model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x, y, idx in tqdm(test_loader, desc="Test"):
        if USE_CHANNELS_LAST:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
