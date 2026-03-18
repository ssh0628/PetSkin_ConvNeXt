import os
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from timm import create_model
from timm.data import resolve_data_config
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# =========================
# ROI view config
# =========================
DROP_MIN_SIDE = 20
N_SCALE = 1.2
K_OFFSET = 0.1
USE_OFF_VIEW = True

# =========================
# Training config
# =========================
NPY_DIR = "/root/project/dataset/cache_npy_sqrt"
SAVE_DIR = "/root/project/Result/multi_roi_trained/A4_train_WCE"
LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

NUM_CLASSES = 8
BATCH_SIZE = 256
NUM_EPOCHS = 200
IMG_SIZE = 224
RANDOM_SEED = 0
PATIENCE = 20
FREEZE_EPOCHS = 5
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_CHANNELS_LAST = True
USE_COMPILE = False

WEIGHT_DECAY = 0.1
LR_HEAD = 1e-3
LR_ALL = 3e-5
PRETRAINED = True
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
DROP_PATH_RATE = 0.2
TARGET_CLASS_NAME = "A4"

os.makedirs(SAVE_DIR, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_device = "cuda" if device.type == "cuda" else "cpu"
amp_dtype = torch.float16
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def find_json_for_image(img_path: Path):
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
                box = info_item.get("box")
                if not box:
                    continue
                locs = box.get("location") or []
                if not locs:
                    continue
                loc = locs[0]
                x = int(loc.get("x"))
                y = int(loc.get("y"))
                w = int(loc.get("width"))
                h = int(loc.get("height"))
                if w <= 0 or h <= 0:
                    continue
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)
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


def crop_view(img: Image.Image, path: str, roi_box, view_type: int, img_size: int):
    img_w, img_h = img.size
    if roi_box is None:
        return None

    x1, y1, x2, y2 = roi_box
    w0 = x2 - x1
    h0 = y2 - y1
    min_side = min(w0, h0)
    if min_side < DROP_MIN_SIDE:
        return None

    cx = x1 + w0 / 2
    cy = y1 + h0 / 2

    if view_type == 0:
        crop = img.crop((x1, y1, x2, y2))
        return crop.resize((img_size, img_size), resample=Image.BICUBIC)

    scale = max(1, int(round(N_SCALE * max(w0, h0))))
    if view_type == 2:
        dx, dy = deterministic_hash_offset(path, scale, K_OFFSET)
        cx = clamp_center_for_bbox(cx + dx, x1, x2, scale, img_w)
        cy = clamp_center_for_bbox(cy + dy, y1, y2, scale, img_h)

    crop = square_crop_clamp(img, cx, cy, scale)
    return crop.resize((img_size, img_size), resample=Image.BICUBIC)


def build_class_weights(labels, num_classes):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


classes_path = Path(NPY_DIR) / "classes.json"
if not classes_path.exists():
    raise FileNotFoundError(f"classes.json not found in {NPY_DIR}")
meta = json.loads(classes_path.read_text(encoding="utf-8"))
classes = meta.get("classes", [])
if not classes:
    raise ValueError("classes.json has empty classes")
NUM_CLASSES = len(classes)

if TARGET_CLASS_NAME not in classes:
    raise ValueError(f"{TARGET_CLASS_NAME} not found in classes: {classes}")
A4_CLASS_IDX = classes.index(TARGET_CLASS_NAME)
print(f"[INFO] A4 class index={A4_CLASS_IDX}, classes={classes}")


class A4MixedViewDataset(Dataset):
    def __init__(self, npy_dir, split, transform=None, use_a4_three_view=False):
        self.transform = transform
        self.npy_dir = Path(npy_dir)
        self.split = split
        self.use_a4_three_view = bool(use_a4_three_view)

        paths_file = self.npy_dir / f"{split}_paths.npy"
        labels_file = self.npy_dir / f"{split}_labels.npy"
        if not paths_file.exists() or not labels_file.exists():
            raise RuntimeError(f"[ERR] Missing NPY files for split '{split}' in {npy_dir}")

        raw_paths = np.load(paths_file, allow_pickle=True)
        raw_labels = np.load(labels_file, allow_pickle=True).astype(np.int64)
        valid_paths = []
        valid_labels = []
        dropped = 0
        for p, y in zip(raw_paths, raw_labels):
            path = str(p)
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    roi_box = extract_roi_box(find_json_for_image(Path(path)), img.width, img.height)
                if roi_box is None:
                    dropped += 1
                    continue
                x1, y1, x2, y2 = roi_box
                if min(x2 - x1, y2 - y1) < DROP_MIN_SIDE:
                    dropped += 1
                    continue
                valid_paths.append(path)
                valid_labels.append(int(y))
            except Exception:
                dropped += 1
        self.paths = np.asarray(valid_paths, dtype=object)
        self.labels = np.asarray(valid_labels, dtype=np.int64)

        self.index_map = []
        for i, y in enumerate(self.labels):
            if self.use_a4_three_view and int(y) == A4_CLASS_IDX:
                view_types = [0, 1, 2] if USE_OFF_VIEW else [0, 1]
            else:
                view_types = [0]
            for v in view_types:
                self.index_map.append((i, v))

        self.effective_labels = np.asarray([int(self.labels[i]) for i, _ in self.index_map], dtype=np.int64)

        base_n = len(self.labels)
        eff_n = len(self.index_map)
        a4_base = int((self.labels == A4_CLASS_IDX).sum())
        a4_eff = int((self.effective_labels == A4_CLASS_IDX).sum())
        print(f"[{split.upper()}] base={base_n}, effective={eff_n}, A4 base={a4_base}, A4 effective={a4_eff}, dropped={dropped}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        base_idx, view_type = self.index_map[idx]
        path = str(self.paths[base_idx])
        label = int(self.labels[base_idx])

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                roi_box = extract_roi_box(find_json_for_image(Path(path)), img.width, img.height)
                img = crop_view(img, path, roi_box, view_type, IMG_SIZE)
                if img is None:
                    raise ValueError(f"invalid roi for {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load valid sample: {path}") from e

        if self.transform:
            img = self.transform(img)
        return img, label, idx


model = create_model(
    MODEL_NAME,
    pretrained=PRETRAINED,
    num_classes=NUM_CLASSES,
    drop_path_rate=DROP_PATH_RATE,
)
if USE_CHANNELS_LAST:
    model = model.to(device, memory_format=torch.channels_last)
else:
    model = model.to(device)
if USE_COMPILE and device.type == "cuda":
    model = torch.compile(model)

for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

data_config = resolve_data_config({}, model=model)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(data_config["mean"], data_config["std"]),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(data_config["mean"], data_config["std"]),
])

train_dataset = A4MixedViewDataset(NPY_DIR, "train", transform=train_transform, use_a4_three_view=True)
val_dataset = A4MixedViewDataset(NPY_DIR, "val", transform=eval_transform, use_a4_three_view=False)
test_dataset = A4MixedViewDataset(NPY_DIR, "test", transform=eval_transform, use_a4_three_view=False)

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

class_weights = build_class_weights(train_dataset.effective_labels, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"[INFO] WCE class_weights={class_weights.detach().cpu().tolist()}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

with open(LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

best_val = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

    if epoch == FREEZE_EPOCHS:
        print(">>> Unfreezing backbone")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR_ALL, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch, eta_min=1e-6)

    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _ in tqdm(train_loader, desc="Train"):
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

    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc="Val"):
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
        csv.writer(f).writerow([epoch + 1, lr, train_loss, train_acc, val_loss, val_acc])

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        patience_counter = 0
        print(f"New Best Validation Accuracy: {best_val:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early Stopping triggered at Epoch {epoch + 1}")
            break

print(f"\n[Test] Loading best model from {SAVE_DIR}/best_model.pth")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location="cpu"))
model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x, y, _ in tqdm(test_loader, desc="Test"):
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
