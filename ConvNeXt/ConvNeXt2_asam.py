import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from utils.asam import ASAM

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
NPY_DIR = "/root/project/dataset/aware_cache_npy_80"
SAVE_DIR = "/root/project/convnext/convnext_aware_80_asam"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

NUM_CLASSES = 8
BATCH_SIZE = 256
NUM_EPOCHS = 200
IMG_SIZE = 224
RANDOM_SEED = 0
PATIENCE = 20
Freeze = 5

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

WEIGHT_DECAY = 0.1
LR1 = 1e-3
LR2 = 3e-5
ASAM_RHO = 0.5
ASAM_ETA = 0.01
USE_ASAM = True

pretrained = True
model_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
# model_name = "convnextv2_femto.fcmae_ft_in1k"
drop_path_rate = 0.2

print(f"[Info] Reading Data from: {NPY_DIR}")
print(f"[Info] USE_ASAM={USE_ASAM} rho={ASAM_RHO} eta={ASAM_ETA}")

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
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        label = int(self.labels[idx])
        
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (return black or error)
            # Here we print and return a black image to avoid crash
            print(f"Error loading {path}: {e}")
            img = Image.new("RGB", (224, 224))

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
).to(device)

# Backbone freeze (Init)
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

data_config = resolve_data_config({}, model=model)

# =========================
# 4. Transform
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# =========================
# 6. Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR1,
    weight_decay=WEIGHT_DECAY
)
asam_optimizer = ASAM(optimizer, model, rho=ASAM_RHO, eta=ASAM_ETA) if USE_ASAM else None
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Keep GradScaler for non-ASAM path. ASAM uses custom ascent/descent steps.
use_grad_scaler = (device.type == "cuda" and amp_dtype == torch.float16 and not USE_ASAM)
scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

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
        asam_optimizer = ASAM(optimizer, model, rho=ASAM_RHO, eta=ASAM_ETA) if USE_ASAM else None
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-epoch, eta_min=1e-6)

    # ---- Train ----
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y, idx in tqdm(train_loader, desc="Train"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if USE_ASAM:
            # ASAM two-pass update: ascent (w+e) -> descent on restored weights.
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            asam_optimizer.ascent_step()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                out_adv = model(x)
                loss_adv = criterion(out_adv, y)
            loss_adv.backward()
            asam_optimizer.descent_step()
            loss_for_log = loss_adv
            out_for_log = out_adv
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                out = model(x)
                loss = criterion(out, y)
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_for_log = loss
            out_for_log = out

        bs = y.size(0)
        loss_sum += float(loss_for_log.detach().cpu()) * bs
        correct += (out_for_log.argmax(1) == y).sum().item()
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
