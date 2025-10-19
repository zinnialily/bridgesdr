"""
Half fine-tuning (Stage 1):
- Loads baseline checkpoint
- Freezes encoder
- Fine-tunes decoder on BRIGHT
- Reduced LR: 1e-4
- Trains for 3 epochs
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from utils.dataset import prepare_dataloaders
from utils.metrics import calculate_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
CHECKPOINT_DIR = "checkpoints/study2/"
USE_EXISTING_CHECKPOINT = True

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load BRIGHT dataset for a stratum (e.g., LIC)
BRIGHT_ROOT = "path/to/BRIGHT_LIC_pseudo"
strata_keywords = []  # Optionally filter by country
train_loader, val_loader = prepare_dataloaders(BRIGHT_ROOT, strata_keywords, 0.7, BATCH_SIZE)

# Load baseline model
baseline_path = os.path.join(CHECKPOINT_DIR, "baseline_unet_best.pth")
model = UNet(in_channels=4, out_classes=4).to(DEVICE)
model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))

# Freeze encoder and bottleneck
for name, param in model.named_parameters():
    if 'enc' in name or 'bottleneck' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

class_weights = torch.tensor([0.1,1.0,1.0,1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

checkpoint_path = os.path.join(CHECKPOINT_DIR, "lic_half_finetuned_unet.pth")

if USE_EXISTING_CHECKPOINT and os.path.exists(checkpoint_path):
    print(f"Loading half-finetuned checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
else:
    print("Fine-tuning decoder from scratch...")
    best_iou = 0
    for epoch in range(EPOCHS):
        model.train()
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_iou, val_dice = 0, 0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
                preds = torch.argmax(model(inputs), dim=1)
                iou, dice, _, _ = calculate_metrics(preds, masks)
                val_iou += iou
                val_dice += dice
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_iou)
        print(f"Epoch {epoch+1}: Val IoU={val_iou:.4f}, Dice={val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)
