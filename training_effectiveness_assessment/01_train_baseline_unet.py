"""
Train baseline U-Net damage segmentation model on xBD only.
- 4-channel input (RGB pre + SAR post)
- 4-class output (none/minor/major/destroyed)
- Optional: compute class weights
- Optional: train from scratch or load checkpoint
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import prepare_dataloaders
from utils.metrics import calculate_metrics

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
CHECKPOINT_DIR = "checkpoints/study2/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
USE_EXISTING_CHECKPOINT = True  # Set False to train from scratch

# Set seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load xBD dataset
train_loader, val_loader = prepare_dataloaders("path/to/xbd", [], 0.7, BATCH_SIZE, is_xbd=True)

# Initialize model
model = UNet(in_channels=4, out_classes=4).to(DEVICE)

# Class weights: downweight "no damage" class to 10%
class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_unet_best.pth")

if USE_EXISTING_CHECKPOINT and os.path.exists(checkpoint_path):
    print(f"Loading existing baseline checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
else:
    print("Training baseline model from scratch...")
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

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "baseline_unet_final.pth"))
