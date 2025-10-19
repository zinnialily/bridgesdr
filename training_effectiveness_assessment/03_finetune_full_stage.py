"""
Full Fine-Tuning (Stage 2) on BRIGHT pseudo-data:
- Loads half-finetuned checkpoint
- Unfreezes all layers
- Fine-tunes entire network
- Saves per-stratum model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from utils.dataset import prepare_dataloaders
from utils.metrics import calculate_metrics

# ======================
# Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
CHECKPOINT_DIR = "checkpoints/study2/"
USE_EXISTING_CHECKPOINT = True

# Strata
STRATA = ["LIC", "MIC", "HIC"]
HALF_PATHS = {
    "LIC": os.path.join(CHECKPOINT_DIR, "lic_half_finetuned_unet.pth"),
    "MIC": os.path.join(CHECKPOINT_DIR, "mic_half_finetuned_unet.pth"),
    "HIC": os.path.join(CHECKPOINT_DIR, "hic_half_finetuned_unet.pth"),
}
FULL_PATHS = {
    "LIC": os.path.join(CHECKPOINT_DIR, "lic_full_finetuned_unet.pth"),
    "MIC": os.path.join(CHECKPOINT_DIR, "mic_full_finetuned_unet.pth"),
    "HIC": os.path.join(CHECKPOINT_DIR, "hic_full_finetuned_unet.pth"),
}

# ======================
# Reproducibility
# ======================
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# Full Fine-Tuning Loop
# ======================
for region in STRATA:
    print(f"\n=== Full Fine-Tuning for {region} ===")
    
    # Dataset
    BRIGHT_ROOT = f"path/to/BRIGHT_{region}_pseudo"
    train_loader, val_loader = prepare_dataloaders(BRIGHT_ROOT, [], 0.7, BATCH_SIZE)

    # Load half-finetuned model
    model = UNet(in_channels=4, out_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(HALF_PATHS[region], map_location=DEVICE))

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Loss, optimizer, scheduler
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0]).to(DEVICE)  # 10% for 'no damage'
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    checkpoint_path = FULL_PATHS[region]

    # Load existing checkpoint if present
    if USE_EXISTING_CHECKPOINT and os.path.exists(checkpoint_path):
        print(f"Loading existing full-finetuned checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        continue

    # Training
    best_iou = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

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

        print(f"{region} - Epoch {epoch+1}: Avg Loss={avg_loss:.4f}, Val IoU={val_iou:.4f}, Dice={val_dice:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved improved model at epoch {epoch+1} to {checkpoint_path}")

    print(f"Finished Full Fine-Tuning for {region}")
