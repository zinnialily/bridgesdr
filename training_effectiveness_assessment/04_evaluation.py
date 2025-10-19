"""
Evaluate model:
- Compute IoU, Dice, Precision, Recall
- Supports baseline, half, or full fine-tuned models
"""

import torch
from models.unet import UNet
from utils.dataset import prepare_dataloaders
from utils.metrics import calculate_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/study2/baseline_unet_best.pth"
DATA_ROOT = "path/to/xbd"
BATCH_SIZE = 16

# Load model
model = UNet(in_channels=4, out_classes=4).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# Load dataset
_, val_loader = prepare_dataloaders(DATA_ROOT, [], 0.7, BATCH_SIZE)

# Initialize metrics
total_iou, total_dice, total_precision, total_recall = 0, 0, 0, 0

with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
        preds = torch.argmax(model(inputs), dim=1)
        iou, dice, precision, recall = calculate_metrics(preds, masks)
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall

num_batches = len(val_loader)
avg_iou = total_iou / num_batches
avg_dice = total_dice / num_batches
avg_precision = total_precision / num_batches
avg_recall = total_recall / num_batches

print(f"Evaluation Metrics:")
print(f"  IoU:      {avg_iou:.4f}")
print(f"  Dice:     {avg_dice:.4f}")
print(f"  Precision:{avg_precision:.4f}")
print(f"  Recall:   {avg_recall:.4f}")
