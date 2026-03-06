"""
_08_shared.py  —  Shared components for Study 2 training scripts
=================================================================
Imported by: 08_train_baseline_unet.py, 08b_train_segformer.py,
             08c_train_changeformer.py, 09_finetune_half_stage.py,
             10_finetune_full_stage.py

Training hyperparameters (Reviewer 1 Concern 8 — explicit config):
  Loss      : CrossEntropyLoss  weights = [0.1, 1.0, 1.5, 2.0]
  Optimizer : AdamW  lr = 1e-3 (baseline) / 1e-4 (fine-tune)
  Scheduler : CosineAnnealingLR  T_max = n_epochs
  Epochs    : 20 (baseline)  /  10 (fine-tune stages)
  Batch     : 8 (GPU)  /  4 (CPU)
  Aug       : HFlip + VFlip + Rotation±30° + ColorJitter (optical only)
  Seed      : 42
"""

import json, random as _random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# Hyper-parameters (canonical single source)
# ─────────────────────────────────────────────
SEED         = 42
BASELINE_LR  = 1e-3
FINETUNE_LR  = 1e-4
WEIGHT_DECAY = 1e-4
BASELINE_EPOCHS  = 20
FINETUNE_EPOCHS  = 10
BATCH_GPU    = 8
BATCH_CPU    = 4
IMG_SIZE     = 512    # BRIGHT native tile size

CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 1.5, 2.0])   # none / minor / major / destroyed
N_CLASSES     = 4

# ─────────────────────────────────────────────
# Project layout & shared data-loading helpers
# ─────────────────────────────────────────────
_BRIGHT_DATA_DIR = Path(__file__).parent.parent / "data" / "bright"
INCOME_LEVELS    = ["lic", "mic", "hic"]
CROSS_MIX_RATIO  = 0.20


def load_manifest(income: str, split: str) -> list:
    """Load train or val manifest for a given income stratum."""
    path = _BRIGHT_DATA_DIR / income / f"manifest_{split}.json"
    if not path.exists():
        print(f"  [warn] manifest not found: {path} — run 02_preprocess_data.py first")
        return []
    with open(path) as fh:
        return json.load(fh)


def load_combined_manifest(income: str) -> list:
    """Load combined manifest (train+val) used by evaluation and visualisation."""
    for fname in ("manifest.json", "manifest_val.json"):
        path = _BRIGHT_DATA_DIR / income / fname
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    return []


def get_augmented_train_records(target_income: str) -> list:
    """Return training records with CROSS_MIX_RATIO cross-stratum augmentation."""
    records = load_manifest(target_income, "train")
    n_cross = int(len(records) * CROSS_MIX_RATIO)
    other   = []
    for inc in INCOME_LEVELS:
        if inc != target_income:
            other.extend(load_manifest(inc, "train"))
    if other and n_cross > 0:
        sample   = _random.sample(other, min(n_cross, len(other)))
        records += sample
        print(f"  Cross-mix: +{len(sample)} tiles from other strata")
    return records


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch_size():
    return BATCH_GPU if torch.cuda.is_available() else BATCH_CPU

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class BRIGHTDataset(Dataset):
    """
    Loads BRIGHT pre+post+mask triplets.

    Pre-event  : 3-channel optical RGB  (.png / .tif)
    Post-event : 1-channel SAR grayscale (.tif)
    Mask       : single-channel PNG with pixel values {0,1,2,3}

    Returns:
        tensor: 4-channel float [0,1] of shape (4, H, W)
                channels = [R, G, B, SAR]
        mask  : long tensor (H, W) with class indices 0-3
    """

    def __init__(self, records: list, augment: bool = False, img_size: int = IMG_SIZE):
        self.records  = records
        self.augment  = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        return torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)   # 3×H×W

    def _load_sar(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        return torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)        # 1×H×W

    def _load_mask(self, path: str) -> torch.Tensor:
        mask = Image.open(path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
        arr = np.clip(np.array(mask, dtype=np.int64), 0, N_CLASSES - 1)
        return torch.from_numpy(arr)                                      # H×W long

    def __getitem__(self, idx):
        rec  = self.records[idx]
        pre  = self._load_rgb(rec["pre_img"])
        sar  = self._load_sar(rec["post_img"])
        mask = self._load_mask(rec["mask"])

        if self.augment:
            # Random horizontal flip
            if _random.random() > 0.5:
                pre  = TF.hflip(pre)
                sar  = TF.hflip(sar)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

            # Random vertical flip
            if _random.random() > 0.5:
                pre  = TF.vflip(pre)
                sar  = TF.vflip(sar)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

            # Random rotation ±30°
            angle = _random.uniform(-30, 30)
            pre  = TF.rotate(pre,  angle, interpolation=TF.InterpolationMode.BILINEAR)
            sar  = TF.rotate(sar,  angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask.unsqueeze(0), angle,
                             interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

            # Color jitter on optical only (not SAR)
            if _random.random() > 0.5:
                pre = TF.adjust_brightness(pre, _random.uniform(0.8, 1.2))
                pre = TF.adjust_contrast  (pre, _random.uniform(0.8, 1.2))
                pre = TF.adjust_saturation(pre, _random.uniform(0.8, 1.2))

        x = torch.cat([pre, sar], dim=0)   # 4×H×W
        return x, mask

# ─────────────────────────────────────────────
# Bitemporal dataset (for ChangeFormer)
# ─────────────────────────────────────────────
class BRIGHTBitemporalDataset(BRIGHTDataset):
    """
    Returns (pre_rgb_3ch, post_sar_replicated_3ch, mask) for ChangeFormer.
    SAR replicated to 3 channels is a known limitation (documented in paper).
    """
    def __getitem__(self, idx):
        x, mask = super().__getitem__(idx)
        pre_3ch  = x[:3]                      # R G B
        sar_3ch  = x[3:4].expand(3, -1, -1)  # SAR replicated
        return pre_3ch, sar_3ch, mask

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    n_classes: int = N_CLASSES) -> dict:
    """
    Per-class and mean IoU / Dice / Precision / Recall.

    Args:
        preds   : (N, H, W) long tensor — predicted class indices
        targets : (N, H, W) long tensor — ground-truth class indices

    Returns dict with keys: iou_per_class, dice_per_class, mean_iou, mean_dice,
                            precision_per_class, recall_per_class
    """
    preds   = preds.cpu().view(-1)
    targets = targets.cpu().view(-1)

    iou_list, dice_list, prec_list, rec_list = [], [], [], []

    for c in range(n_classes):
        p = (preds   == c)
        t = (targets == c)
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()

        iou  = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)

        iou_list.append(iou.item())
        dice_list.append(dice.item())
        prec_list.append(prec.item())
        rec_list.append(rec.item())

    return {
        "iou_per_class":       iou_list,
        "dice_per_class":      dice_list,
        "precision_per_class": prec_list,
        "recall_per_class":    rec_list,
        "mean_iou":            float(np.mean(iou_list)),
        "mean_dice":           float(np.mean(dice_list)),
        "mean_precision":      float(np.mean(prec_list)),
        "mean_recall":         float(np.mean(rec_list)),
    }

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 criterion, n_epochs: int, device, save_path: Path,
                 model_name: str = "model",
                 bitemporal: bool = False) -> dict:
    """
    Generic training loop used by all three models.

    Returns best validation metrics dict.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_iou  = 0.0
    best_metrics = {}

    for epoch in range(1, n_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader,
                          desc=f"  [{model_name}] Epoch {epoch}/{n_epochs} train",
                          leave=False):
            if bitemporal:
                pre, post, masks = batch
                pre, post, masks = pre.to(device), post.to(device), masks.to(device)
                logits = model(pre, post)
            else:
                inputs, masks = batch
                inputs, masks = inputs.to(device), masks.to(device)
                logits = model(inputs)

            # Handle HuggingFace SegFormer output dict
            if hasattr(logits, "logits"):
                logits = logits.logits
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        all_preds, all_masks = [], []

        with torch.no_grad():
            for batch in val_loader:
                if bitemporal:
                    pre, post, masks = batch
                    pre, post = pre.to(device), post.to(device)
                    logits = model(pre, post)
                else:
                    inputs, masks = batch
                    inputs = inputs.to(device)
                    logits = model(inputs)

                if hasattr(logits, "logits"):
                    logits = logits.logits
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_masks.append(masks)

        val_metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_masks))

        print(f"  [{model_name}] Epoch {epoch:2d}  "
              f"loss={avg_train_loss:.4f}  "
              f"mIoU={val_metrics['mean_iou']:.4f}  "
              f"mDice={val_metrics['mean_dice']:.4f}")

        if val_metrics["mean_iou"] > best_iou:
            best_iou     = val_metrics["mean_iou"]
            best_metrics = val_metrics
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics":       val_metrics,
            }, save_path)
            print(f"    ✓ Saved checkpoint (mIoU={best_iou:.4f}) → {save_path.name}")

    return best_metrics
