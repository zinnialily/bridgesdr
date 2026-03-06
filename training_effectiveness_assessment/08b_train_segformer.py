"""
08b_train_segformer.py  —  Study 2: Baseline SegFormer-B2 on BRIGHT
=====================================================================
Addresses Reviewer 1 Concern 6: "Only a single baseline (U-Net) is evaluated."

Architecture : SegFormer-B2  (HuggingFace transformers, pretrained nvidia/mit-b2)
Input        : 4 × H × W  [R, G, B, SAR]
               First patch-embedding conv extended from 3→4 channels
               (SAR weight initialised as mean of RGB weights for warm start)
Output       : N_CLASSES × H × W

Hyperparameters (same protocol as U-Net for fair comparison):
  Loss      : CrossEntropyLoss  weights=[0.1, 1.0, 1.5, 2.0]
  Optimizer : AdamW  lr=6e-5  weight_decay=1e-4   (lower LR for pretrained Transformer)
  Scheduler : CosineAnnealingLR  T_max=20
  Epochs    : 20
  Batch     : 8 (GPU) / 4 (CPU)
  Seed      : 42

Outputs:
  checkpoints/segformer_baseline_<income>.pth
  results/study2/baseline/segformer_baseline_results.json
"""

import sys, json
from pathlib import Path
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _08_shared import (
    BRIGHTDataset, run_training,
    set_seed, get_device, get_batch_size,
    BASELINE_EPOCHS, WEIGHT_DECAY, CLASS_WEIGHTS, SEED,
    INCOME_LEVELS, load_manifest,
)
from _model_registry import load_model

SEGFORMER_LR   = 6e-5      # lower than U-Net: pre-trained transformer
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results" / "study2" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    set_seed(SEED)
    device     = get_device()
    batch_size = get_batch_size()

    print("=" * 60)
    print("08b  —  Baseline SegFormer-B2 Training  (BRIGHT only, Study 2)")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {BASELINE_EPOCHS}")
    print(f"LR (AdamW) : {SEGFORMER_LR}  (lower for pretrained Transformer)")
    print(f"Loss wts   : {CLASS_WEIGHTS.tolist()}")

    all_results = {}

    for income in INCOME_LEVELS:
        print(f"\n{'─'*50}")
        print(f"Stratum: {income.upper()}")
        print(f"{'─'*50}")

        train_recs = load_manifest(income, "train")
        val_recs   = load_manifest(income, "val")

        if not train_recs:
            print(f"  [skip] No training data for {income}")
            continue

        train_ds = BRIGHTDataset(train_recs, augment=True)
        val_ds   = BRIGHTDataset(val_recs,   augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        model     = load_model("segformer").to(device)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=SEGFORMER_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=BASELINE_EPOCHS)

        save_path = CHECKPOINT_DIR / f"segformer_baseline_{income}.pth"

        best = run_training(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion,
            BASELINE_EPOCHS, device, save_path,
            model_name=f"SegFormer[{income.upper()}]",
            bitemporal=False,
        )

        all_results[income] = best
        print(f"  Best mIoU ({income.upper()}): {best.get('mean_iou', 0):.4f}")

    out_path = RESULTS_DIR / "segformer_baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
