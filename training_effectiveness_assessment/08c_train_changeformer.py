"""
08c_train_changeformer.py  —  Study 2: Baseline ChangeFormer on BRIGHT
=======================================================================
Addresses Reviewer 1 Concern 6: "Only a single baseline (U-Net) is evaluated."

Architecture : ChangeFormer  (Siamese Transformer for change detection)
               Bandara & Patel, IGARSS 2022
               https://arxiv.org/abs/2201.01293

Input        : Two separate 3-channel inputs:
                 • pre  : optical RGB  (3 × H × W)
                 • post : SAR replicated to 3 channels (known limitation—documented)
Output       : N_CLASSES × H × W

Rationale for SAR replication:
  ChangeFormer uses a Siamese encoder expecting the same channel count for both
  branches. Since BRIGHT post-event is single-channel SAR, we replicate it to
  3 channels. This is explicitly noted as a limitation in Section 3 and
  Discussion. A proper bi-modal ChangeFormer is left for future work.

Hyperparameters (same protocol as U-Net/SegFormer for fair comparison):
  Loss      : CrossEntropyLoss  weights=[0.1, 1.0, 1.5, 2.0]
  Optimizer : AdamW  lr=6e-5  weight_decay=1e-4
  Scheduler : CosineAnnealingLR  T_max=20
  Epochs    : 20
  Batch     : 8 (GPU) / 4 (CPU)
  Seed      : 42

Outputs:
  checkpoints/changeformer_baseline_<income>.pth
  results/study2/baseline/changeformer_baseline_results.json
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
    BRIGHTBitemporalDataset, run_training,
    set_seed, get_device, get_batch_size,
    BASELINE_EPOCHS, WEIGHT_DECAY, CLASS_WEIGHTS, SEED,
    INCOME_LEVELS, load_manifest,
)
from _model_registry import load_model

CHANGEFORMER_LR = 6e-5
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results" / "study2" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    set_seed(SEED)
    device     = get_device()
    batch_size = get_batch_size()

    print("=" * 60)
    print("08c  —  Baseline ChangeFormer Training  (BRIGHT only, Study 2)")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {BASELINE_EPOCHS}")
    print(f"LR (AdamW) : {CHANGEFORMER_LR}")
    print(f"Loss wts   : {CLASS_WEIGHTS.tolist()}")
    print("NOTE: SAR replicated to 3ch for Siamese encoder (documented limitation)")

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

        # BRIGHTBitemporalDataset returns (pre_3ch, post_3ch, mask)
        train_ds = BRIGHTBitemporalDataset(train_recs, augment=True)
        val_ds   = BRIGHTBitemporalDataset(val_recs,   augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        model     = load_model("changeformer").to(device)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CHANGEFORMER_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=BASELINE_EPOCHS)

        save_path = CHECKPOINT_DIR / f"changeformer_baseline_{income}.pth"

        best = run_training(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion,
            BASELINE_EPOCHS, device, save_path,
            model_name=f"ChangeFormer[{income.upper()}]",
            bitemporal=True,   # ← uses (pre, post, mask) unpacking
        )

        all_results[income] = best
        print(f"  Best mIoU ({income.upper()}): {best.get('mean_iou', 0):.4f}")

    out_path = RESULTS_DIR / "changeformer_baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
