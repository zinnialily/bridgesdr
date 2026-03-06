"""
10_finetune_full_stage.py  —  Study 2: Stage 2 Fine-Tuning (all 3 models)
=========================================================================
Stage 2: ALL layers trainable — loads Stage 1 checkpoints.

Applies to U-Net, SegFormer-B2, and ChangeFormer.

Cross-disaster augmentation (20% from other strata) continues from Stage 1.

Hyperparameters:
  Optimizer : AdamW  lr=1e-4  weight_decay=1e-4
  Scheduler : CosineAnnealingLR  T_max=10
  Epochs    : 10
  Batch     : 8 (GPU) / 4 (CPU)
  Seed      : 42

Inputs  : checkpoints/<model>_stage1_<income>.pth
Outputs : checkpoints/<model>_stage2_<income>.pth
Results : results/study2/finetune_stage2/stage2_results.json
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
    BRIGHTDataset, BRIGHTBitemporalDataset, run_training,
    set_seed, get_device, get_batch_size,
    FINETUNE_LR, FINETUNE_EPOCHS, WEIGHT_DECAY, CLASS_WEIGHTS, SEED,
    INCOME_LEVELS, load_manifest, get_augmented_train_records,
)
from _model_registry import load_model, MODEL_NAMES

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results" / "study2" / "finetune_stage2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    set_seed(SEED)
    device     = get_device()
    batch_size = get_batch_size()

    print("=" * 60)
    print("10  —  Stage 2 Fine-Tuning: All Layers  (all 3 models)")
    print("=" * 60)
    print(f"Models   : {MODEL_NAMES}")
    print(f"LR       : {FINETUNE_LR}")
    print(f"Epochs   : {FINETUNE_EPOCHS}")

    all_results = {}

    for model_name in MODEL_NAMES:
        all_results[model_name] = {}
        bitemporal = (model_name == "changeformer")
        DS = BRIGHTBitemporalDataset if bitemporal else BRIGHTDataset

        for income in INCOME_LEVELS:
            print(f"\n{'─'*50}")
            print(f"Model: {model_name.upper()}  |  Stratum: {income.upper()}")
            print(f"{'─'*50}")

            # Load Stage 1 checkpoint (fallback to baseline if Stage 1 missing)
            ckpt = CHECKPOINT_DIR / f"{model_name}_stage1_{income}.pth"
            if not ckpt.exists():
                ckpt = CHECKPOINT_DIR / f"{model_name}_baseline_{income}.pth"
                if ckpt.exists():
                    print(f"  [info] Stage 1 not found; loading baseline instead")
                else:
                    print(f"  [skip] No checkpoint found for {model_name}/{income}")
                    continue

            train_recs = get_augmented_train_records(income)
            val_recs   = load_manifest(income, "val")

            if not train_recs:
                print(f"  [skip] No training data for {income}")
                continue

            train_ds = DS(train_recs, augment=True)
            val_ds   = DS(val_recs,   augment=False)

            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True,  num_workers=2, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                      shuffle=False, num_workers=2, pin_memory=True)

            # Load Stage 1 model
            model = load_model(model_name).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))

            # Unfreeze ALL parameters
            for param in model.parameters():
                param.requires_grad = True

            criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR,
                                          weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=FINETUNE_EPOCHS)

            save_path = CHECKPOINT_DIR / f"{model_name}_stage2_{income}.pth"

            best = run_training(
                model, train_loader, val_loader,
                optimizer, scheduler, criterion,
                FINETUNE_EPOCHS, device, save_path,
                model_name=f"{model_name}[{income.upper()}] Stage2",
                bitemporal=bitemporal,
            )

            all_results[model_name][income] = best
            print(f"  Best mIoU: {best.get('mean_iou', 0):.4f}")

    out_path = RESULTS_DIR / "stage2_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll results → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
