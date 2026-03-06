"""
11_evaluation.py  —  Study 2: Full Evaluation of All Models × All Stages
=========================================================================
Evaluates U-Net, SegFormer-B2, and ChangeFormer across:
  • 3 income strata (LIC / MIC / HIC)
  • 3 training stages (baseline / stage1 / stage2)

Per-event breakdown enables downstream confound analysis in:
  • 13_sample_size_correlation.py
  • 14_disaster_type_analysis.py

Outputs:
  results/study2/evaluation/all_results.json
  results/study2/evaluation/summary_table.json
"""

import sys, json
from pathlib import Path
from collections import defaultdict
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from _08_shared import (
    BRIGHTDataset, BRIGHTBitemporalDataset, compute_metrics,
    set_seed, get_device, get_batch_size, SEED,
    INCOME_LEVELS, load_combined_manifest,
)
from _model_registry import load_model, MODEL_NAMES

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results" / "study2" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["baseline", "stage1", "stage2"]


def load_checkpoint(model_name: str, stage: str, income: str, device):
    """Load checkpoint; return (model, ok_flag)."""
    ckpt_map = {
        "baseline": f"{model_name}_baseline_{income}.pth",
        "stage1":   f"{model_name}_stage1_{income}.pth",
        "stage2":   f"{model_name}_stage2_{income}.pth",
    }
    ckpt_path = CHECKPOINT_DIR / ckpt_map[stage]
    if not ckpt_path.exists():
        return None, False

    model = load_model(model_name).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model, True


@torch.no_grad()
def evaluate_model(model, loader, device, bitemporal: bool) -> dict:
    """Run inference and return aggregate + per-sample metrics."""
    all_preds, all_masks = [], []

    for batch in tqdm(loader, desc="    eval", leave=False):
        if bitemporal:
            pre, post, masks = batch
            logits = model(pre.to(device), post.to(device))
        else:
            inputs, masks = batch
            logits = model(inputs.to(device))

        if hasattr(logits, "logits"):
            logits = logits.logits
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_masks.append(masks)

    return compute_metrics(torch.cat(all_preds), torch.cat(all_masks))


def main():
    set_seed(SEED)
    device     = get_device()
    batch_size = get_batch_size()

    print("=" * 60)
    print("11  —  Full Evaluation: All Models × All Stages × All Strata")
    print("=" * 60)
    print(f"Models : {MODEL_NAMES}")
    print(f"Stages : {STAGES}")
    print(f"Strata : {INCOME_LEVELS}")

    all_results  = {}
    summary_rows = []

    for model_name in MODEL_NAMES:
        all_results[model_name] = {}
        bitemporal = (model_name == "changeformer")
        DS = BRIGHTBitemporalDataset if bitemporal else BRIGHTDataset

        for income in INCOME_LEVELS:
            records = load_combined_manifest(income)
            if not records:
                print(f"  [skip] No data for {income}")
                continue

            # Per-event breakdown (for confound analysis)
            event_groups: dict = defaultdict(list)
            for r in records:
                event_groups[r.get("event", "unknown")].append(r)

            all_results[model_name][income] = {}

            for stage in STAGES:
                print(f"\n  {model_name.upper()} | {income.upper()} | {stage}")
                model, ok = load_checkpoint(model_name, stage, income, device)
                if not ok:
                    print(f"    [skip] checkpoint not found")
                    continue

                # ── Overall evaluation ────────────────────────────────────
                ds     = DS(records, augment=False)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True)
                metrics = evaluate_model(model, loader, device, bitemporal)
                print(f"    mIoU={metrics['mean_iou']:.4f}  "
                      f"mDice={metrics['mean_dice']:.4f}")

                # ── Per-event evaluation ──────────────────────────────────
                per_event = {}
                for event, evr in event_groups.items():
                    ev_ds  = DS(evr, augment=False)
                    ev_ldr = DataLoader(ev_ds, batch_size=batch_size,
                                        shuffle=False, num_workers=0)
                    ev_met = evaluate_model(model, ev_ldr, device, bitemporal)
                    per_event[event] = {
                        "mean_iou":  ev_met["mean_iou"],
                        "mean_dice": ev_met["mean_dice"],
                        "n_tiles":   len(evr),
                        "income":    income,
                    }

                all_results[model_name][income][stage] = {
                    "overall":   metrics,
                    "per_event": per_event,
                }

                summary_rows.append({
                    "model":      model_name,
                    "income":     income,
                    "stage":      stage,
                    "mean_iou":   metrics["mean_iou"],
                    "mean_dice":  metrics["mean_dice"],
                    "mean_prec":  metrics["mean_precision"],
                    "mean_rec":   metrics["mean_recall"],
                })

    # ── Save ─────────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "all_results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    with open(RESULTS_DIR / "summary_table.json", "w") as fh:
        json.dump(summary_rows, fh, indent=2)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Model':<15} {'Stratum':<8} {'Stage':<10} "
          f"{'mIoU':>8} {'mDice':>8} {'Prec':>8} {'Rec':>8}")
    print("─" * 80)
    for row in summary_rows:
        print(f"{row['model']:<15} {row['income'].upper():<8} {row['stage']:<10} "
              f"{row['mean_iou']:>8.4f} {row['mean_dice']:>8.4f} "
              f"{row['mean_prec']:>8.4f} {row['mean_rec']:>8.4f}")
    print("=" * 80)

    print(f"\nResults → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
