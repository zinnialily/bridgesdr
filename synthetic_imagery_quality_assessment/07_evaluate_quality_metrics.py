"""
07_evaluate_quality_metrics.py  —  Study 1: DisasterGAN Mask Quality Evaluation
=================================================================================
BUG FIXES vs original:
  1. Removed phantom 'F1' key (F1 == Dice; original KeyError on line 143/184).
  2. Added Figure 7 with a CORRECT discrete colormap and unambiguous legend
     using BoundaryNorm + explicit per-class patches (original had mis-ordered
     auto-label colours).

Outputs:
  results/study1/evaluation/<mask_type>/per_image_results.json
  results/study1/evaluation/<mask_type>/income_summaries.json
  results/study1/evaluation/<mask_type>/overall_summary.json
  results/study1/figures/figure7_damage_mask_comparison.png
"""

import sys, json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    masks_dir   = PROJECT_ROOT / "results" / "study1" / "masks"
    output_dir  = PROJECT_ROOT / "results" / "study1" / "evaluation"
    figures_dir = PROJECT_ROOT / "results" / "study1" / "figures"
    income_levels = ["lic", "mic", "hic"]
    # Damage class colour map  (classes 0-3)
    CLASS_COLORS = ["#d9d9d9", "#fee08b", "#f46d43", "#d73027"]
    CLASS_LABELS = ["No damage", "Minor damage", "Major damage", "Destroyed"]
    N_CLASSES    = 4

config = Config()

# ─────────────────────────────────────────────
# Colour-map helpers  (fixes Figure 7 legend bug)
# ─────────────────────────────────────────────
def _damage_cmap_norm():
    cmap   = mcolors.ListedColormap(config.CLASS_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def _damage_patches():
    return [
        mpatches.Patch(facecolor=c, label=l, edgecolor="black", linewidth=0.5)
        for c, l in zip(config.CLASS_COLORS, config.CLASS_LABELS)
    ]

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_mask_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Weighted pixel-level metrics. Keys: IoU, Dice, Precision, Recall.
    NOTE: 'F1' key intentionally absent — it was an alias for Dice and caused
    a KeyError because compute_mask_metrics never populated it.
    """
    p = pred.flatten().astype(int)
    t = true.flatten().astype(int)
    return {
        "IoU":       float(jaccard_score  (t, p, average="weighted", zero_division=0)),
        "Dice":      float(f1_score       (t, p, average="weighted", zero_division=0)),
        "Precision": float(precision_score(t, p, average="weighted", zero_division=0)),
        "Recall":    float(recall_score   (t, p, average="weighted", zero_division=0)),
    }

def _ci(arr):
    a  = np.asarray(arr, dtype=float)
    se = np.std(a, ddof=1) / np.sqrt(len(a))
    return (float(np.mean(a) - 1.96 * se), float(np.mean(a) + 1.96 * se))

def _summarise(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "std":  float(np.std(a, ddof=1)), "95_ci": list(_ci(a))}

# ─────────────────────────────────────────────
# Figure 7
# ─────────────────────────────────────────────
def make_figure7(triplets: list, out_path: Path):
    if not triplets:
        print("  [Figure 7] No triplets — skipping.")
        return

    cmap, norm = _damage_cmap_norm()
    n = len(triplets)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.8 * n), squeeze=False)
    fig.suptitle("Figure 7 — Real vs. Synthetic Damage Masks",
                 fontsize=12, fontweight="bold", y=1.01)
    axes[0][0].set_title("Real mask",      fontsize=10, pad=6)
    axes[0][1].set_title("Synthetic mask", fontsize=10, pad=6)

    for row, trip in enumerate(triplets):
        income = trip.get("income", "")
        event  = trip.get("event", income.upper())
        for col, key in enumerate(["real_mask_path", "syn_mask_path"]):
            ax  = axes[row][col]
            pth = Path(trip[key])
            if pth.exists():
                mask = np.clip(np.array(Image.open(pth).convert("L")),
                               0, config.N_CLASSES - 1)
                ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            ax.axis("off")
        axes[row][0].set_ylabel(f"{event}\n[{income.upper()}]",
                                fontsize=8, rotation=0, labelpad=60, va="center")

    fig.legend(handles=_damage_patches(), title="Damage class",
               title_fontsize=8, fontsize=7, loc="lower center",
               ncol=config.N_CLASSES, bbox_to_anchor=(0.5, -0.02),
               frameon=True, edgecolor="grey")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure 7] → {out_path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def evaluate_quality():
    print("=" * 60)
    print("07  —  Synthetic Image Quality Evaluation  (Study 1)")
    print("=" * 60)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    figure7_triplets = []

    for mask_type in ["binary", "multiclass"]:
        print(f"\n{mask_type.upper()} masks")
        all_results    = []
        income_results = {inc: [] for inc in config.income_levels}

        for income in config.income_levels:
            real_dir = config.masks_dir / "real"      / mask_type / income
            syn_dir  = config.masks_dir / "synthetic" / mask_type / income
            if not real_dir.exists():
                continue

            real_masks = {f.name: f for f in real_dir.glob("*.png")}
            pair_count = 0

            for real_name, real_path in tqdm(real_masks.items(),
                                             desc=f"  {income.upper()}"):
                syn_path = syn_dir / real_name
                if not syn_path.exists():
                    continue
                try:
                    real_mask = np.array(Image.open(real_path).convert("L"))
                    syn_mask  = np.array(Image.open(syn_path ).convert("L"))
                    metrics   = compute_mask_metrics(syn_mask, real_mask)

                    all_results.append({"filename": real_name, "income": income,
                                        **metrics})
                    income_results[income].append(metrics)
                    pair_count += 1

                    if mask_type == "multiclass" and pair_count == 1:
                        figure7_triplets.append({
                            "income": income,
                            "event":  real_name.split("_")[0],
                            "real_mask_path": str(real_path),
                            "syn_mask_path":  str(syn_path),
                        })
                except Exception as exc:
                    print(f"\n  [error] {real_name}: {exc}")

            print(f"    {income.upper()}: {pair_count} pairs evaluated")

        if not all_results:
            print(f"  No results for {mask_type}")
            continue

        # Per-income summaries  (no 'F1' key — fixed)
        income_summaries = {}
        for income in config.income_levels:
            vals = income_results[income]
            if not vals:
                continue
            income_summaries[income] = {
                "n_samples": len(vals),
                "IoU":       _summarise([m["IoU"]       for m in vals]),
                "Dice":      _summarise([m["Dice"]      for m in vals]),
                "Precision": _summarise([m["Precision"] for m in vals]),
                "Recall":    _summarise([m["Recall"]    for m in vals]),
            }

        overall = {
            "n_samples": len(all_results),
            "IoU":       _summarise([r["IoU"]       for r in all_results]),
            "Dice":      _summarise([r["Dice"]      for r in all_results]),
            "Precision": _summarise([r["Precision"] for r in all_results]),
            "Recall":    _summarise([r["Recall"]    for r in all_results]),
        }

        out_dir = config.output_dir / mask_type
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "per_image_results.json", "w") as fh:
            json.dump(all_results, fh, indent=2)
        with open(out_dir / "income_summaries.json", "w") as fh:
            json.dump(income_summaries, fh, indent=2)
        with open(out_dir / "overall_summary.json", "w") as fh:
            json.dump(overall, fh, indent=2)

        # Print
        ov = overall
        print(f"\n  Overall (n={ov['n_samples']}):")
        print(f"    IoU  {ov['IoU']['mean']:.4f} ± {ov['IoU']['std']:.4f}")
        print(f"    Dice {ov['Dice']['mean']:.4f} ± {ov['Dice']['std']:.4f}")
        for inc, sm in income_summaries.items():
            print(f"  {inc.upper()} (n={sm['n_samples']}):  "
                  f"IoU={sm['IoU']['mean']:.4f}  Dice={sm['Dice']['mean']:.4f}")

    print("\nGenerating Figure 7 …")
    make_figure7(figure7_triplets,
                 config.figures_dir / "figure7_damage_mask_comparison.png")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  Results → {config.output_dir}")
    print(f"  Figures → {config.figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_quality()
