"""
13_sample_size_correlation.py  —  Reviewer 1 Concern 1: Sample Size Confound
=============================================================================
Tests whether the observed fine-tuning improvement (ΔIoU) is merely an
artefact of the number of training tiles available per income stratum.

Approach:
  1. Load per-event baseline and stage2 mIoU from 11_evaluation.py output.
  2. Compute ΔIoU = stage2_mIoU − baseline_mIoU  for each event × model.
  3. Spearman ρ(log n_tiles, ΔIoU)  — tests monotonic association.
  4. OLS regression: ΔIoU ~ log(n_tiles) + income_dummies
     → partial effect of income after controlling for sample size.

Outputs:
  results/study2/sample_size/correlation_results.json
  results/study2/sample_size/sample_size_vs_delta_iou.png
"""

import sys, json
from pathlib import Path
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

EVAL_PATH  = PROJECT_ROOT / "results" / "study2" / "evaluation" / "all_results.json"
OUT_DIR    = PROJECT_ROOT / "results" / "study2" / "sample_size"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INCOME_COLORS = {"lic": "#d73027", "mic": "#fee08b", "hic": "#1a9850"}
MODEL_NAMES   = ["unet", "segformer", "changeformer"]


def build_event_table(all_results: dict) -> list:
    """
    For each (model, event) pair, compute:
      n_tiles, baseline_iou, stage2_iou, delta_iou, income
    """
    rows = []
    for model_name, income_data in all_results.items():
        for income, stage_data in income_data.items():
            base_events  = stage_data.get("baseline", {}).get("per_event", {})
            stage2_events = stage_data.get("stage2",   {}).get("per_event", {})

            for event, bmet in base_events.items():
                if event not in stage2_events:
                    continue
                s2met = stage2_events[event]
                rows.append({
                    "model":        model_name,
                    "event":        event,
                    "income":       income,
                    "n_tiles":      bmet.get("n_tiles", 0),
                    "baseline_iou": bmet.get("mean_iou", 0.0),
                    "stage2_iou":   s2met.get("mean_iou", 0.0),
                    "delta_iou":    s2met.get("mean_iou", 0.0) - bmet.get("mean_iou", 0.0),
                })
    return rows


def spearman_analysis(rows: list) -> dict:
    """Spearman ρ per model."""
    results = {}
    for model in MODEL_NAMES:
        subset = [r for r in rows if r["model"] == model and r["n_tiles"] > 0]
        if len(subset) < 3:
            continue
        log_n   = np.log([r["n_tiles"] for r in subset])
        delta   = np.array([r["delta_iou"] for r in subset])
        rho, pval = spearmanr(log_n, delta)
        results[model] = {"rho": float(rho), "p_value": float(pval), "n": len(subset)}
        print(f"  {model:<15}  ρ={rho:+.3f}  p={pval:.4f}  n={len(subset)}")
    return results


def regression_analysis(rows: list) -> dict:
    """OLS: ΔIoU ~ log(n_tiles) + income_dummies — checks partial effect."""
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [warn] statsmodels not installed — skipping OLS. pip install statsmodels")
        return {}

    results = {}
    for model in MODEL_NAMES:
        subset = [r for r in rows if r["model"] == model and r["n_tiles"] > 0]
        if len(subset) < 5:
            continue

        log_n   = np.log([r["n_tiles"]  for r in subset])
        delta   = np.array([r["delta_iou"] for r in subset])
        is_lic  = np.array([1.0 if r["income"] == "lic" else 0.0 for r in subset])
        is_hic  = np.array([1.0 if r["income"] == "hic" else 0.0 for r in subset])
        # MIC is the reference category

        X = sm.add_constant(np.column_stack([log_n, is_lic, is_hic]))
        ols = sm.OLS(delta, X).fit()

        results[model] = {
            "coef_log_n":  float(ols.params[1]),
            "pval_log_n":  float(ols.pvalues[1]),
            "coef_lic":    float(ols.params[2]),
            "coef_hic":    float(ols.params[3]),
            "r_squared":   float(ols.rsquared),
            "n":           len(subset),
        }
        print(f"  {model}  OLS R²={ols.rsquared:.3f}  "
              f"β_log_n={ols.params[1]:+.4f} (p={ols.pvalues[1]:.3f})")
    return results


def plot_scatter(rows: list, out_path: Path):
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5 * len(MODEL_NAMES), 5),
                             sharey=True)
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_NAMES):
        subset = [r for r in rows if r["model"] == model and r["n_tiles"] > 0]
        if not subset:
            ax.set_title(model); continue

        log_n  = np.log([r["n_tiles"]  for r in subset])
        delta  = np.array([r["delta_iou"] for r in subset])
        colors = [INCOME_COLORS[r["income"]] for r in subset]
        labels = [r["event"] for r in subset]

        ax.scatter(log_n, delta, c=colors, s=60, edgecolors="black", linewidths=0.5)

        for x, y, lbl in zip(log_n, delta, labels):
            ax.annotate(lbl.split("-")[0], (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

        # Trend line
        if len(log_n) >= 2:
            m, b = np.polyfit(log_n, delta, 1)
            xs = np.linspace(log_n.min(), log_n.max(), 50)
            ax.plot(xs, m * xs + b, "k--", linewidth=1)

        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel("log(n tiles)", fontsize=9)
        ax.set_ylabel("ΔIoU (stage2 − baseline)", fontsize=9)
        ax.set_title(model.capitalize(), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Shared legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=i.upper(), edgecolor="black")
                  for i, c in INCOME_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Sample Size vs. Fine-Tuning Improvement (ΔIoU)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path.name}")


def main():
    print("=" * 60)
    print("13  —  Sample Size Confound Analysis  (Reviewer 1, Concern 1)")
    print("=" * 60)

    if not EVAL_PATH.exists():
        print(f"  [error] Evaluation results not found at {EVAL_PATH}")
        print("  Run 11_evaluation.py first.")
        return

    with open(EVAL_PATH) as fh:
        all_results = json.load(fh)

    rows = build_event_table(all_results)
    if not rows:
        print("  [warn] No per-event data found in evaluation results.")
        return

    print(f"  Total event×model rows: {len(rows)}")

    print("\nSpearman ρ  (log n_tiles  vs  ΔIoU):")
    spearman_res = spearman_analysis(rows)

    print("\nOLS regression  (ΔIoU ~ log n + income dummies):")
    ols_res = regression_analysis(rows)

    results = {"spearman": spearman_res, "ols": ols_res, "n_rows": len(rows)}

    with open(OUT_DIR / "correlation_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    plot_scatter(rows, OUT_DIR / "sample_size_vs_delta_iou.png")

    print(f"\nResults → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
