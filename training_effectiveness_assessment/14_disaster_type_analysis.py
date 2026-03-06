"""
14_disaster_type_analysis.py  —  Reviewer 1 Concern 2: Disaster Type Confound
==============================================================================
Tests whether income-level IoU differences are driven by disaster TYPE rather
than income. Earthquake-heavy strata may differ from wildfire-heavy strata
independently of income.

Approach:
  1. Map each BRIGHT event to a disaster type.
  2. Kruskal-Wallis test: H0 = mIoU is the same across disaster types.
  3. Two-factor OLS: mIoU ~ income + disaster_type + income×disaster_type
     Compute partial η² for income after controlling for disaster type.
  4. Plot per-event performance bars (coloured by income).
  5. Plot difficulty vs income (difficulty = 1 − mean_baseline_IoU).

Outputs:
  results/study2/disaster_type/disaster_type_analysis.json
  results/study2/disaster_type/per_event_performance.png
  results/study2/disaster_type/difficulty_vs_income.png
"""

import sys, json
from pathlib import Path
from collections import defaultdict
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

EVAL_PATH = PROJECT_ROOT / "results" / "study2" / "evaluation" / "all_results.json"
OUT_DIR   = PROJECT_ROOT / "results" / "study2" / "disaster_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# BRIGHT event → broad disaster type
DISASTER_TYPE = {
    "haiti-earthquake":   "earthquake",
    "congo-volcano":      "volcano",
    "turkey-earthquake":  "earthquake",
    "morocco-earthquake": "earthquake",
    "libya-flood":        "flood",
    "bata-explosion":     "explosion",
    "beirut-explosion":   "explosion",
    "hawaii-wildfire":    "wildfire",
    "la_palma-volcano":   "volcano",
    "noto-earthquake":    "earthquake",
    "marshall-wildfire":  "wildfire",
}

INCOME_COLORS = {"lic": "#d73027", "mic": "#fee08b", "hic": "#1a9850"}
MODEL_NAMES   = ["unet", "segformer", "changeformer"]


def build_event_table(all_results: dict) -> list:
    rows = []
    for model_name, income_data in all_results.items():
        for income, stage_data in income_data.items():
            for stage in ["baseline", "stage1", "stage2"]:
                per_event = stage_data.get(stage, {}).get("per_event", {})
                for event, met in per_event.items():
                    rows.append({
                        "model":        model_name,
                        "income":       income,
                        "stage":        stage,
                        "event":        event,
                        "disaster_type": DISASTER_TYPE.get(event, "unknown"),
                        "mean_iou":     met.get("mean_iou", 0.0),
                        "n_tiles":      met.get("n_tiles", 0),
                    })
    return rows


def kruskal_wallis_by_type(rows: list, model: str, stage: str = "stage2") -> dict:
    """Kruskal-Wallis across disaster types for a given model/stage."""
    subset = [r for r in rows
              if r["model"] == model and r["stage"] == stage
              and r["disaster_type"] != "unknown"]

    groups = defaultdict(list)
    for r in subset:
        groups[r["disaster_type"]].append(r["mean_iou"])

    if len(groups) < 2:
        return {}

    stat, pval = kruskal(*groups.values())
    return {
        "H_statistic": float(stat),
        "p_value":     float(pval),
        "groups":      {k: {"mean": float(np.mean(v)), "n": len(v)}
                        for k, v in groups.items()},
    }


def two_factor_anova(rows: list, model: str, stage: str = "stage2") -> dict:
    """OLS: mIoU ~ income + disaster_type + income:disaster_type"""
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        print("  [warn] statsmodels/pandas not installed. pip install statsmodels pandas")
        return {}

    subset = [r for r in rows
              if r["model"] == model and r["stage"] == stage
              and r["disaster_type"] != "unknown"]
    if len(subset) < 6:
        return {}

    df = pd.DataFrame(subset)
    df["income"] = df["income"].astype("category")
    df["disaster_type"] = df["disaster_type"].astype("category")

    try:
        model_ols = smf.ols(
            "mean_iou ~ C(income) + C(disaster_type) + C(income):C(disaster_type)",
            data=df).fit()
    except Exception as exc:
        print(f"  [warn] OLS failed: {exc}")
        return {}

    # Partial η² for income = SS_income / SS_total
    from statsmodels.stats.anova import anova_lm
    try:
        anova_table = anova_lm(model_ols, typ=2)
        ss_income   = float(anova_table.loc["C(income)", "sum_sq"])
        ss_total    = float(anova_table["sum_sq"].sum())
        partial_eta2 = ss_income / ss_total
    except Exception:
        partial_eta2 = float("nan")

    return {
        "r_squared":           float(model_ols.rsquared),
        "partial_eta2_income": partial_eta2,
        "n":                   len(subset),
    }


def plot_per_event_performance(rows: list, out_path: Path):
    """Horizontal bar chart: per-event mIoU (stage2) coloured by income."""
    subset = [r for r in rows if r["stage"] == "stage2" and r["model"] == "unet"]
    if not subset:
        subset = [r for r in rows if r["stage"] == "baseline"]
    if not subset:
        return

    subset.sort(key=lambda r: r["mean_iou"])
    events  = [r["event"] for r in subset]
    iou_vals = [r["mean_iou"] for r in subset]
    colors  = [INCOME_COLORS[r["income"]] for r in subset]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(events))))
    bars = ax.barh(events, iou_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean IoU (stage2)", fontsize=10)
    ax.set_title("Per-Event Segmentation Performance", fontsize=11, fontweight="bold")
    ax.axvline(np.mean(iou_vals), color="black", linestyle="--",
               linewidth=1, label=f"Mean={np.mean(iou_vals):.3f}")

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=i.upper(), edgecolor="black")
                  for i, c in INCOME_COLORS.items()]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path.name}")


def plot_difficulty_vs_income(rows: list, out_path: Path):
    """Scatter: difficulty (1-baseline_IoU) vs income level for each event."""
    subset = [r for r in rows if r["stage"] == "baseline"]

    # Average across models per event
    by_event: dict = defaultdict(list)
    for r in subset:
        by_event[(r["event"], r["income"])].append(r["mean_iou"])

    events_data = {
        (ev, inc): float(np.mean(vals))
        for (ev, inc), vals in by_event.items()
    }

    income_order = {"lic": 0, "mic": 1, "hic": 2}
    xs, ys, cs, labels = [], [], [], []
    for (ev, inc), mean_iou in events_data.items():
        xs.append(income_order[inc] + np.random.uniform(-0.15, 0.15))
        ys.append(1.0 - mean_iou)   # difficulty
        cs.append(INCOME_COLORS[inc])
        labels.append(ev.split("-")[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=cs, s=70, edgecolors="black", linewidths=0.5)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=6, xytext=(4, 2),
                    textcoords="offset points")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["LIC", "MIC", "HIC"], fontsize=10)
    ax.set_ylabel("Difficulty  (1 − baseline mIoU)", fontsize=10)
    ax.set_title("Disaster Difficulty vs Income Level", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path.name}")


def main():
    print("=" * 60)
    print("14  —  Disaster Type Confound Analysis  (Reviewer 1, Concern 2)")
    print("=" * 60)

    if not EVAL_PATH.exists():
        print(f"  [error] {EVAL_PATH}  — run 11_evaluation.py first")
        return

    with open(EVAL_PATH) as fh:
        all_results = json.load(fh)

    rows = build_event_table(all_results)
    print(f"  Total rows: {len(rows)}")

    results = {}

    for model in MODEL_NAMES:
        print(f"\n  Model: {model.upper()}")
        kw = kruskal_wallis_by_type(rows, model)
        ov = two_factor_anova(rows, model)
        results[model] = {"kruskal_wallis": kw, "two_factor_ols": ov}

        if kw:
            print(f"    Kruskal-Wallis  H={kw['H_statistic']:.2f}  "
                  f"p={kw['p_value']:.4f}")
        if ov:
            print(f"    OLS R²={ov['r_squared']:.3f}  "
                  f"partial η²(income)={ov['partial_eta2_income']:.3f}")

    with open(OUT_DIR / "disaster_type_analysis.json", "w") as fh:
        json.dump(results, fh, indent=2)

    plot_per_event_performance(rows, OUT_DIR / "per_event_performance.png")
    plot_difficulty_vs_income(rows, OUT_DIR / "difficulty_vs_income.png")

    print(f"\nResults → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
