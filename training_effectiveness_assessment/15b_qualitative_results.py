"""
15b_qualitative_results.py  —  Study 2: BRIGHT Segmentation Qualitative Results
=================================================================================
Addresses Reviewer 2 visual concern: "Need more qualitative results."

For each income stratum, produces:
  • Per-model grids:   pre-optical | post-SAR | GT mask | prediction
  • All-model comparison: adds columns for U-Net, SegFormer, ChangeFormer side-by-side

Uses Stage 2 (full fine-tune) checkpoints; falls back to Stage 1 / baseline.

Outputs:
  results/study2/figures/qualitative/
      <income>_<model>_prediction_grid.png
      all_models_<income>_overview.png
"""

import sys, json, random, argparse
from pathlib import Path
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from _08_shared import INCOME_LEVELS, load_combined_manifest

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR    = PROJECT_ROOT / "results" / "study2" / "figures" / "qualitative"

N_EXAMPLES    = 5
MODEL_NAMES   = ["unet", "segformer", "changeformer"]
IMG_SIZE      = (256, 256)

CLASS_COLORS = ["#d9d9d9", "#fee08b", "#f46d43", "#d73027"]
CLASS_LABELS = ["No damage", "Minor damage", "Major damage", "Destroyed"]
N_CLASSES    = 4


def damage_cmap_norm():
    cmap = mcolors.ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm

def damage_patches():
    return [mpatches.Patch(facecolor=c, label=l, edgecolor="black", linewidth=0.5)
            for c, l in zip(CLASS_COLORS, CLASS_LABELS)]

def load_rgb(path):
    p = Path(path)
    if not p.exists():
        return np.full((*IMG_SIZE, 3), 200, dtype=np.uint8)
    return np.array(Image.open(p).convert("RGB").resize(IMG_SIZE, Image.BILINEAR))

def load_sar(path):
    p = Path(path)
    if not p.exists():
        return np.full(IMG_SIZE, 128, dtype=np.uint8)
    arr = np.array(Image.open(p).convert("L").resize(IMG_SIZE, Image.BILINEAR),
                   dtype=np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    return arr.astype(np.uint8)

def load_mask(path):
    p = Path(path)
    if not p.exists():
        return np.zeros(IMG_SIZE, dtype=np.uint8)
    return np.clip(np.array(Image.open(p).convert("L").resize(IMG_SIZE, Image.NEAREST)),
                   0, N_CLASSES - 1)

# ── Model inference ────────────────────────────────────────────────────────
def load_model_from_checkpoint(model_name: str, income: str, device):
    try:
        from _model_registry import load_model
        import torch
    except ImportError:
        return None, None

    for stage in ["stage2", "stage1", "baseline"]:
        ckpt = CHECKPOINT_DIR / f"{model_name}_{stage}_{income}.pth"
        if ckpt.exists():
            model = load_model(model_name).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            model.eval()
            print(f"    Loaded {ckpt.name}")
            return model, device
    return None, None


def run_inference(model, device, pre_img, sar_img, model_name):
    try:
        import torch, torch.nn.functional as F
    except ImportError:
        return np.zeros(IMG_SIZE, dtype=np.uint8)

    if model is None:
        return np.zeros(IMG_SIZE, dtype=np.uint8)

    pre_t = torch.from_numpy(pre_img.astype(np.float32)/255.).permute(2,0,1)
    sar_t = torch.from_numpy(sar_img.astype(np.float32)/255.).unsqueeze(0)

    with torch.no_grad():
        if model_name == "changeformer":
            sar_3 = sar_t.expand(3,-1,-1)
            logits = model(pre_t.unsqueeze(0).to(device),
                           sar_3.unsqueeze(0).to(device))
        else:
            x = torch.cat([pre_t, sar_t], dim=0).unsqueeze(0).to(device)
            logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
            logits = F.interpolate(logits, size=IMG_SIZE, mode="bilinear",
                                   align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


# ── Per-model grid ─────────────────────────────────────────────────────────
def make_model_grid(income, model_name, samples, model, device, out_path):
    cmap, norm = damage_cmap_norm()
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2*n), squeeze=False,
                             gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    for col, title in enumerate(["Pre-event (optical)", "Post-event (SAR)",
                                  "Ground truth", f"{model_name.capitalize()} pred"]):
        axes[0][col].set_title(title, fontsize=9, pad=6, fontweight="bold")

    for row, rec in enumerate(samples):
        pre  = load_rgb (rec.get("pre_img",  ""))
        sar  = load_sar (rec.get("post_img", ""))
        gt   = load_mask(rec.get("mask",     ""))
        pred = run_inference(model, device, pre, sar, model_name)

        for col, (panel, is_mask) in enumerate(
                [(pre,False),(sar,False),(gt,True),(pred,True)]):
            ax = axes[row][col]; ax.axis("off")
            if is_mask:
                ax.imshow(panel, cmap=cmap, norm=norm, interpolation="nearest")
            elif panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
        axes[row][0].set_ylabel(f"{rec.get('event',income)}\n[{income.upper()}]",
                                fontsize=7, rotation=0, labelpad=70, va="center")

    fig.legend(handles=damage_patches(), title="Damage class", title_fontsize=8,
               fontsize=7, loc="lower center", ncol=N_CLASSES,
               bbox_to_anchor=(0.5,-0.01), frameon=True, edgecolor="grey")
    fig.suptitle(f"Study 2 — {model_name.capitalize()} [{income.upper()}]",
                 fontsize=11, fontweight="bold", y=1.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path.name}")


# ── All-model comparison ────────────────────────────────────────────────────
def make_comparison_grid(income, samples, model_map, out_path):
    cmap, norm = damage_cmap_norm()
    n      = len(samples)
    n_cols = 3 + len(MODEL_NAMES)
    fig, axes = plt.subplots(n, n_cols, figsize=(3.5*n_cols, 3.2*n), squeeze=False,
                             gridspec_kw={"hspace": 0.04, "wspace": 0.03})
    col_titles = (["Pre-event\n(optical)", "Post-event\n(SAR)", "Ground truth"]
                  + [m.capitalize() for m in MODEL_NAMES])
    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=9, pad=6, fontweight="bold")

    for row, rec in enumerate(samples):
        pre  = load_rgb (rec.get("pre_img",  ""))
        sar  = load_sar (rec.get("post_img", ""))
        gt   = load_mask(rec.get("mask",     ""))
        preds = {mn: run_inference(*model_map[mn], pre, sar, mn)
                 for mn in MODEL_NAMES}

        panels  = [pre, sar, gt] + [preds[mn] for mn in MODEL_NAMES]
        is_mask = [False, False, True] + [True]*len(MODEL_NAMES)
        for col, (panel, is_m) in enumerate(zip(panels, is_mask)):
            ax = axes[row][col]; ax.axis("off")
            if is_m:
                ax.imshow(panel, cmap=cmap, norm=norm, interpolation="nearest")
            elif panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
        axes[row][0].set_ylabel(f"{rec.get('event',income)}\n[{income.upper()}]",
                                fontsize=7, rotation=0, labelpad=70, va="center")

    fig.legend(handles=damage_patches(), title="Damage class", title_fontsize=8,
               fontsize=7, loc="lower center", ncol=N_CLASSES,
               bbox_to_anchor=(0.5,-0.01), frameon=True, edgecolor="grey")
    fig.suptitle(f"Study 2 — All Models [{income.upper()}]",
                 fontsize=11, fontweight="bold", y=1.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Comparison → {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────
def main(seed=42, n_examples=N_EXAMPLES):
    random.seed(seed); np.random.seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None

    print("=" * 60)
    print("15b  —  BRIGHT Segmentation Qualitative Results  (Study 2)")
    print("=" * 60)

    for income in INCOME_LEVELS:
        print(f"\n{'─'*50}\nStratum: {income.upper()}\n{'─'*50}")
        manifest = load_combined_manifest(income)
        if not manifest:
            print("  [skip] No manifest"); continue

        # Diverse sampling: round-robin across events
        from collections import defaultdict
        eg: dict = defaultdict(list)
        for r in manifest:
            eg[r.get("event", "unk")].append(r)
        evlists = list(eg.values())
        samples, idx = [], 0
        while len(samples) < min(n_examples, len(manifest)):
            g = evlists[idx % len(evlists)]
            if g: samples.append(g.pop(random.randint(0, len(g)-1)))
            idx += 1
            if all(len(g)==0 for g in evlists): break

        print(f"  Sampled {len(samples)} tiles")

        # Load models
        model_map = {}
        for mn in MODEL_NAMES:
            print(f"  Loading {mn} …")
            m, d = load_model_from_checkpoint(mn, income, device)
            model_map[mn] = (m, d)

        # Per-model grids
        for mn in MODEL_NAMES:
            out = FIGURES_DIR / f"{income}_{mn}_prediction_grid.png"
            make_model_grid(income, mn, samples, *model_map[mn], out)

        # All-model comparison
        make_comparison_grid(income, samples, model_map,
                             FIGURES_DIR / f"all_models_{income}_overview.png")

    print(f"\nDone → {FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    args = ap.parse_args()
    main(args.seed, args.n_examples)
