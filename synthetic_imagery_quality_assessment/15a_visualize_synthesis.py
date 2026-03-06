"""
15a_visualize_synthesis.py  —  Study 1: DisasterGAN Qualitative Visualization
===============================================================================
Addresses Reviewer 1 Concern 4: "The paper lacks qualitative visualizations
of the synthetic imagery."

For each xBD disaster type, shows up to N_EXAMPLES rows × 4 columns:
    [Pre-event optical] | [Real post-event optical] |
    [Synthetic (DisasterGAN)] | [Damage mask]

A consistent discrete colormap is used for all mask panels.

Outputs:
    results/study1/figures/synthesis_vis/<type>_synthesis_grid.png
    results/study1/figures/synthesis_vis/all_types_overview.png
"""

import sys, json, random, argparse
from pathlib import Path
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm

RESULTS_DIR   = PROJECT_ROOT / "results" / "study1"
FIGURES_DIR   = RESULTS_DIR / "figures" / "synthesis_vis"
MANIFEST_PATH = RESULTS_DIR / "xbd_test_manifest.json"
XBD_ROOT      = PROJECT_ROOT / "data" / "xbd"
SYNTH_DIR     = RESULTS_DIR / "synthetic_images"

N_EXAMPLES = 5

# Damage class colours (consistent across all scripts)
CLASS_COLORS = ["#d9d9d9", "#fee08b", "#f46d43", "#d73027"]
CLASS_LABELS = ["No damage", "Minor damage", "Major damage", "Destroyed"]
N_CLASSES    = 4

# Disaster type groups (xBD events)
DISASTER_TYPE_GROUPS = {
    "earthquake":       ["mexico-earthquake", "nepal-earthquake", "noto-earthquake"],
    "flood_tsunami":    ["midwest-flooding", "nepal-flooding", "palu-tsunami", "sunda-strait"],
    "hurricane_tornado":["hurricane-florence", "hurricane-harvey", "hurricane-matthew",
                         "hurricane-michael", "joplin-tornado", "moore-tornado",
                         "tuscaloosa-tornado"],
    "wildfire":         ["portugal-wildfire", "santa-rosa-wildfire", "socal-fire",
                         "woolsey-fire"],
    "volcano":          ["guatemala-volcano", "lower-puna-volcano"],
}

XBD_EVENT_INCOME = {
    "guatemala-volcano": "LIC", "hurricane-matthew": "LIC", "nepal-earthquake": "LIC",
    "nepal-flooding": "LIC", "palu-tsunami": "MIC", "sunda-strait": "MIC",
    "mexico-earthquake": "MIC", "portugal-wildfire": "HIC", "hurricane-florence": "HIC",
    "hurricane-harvey": "HIC", "hurricane-michael": "HIC", "joplin-tornado": "HIC",
    "lower-puna-volcano": "HIC", "midwest-flooding": "HIC", "moore-tornado": "HIC",
    "noto-earthquake": "HIC", "santa-rosa-wildfire": "HIC", "socal-fire": "HIC",
    "tuscaloosa-tornado": "HIC", "woolsey-fire": "HIC",
}

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def damage_cmap_norm():
    cmap   = mcolors.ListedColormap(CLASS_COLORS)
    norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm

def damage_patches():
    return [mpatches.Patch(facecolor=c, label=l, edgecolor="black", linewidth=0.5)
            for c, l in zip(CLASS_COLORS, CLASS_LABELS)]

def load_rgb(path, size=(256,256)):
    p = Path(path)
    if not p.exists():
        return np.full((*size, 3), 200, dtype=np.uint8)
    return np.array(Image.open(p).convert("RGB").resize(size, Image.BILINEAR))

def load_mask(path, size=(256,256)):
    p = Path(path)
    if not p.exists():
        return np.zeros(size, dtype=np.uint8)
    return np.clip(np.array(Image.open(p).convert("L").resize(size, Image.NEAREST)),
                   0, N_CLASSES - 1)

def find_synthetic(stem: str) -> Path:
    for p in [SYNTH_DIR / f"{stem}_synthetic.png",
              SYNTH_DIR / f"{stem}.png"]:
        if p.exists():
            return p
    cands = list(SYNTH_DIR.rglob(f"*{stem}*.png"))
    return cands[0] if cands else Path("__missing__")

def load_manifest() -> list:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as fh:
            return json.load(fh)
    # Fallback: scan xBD test directory
    records = []
    for pre in sorted(XBD_ROOT.rglob("*_pre_disaster.*")):
        if pre.suffix.lower() not in EXTS:
            continue
        stem  = pre.stem.replace("_pre_disaster", "")
        post  = pre.parent / f"{stem}_post_disaster{pre.suffix}"
        mask  = pre.parent.parent / "labels" / f"{stem}_post_disaster_mask.png"
        event = "_".join(stem.split("_")[:-1])
        records.append({"event": event, "pre_img": str(pre),
                        "post_img": str(post), "mask": str(mask)})
    return records


def make_type_grid(dtype: str, samples: list, out_path: Path):
    cmap, norm = damage_cmap_norm()
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n), squeeze=False,
                             gridspec_kw={"hspace": 0.05, "wspace": 0.03})
    for col, title in enumerate(["Pre-event\n(optical)", "Real post-event\n(optical)",
                                  "Synthetic\n(DisasterGAN)", "Damage mask\n(labels)"]):
        axes[0][col].set_title(title, fontsize=9, pad=6, fontweight="bold")

    for row, sample in enumerate(samples):
        event = sample.get("event", "")
        income = XBD_EVENT_INCOME.get(event, "?")
        stem  = Path(sample.get("pre_img", "")).stem.replace("_pre_disaster", "")
        pre   = load_rgb(sample.get("pre_img",  "__missing__"))
        post  = load_rgb(sample.get("post_img", "__missing__"))
        synth = load_rgb(find_synthetic(stem))
        mask  = load_mask(sample.get("mask",    "__missing__"))

        for col, (panel, is_mask) in enumerate(
                [(pre, False), (post, False), (synth, False), (mask, True)]):
            ax = axes[row][col]
            ax.axis("off")
            if is_mask:
                ax.imshow(panel, cmap=cmap, norm=norm, interpolation="nearest")
            else:
                ax.imshow(panel)
        axes[row][0].set_ylabel(f"{event}\n[{income}]",
                                fontsize=7, rotation=0, labelpad=70, va="center")

    fig.legend(handles=damage_patches(), title="Damage class", title_fontsize=8,
               fontsize=7, loc="lower center", ncol=N_CLASSES,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor="grey")
    fig.suptitle(f"Study 1 — DisasterGAN: {dtype.replace('_',' ').title()} Events",
                 fontsize=11, fontweight="bold", y=1.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def make_overview(type_paths: dict, out_path: Path):
    valid = {k: v for k, v in type_paths.items() if v.exists()}
    if not valid:
        return
    imgs  = [Image.open(p) for p in valid.values()]
    max_w = max(im.width for im in imgs)
    tot_h = sum(im.height for im in imgs)
    canvas = Image.new("RGB", (max_w, tot_h), (255, 255, 255))
    y = 0
    for im in imgs:
        pad = Image.new("RGB", (max_w, im.height), (255, 255, 255))
        pad.paste(im)
        canvas.paste(pad, (0, y)); y += im.height
    canvas.save(out_path, dpi=(180, 180))
    print(f"  Overview → {out_path.name}")


def main(seed=42, n_examples=N_EXAMPLES):
    random.seed(seed); np.random.seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("15a  —  DisasterGAN Synthesis Visualization  (Study 1)")
    print("=" * 60)

    manifest  = load_manifest()
    event_idx = {}
    for r in manifest:
        event_idx.setdefault(r["event"], []).append(r)

    type_paths = {}
    for dtype, events in DISASTER_TYPE_GROUPS.items():
        print(f"\nType: {dtype.upper()}")
        pool = [r for e in events for r in event_idx.get(e, [])]
        if not pool:
            print("  [warn] No tiles found"); continue
        samples   = random.sample(pool, min(n_examples, len(pool)))
        out_path  = FIGURES_DIR / f"{dtype}_synthesis_grid.png"
        type_paths[dtype] = out_path
        make_type_grid(dtype, samples, out_path)

    make_overview(type_paths, FIGURES_DIR / "all_types_overview.png")
    print(f"\nDone → {FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    args = ap.parse_args()
    main(args.seed, args.n_examples)
