#!/usr/bin/env python3
"""
evaluation.py
Evaluate a trained DisasterGAN generator on pre/post image pairs with:
 - binary damage mask (SAR-like) evaluation
 - multiclass damage mask evaluation (minor/major/destroyed)
 - supports country-strata (LIC / MIC / HIC) filtering by filename
 - saves per-pair JSON, per-disaster summaries, overall summary, and visualization images

Usage examples:
    python evaluation.py --model ./saved_models/G_final.pth --pre-dir ./pre-event --post-dir ./post-event --out ./results/lic_binary --strata LIC --mode binary
    python evaluation.py --model ./saved_models/G_final.pth --pre-dir ./pre-event --post-dir ./post-event --out ./results/lic_multiclass --strata LIC --mode multiclass
"""

import os
import argparse
import json
import glob
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import transforms as T
from torchvision.models import vgg16

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from scipy.ndimage import binary_opening, binary_closing

# -------------------------
# Config (override via CLI)
# -------------------------
DEFAULT_DISASTER_TYPES = [
    'volcano', 'fire', 'tornado', 'tsunami',
    'flooding', 'earthquake', 'hurricane'
]

COUNTRY_STRATA = {
    "LIC": ["haiti", "congo"],
    "MIC": ["turkey", "morocco", "libya"],
    "HIC": ["noto", "la_palma", "hawaii"],
}

# -------------------------
# Dataset-like helper (pair listing)
# -------------------------
def make_pairs_from_dirs(pre_dir, post_dir, filter_countries=None):
    """
    Find matching pre/post files in directories. Optionally filter pre filenames by
    presence of any token in filter_countries list (case-insensitive).
    """
    pre_files = sorted([f for f in os.listdir(pre_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    pairs = []
    for pre_fname in pre_files:
        if filter_countries:
            if not any(tok.lower() in pre_fname.lower() for tok in filter_countries):
                continue
        post_fname = pre_fname.replace('_pre_disaster', '_post_disaster')
        pre_path = os.path.join(pre_dir, pre_fname)
        post_path = os.path.join(post_dir, post_fname)
        if os.path.exists(post_path):
            pairs.append((pre_path, post_path))
    return pairs

# -------------------------
# Recreate the generator architecture (same as training)
# -------------------------
class DisasterGenerator(nn.Module):
    def __init__(self, disaster_types=DEFAULT_DISASTER_TYPES):
        super().__init__()
        # encoder
        self.enc1 = nn.Sequential(spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))

        # decoders
        self.dec_img = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

        self.dec_mask = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid()
        )

        self.disaster_types = disaster_types

    def add_disaster_channel(self, x, disaster):
        batch_size, _, h, w = x.size()
        disaster_map = disaster.view(-1, 1, 1, 1).expand(-1, -1, h, w).float() / max(1, len(self.disaster_types))
        return torch.cat([x, disaster_map], dim=1)

    def forward(self, x, disaster):
        # x: (B,3,H,W) pre image; add disaster to make 4 channels
        x = self.add_disaster_channel(x, disaster)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        img = self.dec_img(e4)
        mask = self.dec_mask(e4)
        return img, mask

# -------------------------
# Utility functions
# -------------------------
transform_rgb = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
transform_gray = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5], [0.5])])

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5) + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    np_img = (tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def infer_disaster_type_from_filename(filename, disaster_types=DEFAULT_DISASTER_TYPES):
    name = os.path.basename(filename).lower()
    for d in disaster_types:
        if d in name:
            return d
    return None

# -------------------------
# SAR-style binary mask function
# -------------------------
def SAR_damage_mask(pre_optical_img: Image.Image, post_sar_img: Image.Image, threshold=0.1, device='cpu'):
    """
    Convert optical to a SAR-like grayscale, compare with post SAR-like, and produce binary mask.
    Returns tensor of shape (1, H, W) as float {0,1}.
    """
    def optical_to_sar_like(img):
        img = img.convert('L')
        img = ImageOps.autocontrast(img, cutoff=2)
        return img

    pre_sar_like = optical_to_sar_like(pre_optical_img)
    post_sar = post_sar_img.convert('L')

    pre_tensor = transform_gray(pre_sar_like).to(device)
    post_tensor = transform_gray(post_sar).to(device)

    with torch.no_grad():
        diff = torch.abs(post_tensor - pre_tensor).unsqueeze(0)  # (1,1,H,W)
        diff = F.avg_pool2d(diff, kernel_size=3, stride=1, padding=1)
        img_std = diff.std().item()
        adaptive_threshold = threshold + (0.1 * img_std)
        mask = (diff > adaptive_threshold).float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return mask.squeeze(0)  # (1,H,W) -> (H,W) if needed by indexing

# -------------------------
# Multiclass mask (0=no,1=minor,2=major,3=destroyed)
# -------------------------
def clean_multiclass_mask(mask_np, min_region_size=20):
    cleaned_mask = np.zeros_like(mask_np)
    for cls in range(1, 4):
        binary = (mask_np == cls)
        binary = binary_opening(binary, structure=np.ones((3,3)))
        binary = binary_closing(binary, structure=np.ones((3,3)))
        cleaned_mask[binary] = cls
    return cleaned_mask

def SAR_damage_mask_multiclass_merged(pre_optical_img: Image.Image, post_sar_img: Image.Image, device='cpu'):
    def optical_to_sar_like(img):
        img = img.convert('L')
        img = ImageOps.autocontrast(img, cutoff=2)
        return img

    pre_sar_like = optical_to_sar_like(pre_optical_img)
    post_sar = post_sar_img.convert('L')

    pre_tensor = transform_gray(pre_sar_like).to(device).squeeze(0)
    post_tensor = transform_gray(post_sar).to(device).squeeze(0)

    with torch.no_grad():
        diff = torch.abs(post_tensor - pre_tensor)  # (H,W)
        diff_smoothed = F.avg_pool2d(diff.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()

        diff_min, diff_max = diff_smoothed.min().item(), diff_smoothed.max().item()
        diff_norm = (diff_smoothed - diff_min) / (diff_max - diff_min + 1e-8)

        thresholds = [0.08, 0.22, 0.45]
        mask = torch.zeros_like(diff_norm, dtype=torch.int64)
        mask = torch.where((diff_norm >= thresholds[0]) & (diff_norm < thresholds[1]), 1, mask)
        mask = torch.where((diff_norm >= thresholds[1]) & (diff_norm < thresholds[2]), 2, mask)
        mask = torch.where(diff_norm >= thresholds[2], 3, mask)

    mask_np = mask.cpu().numpy().astype(np.int32)
    cleaned = clean_multiclass_mask(mask_np)
    return torch.from_numpy(cleaned).long()

# -------------------------
# Metrics
# -------------------------
def compute_binary_metrics(pred_mask, true_mask):
    """
    pred_mask, true_mask: torch tensors (H,W) or (1,H,W) with values 0/1 (or floats)
    returns dict of IoU, Dice (F1), Precision, Recall
    """
    pred = (pred_mask.detach().cpu().numpy().astype(int)).ravel()
    true = (true_mask.detach().cpu().numpy().astype(int)).ravel()
    # If all zeros or trivial, sklearn may error — handle small edge cases
    try:
        iou = jaccard_score(true, pred)
    except Exception:
        iou = float(np.nan)
    try:
        f1 = f1_score(true, pred)
    except Exception:
        f1 = float(np.nan)
    try:
        prec = precision_score(true, pred)
    except Exception:
        prec = float(np.nan)
    try:
        rec = recall_score(true, pred)
    except Exception:
        rec = float(np.nan)
    return {'IoU': iou, 'Dice': f1, 'F1': f1, 'Precision': prec, 'Recall': rec}

def compute_multiclass_metrics(pred_mask, true_mask):
    pred = pred_mask.flatten().cpu().numpy().astype(int)
    true = true_mask.flatten().cpu().numpy().astype(int)
    return {
        'IoU': jaccard_score(true, pred, average='weighted'),
        'Dice': f1_score(true, pred, average='weighted'),
        'F1': f1_score(true, pred, average='weighted'),
        'Precision': precision_score(true, pred, average='weighted'),
        'Recall': recall_score(true, pred, average='weighted')
    }

def compute_ci(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return [float('nan'), float('nan')]
    mean = np.mean(arr)
    std_err = np.std(arr) / np.sqrt(len(arr))
    return [float(mean - 1.96 * std_err), float(mean + 1.96 * std_err)]

# -------------------------
# Visualization helpers
# -------------------------
damage_colors = ['black', 'blue', 'yellow', 'red']  # classes 0..3
cmap = mcolors.ListedColormap(damage_colors)
norm = mcolors.BoundaryNorm(boundaries=[-0.5,0.5,1.5,2.5,3.5], ncolors=len(damage_colors))

# -------------------------
# Main evaluation runner
# -------------------------
def run_evaluation(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    # instantiate generator and load weights
    G = DisasterGenerator()
    G.to(device)
    state = torch.load(args.model, map_location=device)
    # allow state dicts that may be wrapped
    if 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    G.load_state_dict(state)
    G.eval()
    print("Generator loaded.")

    # create pairs based on strata
    filter_countries = None
    if args.strata:
        if args.strata not in COUNTRY_STRATA:
            raise ValueError(f"Unknown strata {args.strata}; choose from {list(COUNTRY_STRATA.keys())}")
        filter_countries = COUNTRY_STRATA[args.strata]
        print(f"Filtering pre filenames for countries: {filter_countries}")

    pairs = make_pairs_from_dirs(args.pre_dir, args.post_dir, filter_countries)
    print(f"Found {len(pairs)} valid pre/post pairs (after filtering).")

    # output folders
    per_disaster_dirs = {}
    for d in DEFAULT_DISASTER_TYPES:
        d_dir = os.path.join(args.out, d)
        os.makedirs(d_dir, exist_ok=True)
        per_disaster_dirs[d] = d_dir

    all_pair_metrics = []
    disaster_type_metrics = {d: [] for d in DEFAULT_DISASTER_TYPES}

    for pair_idx, (pre_path, post_path) in enumerate(tqdm(pairs, desc="pairs")):
        pre_img_full = Image.open(pre_path).convert('RGB')
        post_img_full = Image.open(post_path).convert('RGB')

        # tile the image at 256x256 (same as training)
        tile_size = args.tile_size
        width, height = pre_img_full.size
        tiles = []
        coords = []
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                box = (left, top, left + tile_size, top + tile_size)
                if box[2] <= width and box[3] <= height:
                    tiles.append((pre_img_full.crop(box), post_img_full.crop(box)))
                    coords.append(box)

        tile_metrics = []
        inferred_type = infer_disaster_type_from_filename(pre_path, DEFAULT_DISASTER_TYPES)
        if inferred_type is None:
            # fallback: try parsing from post path
            inferred_type = infer_disaster_type_from_filename(post_path, DEFAULT_DISASTER_TYPES)
        if inferred_type is None:
            inferred_type = "unknown"

        for idx, (pre_tile, post_tile) in enumerate(tiles):
            # prepare input
            input_tensor = transform_rgb(pre_tile).unsqueeze(0).to(device)  # shape (1,3,H,W)
            # infer disaster label index
            if inferred_type in DEFAULT_DISASTER_TYPES:
                disaster_idx = DEFAULT_DISASTER_TYPES.index(inferred_type)
            else:
                disaster_idx = 0
            disaster_label = torch.tensor([ [disaster_idx] ], dtype=torch.long).to(device)

            with torch.no_grad():
                fake_post, pred_mask = G(input_tensor, disaster_label)

            # compute true masks
            if args.mode == 'binary':
                true_mask = SAR_damage_mask(pre_tile, post_tile, threshold=args.threshold, device=device)  # (1,H,W) or (H,W)
                # predicted mask: if generator outputs mask (sigmoid) use it directly; else generate via SAR on fake image
                if pred_mask is not None:
                    # pred_mask from generator is (B,1,H,W) in [0,1]
                    pred_binary = (pred_mask.squeeze(0) > 0.5).float().squeeze(0)
                else:
                    fake_post_img = tensor_to_pil(fake_post)
                    pred_binary = SAR_damage_mask(pre_tile, fake_post_img, threshold=args.threshold, device=device)
                metrics = compute_binary_metrics(pred_binary.cpu(), true_mask.cpu())
            else:
                # multiclass
                true_mask = SAR_damage_mask_multiclass_merged(pre_tile, post_tile, device=device)
                fake_post_img = tensor_to_pil(fake_post)
                pred_mask_mc = SAR_damage_mask_multiclass_merged(pre_tile, fake_post_img, device=device)
                metrics = compute_multiclass_metrics(pred_mask_mc, true_mask)

            tile_metrics.append(metrics)

            # save example visual for every N tiles
            if idx % args.vis_every == 0:
                vis_dir = os.path.join(per_disaster_dirs.get(inferred_type, args.out), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                fig, axs = plt.subplots(1, 4, figsize=(16,4))
                axs[0].imshow(pre_tile); axs[0].set_title("Pre"); axs[0].axis('off')
                axs[1].imshow(post_tile); axs[1].set_title("Post (True)"); axs[1].axis('off')
                axs[2].imshow(tensor_to_pil(fake_post)); axs[2].set_title("Fake Post"); axs[2].axis('off')
                if args.mode == 'binary':
                    axs[3].imshow(pred_binary.cpu(), cmap='gray'); axs[3].set_title("Pred Mask"); axs[3].axis('off')
                else:
                    axs[3].imshow(pred_mask_mc.cpu(), cmap=cmap, norm=norm); axs[3].set_title("Pred Mask (mc)"); axs[3].axis('off')
                plt.suptitle(f"Pair {pair_idx+1} Tile {idx} ({inferred_type})")
                savepath = os.path.join(vis_dir, f"pair{pair_idx+1}_tile{idx}.png")
                plt.savefig(savepath, bbox_inches='tight')
                plt.close(fig)

        # aggregate pair-level metrics
        if len(tile_metrics) == 0:
            continue
        mean_metrics = {k: float(np.nanmean([t[k] for t in tile_metrics])) for k in tile_metrics[0].keys()}
        all_pair_metrics.append({'pair': (os.path.basename(pre_path), os.path.basename(post_path)), 'metrics': mean_metrics})
        if inferred_type in disaster_type_metrics:
            disaster_type_metrics[inferred_type].append(mean_metrics)
        else:
            # if unknown type, store under 'unknown'
            disaster_type_metrics.setdefault(inferred_type, []).append(mean_metrics)

        print(f"Pair {pair_idx+1}: {mean_metrics}")

    # --- overall stats ---
    if len(all_pair_metrics) == 0:
        print("No pairs evaluated. Exiting.")
        return

    # compute lists
    all_iou = [p['metrics']['IoU'] for p in all_pair_metrics if not np.isnan(p['metrics']['IoU'])]
    all_dice = [p['metrics']['Dice'] for p in all_pair_metrics if not np.isnan(p['metrics']['Dice'])]
    all_f1 = [p['metrics']['F1'] for p in all_pair_metrics if not np.isnan(p['metrics']['F1'])]
    all_prec = [p['metrics']['Precision'] for p in all_pair_metrics if not np.isnan(p['metrics']['Precision'])]
    all_rec = [p['metrics']['Recall'] for p in all_pair_metrics if not np.isnan(p['metrics']['Recall'])]

    print("\n=== Overall Summary ===")
    summary_stats = {
        'Average IoU': float(np.nanmean(all_iou)) if len(all_iou) else float('nan'),
        'Median IoU': float(np.nanmedian(all_iou)) if len(all_iou) else float('nan'),
        '95% CI IoU': compute_ci(all_iou),

        'Average Dice': float(np.nanmean(all_dice)) if len(all_dice) else float('nan'),
        'Median Dice': float(np.nanmedian(all_dice)) if len(all_dice) else float('nan'),
        '95% CI Dice': compute_ci(all_dice),

        'Average F1': float(np.nanmean(all_f1)) if len(all_f1) else float('nan'),
        'Median F1': float(np.nanmedian(all_f1)) if len(all_f1) else float('nan'),
        '95% CI F1': compute_ci(all_f1),

        'Average Precision': float(np.nanmean(all_prec)) if len(all_prec) else float('nan'),
        'Median Precision': float(np.nanmedian(all_prec)) if len(all_prec) else float('nan'),
        '95% CI Precision': compute_ci(all_prec),

        'Average Recall': float(np.nanmean(all_rec)) if len(all_rec) else float('nan'),
        'Median Recall': float(np.nanmedian(all_rec)) if len(all_rec) else float('nan'),
        '95% CI Recall': compute_ci(all_rec),
    }

    print(json.dumps(summary_stats, indent=2))
    # save
    with open(os.path.join(args.out, "per_pair_metrics.json"), "w") as f:
        json.dump(all_pair_metrics, f, indent=2)

    with open(os.path.join(args.out, "summary_stats.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)

    # per-disaster summaries
    for d, metrics_list in disaster_type_metrics.items():
        if not metrics_list:
            continue
        iou_vals = [m['IoU'] for m in metrics_list if not np.isnan(m['IoU'])]
        dice_vals = [m['Dice'] for m in metrics_list if not np.isnan(m['Dice'])]
        prec_vals = [m['Precision'] for m in metrics_list if not np.isnan(m['Precision'])]
        rec_vals = [m['Recall'] for m in metrics_list if not np.isnan(m['Recall'])]

        summary = {
            'Average IoU': float(np.nanmean(iou_vals)) if len(iou_vals) else float('nan'),
            'Median IoU': float(np.nanmedian(iou_vals)) if len(iou_vals) else float('nan'),
            '95% CI IoU': compute_ci(iou_vals),

            'Average Dice': float(np.nanmean(dice_vals)) if len(dice_vals) else float('nan'),
            'Median Dice': float(np.nanmedian(dice_vals)) if len(dice_vals) else float('nan'),
            '95% CI Dice': compute_ci(dice_vals),

            'Average Precision': float(np.nanmean(prec_vals)) if len(prec_vals) else float('nan'),
            'Median Precision': float(np.nanmedian(prec_vals)) if len(prec_vals) else float('nan'),
            '95% CI Precision': compute_ci(prec_vals),

            'Average Recall': float(np.nanmean(rec_vals)) if len(rec_vals) else float('nan'),
            'Median Recall': float(np.nanmedian(rec_vals)) if len(rec_vals) else float('nan'),
            '95% CI Recall': compute_ci(rec_vals),
        }
        out_dir = os.path.join(args.out, d)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary_stats.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary for {d} -> {out_dir}")

    print(f"\nSaved overall results to: {args.out}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DisasterGAN generator on pre/post pairs (binary or multiclass).")
    p.add_argument("--model", required=True, help="Path to generator .pth (G) file")
    p.add_argument("--pre-dir", required=True, help="Directory containing pre-disaster tiles/images")
    p.add_argument("--post-dir", required=True, help="Directory containing post-disaster tiles/images")
    p.add_argument("--out", required=True, help="Output directory for results (JSON, visuals)")
    p.add_argument("--strata", choices=list(COUNTRY_STRATA.keys()), default=None, help="Optional: LIC/MIC/HIC filtering by filename tokens")
    p.add_argument("--mode", choices=['binary','multiclass'], default='binary', help="Evaluation mode")
    p.add_argument("--threshold", type=float, default=0.1, help="Base threshold for SAR binary mask function")
    p.add_argument("--vis-every", type=int, default=10, help="Save visualization every N tiles")
    p.add_argument("--tile-size", type=int, default=256, help="Tile size used for tiling large images")
    p.add_argument("--device", type=str, default=None, help="Torch device identifier (e.g. cuda:0 or cpu)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
