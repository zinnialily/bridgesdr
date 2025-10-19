
# ===================================================================
# Generates both binary and multiclass damage masks.
=====================================================================

import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import binary_opening, binary_closing
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if "study1" in str(SCRIPT_DIR) else SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Configuration
# -------------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bright_root = PROJECT_ROOT / "data" / "bright"
    synthetic_dir = PROJECT_ROOT / "results" / "study1" / "synthetic_images"
    output_dir = PROJECT_ROOT / "results" / "study1" / "masks"
    
    income_levels = ['lic', 'mic', 'hic']

config = Config()

# -------------------------
# Transforms (from original code)
# -------------------------
transform_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------
# Mask Generation Functions (from original code)
# -------------------------
def optical_to_sar_like(img):
    """Convert optical RGB image to SAR-like grayscale."""
    img = img.convert('L')
    img = ImageOps.autocontrast(img, cutoff=2)
    return img

def SAR_damage_mask(pre_optical_img, post_sar_img, threshold=0.1):
    """
    Generate binary damage mask from pre-optical and post-SAR images.
    
    Original function from evaluation code - computes difference between
    pre (converted to SAR-like) and post (actual SAR) to detect changes.
    """
    # Convert pre-optical to SAR-like
    pre_sar_like = optical_to_sar_like(pre_optical_img)
    post_sar = post_sar_img.convert('L')

    pre_tensor = transform_gray(pre_sar_like).to(config.device)
    post_tensor = transform_gray(post_sar).to(config.device)

    with torch.no_grad():
        diff = torch.abs(post_tensor - pre_tensor).unsqueeze(0)
        diff = F.avg_pool2d(diff, kernel_size=3, stride=1, padding=1)
        img_std = diff.std()
        adaptive_threshold = threshold + (0.1 * img_std)
        mask = (diff > adaptive_threshold).float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return mask.squeeze(0)

def clean_multiclass_mask(mask_np, min_region_size=20):
    """Clean multiclass mask using morphological operations."""
    cleaned_mask = np.zeros_like(mask_np)
    for cls in range(1, 4):  # Skip 0 (no damage)
        binary = (mask_np == cls)
        binary = binary_opening(binary, structure=np.ones((3,3)))
        binary = binary_closing(binary, structure=np.ones((3,3)))
        cleaned_mask[binary] = cls
    return cleaned_mask

def SAR_damage_mask_multiclass_merged(pre_optical_img, post_sar_img):
    """
    Generate multiclass damage mask (0=none, 1=minor, 2=major, 3=destroyed).
    
    Original function from evaluation code with improved thresholds
    and post-processing.
    """
    # Convert pre-optical to SAR-like
    pre_sar_like = optical_to_sar_like(pre_optical_img)
    post_sar = post_sar_img.convert('L')

    pre_tensor = transform_gray(pre_sar_like).to(config.device).squeeze(0)
    post_tensor = transform_gray(post_sar).to(config.device).squeeze(0)

    with torch.no_grad():
        diff = torch.abs(post_tensor - pre_tensor)  # shape (H, W)
        diff_smoothed = F.avg_pool2d(diff.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()

        # Normalize diff to [0,1]
        diff_min, diff_max = diff_smoothed.min(), diff_smoothed.max()
        diff_norm = (diff_smoothed - diff_min) / (diff_max - diff_min + 1e-8)

        # Thresholds for damage severity
        thresholds = [0.08, 0.22, 0.45]

        # Initial multiclass mask
        mask = torch.zeros_like(diff_norm, dtype=torch.long)
        mask = torch.where((diff_norm >= thresholds[0]) & (diff_norm < thresholds[1]), 1, mask)  # Minor
        mask = torch.where((diff_norm >= thresholds[1]) & (diff_norm < thresholds[2]), 2, mask)  # Major
        mask = torch.where(diff_norm >= thresholds[2], 3, mask)  # Destroyed

    # Post-process: remove specks and smooth classes
    mask_np = mask.cpu().numpy()
    cleaned = clean_multiclass_mask(mask_np)

    return torch.from_numpy(cleaned).long()

# -------------------------
# Main Generation Function
# -------------------------
def generate_all_masks():
    """Generate binary and multiclass masks for real and synthetic images."""
    
    print("="*60)
    print("Generating Damage Masks")
    print("="*60)
    print(f"Device: {config.device}\n")
    
    generation_stats = {
        'real': {'binary': 0, 'multiclass': 0},
        'synthetic': {'binary': 0, 'multiclass': 0}
    }
    
    # Process real BRIGHT images
    print("="*60)
    print("Processing REAL BRIGHT Images")
    print("="*60 + "\n")
    
    for income in config.income_levels:
        income_dir = config.bright_root / income / "images"
        
        if not income_dir.exists():
            print(f"{income_dir} not found, skipping...")
            continue
        
        # Create output directories
        output_binary_dir = config.output_dir / "real" / "binary" / income
        output_multiclass_dir = config.output_dir / "real" / "multiclass" / income
        output_binary_dir.mkdir(parents=True, exist_ok=True)
        output_multiclass_dir.mkdir(parents=True, exist_ok=True)
        
        # Find pre-post pairs
        pre_files = list(income_dir.glob("*_pre_disaster.png"))
        
        print(f"Processing {income.upper()} - {len(pre_files)} image pairs")
        
        for pre_path in tqdm(pre_files, desc=f"Real {income.upper()}"):
            try:
                # Find corresponding post image
                post_path = str(pre_path).replace('_pre_disaster.png', '_post_disaster.tif')
                post_path = Path(post_path)
                
                if not post_path.exists():
                    # Try .png extension
                    post_path = str(pre_path).replace('_pre_disaster.png', '_post_disaster.png')
                    post_path = Path(post_path)
                
                if not post_path.exists():
                    continue
                
                # Load images
                pre_img = Image.open(pre_path).convert('RGB')
                post_img = Image.open(post_path)
                
                # Generate binary mask
                binary_mask = SAR_damage_mask(pre_img, post_img, threshold=0.2)
                binary_mask_pil = Image.fromarray((binary_mask[0].cpu().numpy() * 255).astype(np.uint8))
                binary_filename = pre_path.name.replace('_pre_disaster.png', '_binary_mask.png')
                binary_mask_pil.save(output_binary_dir / binary_filename)
                generation_stats['real']['binary'] += 1
                
                # Generate multiclass mask
                multiclass_mask = SAR_damage_mask_multiclass_merged(pre_img, post_img)
                multiclass_mask_pil = Image.fromarray(multiclass_mask.cpu().numpy().astype(np.uint8))
                multiclass_filename = pre_path.name.replace('_pre_disaster.png', '_multiclass_mask.png')
                multiclass_mask_pil.save(output_multiclass_dir / multiclass_filename)
                generation_stats['real']['multiclass'] += 1
                
            except Exception as e:
                print(f"\nError processing {pre_path.name}: {e}")
                continue
        
        print(f"Generated masks for {income.upper()}\n")
    
    # Process synthetic images
    print("\n" + "="*60)
    print("Processing SYNTHETIC Images")
    print("="*60 + "\n")
    
    for income in config.income_levels:
        synthetic_income_dir = config.synthetic_dir / income
        bright_income_dir = config.bright_root / income / "images"
        
        if not synthetic_income_dir.exists() or not bright_income_dir.exists():
            print(f"Warning: Directories for {income} not found, skipping...")
            continue
        
        # Create output directories
        output_binary_dir = config.output_dir / "synthetic" / "binary" / income
        output_multiclass_dir = config.output_dir / "synthetic" / "multiclass" / income
        output_binary_dir.mkdir(parents=True, exist_ok=True)
        output_multiclass_dir.mkdir(parents=True, exist_ok=True)
        
        # Find synthetic post images
        synthetic_files = list(synthetic_income_dir.glob("*_post_disaster_synthetic.png"))
        
        print(f"Processing {income.upper()} - {len(synthetic_files)} synthetic images")
        
        for synthetic_path in tqdm(synthetic_files, desc=f"Synthetic {income.upper()}"):
            try:
                # Find corresponding pre image from BRIGHT
                pre_filename = synthetic_path.name.replace('_post_disaster_synthetic.png', '_pre_disaster.png')
                pre_path = bright_income_dir / pre_filename
                
                if not pre_path.exists():
                    continue
                
                # Load images
                pre_img = Image.open(pre_path).convert('RGB')
                synthetic_post_img = Image.open(synthetic_path).convert('RGB')
                
                # Generate binary mask
                binary_mask = SAR_damage_mask(pre_img, synthetic_post_img, threshold=0.2)
                binary_mask_pil = Image.fromarray((binary_mask[0].cpu().numpy() * 255).astype(np.uint8))
                binary_filename = synthetic_path.name.replace('_post_disaster_synthetic.png', '_binary_mask.png')
                binary_mask_pil.save(output_binary_dir / binary_filename)
                generation_stats['synthetic']['binary'] += 1
                
                # Generate multiclass mask
                multiclass_mask = SAR_damage_mask_multiclass_merged(pre_img, synthetic_post_img)
                multiclass_mask_pil = Image.fromarray(multiclass_mask.cpu().numpy().astype(np.uint8))
                multiclass_filename = synthetic_path.name.replace('_post_disaster_synthetic.png', '_multiclass_mask.png')
                multiclass_mask_pil.save(output_multiclass_dir / multiclass_filename)
                generation_stats['synthetic']['multiclass'] += 1
                
            except Exception as e:
                print(f"\nError processing {synthetic_path.name}: {e}")
                continue
        
        print(f" Generated masks for synthetic {income.upper()}\n")
    
    # Save statistics
    stats_path = config.output_dir / "mask_generation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(generation_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MASK GENERATION COMPLETE")
    print("="*60)
    print(f"Real images:")
    print(f"  Binary masks: {generation_stats['real']['binary']}")
    print(f"  Multiclass masks: {generation_stats['real']['multiclass']}")
    print(f"\nSynthetic images:")
    print(f"  Binary masks: {generation_stats['synthetic']['binary']}")
    print(f"  Multiclass masks: {generation_stats['synthetic']['multiclass']}")
    print(f"\nOutput directory: {config.output_dir}")
    print(f"Statistics saved to: {stats_path}")
    print("="*60 + "\n")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    generate_all_masks()
