#!/usr/bin/env python3
"""
SAR-to-Optical Approximation Validation
========================================

This script validates the simplified SAR-to-optical conversion methodology
using the Sentinel-1/2 dataset. It evaluates image similarity metrics across
four terrain types following the NeurIPS paper methodology.

Dataset Structure (from Kaggle):
v_2/
  ├── agri/
  │   ├── s1/  (SAR images)
  │   └── s2/  (Optical images)
  ├── barrenland/
  │   ├── s1/
  │   └── s2/
  ├── grassland/
  │   ├── s1/
  │   └── s2/
  └── urban/
      ├── s1/
      └── s2/

Usage:
    python sar_optical_validation.py --download --evaluate
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy import stats

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("Installing scikit-image...")
    os.system('pip install scikit-image --break-system-packages')
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr


class Config:
    """Configuration for SAR-to-Optical validation"""
    BASE_DIR = Path.home() / "sar_validation"
    DATA_DIR = BASE_DIR / "data"
    SENTINEL_ROOT = DATA_DIR / "v_2"
    RESULTS_DIR = BASE_DIR / "results"
    KAGGLE_DATASET = "muhammedfurkan/sentinel1-2-image-pairs-sar-optical"
    
    TERRAIN_MAPPING = {
        'agri': 'agricultural',
        'barrenland': 'barren',
        'grassland': 'grassland',
        'urban': 'urban'
    }
    
    SAMPLE_SIZE = 100
    IMAGE_SIZE = (256, 256)
    CONFIDENCE_LEVEL = 0.95
    
    def __init__(self):
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n" + "="*70)
        print("Kaggle API Credentials Not Found")
        print("="*70)
        print("\nSteps to set up:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Move kaggle.json to ~/.kaggle/")
        print("4. chmod 600 ~/.kaggle/kaggle.json")
        print("="*70 + "\n")
        sys.exit(1)
    
    kaggle_json.chmod(0o600)
    print("✓ Kaggle credentials found")


def download_sentinel_dataset(config: Config, force: bool = False):
    if config.SENTINEL_ROOT.exists() and not force:
        print(f"\n✓ Dataset already exists at {config.SENTINEL_ROOT}")
        return True
    
    try:
        import kaggle
    except ImportError:
        print("Installing Kaggle API...")
        os.system('pip install kaggle --break-system-packages')
        import kaggle
    
    print("\n" + "="*70)
    print("Downloading Sentinel-1/2 Dataset")
    print("="*70)
    
    try:
        kaggle.api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=config.DATA_DIR,
            unzip=True,
            quiet=False
        )
        print("\n✓ Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def verify_dataset_structure(config: Config) -> bool:
    print("\n" + "="*70)
    print("Verifying Dataset Structure")
    print("="*70)
    
    if not config.SENTINEL_ROOT.exists():
        print(f"\n✗ Dataset root not found: {config.SENTINEL_ROOT}")
        return False
    
    print(f"\n✓ Dataset root found: {config.SENTINEL_ROOT}")
    
    all_valid = True
    for terrain_folder in config.TERRAIN_MAPPING.keys():
        terrain_path = config.SENTINEL_ROOT / terrain_folder
        s1_path = terrain_path / "s1"
        s2_path = terrain_path / "s2"
        
        if not all([terrain_path.exists(), s1_path.exists(), s2_path.exists()]):
            print(f"✗ Missing folders for: {terrain_folder}")
            all_valid = False
            continue
        
        s1_images = list(s1_path.glob("*"))
        s2_images = list(s2_path.glob("*"))
        print(f"✓ {terrain_folder:12} - S1: {len(s1_images):4d}, S2: {len(s2_images):4d}")
    
    return all_valid


def optical_to_sar_approximation(image: Image.Image) -> Image.Image:
    """Convert optical to SAR-like grayscale (NeurIPS paper methodology)"""
    if image.mode != 'L':
        image = image.convert('L')
    image = ImageOps.autocontrast(image, cutoff=2)
    return image


def match_image_pairs(sar_dir: Path, optical_dir: Path, max_pairs: int = 100) -> List[Tuple[Path, Path]]:
    sar_files = sorted([f for f in sar_dir.iterdir() if f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
    optical_files = sorted([f for f in optical_dir.iterdir() if f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
    
    pairs = []
    for sar_file in sar_files[:max_pairs]:
        sar_stem = sar_file.stem
        optical_match = None
        
        for opt_file in optical_files:
            if opt_file.stem == sar_stem:
                optical_match = opt_file
                break
        
        if optical_match is None and optical_files:
            idx = sar_files.index(sar_file)
            if idx < len(optical_files):
                optical_match = optical_files[idx]
        
        if optical_match:
            pairs.append((sar_file, optical_match))
        
        if len(pairs) >= max_pairs:
            break
    
    return pairs


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    try:
        return psnr(image1, image2, data_range=255)
    except:
        return 0.0


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    try:
        return ssim(image1, image2, data_range=255)
    except:
        return 0.0


def compute_specificity(pred: np.ndarray, target: np.ndarray, threshold: int = 128) -> float:
    pred_binary = (pred < threshold).astype(int)
    target_binary = (target < threshold).astype(int)
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_mcc(pred: np.ndarray, target: np.ndarray, threshold: int = 128) -> float:
    pred_binary = (pred < threshold).astype(int).flatten()
    target_binary = (target < threshold).astype(int).flatten()
    try:
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(target_binary, pred_binary)
    except:
        return 0.0


def evaluate_image_pair(sar_path: Path, optical_path: Path, image_size: Tuple[int, int]) -> Optional[Dict[str, float]]:
    try:
        sar_img = Image.open(sar_path).resize(image_size, Image.Resampling.LANCZOS)
        optical_img = Image.open(optical_path).resize(image_size, Image.Resampling.LANCZOS)
        optical_to_sar_img = optical_to_sar_approximation(optical_img)
        
        if sar_img.mode != 'L':
            sar_img = sar_img.convert('L')
        
        sar_array = np.array(sar_img)
        approx_array = np.array(optical_to_sar_img)
        
        return {
            'psnr': compute_psnr(approx_array, sar_array),
            'ssim': compute_ssim(approx_array, sar_array),
            'specificity': compute_specificity(approx_array, sar_array),
            'mcc': compute_mcc(approx_array, sar_array)
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def evaluate_terrain_type(config: Config, terrain_folder: str, terrain_name: str) -> pd.DataFrame:
    print(f"\nEvaluating {terrain_name} terrain...")
    
    sar_dir = config.SENTINEL_ROOT / terrain_folder / "s1"
    optical_dir = config.SENTINEL_ROOT / terrain_folder / "s2"
    pairs = match_image_pairs(sar_dir, optical_dir, max_pairs=config.SAMPLE_SIZE)
    
    if not pairs:
        print(f"  ✗ No pairs found")
        return pd.DataFrame()
    
    print(f"  Found {len(pairs)} image pairs")
    
    results = []
    for sar_path, optical_path in tqdm(pairs, desc=f"  Processing", leave=False):
        metrics = evaluate_image_pair(sar_path, optical_path, config.IMAGE_SIZE)
        if metrics:
            metrics['terrain'] = terrain_name
            results.append(metrics)
    
    print(f"  ✓ Evaluated {len(results)} pairs")
    return pd.DataFrame(results)


def evaluate_all_terrains(config: Config) -> pd.DataFrame:
    print("\n" + "="*70)
    print("Evaluating SAR-to-Optical Approximation")
    print("="*70)
    
    all_results = []
    for terrain_folder, terrain_name in config.TERRAIN_MAPPING.items():
        df = evaluate_terrain_type(config, terrain_folder, terrain_name)
        if not df.empty:
            all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    output_path = config.RESULTS_DIR / "raw_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Raw results: {output_path}")
    
    return results_df


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    if len(data) == 0:
        return (0.0, 0.0)
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return (mean - interval, mean + interval)


def generate_summary_statistics(results_df: pd.DataFrame, config: Config):
    print("\n" + "="*70)
    print("Computing Summary Statistics")
    print("="*70)
    
    summary_data = []
    for terrain in results_df['terrain'].unique():
        terrain_data = results_df[results_df['terrain'] == terrain]
        row = {'Terrain': terrain.capitalize()}
        
        for metric in ['psnr', 'specificity', 'mcc', 'ssim']:
            values = terrain_data[metric].values
            mean = np.mean(values)
            ci_lower, ci_upper = compute_confidence_interval(values, config.CONFIDENCE_LEVEL)
            row[f'{metric.upper()}'] = f"{mean:.2f}"
            row[f'{metric.upper()}_ci'] = f"({ci_lower:.2f}, {ci_upper:.2f})"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    paper_table = pd.DataFrame()
    paper_table['Terrain'] = summary_df['Terrain']
    paper_table['PSNR (dB)'] = summary_df['PSNR'] + ' ' + summary_df['PSNR_ci']
    paper_table['Specificity'] = summary_df['SPECIFICITY'] + ' ' + summary_df['SPECIFICITY_ci']
    paper_table['MCC'] = summary_df['MCC'] + ' ' + summary_df['MCC_ci']
    paper_table['SSIM'] = summary_df['SSIM'] + ' ' + summary_df['SSIM_ci']
    
    summary_df.to_csv(config.RESULTS_DIR / "summary_statistics.csv", index=False)
    paper_table.to_csv(config.RESULTS_DIR / "paper_table.csv", index=False)
    
    print("\n" + "="*70)
    print("SUMMARY RESULTS (NeurIPS paper format)")
    print("="*70)
    print("\n" + paper_table.to_string(index=False))
    print()
    
    return summary_df, paper_table


def main():
    parser = argparse.ArgumentParser(description='SAR-to-Optical Validation')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--sample-size', type=int, default=100, help='Samples per terrain')
    args = parser.parse_args()
    
    if not (args.download or args.evaluate):
        args.download = True
        args.evaluate = True
    
    config = Config()
    if args.sample_size:
        config.SAMPLE_SIZE = args.sample_size
    
    print("\n" + "="*70)
    print("SAR-to-Optical Approximation Validation")
    print("="*70)
    
    if args.download:
        setup_kaggle_credentials()
        if not download_sentinel_dataset(config, force=args.force):
            sys.exit(1)
    
    if args.evaluate:
        if not verify_dataset_structure(config):
            print("\n✗ Run with --download first")
            sys.exit(1)
        
        results_df = evaluate_all_terrains(config)
        if results_df.empty:
            sys.exit(1)
        
        generate_summary_statistics(results_df, config)
        
        print("\n" + "="*70)
        print("Complete! Results in:", config.RESULTS_DIR)
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
