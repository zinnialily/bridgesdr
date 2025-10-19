#!/usr/bin/env python3
"""
Dataset Preprocessing Script for xBD and BRIGHT
xBD Structure:
    data/xbd/{train|tier1|tier3|hold|test}/images/*.png
    data/xbd/{train|tier1|tier3|hold|test}/labels/*.json
    
BRIGHT Structure:
    data/bright/{lic|mic|hic}/images/*_pre_disaster.{png|tif}
    data/bright/{lic|mic|hic}/images/*_post_disaster.{png|tif}
    data/bright/{lic|mic|hic}/masks/*_damage_mask.{png|tif}

Usage:
    python scripts/02_preprocess_data.py [--xbd-only | --bright-only] [--verify]
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

################################################################################
# Configuration
################################################################################

DATA_DIR = PROJECT_ROOT / "data"
XBD_RAW = DATA_DIR / "xbd_raw"
BRIGHT_RAW = DATA_DIR / "bright_raw"
XBD_PROCESSED = DATA_DIR / "xbd"
BRIGHT_PROCESSED = DATA_DIR / "bright"

# Country to income level mapping (matches COUNTRY_STRATA in code)
COUNTRY_STRATA = {
    "LIC": ["haiti", "congo"],
    "MIC": ["turkey", "morocco", "libya"],
    "HIC": ["noto", "la_palma", "hawaii"],
}

# Reverse mapping for quick lookups
COUNTRY_TO_INCOME = {}
for income, countries in COUNTRY_STRATA.items():
    for country in countries:
        COUNTRY_TO_INCOME[country.lower()] = income.lower()

# Disaster type mappings
DISASTER_TYPES = [
    'volcano', 'fire', 'tornado', 'tsunami',
    'flooding', 'earthquake', 'hurricane', 'wildfire',
    'explosion', 'eruption', 'flood'
]

################################################################################
# Helper Functions
################################################################################

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_section(text: str):
    """Print a formatted section header"""
    print(f"\n{Colors.GREEN}{'='*70}{Colors.NC}")
    print(f"{Colors.GREEN}{text}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*70}{Colors.NC}\n")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}ERROR: {text}{Colors.NC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}WARNING: {text}{Colors.NC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}SUCCESS: {text}{Colors.NC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}INFO: {text}{Colors.NC}")

def extract_disaster_type(filename: str) -> str:
    """Extract disaster type from filename"""
    filename_lower = filename.lower()
    for disaster in DISASTER_TYPES:
        if disaster in filename_lower:
            return disaster
    return "unknown"

def extract_country_from_filename(filename: str) -> Optional[str]:
    """
    Extract country/location from BRIGHT filename
    
    Expected patterns in BRIGHT filenames from Zenodo:
    - Contains country name somewhere in the filename
    - May have various separators (_, -, space)
    """
    filename_lower = filename.lower()
    
    # Check for known countries in the COUNTRY_STRATA mapping
    for income_level, countries in COUNTRY_STRATA.items():
        for country in countries:
            # Handle multi-word country names (e.g., "la_palma")
            country_variations = [
                country,
                country.replace('_', ''),
                country.replace('_', '-'),
                country.replace('_', ' '),
            ]
            for variation in country_variations:
                if variation in filename_lower:
                    return country
    
    # Special case handling for countries that might appear differently
    if any(x in filename_lower for x in ['lapalma', 'la_palma', 'la-palma', 'la palma']):
        return "la_palma"
    
    if any(x in filename_lower for x in ['noto', 'japan']):
        return "noto"
    
    return None

def classify_by_income(country: str) -> Optional[str]:
    """Classify country by income level using COUNTRY_STRATA"""
    if country is None:
        return None
    
    country_lower = country.lower()
    
    # Direct lookup
    if country_lower in COUNTRY_TO_INCOME:
        return COUNTRY_TO_INCOME[country_lower]
    
    return None

################################################################################
# xBD Preprocessing
################################################################################

def preprocess_xbd():
    """
    Organize xBD dataset into required structure.
    
    Expected output structure:
        data/xbd/
        ├── train/
        │   ├── images/
        │   │   ├── disaster-location_NNNNNNNN_pre_disaster.png
        │   │   └── disaster-location_NNNNNNNN_post_disaster.png
        │   └── labels/
        │       ├── disaster-location_NNNNNNNN_pre_disaster.json
        │       └── disaster-location_NNNNNNNN_post_disaster.json
        └── (tier1, tier3, hold, test with same structure)
    """
    print_section("Preprocessing xBD Dataset")
    
    if not XBD_RAW.exists():
        print_error(f"xBD raw data not found at {XBD_RAW}")
        print_info("Please run download script first: bash scripts/01_download_datasets.sh")
        return False
    
    # Create output directory structure
    splits = ['train', 'tier1', 'tier3', 'hold', 'test']
    for split in splits:
        (XBD_PROCESSED / split / "images").mkdir(parents=True, exist_ok=True)
        (XBD_PROCESSED / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Find xBD data structure
    print_info("Analyzing xBD directory structure...")
    
    # Common xBD directory patterns from Kaggle
    possible_roots = [
        XBD_RAW,
        XBD_RAW / "xview2",
        XBD_RAW / "xBD",
        XBD_RAW / "xbd-dataset",
        XBD_RAW / "xbd",
    ]
    
    xbd_root = None
    for root in possible_roots:
        if root.exists():
            # Check if it contains expected subdirectories
            has_splits = any((root / split).exists() for split in splits)
            if has_splits:
                xbd_root = root
                break
            # Also check for just 'train' which is minimum requirement
            if (root / "train").exists():
                xbd_root = root
                break
    
    if xbd_root is None:
        # Try to find images directory recursively
        print_info("Searching for xBD structure...")
        image_dirs = list(XBD_RAW.rglob("*images*"))
        if image_dirs:
            # Go up two levels (images -> split -> root)
            xbd_root = image_dirs[0].parent.parent
        else:
            print_error("Could not determine xBD directory structure")
            print_info("Expected structure: xbd_raw/{train,tier1,tier3,hold,test}/{images,labels}/")
            return False
    
    print_success(f"Found xBD root at: {xbd_root}")
    
    # Process each split
    stats = defaultdict(lambda: {"images": 0, "labels": 0, "pre": 0, "post": 0})
    
    for split in splits:
        split_dir = xbd_root / split
        if not split_dir.exists():
            print_warning(f"Split '{split}' not found, skipping...")
            continue
        
        print_info(f"Processing {split} split...")
        
        # Find images and labels
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists():
            print_warning(f"Images directory not found for {split}")
            continue
        
        # Copy images (preserve original naming which should match pattern)
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        for img_file in tqdm(image_files, desc=f"  Copying {split} images", leave=False):
            dest = XBD_PROCESSED / split / "images" / img_file.name
            shutil.copy2(img_file, dest)
            stats[split]["images"] += 1
            
            # Count pre/post for verification
            if "_pre_disaster" in img_file.name:
                stats[split]["pre"] += 1
            elif "_post_disaster" in img_file.name:
                stats[split]["post"] += 1
        
        # Copy labels if they exist
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.json"))
            for label_file in tqdm(label_files, desc=f"  Copying {split} labels", leave=False):
                dest = XBD_PROCESSED / split / "labels" / label_file.name
                shutil.copy2(label_file, dest)
                stats[split]["labels"] += 1
    
    # Print statistics
    print_section("xBD Processing Statistics")
    total_images = 0
    total_labels = 0
    total_pre = 0
    total_post = 0
    
    for split in splits:
        images = stats[split]["images"]
        labels = stats[split]["labels"]
        pre = stats[split]["pre"]
        post = stats[split]["post"]
        
        total_images += images
        total_labels += labels
        total_pre += pre
        total_post += post
        
        if images > 0 or labels > 0:
            print(f"  {split:10s}: {images:6d} images ({pre:5d} pre, {post:5d} post), {labels:6d} labels")
    
    print(f"\n  {'TOTAL':10s}: {total_images:6d} images ({total_pre:5d} pre, {total_post:5d} post), {total_labels:6d} labels")
    
    # Verify naming convention
    if total_pre != total_post:
        print_warning(f"Pre/post image count mismatch: {total_pre} pre vs {total_post} post")
    else:
        print_success(f"Pre/post images matched: {total_pre} pairs")
    
    print_success(f"xBD dataset organized at: {XBD_PROCESSED}")
    return True

################################################################################
# BRIGHT Preprocessing
################################################################################

def preprocess_bright():
    """
    Organize BRIGHT dataset by income level based on country classification.
    
    Expected output structure:
        data/bright/
        ├── lic/
        │   ├── images/
        │   │   ├── country_disaster_00000_pre_disaster.png  (RGB optical)
        │   │   └── country_disaster_00000_post_disaster.tif (SAR)
        │   └── masks/
        │       └── country_disaster_00000_damage_mask.png
        └── (mic, hic with same structure)
    """
    print_section("Preprocessing BRIGHT Dataset")
    
    if not BRIGHT_RAW.exists():
        print_error(f"BRIGHT raw data not found at {BRIGHT_RAW}")
        print_info("Please run download script first: bash scripts/01_download_datasets.sh")
        return False
    
    # Create output directory structure
    for income in ['lic', 'mic', 'hic']:
        (BRIGHT_PROCESSED / income / "images").mkdir(parents=True, exist_ok=True)
        (BRIGHT_PROCESSED / income / "masks").mkdir(parents=True, exist_ok=True)
    
    print_info("Analyzing BRIGHT directory structure...")
    
    # Find pre-event, post-event, and target directories
    pre_event_dir = None
    post_event_dir = None
    target_dir = None
    
    # Search for directories (case-insensitive)
    for subdir in BRIGHT_RAW.rglob("*"):
        if subdir.is_dir():
            name = subdir.name.lower()
            if 'pre' in name and 'event' in name and pre_event_dir is None:
                pre_event_dir = subdir
            elif 'post' in name and 'event' in name and post_event_dir is None:
                post_event_dir = subdir
            elif ('target' in name or 'label' in name) and target_dir is None:
                target_dir = subdir
    
    if not all([pre_event_dir, post_event_dir, target_dir]):
        print_error("Could not find all required BRIGHT directories")
        print_info(f"Pre-event: {pre_event_dir}")
        print_info(f"Post-event: {post_event_dir}")
        print_info(f"Target: {target_dir}")
        print_info("\nExpected after unzipping Zenodo files:")
        print_info("  - pre-event/  (from pre-event.zip)")
        print_info("  - post-event/ (from post-event.zip)")
        print_info("  - target/     (from target.zip)")
        return False
    
    print_success(f"Pre-event directory: {pre_event_dir}")
    print_success(f"Post-event directory: {post_event_dir}")
    print_success(f"Target directory: {target_dir}")
    
    # Get all files from each directory
    pre_files = list(pre_event_dir.glob("*.*"))
    pre_files = [f for f in pre_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    post_files = list(post_event_dir.glob("*.*"))
    post_files = [f for f in post_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    target_files = list(target_dir.glob("*.*"))
    target_files = [f for f in target_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    print_info(f"Found {len(pre_files)} pre-disaster images")
    print_info(f"Found {len(post_files)} post-disaster images")
    print_info(f"Found {len(target_files)} target masks")
    
    # Create lookup dictionaries by base name
    print_info("Building file mappings...")
    
    def get_base_identifier(filepath: Path) -> str:
        """
        Extract base identifier from filename for matching.
        Removes common suffixes and normalizes.
        """
        name = filepath.stem.lower()
        
        # Remove common suffixes
        for suffix in ['_pre', '_post', '_target', '_label', '_mask', 
                      '-pre', '-post', '-target', '-label', '-mask',
                      'pre', 'post', 'target', 'label', 'mask']:
            name = name.replace(suffix, '')
        
        # Remove trailing/leading underscores and dashes
        name = name.strip('_-')
        
        return name
    
    # Build lookups
    post_lookup = {}
    for f in post_files:
        base = get_base_identifier(f)
        post_lookup[base] = f
    
    target_lookup = {}
    for f in target_files:
        base = get_base_identifier(f)
        target_lookup[base] = f
    
    # Process each pre-disaster image
    stats = defaultdict(lambda: {"pre": 0, "post": 0, "mask": 0})
    skipped = {"no_country": 0, "no_post": 0, "no_mask": 0}
    sequence_counters = defaultdict(lambda: defaultdict(int))
    unmatched_files = []
    
    print_info("\nProcessing and organizing files by country...")
    
    for pre_file in tqdm(pre_files, desc="Processing BRIGHT images"):
        # Extract country from filename
        country = extract_country_from_filename(pre_file.name)
        
        if country is None:
            skipped["no_country"] += 1
            unmatched_files.append(pre_file.name)
            continue
        
        # Classify by income level
        income_level = classify_by_income(country)
        
        if income_level is None:
            skipped["no_country"] += 1
            print_warning(f"Country '{country}' not in COUNTRY_STRATA: {pre_file.name}")
            continue
        
        # Extract disaster type
        disaster_type = extract_disaster_type(pre_file.name)
        
        # Get sequence number for this country-disaster combination
        key = f"{country}_{disaster_type}"
        sequence = sequence_counters[income_level][key]
        sequence_counters[income_level][key] += 1
        
        # Create standardized filename (matches GitHub loading code expectations)
        # Format: {country}_{disaster_type}_{sequence:05d}_{timing}.{ext}
        new_base = f"{country}_{disaster_type}_{sequence:05d}"
        
        # Copy pre-disaster image (should be .png for optical)
        pre_dest = BRIGHT_PROCESSED / income_level / "images" / f"{new_base}_pre_disaster{pre_file.suffix}"
        shutil.copy2(pre_file, pre_dest)
        stats[income_level]["pre"] += 1
        
        # Find and copy corresponding post-disaster image (should be .tif for SAR)
        base_identifier = get_base_identifier(pre_file)
        
        if base_identifier in post_lookup:
            post_file = post_lookup[base_identifier]
            post_dest = BRIGHT_PROCESSED / income_level / "images" / f"{new_base}_post_disaster{post_file.suffix}"
            shutil.copy2(post_file, post_dest)
            stats[income_level]["post"] += 1
        else:
            skipped["no_post"] += 1
        
        # Find and copy corresponding mask
        if base_identifier in target_lookup:
            mask_file = target_lookup[base_identifier]
            mask_dest = BRIGHT_PROCESSED / income_level / "masks" / f"{new_base}_damage_mask{mask_file.suffix}"
            shutil.copy2(mask_file, mask_dest)
            stats[income_level]["mask"] += 1
        else:
            skipped["no_mask"] += 1
    
    # Print statistics
    print_section("BRIGHT Processing Statistics")
    
    print(f"Country Classification (COUNTRY_STRATA):")
    for income in ['LIC', 'MIC', 'HIC']:
        countries = ', '.join(COUNTRY_STRATA[income])
        print(f"  {income}: {countries}")
    
    print(f"\nProcessed Files by Income Level:")
    for income_level in ['lic', 'mic', 'hic']:
        print(f"\n{income_level.upper()}:")
        print(f"  Pre-disaster images:  {stats[income_level]['pre']:6d}")
        print(f"  Post-disaster images: {stats[income_level]['post']:6d}")
        print(f"  Damage masks:         {stats[income_level]['mask']:6d}")
    
    total_pre = sum(stats[level]["pre"] for level in ['lic', 'mic', 'hic'])
    total_post = sum(stats[level]["post"] for level in ['lic', 'mic', 'hic'])
    total_mask = sum(stats[level]["mask"] for level in ['lic', 'mic', 'hic'])
    
    print(f"\n{'TOTAL':3s}:")
    print(f"  Pre-disaster images:  {total_pre:6d}")
    print(f"  Post-disaster images: {total_post:6d}")
    print(f"  Damage masks:         {total_mask:6d}")
    
    # Print skipped files
    if any(skipped.values()):
        print(f"\nSkipped Files:")
        print(f"  No country identified: {skipped['no_country']:6d}")
        print(f"  No matching post:      {skipped['no_post']:6d}")
        print(f"  No matching mask:      {skipped['no_mask']:6d}")
        
        if unmatched_files and len(unmatched_files) <= 10:
            print(f"\nUnmatched filenames (sample):")
            for fname in unmatched_files[:10]:
                print(f"  - {fname}")
    
    # Calculate distribution percentages
    if total_pre > 0:
        print(f"\nIncome Distribution:")
        for income_level in ['lic', 'mic', 'hic']:
            count = stats[income_level]['pre']
            pct = (count / total_pre) * 100
            print(f"  {income_level.upper()}: {count:4d} events ({pct:5.1f}%)")
    
    print_success(f"\nBRIGHT dataset organized at: {BRIGHT_PROCESSED}")
    
    # Verify files match the expected naming convention
    print_info("\nVerifying naming convention...")
    sample_files = list((BRIGHT_PROCESSED / "lic" / "images").glob("*_pre_disaster.*"))[:3]
    if sample_files:
        print("Sample filenames:")
        for f in sample_files:
            print(f"  {f.name}")
    
    return True

################################################################################
# Verification
################################################################################

def verify_dataset_structure():
    """Verify that datasets match GitHub code expectations"""
    print_section("Verifying Dataset Structure")
    
    issues = []
    warnings = []
    
    # Verify xBD structure
    if XBD_PROCESSED.exists():
        print_info("Checking xBD structure...")
        for split in ['train', 'tier1', 'tier3', 'hold', 'test']:
            images_dir = XBD_PROCESSED / split / "images"
            labels_dir = XBD_PROCESSED / split / "labels"
            
            if not images_dir.exists():
                issues.append(f"Missing xBD/{split}/images directory")
            if not labels_dir.exists():
                issues.append(f"Missing xBD/{split}/labels directory")
            
            # Check for pre/post pairs and naming convention
            if images_dir.exists():
                pre_images = list(images_dir.glob("*_pre_disaster.*"))
                post_images = list(images_dir.glob("*_post_disaster.*"))
                
                if len(pre_images) != len(post_images):
                    warnings.append(f"xBD/{split}: Mismatch between pre ({len(pre_images)}) and post ({len(post_images)}) images")
                
                # Verify naming convention matches GitHub code expectations
                if pre_images:
                    sample = pre_images[0].name
                    if not re.match(r'.+_.+_pre_disaster\.(png|jpg)', sample):
                        warnings.append(f"xBD/{split}: Filename may not match expected pattern: {sample}")
    else:
        print_info("xBD not processed, skipping verification")
    
    # Verify BRIGHT structure
    if BRIGHT_PROCESSED.exists():
        print_info("Checking BRIGHT structure...")
        for income in ['lic', 'mic', 'hic']:
            images_dir = BRIGHT_PROCESSED / income / "images"
            masks_dir = BRIGHT_PROCESSED / income / "masks"
            
            if not images_dir.exists():
                issues.append(f"Missing BRIGHT/{income}/images directory")
            if not masks_dir.exists():
                issues.append(f"Missing BRIGHT/{income}/masks directory")
            
            # Check for pre/post pairs and naming convention
            if images_dir.exists():
                pre_images = list(images_dir.glob("*_pre_disaster.*"))
                post_images = list(images_dir.glob("*_post_disaster.*"))
                masks = list(masks_dir.glob("*_damage_mask.*")) if masks_dir.exists() else []
                
                print(f"  {income.upper()}: {len(pre_images)} pre, {len(post_images)} post, {len(masks)} masks")
                
                # Verify naming convention matches GitHub code expectations
                if pre_images:
                    sample = pre_images[0].name
                    # Expected: {country}_{disaster}_{sequence}_pre_disaster.{ext}
                    if not re.match(r'[a-z_]+_[a-z]+_\d+_pre_disaster\.(png|tif|jpg)', sample):
                        warnings.append(f"BRIGHT/{income}: Filename may not match expected pattern: {sample}")
                
                # Allow some mismatch since not all images may have all components
                if abs(len(pre_images) - len(post_images)) > max(1, len(pre_images) * 0.1):
                    warnings.append(f"BRIGHT/{income}: Significant mismatch between pre ({len(pre_images)}) and post ({len(post_images)}) images")
    else:
        print_info("BRIGHT not processed, skipping verification")
    
    if not issues and not warnings:
        print_success("All dataset structures verified successfully!")
        return True
    elif not issues:
        print_success("No critical issues found (warnings may be acceptable)")
        return True
    else:
        return False

################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess xBD and BRIGHT datasets to match GitHub code structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/02_preprocess_data.py                    # Process both datasets
  python scripts/02_preprocess_data.py --xbd-only         # Process only xBD
  python scripts/02_preprocess_data.py --bright-only      # Process only BRIGHT
  python scripts/02_preprocess_data.py --verify           # Process and verify

Country Classification (COUNTRY_STRATA):
  LIC: haiti, congo
  MIC: turkey, morocco, libya
  HIC: noto, la_palma, hawaii

Output Structure:
  xBD:    data/xbd/{train,tier1,tier3,hold,test}/{images,labels}/
  BRIGHT: data/bright/{lic,mic,hic}/{images,masks}/
        """
    )
    parser.add_argument('--xbd-only', action='store_true',
                       help='Process only xBD dataset')
    parser.add_argument('--bright-only', action='store_true',
                       help='Process only BRIGHT dataset')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset structure after preprocessing')
    
    args = parser.parse_args()
    
    # Determine what to process
    process_xbd = not args.bright_only
    process_bright = not args.xbd_only
    
    print_section("Dataset Preprocessing for GitHub Codebase")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Process xBD: {process_xbd}")
    print(f"Process BRIGHT: {process_bright}")
    
    if process_bright:
        print(f"\nCountry to Income Classification:")
        for income, countries in COUNTRY_STRATA.items():
            print(f"  {income}: {', '.join(countries)}")
    
    # Process datasets
    success = True
    
    if process_xbd:
        if not preprocess_xbd():
            success = False
    
    if process_bright:
        if not preprocess_bright():
            success = False
    
    # Always verify to ensure GitHub compatibility
    print_info("\nRunning automatic verification...")
    if not verify_dataset_structure():
        print_warning("Verification found issues - please review")
    
    # Final summary
    print_section("Preprocessing Complete")
    
    if success:
        print_success("All datasets processed successfully!")
        print("\nDataset locations:")
        if process_xbd and XBD_PROCESSED.exists():
            print(f"  xBD:    {XBD_PROCESSED}")
        if process_bright and BRIGHT_PROCESSED.exists():
            print(f"  BRIGHT: {BRIGHT_PROCESSED}")
        
        print("\nDataset structure matches GitHub code expectations:")
        print("  File naming conventions")
        print("  Directory organization")
        print("  Pre/post/mask pairing")
        
        # Write report
        report_file = DATA_DIR / "preprocessing_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Preprocessing completed: {Path.cwd()}\n")
            f.write(f"Timestamp: {Path.cwd()}\n\n")
            f.write(f"xBD processed: {process_xbd}\n")
            f.write(f"BRIGHT processed: {process_bright}\n\n")
            if process_xbd:
                f.write(f"xBD location: {XBD_PROCESSED}\n")
            if process_bright:
                f.write(f"BRIGHT location: {BRIGHT_PROCESSED}\n")
                f.write(f"\nCountry Classification:\n")
                for income, countries in COUNTRY_STRATA.items():
                    f.write(f"  {income}: {', '.join(countries)}\n")
        
        return 0
    else:
        print_error("Preprocessing completed with errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
