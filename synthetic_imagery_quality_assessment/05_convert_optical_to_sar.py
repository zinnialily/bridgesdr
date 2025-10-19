import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if "study1" in str(SCRIPT_DIR) else SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Configuration
# -------------------------
class Config:
    synthetic_images_dir = PROJECT_ROOT / "results" / "study1" / "synthetic_images"
    output_dir = PROJECT_ROOT / "results" / "study1" / "synthetic_sar"
    
    income_levels = ['lic', 'mic', 'hic']

config = Config()

# -------------------------
# Conversion Function (from original code)
# -------------------------
def optical_to_sar_like(img):
    """
    Convert optical RGB image to SAR-like grayscale.
    
    This matches the conversion used in evaluation:
    1. Convert to grayscale
    2. Apply autocontrast to enhance local intensity differences
    
    Args:
        img: PIL Image in RGB format
    
    Returns:
        PIL Image in grayscale (SAR-like)
    """
    img = img.convert('L')
    img = ImageOps.autocontrast(img, cutoff=2)
    return img

# -------------------------
# Main Conversion Function
# -------------------------
def convert_synthetic_to_sar():
    """Convert all synthetic optical images to SAR-like format."""
    
    if not config.synthetic_images_dir.exists():
        print(f"Synthetic images directory not found at {config.synthetic_images_dir}")
        print("Please run 04_generate_synthetic_images.py first.")
        return
    
    print("="*60)
    print("Converting Synthetic Optical Images to SAR-like Format")
    print("="*60)
    print(f"Input: {config.synthetic_images_dir}")
    print(f"Output: {config.output_dir}\n")
    
    conversion_stats = {
        'total_converted': 0,
        'by_income': {'lic': 0, 'mic': 0, 'hic': 0}
    }
    
    # Process each income level
    for income in config.income_levels:
        input_income_dir = config.synthetic_images_dir / income
        
        if not input_income_dir.exists():
            print(f"Warning: {input_income_dir} not found, skipping...")
            continue
        
        # Create output directory
        output_income_dir = config.output_dir / income
        output_income_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all synthetic post-disaster images (not masks)
        synthetic_files = [f for f in input_income_dir.glob("*_post_disaster_synthetic.png")]
        
        if len(synthetic_files) == 0:
            print(f"No synthetic images found in {input_income_dir}")
            continue
        
        print(f"{'='*60}")
        print(f"Processing {income.upper()} - {len(synthetic_files)} images")
        print(f"{'='*60}\n")
        
        # Convert each image
        for img_path in tqdm(synthetic_files, desc=f"Converting {income.upper()}"):
            try:
                # Load optical image
                optical_img = Image.open(img_path).convert('RGB')
                
                # Convert to SAR-like
                sar_like_img = optical_to_sar_like(optical_img)
                
                # Save
                output_filename = img_path.name.replace('.png', '_sar.png')
                output_path = output_income_dir / output_filename
                sar_like_img.save(output_path)
                
                # Update statistics
                conversion_stats['total_converted'] += 1
                conversion_stats['by_income'][income] += 1
                
            except Exception as e:
                print(f"\nError converting {img_path.name}: {e}")
                continue
        
        print(f"\nConverted {conversion_stats['by_income'][income]} images for {income.upper()}\n")
    
    # Save statistics
    stats_path = config.output_dir / "conversion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(conversion_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total images converted: {conversion_stats['total_converted']}")
    print(f"\nBy income level:")
    for income, count in conversion_stats['by_income'].items():
        print(f"  {income.upper()}: {count}")
    print(f"\nOutput directory: {config.output_dir}")
    print(f"Statistics saved to: {stats_path}")
    print("="*60 + "\n")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    convert_synthetic_to_sar()
