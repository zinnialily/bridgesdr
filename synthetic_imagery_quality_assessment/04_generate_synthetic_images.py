# Input: BRIGHT pre-disaster optical images from preprocessing
# Output: Synthetic post-disaster optical images
# ===================================================================
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision import transforms
from PIL import Image
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if "study1" in str(SCRIPT_DIR) else SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Configuration
# -------------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    
    # Paths
    bright_root = PROJECT_ROOT / "data" / "bright"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "disastergan" / "disastergan_generator_final.pth"
    output_dir = PROJECT_ROOT / "results" / "study1" / "synthetic_images"
    
    disaster_types = [
        'volcano', 'fire', 'tornado', 'tsunami',
        'flooding', 'earthquake', 'hurricane', 'wildfire'
    ]
    
    # Economic strata
    income_levels = ['lic', 'mic', 'hic']

config = Config()

# -------------------------
# Generator Architecture (same as training)
# -------------------------
class DisasterGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoders
        self.dec_img = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        self.dec_mask = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def add_disaster_channel(self, x, disaster):
        batch_size, _, h, w = x.size()
        disaster_map = disaster.view(-1, 1, 1, 1).expand(-1, -1, h, w).float() / len(config.disaster_types)
        return torch.cat([x, disaster_map], dim=1)

    def forward(self, x, disaster):
        x = self.add_disaster_channel(x, disaster)
        
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoding
        img = self.dec_img(e4)
        mask = self.dec_mask(e4)
        return img, mask

# -------------------------
# Utility Functions
# -------------------------
def infer_disaster_type(filename):
    """Infer disaster type from filename."""
    filename_lower = filename.lower()
    for d in config.disaster_types:
        if d in filename_lower:
            return d
    # Default to first disaster type if not found
    return config.disaster_types[0]

def tensor_to_pil(tensor):
    """
    Convert a normalized tensor (C,H,W) with values in [-1,1] to a PIL RGB image.
    """
    tensor = tensor.squeeze(0).cpu()  # Remove batch dim, move to CPU
    tensor = (tensor * 0.5) + 0.5     # Denormalize from [-1,1] to [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.permute(1, 2, 0).numpy() * 255
    np_img = np_img.astype(np.uint8)
    return Image.fromarray(np_img)

# -------------------------
# Main Generation Function
# -------------------------
def generate_synthetic_images():
    """Generate synthetic post-disaster images for BRIGHT dataset."""
    
    # Check if checkpoint exists
    if not config.checkpoint_path.exists():
        print(f"ERROR: Generator checkpoint not found at {config.checkpoint_path}")
        print("Please run training script first or check checkpoint path.")
        return
    
    # Load generator
    print("="*60)
    print("Loading DisasterGAN Generator")
    print("="*60)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Device: {config.device}\n")
    
    G = DisasterGenerator().to(config.device)
    G.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    G.eval()
    print("Generator loaded successfully\n")
    
    # Image transform
    transform_rgb = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Statistics tracking
    generation_stats = {
        'total_generated': 0,
        'by_income': {'lic': 0, 'mic': 0, 'hic': 0},
        'by_disaster': {d: 0 for d in config.disaster_types}
    }
    
    metadata_records = []
    
    # Process each income level
    for income in config.income_levels:
        income_dir = config.bright_root / income / "images"
        
        if not income_dir.exists():
            print(f"Warning: {income_dir} not found, skipping...")
            continue
        
        # Create output directory
        output_income_dir = config.output_dir / income
        output_income_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all pre-disaster images
        pre_files = list(income_dir.glob("*_pre_disaster.png"))
        
        if len(pre_files) == 0:
            print(f"Warning: No pre-disaster images found in {income_dir}")
            continue
        
        print(f"{'='*60}")
        print(f"Processing {income.upper()} - {len(pre_files)} images")
        print(f"{'='*60}\n")
        
        # Generate synthetic images
        for pre_path in tqdm(pre_files, desc=f"Generating {income.upper()}"):
            try:
                # Load pre-disaster image
                pre_img = Image.open(pre_path).convert('RGB')
                input_tensor = transform_rgb(pre_img).unsqueeze(0).to(config.device)
                
                # Infer disaster type from filename
                disaster_type = infer_disaster_type(pre_path.name)
                disaster_idx = config.disaster_types.index(disaster_type)
                disaster_label = torch.tensor([[disaster_idx]], device=config.device)
                
                # Generate synthetic post-disaster image
                with torch.no_grad():
                    fake_post, pred_mask = G(input_tensor, disaster_label)
                
                # Convert to PIL
                fake_post_img = tensor_to_pil(fake_post)
                pred_mask_img = tensor_to_pil(pred_mask.repeat(1, 3, 1, 1))  # Convert mask to RGB for visualization
                
                # Save synthetic image
                output_filename = pre_path.name.replace('_pre_disaster', '_post_disaster_synthetic')
                output_path = output_income_dir / output_filename
                fake_post_img.save(output_path)
                
                # Save mask
                mask_filename = pre_path.name.replace('_pre_disaster.png', '_mask_synthetic.png')
                mask_path = output_income_dir / mask_filename
                pred_mask_img.save(mask_path)
                
                # Update statistics
                generation_stats['total_generated'] += 1
                generation_stats['by_income'][income] += 1
                generation_stats['by_disaster'][disaster_type] += 1
                
                # Record metadata
                metadata_records.append({
                    'original_file': str(pre_path.name),
                    'synthetic_file': output_filename,
                    'mask_file': mask_filename,
                    'income_level': income,
                    'disaster_type': disaster_type,
                    'disaster_index': disaster_idx
                })
                
            except Exception as e:
                print(f"\nError processing {pre_path.name}: {e}")
                continue
        
        print(f"\n✓ Generated {generation_stats['by_income'][income]} synthetic images for {income.upper()}\n")
    
    # Save metadata
    metadata_path = config.output_dir / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_records, f, indent=2)
    
    # Save statistics
    stats_path = config.output_dir / "generation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(generation_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total synthetic images generated: {generation_stats['total_generated']}")
    print(f"\nBy income level:")
    for income, count in generation_stats['by_income'].items():
        print(f"  {income.upper()}: {count}")
    print(f"\nBy disaster type:")
    for disaster, count in generation_stats['by_disaster'].items():
        if count > 0:
            print(f"  {disaster}: {count}")
    print(f"\nOutput directory: {config.output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Statistics saved to: {stats_path}")
    print("="*60 + "\n")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic post-disaster images using DisasterGAN"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to generator checkpoint (default: checkpoints/disastergan/disastergan_generator_final.pth)'
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        config.checkpoint_path = Path(args.checkpoint)
    
    generate_synthetic_images()
