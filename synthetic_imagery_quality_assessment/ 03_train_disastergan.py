# ===================================================================
# XBD DisasterGAN - Training Script
# ===================================================================
#  TRAINING IS OPTIONAL - Pre-trained checkpoints are available at:
#    checkpoints/disastergan/disastergan_generator_final.pth
#    checkpoints/disastergan/disastergan_discriminator_final.pth
#
# These checkpoints are already saved in the repository and can be
# loaded directly for evaluation. Only run this script if you want to
# retrain the model from scratch or experiment with different settings.
#
# This script trains DisasterGAN on the preprocessed xBD dataset from
# 02_preprocess_data.py, using the 64×64 tiled images for efficient
# training following the DisasterGAN methodology.
#
# Dataset: XBD (https://xview2.org/dataset)
# DisasterGAN base configuration inspired by:
# "DisasterGAN: Generating Post-Disaster Images" by Adhoppin 
# Kaggle kernel: https://www.kaggle.com/code/adhoppin/disastergan-generating-post-disaster-images
# ===================================================================

# -------------------------
# Imports
# -------------------------
import os
import sys
import glob
import json
import argparse
from pathlib import Path
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch.nn.utils import spectral_norm
from torchvision.models import vgg16

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "reproduction_scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Configuration
# -------------------------
class Config:
    seed = 42
    tile_size = 64          # Use 64×64 tiles from preprocessing
    original_size = 256     # Original image size before tiling
    batch_size = 16
    epochs = 7
    lr = 2e-4
    betas = (0.5, 0.999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_cls = 1
    lambda_mask = 10
    lambda_gp = 10
    lambda_cycle = 10
  
    # Paths from preprocessing output
    data_root = PROJECT_ROOT / "data" / "xbd"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "disastergan"
    samples_dir = PROJECT_ROOT / "results" / "study1" / "training_samples"
    plots_dir = PROJECT_ROOT / "results" / "study1" / "training_plots"
    
    # Use tiled images for training (64×64)
    use_tiled = True

    disaster_types = [
        'volcano', 'fire', 'tornado', 'tsunami',
        'flooding', 'earthquake', 'hurricane', 'wildfire'
    ]

config = Config()
config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
config.samples_dir.mkdir(parents=True, exist_ok=True)
config.plots_dir.mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

print(f"Using device: {config.device}")
print(f"Data root: {config.data_root}")
print(f"Checkpoint directory: {config.checkpoint_dir}")

# -------------------------
# Dataset
# -------------------------
class XBDDisasterGANDataset(Dataset):
    def __init__(self, split_names=("train", "tier1", "tier3"), use_tiled=True):
        """
        Args:
            split_names: Which splits to use from xBD
            use_tiled: If True, use 64×64 tiles; if False, use 256×256 originals
        """
        self.use_tiled = use_tiled
        self.pairs = []
        
        for split in split_names:
            split_path = config.data_root / split
            if not split_path.exists():
                print(f"Warning: Split '{split}' not found at {split_path}, skipping...")
                continue
            
            # Choose tiled or original images
            if use_tiled:
                images_dir = split_path / "images_tiled"
                pattern = "*_pre_disaster_tile_*.png"
            else:
                images_dir = split_path / "images"
                pattern = "*_pre_disaster.png"
            
            labels_dir = split_path / "labels"
            
            if not images_dir.exists():
                print(f"Warning: Images directory not found at {images_dir}, skipping...")
                continue
            
            pre_images = list(images_dir.glob(pattern))
            
            for pre_path in pre_images:
                # Construct corresponding post and label paths
                if use_tiled:
                    post_path = str(pre_path).replace("_pre_disaster_", "_post_disaster_")
                    # Get base name without tile suffix for label
                    base_name = pre_path.name.split("_tile_")[0]
                    label_path = labels_dir / f"{base_name}_post_disaster.json"
                else:
                    base = pre_path.stem.replace("_pre_disaster", "")
                    post_path = images_dir / f"{base}_post_disaster.png"
                    label_path = labels_dir / f"{base}_post_disaster.json"
                
                post_path = Path(post_path)
                
                if post_path.exists():
                    self.pairs.append({
                        "pre": str(pre_path),
                        "post": str(post_path),
                        "label": str(label_path) if label_path.exists() else None
                    })
        
        print(f"Found {len(self.pairs)} image pairs across {split_names}")
        
        # Transform based on whether using tiles or originals
        if use_tiled:
            # Tiles are already 64×64, just normalize
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            # Resize originals to 64×64 for consistency with DisasterGAN
            self.transform = transforms.Compose([
                transforms.Resize(config.tile_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def parse_disaster(self, label_path):
        """Extract disaster type from JSON label file."""
        if label_path is None:
            return torch.tensor(0, dtype=torch.long)
        
        try:
            with open(label_path) as f:
                data = json.load(f)
            disaster = data['metadata']['disaster_type'].lower()
            
            # Try to match with known disaster types
            for idx, dtype in enumerate(config.disaster_types):
                if dtype in disaster or disaster in dtype:
                    return torch.tensor(idx, dtype=torch.long)
            
            # Default to first type if not found
            return torch.tensor(0, dtype=torch.long)
        except Exception as e:
            # Default to first disaster type on error
            return torch.tensor(0, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        pre_img = self.transform(Image.open(pair["pre"]).convert('RGB'))
        post_img = self.transform(Image.open(pair["post"]).convert('RGB'))
        
        # Get disaster label
        disaster_label = self.parse_disaster(pair["label"])
        
        # Generate binary damage mask from image difference
        with torch.no_grad():
            diff = torch.abs(post_img - pre_img).mean(dim=0, keepdim=True)
            mask = (diff > 0.1).float()
        
        return {
            'pre': pre_img,
            'post': post_img,
            'disaster': disaster_label,
            'mask': mask
        }

# -------------------------
# Generator
# -------------------------
class DisasterGenerator(nn.Module):
    """
    DisasterGAN Generator that takes pre-disaster image + disaster type
    and outputs post-disaster image + damage mask.
    
    Input: (B, 4, H, W) - 3 channels RGB + 1 channel disaster type
    Output: 
        - Generated post-disaster image (B, 3, H, W)
        - Binary damage mask (B, 1, H, W)
    """
    
    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
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

        # Decoder for post-disaster image
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

        # Decoder for damage mask
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
        """Add disaster type as additional input channel."""
        batch_size, _, h, w = x.size()
        # Normalize disaster index to [0, 1] range
        disaster_map = disaster.view(-1, 1, 1, 1).expand(-1, -1, h, w).float() / len(config.disaster_types)
        return torch.cat([x, disaster_map], dim=1)

    def forward(self, x, disaster):
        """
        Args:
            x: Pre-disaster RGB image (B, 3, H, W)
            disaster: Disaster type index (B,)
        
        Returns:
            img: Generated post-disaster image (B, 3, H, W)
            mask: Generated damage mask (B, 1, H, W)
        """
        x = self.add_disaster_channel(x, disaster)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        img = self.dec_img(e4)
        mask = self.dec_mask(e4)
        return img, mask

# -------------------------
# Discriminator
# -------------------------
class DisasterDiscriminator(nn.Module):
    """
    DisasterGAN Discriminator with auxiliary classifier for disaster type.
    
    Outputs:
        - Source prediction: Real/Fake classification
        - Class prediction: Disaster type classification
    """
    
    def __init__(self):
        super().__init__()
        # Main feature extractor
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Real/Fake classification head
        self.src = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        
        # Disaster type classification head
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(512, len(config.disaster_types), 1)),
            nn.Flatten()
        )

    def forward(self, x):
        """
        Args:
            x: Post-disaster image (B, 3, H, W)
        
        Returns:
            src: Real/Fake score (B, 1, H', W')
            cls: Disaster type logits (B, num_disaster_types)
        """
        features = self.main(x)
        return self.src(features), self.cls(features)

# -------------------------
# Training Utilities
# -------------------------
def compute_gp(D, real, fake):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def perceptual_loss(fake, real, vgg):
    """VGG-based perceptual loss."""
    return nn.functional.l1_loss(vgg(fake), vgg(real))

def total_variation(x):
    """Total variation loss for smoothness."""
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

def load_pretrained_checkpoints(G, D, checkpoint_dir):
    """
    Load pre-trained checkpoints if available.
    
    Returns:
        bool: True if checkpoints were loaded successfully
    """
    gen_path = checkpoint_dir / "disastergan_generator_final.pth"
    disc_path = checkpoint_dir / "disastergan_discriminator_final.pth"
    
    if gen_path.exists() and disc_path.exists():
        print("\n" + "="*60)
        print("Found pre-trained checkpoints!")
        print(f"  Generator: {gen_path}")
        print(f"  Discriminator: {disc_path}")
        
        response = input("Load pre-trained weights? (y/n): ").lower().strip()
        if response == 'y':
            G.load_state_dict(torch.load(gen_path, map_location=config.device))
            D.load_state_dict(torch.load(disc_path, map_location=config.device))
            print("✓ Loaded pre-trained checkpoints successfully!")
            print("="*60 + "\n")
            return True
    
    return False

# -------------------------
# Main Training Function
# -------------------------
def train():
    """Main training loop for DisasterGAN."""
    
    # Initialize models
    print("\n" + "="*60)
    print("Initializing DisasterGAN models...")
    print("="*60 + "\n")
    
    G = DisasterGenerator().to(config.device)
    D = DisasterDiscriminator().to(config.device)
    
    # Try to load pre-trained checkpoints
    if load_pretrained_checkpoints(G, D, config.checkpoint_dir):
        response = input("Continue training from checkpoint? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting. Use the loaded checkpoints for evaluation.")
            return
    
    # Initialize optimizers
    opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=config.betas)
    opt_D = optim.Adam(D.parameters(), lr=config.lr, betas=config.betas)

    # Initialize perceptual loss network (VGG16)
    print("Loading VGG16 for perceptual loss...")
    vgg = vgg16(pretrained=True).features[:16].eval().to(config.device)
    for param in vgg.parameters():
        param.requires_grad = False

    # Loss functions
    L1 = nn.L1Loss()
    CE = nn.CrossEntropyLoss()

    # Create dataset and dataloader
    print(f"\nLoading xBD dataset from: {config.data_root}")
    print(f"Using {'tiled (64×64)' if config.use_tiled else 'original (256×256)'} images")
    
    dataset = XBDDisasterGANDataset(
        split_names=["train", "tier1", "tier3"],
        use_tiled=config.use_tiled
    )
    
    if len(dataset) == 0:
        print("\nERROR: No data found! Please run preprocessing first:")
        print("   python reproduction_scripts/02_preprocess_data.py")
        return
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device.type == 'cuda' else False
    )

    visual_dir = config.samples_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'epoch': []
    }

    # -------------------------
    # Training Loop
    # -------------------------
    print("\n" + "="*60)
    print(f"Starting training for {config.epochs} epochs")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Device: {config.device}")
    print("="*60 + "\n")

    for epoch in range(config.epochs):
        G.train()
        D.train()
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            real_pre = batch['pre'].to(config.device)
            real_post = batch['post'].to(config.device)
            disaster = batch['disaster'].to(config.device)
            real_mask = batch['mask'].to(config.device)

            # -------------------
            # Update Discriminator
            # -------------------
            opt_D.zero_grad()
            
            # Real images
            src_real, cls_real = D(real_post)
            loss_real = -torch.mean(src_real)
            loss_cls_real = CE(cls_real, disaster)

            # Fake images
            with torch.no_grad():
                fake_post, _ = G(real_pre, disaster)
            src_fake, _ = D(fake_post)
            loss_fake = torch.mean(src_fake)

            # Gradient penalty
            gp = compute_gp(D, real_post, fake_post)
            
            # Total discriminator loss
            loss_D = loss_real + loss_fake + config.lambda_gp * gp + config.lambda_cls * loss_cls_real
            loss_D.backward()
            opt_D.step()

            # -------------------
            # Update Generator
            # -------------------
            opt_G.zero_grad()
            
            fake_post, pred_mask = G(real_pre, disaster)
            src_fake, cls_fake = D(fake_post)

            # Adversarial loss
            loss_adv = -torch.mean(src_fake)
            
            # Classification loss
            loss_cls = CE(cls_fake, disaster)
            
            # Mask loss
            loss_mask = L1(pred_mask, real_mask)
            
            # Cycle consistency loss
            loss_cycle = L1(
                G(fake_post, torch.zeros_like(disaster).to(config.device))[0],
                real_pre
            )
            
            # Pixel-level loss (L1 + perceptual)
            loss_pixel = 3.0 * perceptual_loss(fake_post, real_post, vgg) + \
                        0.05 * L1(fake_post, real_post)
            
            # Total variation for smoothness
            loss_tv = total_variation(fake_post)

            # Progressive weighting schedule
            progress = epoch / config.epochs
            current_lambda_adv = 0.05 + 0.95 * progress
            current_lambda_pixel = max(150 * (1 - progress), 20)

            # Total generator loss
            loss_G = (current_lambda_adv * loss_adv +
                     config.lambda_cls * loss_cls +
                     current_lambda_pixel * loss_pixel +
                     config.lambda_mask * loss_mask +
                     config.lambda_cycle * loss_cycle +
                     0.1 * loss_tv)
            
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })

        # Epoch summary
        avg_g_loss = epoch_g_loss / len(loader)
        avg_d_loss = epoch_d_loss / len(loader)
        
        history['epoch'].append(epoch + 1)
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Generator Loss: {avg_g_loss:.4f}")
        print(f"  Discriminator Loss: {avg_d_loss:.4f}")

        # -------------------
        # Save Visual Samples
        # -------------------
        if (epoch + 1) % 1 == 0:  # Save every epoch
            G.eval()
            with torch.no_grad():
                # Get sample batch
                sample_batch = next(iter(loader))
                sample_pre = sample_batch['pre'][:8].to(config.device)
                sample_post = sample_batch['post'][:8].to(config.device)
                sample_disaster = sample_batch['disaster'][:8].to(config.device)
                
                # Generate fake post-disaster images
                fake_post, fake_mask = G(sample_pre, sample_disaster)
                
                # Create visualization grid: [pre, fake, real]
                visuals = torch.cat([
                    sample_pre.cpu(),
                    fake_post.cpu(),
                    sample_post.cpu()
                ], dim=0)
                
                grid = vutils.make_grid(
                    visuals,
                    nrow=8,
                    normalize=True,
                    scale_each=True,
                    padding=2
                )
                
                visual_path = visual_dir / f"epoch_{epoch+1:03d}.png"
                vutils.save_image(grid, visual_path)
                print(f"  Saved visual samples to: {visual_path}")

        # -------------------
        # Save Checkpoints
        # -------------------
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            checkpoint_path_g = config.checkpoint_dir / f"disastergan_generator_epoch_{epoch+1}.pth"
            checkpoint_path_d = config.checkpoint_dir / f"disastergan_discriminator_epoch_{epoch+1}.pth"
            
            torch.save(G.state_dict(), checkpoint_path_g)
            torch.save(D.state_dict(), checkpoint_path_d)
            print(f"  Saved checkpoints at epoch {epoch+1}")

    # -------------------
    # Save Final Models
    # -------------------
    final_gen_path = config.checkpoint_dir / "disastergan_generator_final.pth"
    final_disc_path = config.checkpoint_dir / "disastergan_discriminator_final.pth"
    
    torch.save(G.state_dict(), final_gen_path)
    torch.save(D.state_dict(), final_disc_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"  Final generator saved to: {final_gen_path}")
    print(f"  Final discriminator saved to: {final_disc_path}")
    print(f"  Training samples saved to: {visual_dir}")
    print("="*60 + "\n")

    # -------------------
    # Plot Training Curves
    # -------------------
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['g_loss'], label='Generator Loss')
    plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DisasterGAN Training Curves')
    plt.legend()
    plt.grid(True)
    
    plot_path = config.plots_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    plt.close()


# -------------------------
# Command Line Interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DisasterGAN on preprocessed xBD dataset (OPTIONAL - pre-trained checkpoints available)"
    )
    parser.add_argument(
        '--force-train',
        action='store_true',
        help='Force training even if checkpoints exist'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--no-tiled',
        action='store_true',
        help='Use 256×256 original images instead of 64×64 tiles'
    )
    
    args = parser.parse_args()
    
    # Update config from args
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.use_tiled = not args.no_tiled
    
    # Check if training is necessary
    gen_exists = (config.checkpoint_dir / "disastergan_generator_final.pth").exists()
    disc_exists = (config.checkpoint_dir / "disastergan_discriminator_final.pth").exists()
    
    if gen_exists and disc_exists and not args.force_train:
        print("\n" + "="*60)
        print("PRE-TRAINED CHECKPOINTS DETECTED")
        print("="*60)
        print("\nDisasterGAN checkpoints are already available at:")
        print(f"  {config.checkpoint_dir}/disastergan_generator_final.pth")
        print(f"  {config.checkpoint_dir}/disastergan_discriminator_final.pth")
        print("\nYou can use these checkpoints directly for evaluation.")
        print("Training is OPTIONAL and not required for reproduction.")
        print("\nOptions:")
        print("  1. Skip training and use existing checkpoints (recommended)")
        print("  2. Retrain from scratch (overwrites checkpoints)")
        print("  3. Continue training from checkpoint")
        print("="*60 + "\n")
        
        response = input("Enter choice (1/2/3): ").strip()
        
        if response == '1':
            print("\nSkipping training. Use existing checkpoints for evaluation.")
            sys.exit(0)
        elif response == '2':
            print("\nRetraining from scratch...")
        elif response == '3':
            print("\nContinuing from checkpoint...")
        else:
            print("\nInvalid choice. Exiting.")
            sys.exit(1)
    
    # Run training
    train()
