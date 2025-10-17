# ===================================================================
# XBD DisasterGAN - Full Journal Reproducibility Script
# ===================================================================
# This script reproduces the GAN training for post-disaster image 
# generation and mask prediction as described in the original 
# methodology.
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
import glob
import json
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

# -------------------------
# Configuration
# -------------------------
class Config:
    seed = 42
    img_size = 256
    batch_size = 16
    epochs = 10
    lr = 2e-4
    betas = (0.5, 0.999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_cls = 1
    lambda_mask = 10
    lambda_gp = 10
    lambda_cycle = 10
  
    data_root = "./data/xbd"        
    save_dir = "./saved_models"     
    samples_dir = "./samples"
    plots_dir = "./plots"

    disaster_types = [
        'volcano', 'fire', 'tornado', 'tsunami',
        'flooding', 'earthquake', 'hurricane'
    ]

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.samples_dir, exist_ok=True)
os.makedirs(config.plots_dir, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class XBDDataset(Dataset):
    def __init__(self, split_names=("train", "tier1", "tier3")):
        self.pairs = []
        for split in split_names:
            split_path = os.path.join(config.data_root, split)
            images_dir = os.path.join(split_path, "images")
            labels_dir = os.path.join(split_path, "labels")
            pre_images = glob.glob(os.path.join(images_dir, "*_pre_disaster.png"))
            for pre_path in pre_images:
                base = os.path.basename(pre_path).replace("_pre_disaster.png", "")
                post_path = os.path.join(images_dir, f"{base}_post_disaster.png")
                label_path = os.path.join(labels_dir, f"{base}_post_disaster.json")
                if os.path.exists(post_path):
                    self.pairs.append({
                        "pre": pre_path,
                        "post": post_path,
                        "label": label_path if os.path.exists(label_path) else None
                    })

        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.pairs)

    def parse_disaster(self, label_path):
        try:
            with open(label_path) as f:
                data = json.load(f)
            disaster = data['metadata']['disaster_type']
            return torch.tensor(config.disaster_types.index(disaster), dtype=torch.long)
        except:
            return torch.tensor(0, dtype=torch.long)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        pre_img = self.transform(Image.open(pair["pre"]).convert('RGB'))
        post_img = self.transform(Image.open(pair["post"]).convert('RGB'))
        disaster_label = self.parse_disaster(pair["label"])
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
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2))

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

    def add_disaster_channel(self, x, disaster):
        batch_size, _, h, w = x.size()
        disaster_map = disaster.view(-1, 1, 1, 1).expand(-1, -1, h, w).float() / len(config.disaster_types)
        return torch.cat([x, disaster_map], dim=1)

    def forward(self, x, disaster):
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
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2)
        )
        self.src = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), spectral_norm(nn.Conv2d(512, len(config.disaster_types), 1)), nn.Flatten())

    def forward(self, x):
        features = self.main(x)
        return self.src(features), self.cls(features)

# -------------------------
# Training Utilities
# -------------------------
def compute_gp(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def perceptual_loss(fake, real, vgg):
    return nn.functional.l1_loss(vgg(fake), vgg(real))

def total_variation(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

# -------------------------
# Initialize Models and Optimizers
# -------------------------
G = DisasterGenerator().to(config.device)
D = DisasterDiscriminator().to(config.device)
opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=config.betas)
opt_D = optim.Adam(D.parameters(), lr=config.lr, betas=config.betas)

vgg = vgg16(pretrained=True).features[:16].eval().to(config.device)
for param in vgg.parameters():
    param.requires_grad = False

L1 = nn.L1Loss()
CE = nn.CrossEntropyLoss()

dataset = XBDDataset()
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

visual_dir = os.path.join(config.samples_dir, "visuals")
os.makedirs(visual_dir, exist_ok=True)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(config.epochs):
    G.train()
    D.train()
    epoch_g_loss = 0
    epoch_d_loss = 0
    print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")

    for batch_idx, batch in enumerate(tqdm(loader)):
        real_pre = batch['pre'].to(config.device)
        real_post = batch['post'].to(config.device)
        disaster = batch['disaster'].to(config.device)
        real_mask = batch['mask'].to(config.device)

        # -------------------
        # Update Discriminator
        # -------------------
        opt_D.zero_grad()
        src_real, cls_real = D(real_post)
        loss_real = -torch.mean(src_real)
        loss_cls_real = CE(cls_real, disaster)

        with torch.no_grad():
            fake_post, _ = G(real_pre, disaster)
        src_fake, _ = D(fake_post)
        loss_fake = torch.mean(src_fake)

        gp = compute_gp(D, real_post, fake_post)
        loss_D = loss_real + loss_fake + config.lambda_gp * gp + config.lambda_cls * loss_cls_real
        loss_D.backward()
        opt_D.step()

        # -------------------
        # Update Generator
        # -------------------
        opt_G.zero_grad()
        fake_post, pred_mask = G(real_pre, disaster)
        src_fake, cls_fake = D(fake_post)

        loss_adv = -torch.mean(src_fake)
        loss_cls = CE(cls_fake, disaster)
        loss_mask = L1(pred_mask, real_mask)
        loss_cycle = L1(G(fake_post, torch.zeros_like(disaster).to(config.device))[0], real_pre)
        loss_pixel = 3.0 * perceptual_loss(fake_post, real_post, vgg) + 0.05 * L1(fake_post, real_post)
        loss_tv = total_variation(fake_post)

        # Linear schedule for adv/pixel weights
        progress = epoch / config.epochs
        current_lambda_adv = 0.05 + 0.95 * progress
        current_lambda_pixel = max(150 * (1 - progress), 20)

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

    print(f"Epoch {epoch+1} summary — G_loss: {epoch_g_loss/len(loader):.4f}, D_loss: {epoch_d_loss/len(loader):.4f}")

    # -------------------
    # Save visuals
    # -------------------
    G.eval()
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_pre = sample_batch['pre'][:4].to(config.device)
        sample_post = sample_batch['post'][:4].to(config.device)
        sample_disaster = sample_batch['disaster'][:4].to(config.device)
        fake_post, _ = G(sample_pre, sample_disaster)
        visuals = torch.cat([sample_pre.cpu(), fake_post.cpu(), sample_post.cpu()], dim=0)
        grid = vutils.make_grid(visuals, nrow=4, normalize=True, scale_each=True)
        visual_path = os.path.join(visual_dir, f"epoch_{epoch+1}.png")
        vutils.save_image(grid, visual_path)

# -------------------------
# Save final models
# -------------------------
torch.save(G.state_dict(), os.path.join(config.save_dir, "G_final.pth"))
torch.save(D.state_dict(), os.path.join(config.save_dir, "D_final.pth"))
print("Training complete. Models and visuals saved.")
