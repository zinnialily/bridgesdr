# ==============================================================================
# TRAINING EFFECTIVENESS ASSESSMENT - COMPLETE REPRODUCTION SCRIPT
# Stage 0: Baseline training on xBD (7 epochs, 1e-3 LR)
# Stage 1: Half fine-tuning - Decoder only (10 epochs, 1e-4 LR)
# Stage 2: Full fine-tuning - Entire network (20 epochs, 1e-4 LR)
# ==============================================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import transforms as T
import numpy as np
from PIL import Image, ImageOps
import random
import copy
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
CONFIG = {
    # Directory Paths
    "XBD_DATA_ROOT": "path/to/xbd/dataset",
    "BRIGHT_LIC_ROOT": "path/to/BRIGHT_LIC_pseudo",
    "BRIGHT_MIC_ROOT": "path/to/BRIGHT_MIC_pseudo",
    "BRIGHT_HIC_ROOT": "path/to/BRIGHT_HIC_pseudo",
    "MODEL_SAVE_DIR": "models/",
    "RESULTS_SAVE_DIR": "results/",

    # Training Hyperparameters
    "BATCH_SIZE": 16,
    "BASELINE_LEARNING_RATE": 1e-3,
    "FINETUNE_LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "BASELINE_EPOCHS": 7,
    "FINETUNE_EPOCHS": 3,
    
    # Reproducibility
    "RANDOM_SEED": 42,
    
    # Learning Rate Scheduler
    "SCHEDULER_PATIENCE": 3,
    "SCHEDULER_FACTOR": 0.5,

    # Model Architecture
    "IN_CHANNELS": 4,
    "OUT_CLASSES": 4,

    # Data Configuration
    "IMAGE_SIZE": (256, 256),
    "TRAIN_TEST_SPLIT": 0.7,
}

# Economic Stratification by Country
COUNTRY_STRATA = {
    "LIC": ["haiti", "congo"],
    "MIC": ["turkey", "morocco", "libya"],
    "HIC": ["noto", "la_palma", "hawaii"],
}


# ==============================================================================
# 1. MODEL ARCHITECTURE: U-NET
# ==============================================================================
class UNet(nn.Module):
    """
    U-Net architecture for pixel-wise damage segmentation.
    
    Input: 256×256, 4 channels (RGB pre-disaster + SAR post-disaster)
    Output: 256×256, 4-class damage mask
    Encoder: 4 convolutional blocks with max pooling
    Bottleneck: 1024 channels
    Decoder: 4 transposed convolution blocks with skip connections
    """
    def __init__(self, in_channels=4, out_classes=4):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder Path
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder Path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Output Layer
        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections saved
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ==============================================================================
# 2. DATASET PREPARATION
# ==============================================================================
class DisasterDataset(Dataset):
    """
    Unified dataset for both xBD (real) and BRIGHT (synthetic) disaster imagery.
    
    Pre-disaster: RGB optical imagery (3 channels)
    Post-disaster: Converted to SAR-like (1 channel) via grayscale + autocontrast
    Masks: 4-class damage classification via RGB color mapping
    """
    def __init__(self, pre_dir, post_dir, mask_dir, file_list, is_xbd=False):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir
        self.files = file_list
        self.is_xbd = is_xbd

        self.transform_pre = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_post = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.files)

    def _optical_to_sar_like(self, img):
        """Convert optical post-disaster image to SAR-like appearance."""
        img = img.convert('L')
        img = ImageOps.autocontrast(img, cutoff=2)
        return img

    def _convert_mask_multiclass(self, mask_rgb_img):
        """Convert RGB damage mask to class labels."""
        mask_np = np.array(mask_rgb_img)
        label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
        
        color_to_label = {
            (0, 0, 0): 0,
            (0, 255, 255): 0,
            (0, 0, 255): 1,
            (255, 255, 0): 2,
            (255, 0, 0): 3,
            (211, 211, 211): 0
        }
        
        for rgb, label in color_to_label.items():
            mask = np.all(mask_np == rgb, axis=-1)
            label_mask[mask] = label
        
        return label_mask

    def __getitem__(self, idx):
        pre_file_name = self.files[idx]
        post_file_name = pre_file_name.replace('pre', 'post')
        
        if self.is_xbd:
            mask_file_name = post_file_name.replace('.png', '_rgb.png')
        else:
            mask_file_name = post_file_name

        pre_path = os.path.join(self.pre_dir, pre_file_name)
        post_path = os.path.join(self.post_dir, post_file_name)
        mask_path = os.path.join(self.mask_dir, mask_file_name)

        pre_img = Image.open(pre_path).convert("RGB")
        post_img_optical = Image.open(post_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("RGB")

        post_img_sar = self._optical_to_sar_like(post_img_optical)

        pre_tensor = self.transform_pre(pre_img)
        post_tensor_sar = self.transform_post(post_img_sar)

        input_tensor = torch.cat([pre_tensor, post_tensor_sar], dim=0)

        mask_np = self._convert_mask_multiclass(mask_img)
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)

        return input_tensor, mask_tensor


def calculate_class_weights(dataloader, num_classes):
    """Calculate class weights for addressing class imbalance."""
    print("Calculating class weights for loss balancing...")
    counts = torch.zeros(num_classes, dtype=torch.double)
    
    for _, masks in tqdm(dataloader, desc="Calculating Class Weights"):
        counts += torch.bincount(masks.flatten(), minlength=num_classes)
    
    total_pixels = counts.sum()
    weights = total_pixels / (num_classes * counts)
    weights[torch.isinf(weights)] = 0

    print(f"Class weights: {weights}")
    return weights.float()


def prepare_dataloaders(root_dir, strata_keywords, split_ratio, batch_size, is_xbd=False):
    """Prepare train and test dataloaders for a given economic stratum."""
    if is_xbd:
        pre_dir = os.path.join(root_dir, "images")
        post_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")
    else:
        pre_dir = os.path.join(root_dir, "train", "pre")
        post_dir = os.path.join(root_dir, "train", "post")
        mask_dir = os.path.join(root_dir, "train", "mask")
    
    all_files = [f for f in os.listdir(pre_dir) if "pre" in f]
    
    if is_xbd:
        file_list = all_files
    else:
        file_list = [f for f in all_files if any(keyword in f for keyword in strata_keywords)]

    dataset = DisasterDataset(pre_dir, post_dir, mask_dir, file_list, is_xbd)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(CONFIG["RANDOM_SEED"])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: {train_size} training, {test_size} testing samples")
    
    return train_loader, test_loader


# ==============================================================================
# 3. TRAINING FUNCTIONS
# ==============================================================================
def train_model(model, train_loader, device, num_epochs, save_name):
    """Baseline model training (Stage 0)."""
    print(f"\n{'='*70}")
    print(f"STAGE 0: Baseline Training")
    print(f"{'='*70}")
    print(f"Dataset: xBD")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {CONFIG['BASELINE_LEARNING_RATE']}")
    
    class_weights = calculate_class_weights(train_loader, CONFIG["OUT_CLASSES"]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG["BASELINE_LEARNING_RATE"], 
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=CONFIG["SCHEDULER_PATIENCE"], 
        factor=CONFIG["SCHEDULER_FACTOR"]
    )
    
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, masks in progress_bar:
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(progress_bar))
        
        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    save_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], f"{save_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nBaseline model saved: {save_path}")


def fine_tune_model(model, train_loader, device, num_epochs, save_name, stage='half'):
    """Progressive fine-tuning (Stage 1 or Stage 2)."""
    class_weights = calculate_class_weights(train_loader, CONFIG["OUT_CLASSES"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if stage == 'half':
        print(f"\n{'='*70}")
        print(f"STAGE 1: Half Fine-Tuning (Decoder Only)")
        print(f"{'='*70}")
        print("Layers: Decoder only (encoder frozen)")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {CONFIG['FINETUNE_LEARNING_RATE']}")
        
        for name, param in model.named_parameters():
            if 'enc' in name or 'bottleneck' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=CONFIG["FINETUNE_LEARNING_RATE"], 
            weight_decay=CONFIG["WEIGHT_DECAY"]
        )
    
    elif stage == 'full':
        print(f"\n{'='*70}")
        print(f"STAGE 2: Full Fine-Tuning (Entire Network)")
        print(f"{'='*70}")
        print("Layers: All layers (encoder + decoder)")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {CONFIG['FINETUNE_LEARNING_RATE']}")
        
        for param in model.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable: {trainable_params:,} (100%)")
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=CONFIG["FINETUNE_LEARNING_RATE"], 
            weight_decay=CONFIG["WEIGHT_DECAY"]
        )
    
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 'half' or 'full'.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=CONFIG["SCHEDULER_PATIENCE"], 
        factor=CONFIG["SCHEDULER_FACTOR"]
    )
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, masks in progress_bar:
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(progress_bar))

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    save_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], f"{save_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned model saved: {save_path}")


# ==============================================================================
# 4. EVALUATION FRAMEWORK
# ==============================================================================
def calculate_metrics(preds, labels):
    """Calculate IoU, Dice, Precision, and Recall for binary damage detection."""
    preds_binary = (preds > 0).cpu().numpy().flatten()
    labels_binary = (labels > 0).cpu().numpy().flatten()

    intersection = np.sum(preds_binary * labels_binary)
    union = np.sum(preds_binary) + np.sum(labels_binary) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (np.sum(preds_binary) + np.sum(labels_binary) + 1e-6)
    
    precision = precision_score(labels_binary, preds_binary, zero_division=0)
    recall = recall_score(labels_binary, preds_binary, zero_division=0)

    return iou, dice, precision, recall


def evaluate_model(model, test_loader, device, label=""):
    """Evaluate model on test set."""
    model.to(device)
    model.eval()
    
    total_iou, total_dice, total_precision, total_recall = 0, 0, 0, 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Evaluating {label}")
        for inputs, masks in progress_bar:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            iou, dice, precision, recall = calculate_metrics(preds, masks)
            total_iou += iou
            total_dice += dice
            total_precision += precision
            total_recall += recall
    
    num_batches = len(test_loader)
    results = {
        "label": label,
        "IoU": total_iou / num_batches,
        "Dice": total_dice / num_batches,
        "Precision": total_precision / num_batches,
        "Recall": total_recall / num_batches
    }
    
    return results


# ==============================================================================
# 5. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("PERFORMANCE BIAS METHODOLOGY - REPRODUCTION SCRIPT")
    print("="*70)
    print("\nThree-Stage Progressive Fine-Tuning:")
    print("  Stage 0: Baseline training on xBD")
    print("  Stage 1: Half fine-tuning (decoder only)")
    print("  Stage 2: Full fine-tuning (entire network)")
    print("="*70)
    
    # Set random seeds
    random.seed(CONFIG["RANDOM_SEED"])
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"\nRandom seed: {CONFIG['RANDOM_SEED']}")

    os.makedirs(CONFIG["MODEL_SAVE_DIR"], exist_ok=True)
    os.makedirs(CONFIG["RESULTS_SAVE_DIR"], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = []
    
    # Stage 0: Baseline Training
    print("\n" + "="*70)
    print("STAGE 0: BASELINE MODEL TRAINING")
    print("="*70)
    
    xbd_train_loader, xbd_test_loader = prepare_dataloaders(
        CONFIG["XBD_DATA_ROOT"], 
        [], 
        CONFIG["TRAIN_TEST_SPLIT"], 
        CONFIG["BATCH_SIZE"], 
        is_xbd=True
    )
    
    baseline_model = UNet(CONFIG["IN_CHANNELS"], CONFIG["OUT_CLASSES"])
    baseline_model_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], "baseline_model.pth")

    if not os.path.exists(baseline_model_path):
        train_model(
            baseline_model, 
            xbd_train_loader, 
            device, 
            CONFIG["BASELINE_EPOCHS"], 
            "baseline_model"
        )
    else:
        print(f"\nLoading existing baseline model from: {baseline_model_path}")
        baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))

    # Stages 1 & 2: Progressive Fine-Tuning
    strata_roots = {
        "LIC": CONFIG["BRIGHT_LIC_ROOT"],
        "MIC": CONFIG["BRIGHT_MIC_ROOT"],
        "HIC": CONFIG["BRIGHT_HIC_ROOT"],
    }
    
    for stratum, root in strata_roots.items():
        print("\n" + "="*70)
        print(f"PROCESSING {stratum} ECONOMIC STRATUM")
        print("="*70)

        stratum_train_loader, stratum_test_loader = prepare_dataloaders(
            root, 
            COUNTRY_STRATA[stratum], 
            CONFIG["TRAIN_TEST_SPLIT"], 
            CONFIG["BATCH_SIZE"]
        )
        
        # Stage 1: Half Fine-Tuning
        model_half = copy.deepcopy(baseline_model)
        model_half_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], f"{stratum}_stage1_half.pth")
        
        if not os.path.exists(model_half_path):
            fine_tune_model(
                model_half, 
                stratum_train_loader,
                device, 
                CONFIG["FINETUNE_EPOCHS"],
                f"{stratum}_stage1_half", 
                stage='half'
            )
        else:
            print(f"\nLoading existing Stage 1 model from: {model_half_path}")
            model_half.load_state_dict(torch.load(model_half_path, map_location=device))

        # Stage 2: Full Fine-Tuning
        model_full = copy.deepcopy(baseline_model)
        model_full_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], f"{stratum}_stage2_full.pth")
        
        if not os.path.exists(model_full_path):
            fine_tune_model(
                model_full, 
                stratum_train_loader,
                device, 
                CONFIG["FINETUNE_EPOCHS"],
                f"{stratum}_stage2_full", 
                stage='full'
            )
        else:
            print(f"\nLoading existing Stage 2 model from: {model_full_path}")
            model_full.load_state_dict(torch.load(model_full_path, map_location=device))

        # Evaluation on BRIGHT Test Set
        print(f"\n{'='*70}")
        print(f"EVALUATION: {stratum} BRIGHT Test Set")
        print(f"{'='*70}")
        
        baseline_results = evaluate_model(
            baseline_model, 
            stratum_test_loader, 
            device, 
            label=f"Baseline_on_{stratum}_BRIGHT"
        )
        half_results = evaluate_model(
            model_half, 
            stratum_test_loader, 
            device, 
            label=f"Stage1_Half_{stratum}_on_BRIGHT"
        )
        full_results = evaluate_model(
            model_full, 
            stratum_test_loader, 
            device, 
            label=f"Stage2_Full_{stratum}_on_BRIGHT"
        )
        
        all_results.extend([baseline_results, half_results, full_results])
        
        print(f"\n{stratum} Results:")
        print(f"  Baseline: IoU={baseline_results['IoU']:.4f}, Dice={baseline_results['Dice']:.4f}")
        print(f"  Stage 1:  IoU={half_results['IoU']:.4f}, Dice={half_results['Dice']:.4f}")
        print(f"  Stage 2:  IoU={full_results['IoU']:.4f}, Dice={full_results['Dice']:.4f}")

        # Cross-Dataset Validation on xBD
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION: {stratum} Models on xBD")
        print(f"{'='*70}")
        
        half_xbd_results = evaluate_model(
            model_half, 
            xbd_test_loader, 
            device, 
            label=f"Stage1_Half_{stratum}_on_xBD"
        )
        full_xbd_results = evaluate_model(
            model_full, 
            xbd_test_loader, 
            device, 
            label=f"Stage2_Full_{stratum}_on_xBD"
        )
        
        all_results.extend([half_xbd_results, full_xbd_results])

    # Baseline Evaluation on xBD
    print(f"\n{'='*70}")
    print("BASELINE EVALUATION ON xBD")
    print(f"{'='*70}")
    
    baseline_on_xbd_results = evaluate_model(
        baseline_model, 
        xbd_test_loader, 
        device, 
        label="Baseline_on_xBD"
    )
    all_results.append(baseline_on_xbd_results)

    # Final Results Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    results_str = ""
    results_str += f"{'Label':<35} | {'IoU':>8} | {'Dice':>8} | {'Precision':>9} | {'Recall':>8}\n"
    results_str += "-"*86 + "\n"
    
    for res in all_results:
        line = (
            f"{res['label']:<35} | "
            f"{res['IoU']:>8.4f} | "
            f"{res['Dice']:>8.4f} | "
            f"{res['Precision']:>9.4f} | "
            f"{res['Recall']:>8.4f}"
        )
        print(line)
        results_str += line + "\n"
    
    # Save Results
    results_file = os.path.join(CONFIG["RESULTS_SAVE_DIR"], "final_results.txt")
    with open(results_file, "w") as f:
        f.write("="*86 + "\n")
        f.write("PERFORMANCE BIAS REPRODUCTION RESULTS\n")
        f.write("="*86 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Baseline Epochs: {CONFIG['BASELINE_EPOCHS']}\n")
        f.write(f"  Fine-tuning Epochs: {CONFIG['FINETUNE_EPOCHS']}\n")
        f.write(f"  Baseline LR: {CONFIG['BASELINE_LEARNING_RATE']}\n")
        f.write(f"  Fine-tuning LR: {CONFIG['FINETUNE_LEARNING_RATE']}\n")
        f.write(f"  Batch Size: {CONFIG['BATCH_SIZE']}\n")
        f.write(f"  Random Seed: {CONFIG['RANDOM_SEED']}\n\n")
        f.write(results_str)
        
    print(f"\nResults saved to: {results_file}")
    
    # Diminishing Returns Analysis
    print("\n" + "="*70)
    print("DIMINISHING RETURNS ANALYSIS")
    print("="*70)
    
    for stratum in ["LIC", "MIC", "HIC"]:
        baseline_res = next(r for r in all_results if r['label'] == f"Baseline_on_{stratum}_BRIGHT")
        half_res = next(r for r in all_results if r['label'] == f"Stage1_Half_{stratum}_on_BRIGHT")
        full_res = next(r for r in all_results if r['label'] == f"Stage2_Full_{stratum}_on_BRIGHT")
        
        baseline_iou = baseline_res['IoU']
        half_iou = half_res['IoU']
        full_iou = full_res['IoU']
        
        stage1_gain = half_iou - baseline_iou
        stage2_gain = full_iou - half_iou
        total_gain = full_iou - baseline_iou
        
        if total_gain > 0:
            stage1_pct = (stage1_gain / total_gain) * 100
            stage2_pct = (stage2_gain / total_gain) * 100
        else:
            stage1_pct = stage2_pct = 0
        
        print(f"\n{stratum}:")
        print(f"  Baseline IoU: {baseline_iou:.4f}")
        print(f"  Stage 1 IoU:  {half_iou:.4f} (+{stage1_gain:.4f})")
        print(f"  Stage 2 IoU:  {full_iou:.4f} (+{stage2_gain:.4f})")
        print(f"  Total gain:   {total_gain:.4f}")
        print(f"  Stage 1 contributes: {stage1_pct:.1f}% of total gain")
        print(f"  Stage 2 contributes: {stage2_pct:.1f}% of total gain")
    
    # Cross-Dataset Generalization Analysis
    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION ANALYSIS")
    print("="*70)
    
    baseline_xbd = baseline_on_xbd_results['IoU']
    
    for stratum in ["LIC", "MIC", "HIC"]:
        bright_half = next(r for r in all_results if r['label'] == f"Stage1_Half_{stratum}_on_BRIGHT")['IoU']
        bright_full = next(r for r in all_results if r['label'] == f"Stage2_Full_{stratum}_on_BRIGHT")['IoU']
        xbd_half = next(r for r in all_results if r['label'] == f"Stage1_Half_{stratum}_on_xBD")['IoU']
        xbd_full = next(r for r in all_results if r['label'] == f"Stage2_Full_{stratum}_on_xBD")['IoU']
        
        degradation_half = ((bright_half - xbd_half) / bright_half) * 100 if bright_half > 0 else 0
        degradation_full = ((bright_full - xbd_full) / bright_full) * 100 if bright_full > 0 else 0
        
        print(f"\n{stratum}:")
        print(f"  Stage 1 - BRIGHT: {bright_half:.4f} | xBD: {xbd_half:.4f} | Degradation: {degradation_half:.1f}%")
        print(f"  Stage 2 - BRIGHT: {bright_full:.4f} | xBD: {xbd_full:.4f} | Degradation: {degradation_full:.1f}%")
    
    print(f"\nBaseline (xBD→xBD): {baseline_xbd:.4f}")
    

    print(f"\nModels saved in: {CONFIG['MODEL_SAVE_DIR']}")
    print(f"Results saved in: {CONFIG['RESULTS_SAVE_DIR']}")
