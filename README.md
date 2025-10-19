**Link to the full paper:** [https://dx.doi.org/10.2139/ssrn.5385441](https://dx.doi.org/10.2139/ssrn.5385441)
# Bridging the Post-Disaster Imagery Gap: Leveraging Synthetic Data for Disaster Response across Economic Spectra

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/zinnialily/bridgesdr)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202024-red)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This repository contains the implementation code for evaluating synthetic disaster imagery effectiveness across low-, middle-, and high-income countries (LIC, MIC, HIC). The research addresses a critical gap in disaster response AI: **how can synthetic imagery improve damage assessment models across different economic contexts?**

**Key Findings:**
- **LIC contexts**: Full synthetic imagery fine-tuning improves performance
- **MIC contexts**: Half synthetic imagery fine-tuning achieves 297% improvement
- **HIC contexts**: Models maintain generalization without synthetic augmentation
- **Cross-context performance**: 79% degradation when models trained in one economic context are applied to another

This work provides the first comprehensive evaluation of synthetic disaster imagery effectiveness stratified by economic development level, offering both technical insights and practical guidance for equitable AI deployment in disaster response.

---

## Table of Contents

- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Reproduction Scripts](#reproduction-scripts)
- [Requirements](#requirements)
- [Citations](#citations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Datasets

This research utilizes two complementary disaster imagery datasets:

### 1. xBD Dataset (xView2)

**Description:** High-resolution optical imagery dataset for building damage assessment from satellite imagery.

**Usage in this project:**
- **Synthetic Imagery Quality Assessment:** Training DisasterGAN with paired pre/post-disaster imagery
- **Training Effectiveness Assessment:** Baseline model training

**Image Modalities:** 
- Pre-disaster RGB optical imagery
- Post-disaster RGB optical imagery  
- Multi-class building-level damage masks (4 classes: none, minor, major, destroyed)

**DOI/URL:** https://xview2.org/dataset

**Kaggle Download:** https://www.kaggle.com/datasets/qianlanzz/xbd-dataset

**Citation:**
```
Gupta, V., Dhu, R., Campbell, R., Truong, A., Xu, T., Erickson, D., ... & Hamann, B. (2019). 
xBD: A dataset for assessing building damage from satellite imagery. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 10-17).
```

**Preprocessing:**
- Images partitioned into 4×4 sub-images (64×64 pixels each) following DisasterGAN methodology
- Enables computational efficiency while preserving spatial context
- Strategic subsampling: All damage patches retained, 10% of undamaged patches sampled

### 2. BRIGHT Dataset

**Description:** Globally distributed multimodal building damage assessment dataset with very-high-resolution imagery for all-weather disaster response.

**Usage in this project:**
- **Synthetic Imagery Quality Assessment:** Test set for evaluating DisasterGAN output realism
- **Training Effectiveness Assessment:** Fine-tuning and testing data stratified by economic context

**Image Modalities:**
- Pre-disaster optical imagery
- Post-disaster SAR (Synthetic Aperture Radar) imagery
- Pixelated damage masks

**Economic Stratification:**
- **LIC (Low-Income Countries):** Haiti, Congo
- **MIC (Middle-Income Countries):** Turkey, Morocco, Libya  
- **HIC (High-Income Countries):** Noto (Italy), La Palma (Spain), Hawaii (USA)

**DOI:** https://zenodo.org/records/15385983

**Citation:**
```
Chen, H., Song, J., Dietrich, O., Broni-Bediako, C., Xuan, W., Wang, J., ... & Yokoya, N. (2025). 
BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution 
for all-weather disaster response.
```

---

## Project Structure

```
bridgesdr/
├── data/                                   
│   ├── xbd_raw/                          
│   ├── bright_raw/                      
│   ├── xbd/                              
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── test/images/
│   │   ├── test/labels/
│   │   ├── tier1/, tier3/, hold/         
│   └── bright/                      
│       ├── lic/images/, lic/masks/
│       ├── mic/images/, mic/masks/
│       └── hic/images/, hic/masks/
│
├── reproduction_scripts/
│ ├── 01_download_datasets.sh # Downloads xBD and BRIGHT
│ ├── 02_preprocess_data.py # Organizes data into standard format
│ │
│ ├── synthetic_imagery_quality_assessment/ # Synthetic generation & validation
│ │ ├── 03_train_disastergan.py
│ │ ├── 04_generate_synthetic_images.py
│ │ ├── 05_convert_optical_to_sar.py
│ │ ├── 06_generate_damage_masks.py
│ │ └── 07_evaluate_quality_metrics.py
│ │
│ └── training_effectiveness_assessment/ # Fine-tuning and evaluation pipeline
│ ├── 08_train_baseline_unet.py
│ ├── 09_finetune_half_stage.py
│ ├── 10_finetune_full_stage.py
│ └── 11_evaluation.py
│
├── checkpoints/
├── ATTRIBUTION.md
├── requirements.txt
└── README.md # This file
```

---

## Installation

### Prerequisites

- **Python:** 3.8 - 3.10
- **CUDA:** 11.3+ (for GPU support)
- **RAM:** Minimum 16GB
- **GPU:** NVIDIA GPU with 8GB+ VRAM recommended (RTX 3090, Tesla V100, or equivalent)
- **OS:** Linux (Ubuntu 20.04+), Windows 10+, or macOS

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/zinnialily/bridgesdr.git
cd bridgesdr
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify GPU availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

5. **Set up Kaggle credentials (for xBD download):**
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage Instructions

### Step 1: Download Datasets

```bash
# Download both datasets
bash reproduction_scripts/01_download_datasets.sh

# Or download individually
bash reproduction_scripts/01_download_datasets.sh --xbd-only
bash reproduction_scripts/01_download_datasets.sh --bright-only
```

**Note:** xBD download requires Kaggle API credentials. BRIGHT downloads from Zenodo (~15GB).

**Output:**
- `data/xbd_raw/` - Raw xBD dataset
- `data/bright_raw/` - Raw BRIGHT dataset with pre-event/, post-event/, target/ folders

### Step 2: Preprocess Data

```bash
# Process both datasets
python reproduction_scripts/02_preprocess_data.py

# Process specific datasets
python reproduction_scripts/02_preprocess_data.py --xbd-only
python reproduction_scripts/02_preprocess_data.py --bright-only

# Verify structure
python reproduction_scripts/02_preprocess_data.py --verify
```

**What this does:**
- Organizes xBD by splits (train/test/tier1/tier3/hold) with pre/post disaster pairs
- Stratifies BRIGHT by economic level (LIC/MIC/HIC)
- Matches pre-disaster, post-disaster, and damage mask triplets
- Validates naming conventions

**Output:**
- `data/xbd/{split}/images/*_pre_disaster.png`
- `data/xbd/{split}/images/*_post_disaster.png`
- `data/xbd/{split}/labels/*.json`
- `data/bright/{lic,mic,hic}/images/*_pre_disaster.*`
- `data/bright/{lic,mic,hic}/images/*_post_disaster.tif`
- `data/bright/{lic,mic,hic}/masks/*_damage_mask.png`

### Step 3: Synthetic Imagery Quality Assessment

**Train DisasterGAN:**
```bash
cd reproduction_scripts/synthetic_imagery_quality_assessment
python reproduce_disastergan_xbd.py
```

**Configuration in script:**
- Epochs: 7 (increase for better results)
- Batch size: 16
- Learning rate: 1e-4
- Image size: 256×256
- Disaster types: volcano, fire, tornado, tsunami, flooding, earthquake, hurricane

**Output:**
- `saved_models/G_epoch_*.pth` - Generator checkpoints
- `saved_models/D_epoch_*.pth` - Discriminator checkpoints
- `samples/` - Generated sample images
- `plots/` - Training loss curves

**Evaluate Synthetic Quality:**
```bash
python evaluation.py \
    --model ./saved_models/G_final.pth \
    --pre-dir ../../data/bright/lic/images \
    --post-dir ../../data/bright/lic/images \
    --out ./results/lic_multiclass \
    --strata LIC \
    --mode multiclass
```

**Evaluation Modes:**
- `binary`: Simple damage/no-damage classification
- `multiclass`: 4-class damage assessment (none/minor/major/destroyed)

**Output:**
- JSON files with per-image metrics (IoU, Dice, Precision, Recall)
- Summary statistics per disaster type
- Visualization images comparing real vs synthetic

### Step 4: Training Effectiveness Assessment

**Train Baseline U-Net (Stage 0):**
```bash
cd reproduction_scripts/training_effectiveness_assessment
python 01_train_baseline_unet.py
```

**Configuration:**
- Training on xBD only
- Epochs: 7 (configurable)
- Batch size: 16
- Learning rate: 1e-3
- Class weights: [0.1, 1.0, 1.0, 1.0] (downweight no-damage class)

**Output:**
- `checkpoints/study2/baseline_unet_best.pth`
- `checkpoints/study2/baseline_unet_final.pth`

**Stage 1: Half Fine-Tuning (Decoder Only):**
```bash
python 02_finetune_half_stage.py
```

**What this does:**
- Loads baseline checkpoint
- Freezes encoder layers
- Fine-tunes decoder only on BRIGHT data (stratified by LIC/MIC/HIC)
- Reduced epochs: 3
- Reduced learning rate: 1e-4

**Output:**
- `checkpoints/study2/lic_half_finetuned_unet.pth`
- `checkpoints/study2/mic_half_finetuned_unet.pth`
- `checkpoints/study2/hic_half_finetuned_unet.pth`

**Stage 2: Full Fine-Tuning:**
```bash
python 03_finetune_full_stage.py
```

**What this does:**
- Loads half-finetuned checkpoint
- Unfreezes all layers
- Full network fine-tuning on BRIGHT data
- Epochs: 3 (configurable)

**Output:**
- `checkpoints/study2/lic_full_finetuned_unet.pth`
- `checkpoints/study2/mic_full_finetuned_unet.pth`
- `checkpoints/study2/hic_full_finetuned_unet.pth`

---

## Methodology

### Experimental Design

This research employs a two-part experimental framework:

1. **Synthetic Imagery Quality Assessment:** Evaluates how well synthetic post-disaster images capture damage characteristics through damage mask comparison (IoU, Dice coefficient)

2. **Training Effectiveness Assessment:** Determines whether synthetic imagery improves operational disaster response model performance through progressive fine-tuning experiments

### Economic Stratification

BRIGHT dataset events are classified by World Bank GNI per capita:

| Income Level | Countries | Classification Criteria |
|-------------|-----------|------------------------|
| **LIC** | Haiti, Congo | GNI per capita < $1,135 |
| **MIC** | Turkey, Morocco, Libya | $1,136 - $13,845 |
| **HIC** | Noto, La Palma, Hawaii | > $13,846 |

### Progressive Fine-Tuning Protocol

**Stage 0 (Baseline):** Train U-Net on xBD only (no BRIGHT domain adaptation)

**Stage 1 (Half Fine-Tuning):** 
- Freeze encoder layers
- Fine-tune decoder only
- Reduced epochs (10) and learning rate (1e-4)
- Minimal resource investment scenario

**Stage 2 (Full Fine-Tuning):**
- Unfreeze all layers
- Full network fine-tuning
- Complete epochs (20) with learning rate decay
- Comprehensive adaptation scenario

---

## Model Architecture

### U-Net for Damage Segmentation

**Input:** 256×256 image with 4 channels (RGB pre-disaster + SAR post-disaster)

**Output:** 256×256 segmentation mask with 4 classes:
- Class 0: No damage
- Class 1: Minor damage
- Class 2: Major damage
- Class 3: Destroyed

**Architecture:**
- **Encoder:** 4 downsampling blocks (64, 128, 256, 512 channels)
- **Bottleneck:** 1024 channels at 16×16 resolution
- **Decoder:** 4 upsampling blocks with skip connections
- **Output:** 1×1 convolution to 4-class segmentation

**Training Hyperparameters:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Loss: CrossEntropyLoss with class weights [0.1, 1.0, 1.0, 1.0]
- Batch size: 16
- LR scheduler: ReduceLROnPlateau (patience=3, factor=0.5)

### DisasterGAN Architecture

**Generator:** Takes pre-disaster optical imagery → generates post-disaster synthetic imagery

**Discriminator:** Distinguishes real vs. synthetic post-disaster images

**Training:**
- Optimizer: Adam (lr=2e-4, betas=(0.5, 0.999))
- Losses: Adversarial + Classification + Mask + Gradient Penalty + Cycle Consistency
- Disaster types: 7 classes (volcano, fire, tornado, tsunami, flooding, earthquake, hurricane)

---

## Reproduction Scripts

### Dataset Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_download_datasets.sh` | Download xBD and BRIGHT datasets | 1-2 hours |
| `02_preprocess_data.py` | Organize and stratify datasets | 30 minutes |

### Synthetic Quality Assessment

| Script | Purpose | Runtime |
|--------|---------|---------|
| `reproduce_disastergan_xbd.py` | Train DisasterGAN | 1-3 days (GPU) |
| `evaluation.py` | Evaluate synthetic image quality | 1-2 hours |

### Training Effectiveness

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_train_baseline_unet.py` | Train baseline U-Net | 1-2 days (GPU) |
| `02_finetune_half_stage.py` | Stage 1 fine-tuning | 12-24 hours |
| `03_finetune_full_stage.py` | Stage 2 fine-tuning | 6-12 hours |

---

## Requirements

See `requirements.txt` for complete list. Key dependencies:

- **PyTorch** >= 1.12.0
- **torchvision** >= 0.13.0
- **numpy** >= 1.21.0
- **Pillow** >= 8.0.0
- **matplotlib** >= 3.4.0
- **scikit-learn** >= 1.0.0
- **scipy** >= 1.7.0
- **tqdm** >= 4.60.0

---

## Citations

### This Work

```bibtex
@misc{singh2025bridging,
  title = {Bridging the Post-Disaster Imagery Gap: Leveraging Synthetic Data for Disaster Response across Economic Spectra},
  author = {Singh, Aanya},
  year = {2025},
  month = {August 09},
  note = {Available at SSRN: https://ssrn.com/abstract=5385441 or http://dx.doi.org/10.2139/ssrn.5385441}
}
```

### Datasets

**xBD:**
```bibtex
@inproceedings{gupta2019xbd,
  title={xBD: A dataset for assessing building damage from satellite imagery},
  author={Gupta, Vishal and others},
  booktitle={CVPR Workshops},
  year={2019}
}
```

**BRIGHT:**
```bibtex
@article{chen2025bright,
  title={BRIGHT: A globally distributed multimodal building damage assessment dataset},
  author={Chen, Hongruixuan and others},
  year={2025}
}
```

### Code Attribution

This implementation is inspired by the DisasterGAN Kaggle kernel:
- **Adhoppin.** "DisasterGAN — Generating Post-Disaster Images"
- URL: https://www.kaggle.com/code/adhoppin/disastergan-generating-post-disaster-images

See `ATTRIBUTION.md` for complete attribution details.

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

**Dataset Licenses:**
- xBD: See https://xview2.org/dataset for terms of use
- BRIGHT: See Zenodo record for license information

---

## Acknowledgments

- **xView2 Challenge** for providing the xBD dataset
- **BRIGHT dataset team** for multimodal disaster imagery
- **DisasterGAN authors** for methodology inspiration
- **Humanitarian OpenStreetMap Team** for practical disaster response insights

---
## Troubleshooting

### Common Issues

**1. Kaggle API credentials error:**
```bash
# Ensure kaggle.json is in correct location
ls ~/.kaggle/kaggle.json
# Should exist with 600 permissions
```

**2. CUDA out of memory:**
- Reduce batch size in training scripts
- Use gradient accumulation
- Try mixed precision training

**3. Dataset file not found:**
- Run preprocessing script with `--verify` flag
- Check that downloads completed successfully
- Verify directory structure matches expected format

**4. Model checkpoint loading error:**
- Ensure you're using the correct checkpoint for each stage
- Check if training completed successfully
- Verify checkpoint file exists and isn't corrupted

---

## Future Work

- Expand to additional disaster types and regions
- Incorporate temporal dynamics in synthetic generation
- Develop real-time deployment pipelines for field use
- Integration with humanitarian response platforms
