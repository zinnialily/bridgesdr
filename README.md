# Bridging the Post-Disaster Imagery Gap: Leveraging Synthetic Data for Disaster Response across Economic Spectra

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/zinnialily/bridgesdr)
[![Paper](https://img.shields.io/badge/Paper-SSRN%202025-red)](https://dx.doi.org/10.2139/ssrn.5385441)
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

This research utilizes three complementary disaster imagery datasets:

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

### 3. Sentinel-1/2 Image Pairs Dataset

**Description:** Paired SAR (Sentinel-1) and optical (Sentinel-2) satellite imagery across diverse terrain types for cross-modal analysis.

**Usage in this project:**
- **Appendix Testing:** Validation of the simplified SAR-to-optical approximation methodology
- Evaluates conversion quality across four terrain types: urban, agricultural, grassland, and barren land

**Image Modalities:**
- Sentinel-1 SAR imagery (S1 - C-band synthetic aperture radar)
- Sentinel-2 optical imagery (S2 - multispectral optical)
- Organized by terrain type for systematic evaluation

**Terrain Types:**
- Urban
- Agricultural
- Grassland
- Barren land

**Kaggle Download:** https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/data

**Original Source:** https://mediatum.ub.tum.de/1436631

**Citation:**
```
Tiwari, R. K., Gupta, R. P., & Arora, M. K. (2022). 
Sentinel-1 and Sentinel-2 Data for Land Use/Land Cover Mapping. 
Technical University of Munich, Mediatum.
```


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
│   ├── 00_* (optional setup)
│   ├── 01_download_datasets.sh
│   └── 02_preprocess_data.py
│
├── synthetic_imagery_quality_assessment/
│   ├── 03_train_disastergan.py
│   ├── 04_generate_synthetic_images.py
│   ├── 05_convert_optical_to_sar.py
│   ├── 06_generate_damage_masks.py
│   └── 07_evaluate_quality_metrics.py
│
├── training_effectiveness_assessment/
│   ├── 08_train_baseline_unet.py
│   ├── 09_finetune_half_stage.py
│   ├── 10_finetune_full_stage.py
│   └── 11_evaluation.py
│
├── checkpoints/
├── ATTRIBUTION.md
├── requirements.txt
└── README.md

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

#### Important: Pre-Trained Checkpoints Available

**DisasterGAN checkpoints are already provided in the repository:**
- `checkpoints/disastergan/disastergan_generator_final.pth`
- `checkpoints/disastergan/disastergan_discriminator_final.pth`

**Training is OPTIONAL.** You can skip directly to image generation if you want to use the pre-trained models.

If using pre-trained checkpoints (reccomended):
```bash
cd synthetic_imagery_quality_assessment

# Generate synthetic images
python 04_generate_synthetic_images.py

# Convert to SAR
python 05_convert_optical_to_sar.py

# Generate damage masks
python 06_generate_damage_masks.py

# Evaluate quality
python 07_evaluate_quality_metrics.py
```
If training from scratch:
- Add python ```bash 03_train_disastergan.py``` at the beginning
- Follow the same steps above for generation and evaluation

---
### Step 4: Training Effectiveness Assessment
#### Option A: Skip Training (Use Existing Checkpoints)
There are already trained model checkpoints in checkpoints/unet_study2/, so you can skip directly to evaluation.
Evaluate Each Checkpoint:
Run 11_evaluation.py separately for each checkpoint by modifying the CHECKPOINT variable in the script:
bash
```cd reproduction_scripts/training_effectiveness_assessment
# Evaluate baseline model
# Edit line 14 in 11_evaluation.py: CHECKPOINT = "checkpoints/unet_study2/baseline_unet.pth"
python 11_evaluation.py

# Evaluate LIC half-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/lic_half_finetuned_unet.pth"
python 11_evaluation.py

# Evaluate LIC full-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/lic_full_finetuned_unet.pth"
python 11_evaluation.py

# Evaluate MIC half-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/mic_half_finetuned_unet.pth"
python 11_evaluation.py

# Evaluate MIC full-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/mic_full_finetuned_unet.pth"
python 11_evaluation.py

# Evaluate HIC half-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/hic_half_finetuned_unet.pth"
python 11_evaluation.py

# Evaluate HIC full-finetuned
# Edit line 14: CHECKPOINT = "checkpoints/unet_study2/hic_full_finetuned_unet.pth"
python 11_evaluation.py
```
#### Option B: Train Models from Scratch (Optional)
If you need to train the models yourself, follow these steps. Note: The training scripts currently output to checkpoints/study2/, so you'll need to either:
Modify CHECKPOINT_DIR in scripts 08, 09, and 10 to "checkpoints/unet_study2/", OR
Move the generated checkpoints after training

bash
```cd reproduction_scripts/training_effectiveness_assessment
python 08_train_baseline_unet.py
python 09_finetune_half_stage.py
python 10_finetune_full_stage.py
```
After training (or if using existing checkpoints), evaluate each model using:
bash ```python 11_evaluation.py```
Remember to modify the CHECKPOINT variable in the script for each model you want to evaluate.

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
