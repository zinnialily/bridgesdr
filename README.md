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
- [Model Architectures](#model-architectures)
- [Citations](#citations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Datasets

This research utilizes three complementary disaster imagery datasets:

### 1. xBD Dataset (xView2)

**Description:** High-resolution optical imagery dataset for building damage assessment from satellite imagery.

**Usage in this project:**
- **Synthetic Imagery Quality Assessment (Study 1):** Training DisasterGAN with paired pre/post-disaster imagery

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
- **Synthetic Imagery Quality Assessment (Study 1):** Test set for evaluating DisasterGAN output realism
- **Training Effectiveness Assessment (Study 2):** Fine-tuning and testing data stratified by economic context

**Image Modalities:**
- Pre-disaster optical imagery
- Post-disaster SAR (Synthetic Aperture Radar) imagery
- Pixelated damage masks

**Economic Stratification:**
- **LIC (Low-Income Countries):** Haiti, Congo
- **MIC (Middle-Income Countries):** Turkey, Morocco, Libya, Bata (Equatorial Guinea), Beirut
- **HIC (High-Income Countries):** Noto (Italy), La Palma (Spain), Hawaii (USA), Marshall Islands

**DOI:** https://zenodo.org/records/15385983

**Citation:**
```
Chen, H., Song, J., Dietrich, O., Broni-Bediako, C., Xuan, W., Wang, J., ... & Yokoya, N. (2025).
BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution
for all-weather disaster response.
```

### 3. Sentinel-1/2 Image Pairs Dataset

**Description:** Paired SAR (Sentinel-1) and optical (Sentinel-2) satellite imagery across diverse terrain types for cross-modal validation.

**Usage in this project:**
- **Appendix Testing:** Validation of the SAR-to-optical approximation methodology

**Terrain Types:** Urban, Agricultural, Grassland, Barren land

**Kaggle Download:** https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/data

**Original Source:** https://mediatum.ub.tum.de/1436631

**Citation:**
```
Tiwari, R. K., Gupta, R. P., & Arora, M. K. (2022).
Sentinel-1 and Sentinel-2 Data for Land Use/Land Cover Mapping.
Technical University of Munich, Mediatum.
```

---

## Project Structure

```
bridgesdr/
├── data/
│   ├── xbd_raw/
│   ├── bright_raw/
│   ├── xbd/
│   │   ├── train/images/, train/labels/
│   │   └── test/images/,  test/labels/
│   └── bright/
│       ├── lic/manifest_train.json, lic/manifest_val.json, lic/manifest.json
│       ├── mic/manifest_train.json, mic/manifest_val.json, mic/manifest.json
│       └── hic/manifest_train.json, hic/manifest_val.json, hic/manifest.json
│
├── reproduction_scripts/
│   ├── 00_setup_environment.sh       (optional)
│   ├── 01_download_datasets.sh
│   └── 02_preprocess_data.py
│
├── synthetic_imagery_quality_assessment/    ← Study 1: DisasterGAN (optical domain)
│   ├── 03_train_disastergan.py
│   ├── 04_generate_synthetic_images.py
│   ├── 04b_compute_fid_is.py              ← FID & Inception Score
│   ├── 06_generate_damage_masks.py
│   ├── 07_evaluate_quality_metrics.py
│   └── 15a_visualize_synthesis.py         ← Qualitative synthesis grids
│
├── training_effectiveness_assessment/      ← Study 2: BRIGHT segmentation
│   ├── _08_shared.py                      ← Shared dataset, training loop, metrics
│   ├── _model_registry.py                 ← U-Net, SegFormer-B2, ChangeFormer
│   ├── 08_train_baseline_unet.py
│   ├── 08b_train_segformer.py
│   ├── 08c_train_changeformer.py
│   ├── 09_finetune_half_stage.py          ← Stage 1: freeze encoder
│   ├── 10_finetune_full_stage.py          ← Stage 2: all layers
│   ├── 11_evaluation.py                   ← Full evaluation (all models × stages)
│   ├── 13_sample_size_correlation.py      ← Confound analysis: sample size
│   ├── 14_disaster_type_analysis.py       ← Confound analysis: disaster type
│   └── 15b_qualitative_results.py        ← Qualitative segmentation grids
│
├── appendix_testing/
│   └── 12_sar_to_optical_validation.py
│
├── checkpoints/
│   ├── disastergan/
│   │   ├── disastergan_generator_final.pth
│   │   └── disastergan_discriminator_final.pth
│   └── unet_study2/                       ← Pre-trained U-Net checkpoints
│
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
bash reproduction_scripts/01_download_datasets.sh
```

**Note:** xBD download requires Kaggle API credentials. BRIGHT downloads from Zenodo (~15GB).

**Output:**
- `data/xbd_raw/` — Raw xBD dataset
- `data/bright_raw/` — Raw BRIGHT dataset with pre-event/, post-event/, target/ folders

### Step 2: Preprocess Data

```bash
# Process both datasets
python reproduction_scripts/02_preprocess_data.py

# Or selectively
python reproduction_scripts/02_preprocess_data.py --xbd-only
python reproduction_scripts/02_preprocess_data.py --bright-only

# Verify structure
python reproduction_scripts/02_preprocess_data.py --verify
```

**What this does:**
- **BRIGHT:** Scans 11 unrestricted events, builds train/val manifests per income stratum (split by event, not tile, to prevent data leakage)
- **xBD:** Organises pre/post pairs into train/test splits for DisasterGAN (Study 1 only — never mixed into BRIGHT)

### Step 3: Synthetic Imagery Quality Assessment (Study 1)

> **Pre-trained DisasterGAN checkpoints are already provided** in `checkpoints/disastergan/`. Training is optional.

**Using pre-trained checkpoints (recommended):**
```bash
cd synthetic_imagery_quality_assessment

# Generate synthetic post-disaster images
python 04_generate_synthetic_images.py

# Compute FID and Inception Score
python 04b_compute_fid_is.py

# Generate damage masks from synthetic images
python 06_generate_damage_masks.py

# Evaluate mask quality (IoU, Dice)
python 07_evaluate_quality_metrics.py

# Qualitative visualisation grids
python 15a_visualize_synthesis.py
```

**Training DisasterGAN from scratch (optional):**
```bash
python 03_train_disastergan.py  # then run steps above
```

### Step 4: Training Effectiveness Assessment (Study 2)

Study 2 uses **BRIGHT only** — no xBD data is mixed in. Three models are evaluated: U-Net, SegFormer-B2, and ChangeFormer.

#### Option A: Use existing checkpoints (skip to evaluation)

Pre-trained U-Net checkpoints are available in `checkpoints/unet_study2/`. Run evaluation directly:

```bash
cd training_effectiveness_assessment
python 11_evaluation.py
```

#### Option B: Train all models from scratch

```bash
cd training_effectiveness_assessment

# Baseline training (all 3 models × 3 strata)
python 08_train_baseline_unet.py
python 08b_train_segformer.py
python 08c_train_changeformer.py

# Stage 1: freeze encoder, fine-tune decoder
python 09_finetune_half_stage.py

# Stage 2: full fine-tuning, all layers
python 10_finetune_full_stage.py

# Full evaluation (all models × stages × strata)
python 11_evaluation.py
```

**Checkpoint naming convention:**
- `checkpoints/<model>_baseline_<income>.pth`
- `checkpoints/<model>_stage1_<income>.pth`
- `checkpoints/<model>_stage2_<income>.pth`

where `<model>` ∈ {unet, segformer, changeformer} and `<income>` ∈ {lic, mic, hic}.

#### Confound Analyses (requires 11_evaluation.py output)

```bash
# Sample size confound (Spearman ρ + OLS)
python 13_sample_size_correlation.py

# Disaster type confound (Kruskal-Wallis + two-factor OLS)
python 14_disaster_type_analysis.py
```

#### Qualitative Visualisation

```bash
# Segmentation grids: pre-optical | post-SAR | GT mask | prediction
python 15b_qualitative_results.py
python 15b_qualitative_results.py --n_examples 8 --seed 123
```

---

## Methodology

### Two-Study Design

This research uses a clean two-study framework to avoid cross-modality contamination:

| Study | Domain | Data | Purpose |
|-------|--------|------|---------|
| **Study 1** | Optical → Optical | xBD only | Evaluate DisasterGAN synthesis quality (FID, IS, mask IoU) |
| **Study 2** | Optical + SAR → Mask | BRIGHT only | Evaluate model training effectiveness across income strata |

### Economic Stratification

BRIGHT dataset events are classified by World Bank GNI per capita:

| Income Level | Events | Classification Criteria |
|-------------|--------|------------------------|
| **LIC** | Haiti-earthquake, Congo-volcano | GNI per capita < $1,135 |
| **MIC** | Turkey-earthquake, Morocco-earthquake, Libya-flood, Bata-explosion, Beirut-explosion | $1,136 – $13,845 |
| **HIC** | Hawaii-wildfire, La Palma-volcano, Noto-earthquake, Marshall-wildfire | > $13,846 |

### Progressive Fine-Tuning Protocol

**Stage 0 (Baseline):** Train on all BRIGHT strata combined — no stratum-specific adaptation

**Stage 1 (Half Fine-Tuning):**
- Freeze encoder layers
- Fine-tune decoder only
- 20% cross-stratum augmentation from other income levels

**Stage 2 (Full Fine-Tuning):**
- Unfreeze all layers
- Full network fine-tuning with cosine LR decay
- 20% cross-stratum augmentation continues

---

## Model Architectures

### U-Net for Damage Segmentation

**Input:** 4-channel tensor (RGB pre-disaster + SAR post-disaster), 512×512

**Output:** 4-class segmentation mask (none / minor / major / destroyed)

**Architecture:**
- Encoder: 4 downsampling blocks (64, 128, 256, 512 channels)
- Bottleneck: 1024 channels
- Decoder: 4 upsampling blocks with skip connections

**Training Hyperparameters:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropyLoss with class weights [0.1, 1.0, 1.5, 2.0]
- Epochs: 20 (baseline), 10 (fine-tune stages)
- LR scheduler: CosineAnnealingLR

### SegFormer-B2

**Input:** 4-channel (RGB + SAR), patch embedding extended from 3→4 channels via weight warm-start

**Backbone:** nvidia/mit-b2 (HuggingFace transformers)

**Training:** AdamW lr=6e-5 (lower than U-Net — pre-trained transformer)

### ChangeFormer (Siamese Transformer)

**Input:** Two separate 3-channel inputs — pre-event RGB and post-event SAR (replicated to 3 channels)

**Architecture:** Siamese hierarchical encoder + difference-feature MLP decoder

**Note:** SAR replication to 3 channels is a documented limitation; a proper bi-modal encoder is left for future work.

### DisasterGAN

**Generator:** Pre-disaster optical → synthetic post-disaster optical

**Discriminator:** Real vs. synthetic post-disaster classification

**Training:**
- Optimizer: Adam (lr=2e-4, betas=(0.5, 0.999))
- Losses: Adversarial + Classification + Mask + Gradient Penalty + Cycle Consistency
- Disaster types: 7 classes (volcano, fire, tornado, tsunami, flooding, earthquake, hurricane)

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

### Code Attribution

This implementation is inspired by the DisasterGAN Kaggle kernel:
- **Adhoppin.** "DisasterGAN — Generating Post-Disaster Images"
- URL: https://www.kaggle.com/code/adhoppin/disastergan-generating-post-disaster-images

See `ATTRIBUTION.md` for complete attribution details.

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

**Dataset Licenses:**
- **xBD** — [Terms of Use](https://xview2.org/dataset)
- **BRIGHT** — See Zenodo record for license information
- **Sentinel-1 & 2 Image Pairs** — [Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

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
ls ~/.kaggle/kaggle.json   # should exist with 600 permissions
```

**2. CUDA out of memory:**
- Reduce batch size in `_08_shared.py` (`BATCH_GPU`)
- Use gradient accumulation
- Try mixed precision training

**3. Dataset file not found:**
- Run `python reproduction_scripts/02_preprocess_data.py --verify`
- Check that downloads completed successfully

**4. Model checkpoint not found:**
- Verify checkpoint naming: `<model>_<stage>_<income>.pth`
- Check that the preceding training stage completed
- Checkpoints are saved in `checkpoints/`

---

## Future Work

- Expand to additional disaster types and regions
- Incorporate temporal dynamics in synthetic generation
- Develop real-time deployment pipelines for field use
- Integration with humanitarian response platforms
