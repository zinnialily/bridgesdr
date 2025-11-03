# Attribution

This project builds upon the work of several researchers and datasets. We gratefully acknowledge the following contributions:

## Code Attribution

### DisasterGAN Implementation
This implementation is inspired by and adapted from the DisasterGAN Kaggle kernel:

* **Author:** Adhoppin
* **Title:** "DisasterGAN — Generating Post-Disaster Images"
* **URL:** https://www.kaggle.com/code/adhoppin/disastergan-generating-post-disaster-images
* **Usage:** Core GAN architecture, training methodology, and image generation pipeline for synthetic post-disaster imagery

Our implementation extends the original work with:
- Economic stratification across LIC/MIC/HIC contexts
- Progressive fine-tuning methodology
- Cross-dataset evaluation framework
- Multi-class damage assessment integration

### U-Net Architecture
* **Original Paper:** Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
* **Conference:** International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)
* **Usage:** Segmentation model for damage assessment

### VGG16 Perceptual Loss
* **Original Paper:** Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
* **Usage:** Perceptual loss computation in GAN training via pre-trained VGG16 features

## Dataset Attribution

### xBD Dataset (xView2)
```bibtex
@inproceedings{gupta2019xbd,
  title={xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author={Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2019}
}
```

* **Description:** High-resolution satellite imagery for building damage assessment
* **URL:** https://xview2.org/dataset
* **License:** See https://xview2.org/dataset for terms of use
* **Usage:** Training DisasterGAN, baseline model training, and evaluation

### BRIGHT Dataset
```bibtex
@article{chen2025bright,
  title={BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset with Very-High-Resolution for All-Weather Disaster Response},
  author={Chen, Hongruixuan and Song, Jian and Dietrich, Olaf and Broni-Bediako, Clifford and Xuan, Weiming and Wang, Junshi and Yokoya, Naoto},
  journal={Scientific Data},
  year={2025},
  publisher={Nature Publishing Group}
}
```

* **Description:** Multimodal building damage assessment dataset with VHR imagery
* **DOI:** https://zenodo.org/records/15385983
* **License:** See Zenodo record for license information
* **Usage:** Economic stratification (LIC/MIC/HIC), synthetic imagery evaluation, cross-context fine-tuning

## Methodological Inspirations

### SAR-to-Optical Conversion
* **Reference:** Shakthi Priya et al. (2024). Contrast enhancement techniques for SAR imagery
* **Usage:** Simplified SAR-to-optical approximation for damage mask generation

### Damage Assessment Protocols
* **Reference:** Kim et al. (2022), Singh et al. (2025). Building damage classification frameworks
* **Usage:** Multi-class damage classification scheme (no damage, minor, major, destroyed)

## Library Dependencies

This project uses the following open-source libraries:
- **PyTorch:** Deep learning framework (BSD-style license)
- **torchvision:** Computer vision utilities (BSD license)
- **NumPy:** Numerical computing (BSD license)
- **scikit-learn:** Machine learning tools (BSD license)
- **Matplotlib:** Visualization library (PSF license)
- **Pillow:** Image processing (HPND license)
- **scipy:** Scientific computing (BSD license)
- **tqdm:** Progress bars (MPL-2.0/MIT licenses)

## Acknowledgments

We acknowledge:
- **xView2 Challenge organizers** for providing the xBD dataset and establishing damage assessment benchmarks
- **BRIGHT dataset team** for creating a globally distributed multimodal disaster imagery dataset
- **Humanitarian OpenStreetMap Team** for practical insights into disaster response workflows
- **Kaggle community** for sharing DisasterGAN implementation and fostering open research

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

**Note:** While this project code is MIT licensed, the datasets (xBD and BRIGHT) have their own respective licenses. Please refer to the original dataset sources for usage terms.

---

**Project Paper:**
Singh, Aanya (2025). "Bridging the Post-Disaster Imagery Gap: Leveraging Synthetic Data for Disaster Response across Economic Spectra"
Available at SSRN: https://ssrn.com/abstract=5385441 or http://dx.doi.org/10.2139/ssrn.5385441
