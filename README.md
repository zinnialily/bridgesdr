# Non-Trivial Fellowship Project

## Abstract

Low- and middle-income countries (LMICs) face particular challenges in disaster response due to limited access to high-quality post-disaster imagery and models trained predominantly on high-income country (HIC) datasets. Current computer vision damage assessment models exhibit significant performance disparities across economic contexts, creating equity concerns in global disaster management.

This research evaluates synthetic disaster imagery generation as a solution to these geographic biases, using **DisasterGAN** to generate post-disaster optical imagery and comparing damage assessment performance across economic regions.

The methodology employs a two-part approach:

1. **Assessing synthetic imagery quality** through damage mask comparison using IoU, Dice coefficient, precision, and recall metrics.
2. **Implementing progressive U-Net fine-tuning** to evaluate synthetic data’s effectiveness as training augmentation across low-, middle-, and high-income contexts.

Counterintuitively, low-income countries achieved superior performance in multi-class damage segmentation (IoU: 0.403 vs. 0.308–0.329 for MIC/HIC), while middle-income countries showed the greatest improvement from fine-tuning (297% IoU increase). However, specialized models exhibited severe performance degradation on general benchmarks, with HIC-adapted models losing 79% accuracy on cross-dataset evaluation.

These findings suggest economic context fundamentally shapes optimal disaster response ML strategies, though results may reflect dataset limitations and simplified SAR conversion methods rather than generalizable patterns. The results indicate the need for region-specific model development rather than universal solutions.

Future work should focus on:

- Expanding multi-regional training datasets.
- Developing economic-adaptive architectures.
- Establishing standardized evaluation protocols that account for infrastructure and data availability constraints across different economic contexts.

---

**Link to the full paper:** [https://pdflink.to/aanya-singh-non-trivial/](https://pdflink.to/aanya-singh-non-trivial/)
