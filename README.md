This is the code for the Non-Trivial Fellowship. An absrtact of the project is provided below:
Low- and middle-income countries (LMICs) face particular challenges in disaster response due to
limited access to high-quality post-disaster imagery and models trained predominantly on high-
income country (HIC) datasets. Current computer vision damage assessment models exhibit signif-
icant performance disparities across economic contexts, creating equity concerns in global disaster
management. This research evaluates synthetic disaster imagery generation as a solution to these
geographic biases, using DisasterGAN to generate post-disaster optical imagery and comparing
damage assessment performance across economic regions. The methodology employs a two-part
approach: first assessing synthetic imagery quality through damage mask comparison using IoU,
Dice coefficient, precision, and recall metrics; second, implementing progressive U-Net fine-tuning
to evaluate synthetic data’s effectiveness as training augmentation across low-, middle-, and high-
income contexts. Counterintuitively, low-income countries achieved superior performance in multi-
class damage segmentation (IoU: 0.403 vs 0.308-0.329 for MIC/HIC), while middle-income countries
showed the greatest improvement from fine-tuning (297% IoU increase). However, specialized mod-
els exhibited severe performance degradation on general benchmarks, with HIC-adapted models
losing 79% accuracy on cross-dataset evaluation. These findings suggest economic context funda-
mentally shapes optimal disaster response ML strategies, though results may reflect dataset limi-
tations and simplified SAR conversion methods rather than generalizable patterns. These findings
suggest economic context fundamentally shapes optimal disaster response ML strategies, indicat-
ing the need for region-specific model development rather than universal solutions. Future work
should focus on expanding multi-regional training datasets, developing economic-adaptive architec-
tures, and establishing standardized evaluation protocols that account for infrastructure and data
availability constraints across different economic contexts.

Link to the full paper: https://pdflink.to/aanya-singh-non-trivial/
