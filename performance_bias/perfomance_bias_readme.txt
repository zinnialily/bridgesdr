This project uses the xBD dataset to build a model that predicts binary damage masks from pre- and post-disaster optical satellite imagery. Building annotation polygons provided in xBD are used to create more accurate target masks.

The main goal is to evaluate whether synthetic post-disaster imagery can improve model performance, particularly in Low-Income Countries (LICs) and in underrepresented disaster scenarios.

Objectives
    - Train a baseline model using xBD to predict damage masks from optical imagery.
    - Use xBD building annotations to generate binary damage masks:
      - white (1) = damaged (minor, major, or destroyed)
      - black (0) = undamaged (no-damage)
    - Test whether synthetic imagery improves model generalization in:
      - LICs
      - MICs (Middle-Income Countries)
      - HICs (High-Income Countries)
    - Conduct case studies, such as evaluating improvements for a Congo volcanic eruption using few-shot fine-tuning.

Experimental Setup
    1. Baseline Model
    - A U-Net trained on pre- and post-disaster optical images from xBD.
    - Supervised with binary masks from building annotations.
    
    2. Evaluation by Income Level
    - Evaluate model performance separately on LIC, MIC, and HIC regions.
    - Measure baseline generalization gaps and identify where synthetic imagery helps most.
    
    3. Few-Shot Synthetic Augmentation
    - For LIC, MIC, and HIC regions:
      - Train on 20% of real post-disaster images + synthetic imagery.
      - Evaluate on the remaining 80% to assess improvements.
      - If possible, include actual building damage annotations from those specific ones (hand-annotated using the BRIGHT dataset for case studies)
    - Example: Train on 20% of LIC real imagery plus synthetic, evaluate on 80%.

Data Preprocessing
    - Convert xBD building annotations (GeoJSON format) into binary damage masks.
    - Resize all input images and masks to a standard resolution (e.g., 256x256).
    - Apply consistent tiling and cropping strategies (crop each image into 4x4 tiles).
    - Optionally convert to heatmaps in future experiments to capture damage severity.

 Future Work
    - Extend binary classification to heatmap-based damage level predictions.
    - Incorporate SAR-like simulated difference maps for augmentation.
    - Investigate domain adaptation and active learning for low-data regions.