# Composition-Driven Discovery and Screening of High-Entropy Alloy Catalysts

The `RF_single.py` script in this directory provides a simple example for predicting high-entropy alloy (HEA) properties at equimolar compositions using a Random Forest regression model.

- **Input:** Precomputed elemental-composition features and target properties (e.g., synthesizability metrics or cosine similarity to Pt) for equimolar HEAs.
- **Feature Engineering:** Automated extraction and aggregation of elemental descriptors (atomic mass, electronegativity, radius, etc.).
- **Model:** Random Forest regressor with fixed hyperparameters (no hyperparameter optimization included).
- **Output:** Cross-validation performance metrics, trained model artifacts, and prediction results.

This script is intended as a minimal working reference and starting point for composition-driven machine learning workflows in HEA discovery.

Copyright © 2025 Johns Hopkins.

This work is provided for review purposes only.
No permission is granted to copy, use, modify, or distribute this code or any part thereof until the conclusion of the peer-review process and explicit relicensing.

If you wish to use this work or have any questions regarding its license, please contact the author.

Unauthorized use is strictly prohibited during the review period.