# Composition-Driven Discovery and Screening of High-Entropy Alloy Catalysts

The `RF_single.py` script in this directory provides a simple example for predicting high-entropy alloy (HEA) properties at equimolar compositions using a Random Forest regression model.

- **Input:** Precomputed elemental-composition features and target properties (e.g., synthesizability metrics or cosine similarity to Pt) for equimolar HEAs.
- **Feature Engineering:** Automated extraction and aggregation of elemental descriptors (atomic mass, electronegativity, radius, etc.).
- **Model:** Random Forest regressor with fixed hyperparameters (no hyperparameter optimization included).
- **Output:** Cross-validation performance metrics, trained model artifacts, and prediction results.

This script is intended as a minimal working reference and starting point for composition-driven machine learning workflows in HEA discovery.

## Citation

Published as:

> G. Han, T. Li, X. Xu, J. Lee, G. Qiu, S. Sequeira, A. Ajith, and C. Oses,
> *The search for high-entropy fuel-cell catalysts using disorder descriptors*,
> Nano Futures **9**, 045001 (2025).
> [doi:10.1088/2399-1984/ae19b0](https://doi.org/10.1088/2399-1984/ae19b0)

Open access. BibTeX is in `IOPEXPORT_BIB.bib`.

## License

Copyright © 2025 Entropy for Energy Lab, Johns Hopkins University.

Released under the MIT License; see `LICENSE`.