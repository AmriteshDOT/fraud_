
# IEEE-CIS Fraud Transaction Detection

End-to-end Card-Not-Present (CNP) fraud detection pipeline that processes 590K financial records. It utilizes temporal state-tracking, strict feature pruning, and a memory-optimized LightGBM engine optimized for Precision-Recall AUC to minimize financial chargebacks.


## What the project does ?

- Memory Optimization: Merges massive train_transaction and train_identity tables, utilizing aggressive 16-bit and 32-bit downcasting to reduce pandas DataFrame memory footprint by ~70% for efficient localized processing.

- Behavioral Feature Engineering: Projects raw timestamp integers into cyclical temporal features (hour, day of week) and derives rolling transaction velocity metrics (daily card frequency and spend amounts) to track anomalous purchasing bursts.

- Strict Pruning: Defends against overfitting and curse of dimensionality by eliminating near-zero variance columns and pruning highly collinear features (redundant Vesta characteristics) via sampled correlation matrix thresholding.

- Out-Of-Time (OOT) Engine: Enforces strict chronological data partitioning to prevent future-leakage of fraud signatures. Trains a LightGBM gradient boosting machine utilizing native histogram binning for rapid, memory-efficient tree building.

- Imbalance Mitigation: Counteracts the severe 3.5% fraud class imbalance by dynamically scaling positive weights in the objective function, optimizing the model directly for Precision-Recall AUC (PR-AUC) rather than standard ROC to heavily penalize False Negatives.
## Components & Flow

- **Preprocess** : Executes a left-join of identity hardware/browser data onto core financial transactions. Runs dynamic boundary checks to downcast float64/int64 types to their minimal 16-bit or 32-bit equivalents. Saves outputs to compressed .parquet format.

- **Feature Engineering** : Applies modulo arithmetic to extract time projections, applies groupby operations for daily card-level velocity and monetary aggregates, and maps frequency-encoding to high-cardinality categorical strings.

- **Select** : Iterates through features to drop variables where >99% of values are identical. Generates an absolute correlation matrix on a 50k sub-sample to identify and eliminate twin variables with >90% collinearity.

- **Train** : Executes an 80/20 chronological split. Feeds continuous variables directly into LightGBM's Dataset to leverage native 255-bin histograms. Monitors average_precision via early stopping to yield the finalized temporally-stable weights.