
# Fraud Detection System using Gradient Boosting

Developed a fraud detection pipeline on 1.3M credit card transactions using LightGBM and XGBoost, handling extreme class imbalance with weighted loss and time-aware cross-validation.


## Data Processing & Feature Engineering

- Extracted temporal features: hour, day, weekday, month, time_since_last_txn, and velocity features (txns_last_1h, txns_last_1d, txns_last_7d).
- Engineered behavioral and interaction features: amount ratios, rolling statistics (mean_amt_last_3, median_amt_last_7), distance between cardholder and merchant, and combined categorical features (state_gender, gender_job).
- Added cyclical encoding for time (day_sin, day_cos) and flags for weekends, holidays, and peak hours.
- Encoded categorical features using LabelEncoder and frequency encoding for high-cardinality columns.
- Calculated normalized entropy for categorical relationships to quantify variability in user behavior.
## Modelling

- Implemented LightGBM and XGBoost with TimeSeriesSplit for temporally consistent validation.
- Performed Optuna hyperparameter tuning for both LightGBM and XGBoost to maximize ROC-AUC.
- Ensemble prediction by averaging LightGBM and XGBoost out-of-fold predictions.
- Evaluated using ROC-AUC, PR-AUC, F1-score, precision, and recall.
## Results

- Overall ROC-AUC: 0.998
- F1-score: 0.938
- Precision: 0.960, Recall: 0.918
- SHAP analysis used for feature importance and interpretability.