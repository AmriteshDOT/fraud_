import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from src.config import ARTIFACTS_DIR, TARGET, PK

PRUNED_DATA = f"{ARTIFACTS_DIR}/pruned_features.parquet"
MODEL_OUT = f"{ARTIFACTS_DIR}/lgb_model.txt"


def run_training():
    df = pd.read_parquet(PRUNED_DATA)

    # Enforce strict chronological ordering for OOT split
    df = df.sort_values(PK).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    cols = [c for c in df.columns if c not in [PK, TARGET]]

    X_tr, y_tr = train_df[cols], train_df[TARGET]
    X_val, y_val = val_df[cols], val_df[TARGET]

    d_tr = lgb.Dataset(X_tr, label=y_tr)
    d_val = lgb.Dataset(X_val, label=y_val, reference=d_tr)

    neg_ct = (y_tr == 0).sum()
    pos_ct = (y_tr == 1).sum()
    imbalance_wt = neg_ct / pos_ct

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "scale_pos_weight": imbalance_wt,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 256,
        "max_bin": 255,
        "verbose": -1,
    }

    print("Training LightGBM engine...")
    model = lgb.train(
        params,
        d_tr,
        num_boost_round=1000,
        valid_sets=[d_tr, d_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    val_preds = model.predict(X_val)
    pr_auc = average_precision_score(y_val, val_preds)

    print(f"OOT Validation PR-AUC: {pr_auc:.4f}")

    model.save_model(MODEL_OUT)
    print(f"Model weights saved to {MODEL_OUT}")


if __name__ == "__main__":
    run_training()
