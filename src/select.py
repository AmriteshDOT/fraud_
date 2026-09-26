import pandas as pd
import numpy as np
from src.config import ARTIFACTS_DIR, TARGET, PK

FEAT_DATA_PATH = f"{ARTIFACTS_DIR}/features.parquet"
PRUNED_DATA_PATH = f"{ARTIFACTS_DIR}/pruned_features.parquet"


def drop_low_var(df, threshold=0.99):
    to_drop = []
    total_rows = len(df)

    for c in df.columns:
        if c in [PK, TARGET]:
            continue

        top_val_pct = df[c].value_counts(normalize=True, dropna=False).values[0]
        if top_val_pct >= threshold:
            to_drop.append(c)

    df.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} low variance columns")
    return df


def drop_high_corr(df, threshold=0.90):
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.drop(columns=[PK, TARGET], errors="ignore")

    # OOM crashSafe
    sample_df = num_df.sample(n=min(50000, len(num_df)), random_state=42)
    corr_mat = sample_df.corr().abs()

    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    df.drop(columns=to_drop, inplace=True)

    print(f"Dropped {len(to_drop)} highly correlated columns")
    return df


def run_pruning():
    print("Loading features...")
    df = pd.read_parquet(FEAT_DATA_PATH)

    start_cols = df.shape[1]

    print("Applying variance threshold...")
    df = drop_low_var(df, threshold=0.99)

    print("Applying collinearity elimination...")
    df = drop_high_corr(df, threshold=0.90)

    end_cols = df.shape[1]
    print(f"Pruning complete: {start_cols} -> {end_cols} columns")

    df.to_parquet(PRUNED_DATA_PATH, index=False)
    print(f"Pruned data saved to {PRUNED_DATA_PATH}")


if __name__ == "__main__":
    run_pruning()
