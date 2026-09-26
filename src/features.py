import pandas as pd
from src.config import PROCESSED_DATA_PATH, ARTIFACTS_DIR, TIME_COL, PK

FEAT_DATA_PATH = f"{ARTIFACTS_DIR}/features.parquet"


def add_time(df):
    df["hr"] = (df[TIME_COL] // 3600) % 24
    df["dw"] = (df[TIME_COL] // (3600 * 24)) % 7
    df["day"] = df[TIME_COL] // (3600 * 24)
    return df


def add_velocity(df):
    # TodayTime
    card_ct = df.groupby(["card1", "day"])[PK].count().rename("card_daily_ct")
    df = df.merge(card_ct, on=["card1", "day"], how="left")

    # TodayAmount
    card_amt = (
        df.groupby(["card1", "day"])["TransactionAmt"].sum().rename("card_daily_amt")
    )
    df = df.merge(card_amt, on=["card1", "day"], how="left")

    return df


def encode_freq(df):
    high_cardinality = ["card1", "card2", "addr1", "P_emaildomain"]

    for c in high_cardinality:
        if c in df.columns:
            freq = df[c].value_counts().to_dict()
            df[f"{c}_freq"] = df[c].map(freq)

    # LightGBM category handling
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category")

    return df


def run_features():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    print("Extracting time...")
    df = add_time(df)

    print("Building velocity state-tracks...")
    df = add_velocity(df)

    print("Encoding categoricals...")
    df = encode_freq(df)

    print("Dropping redundant columns...")
    df.drop(columns=[TIME_COL, "day"], inplace=True, errors="ignore")

    df.to_parquet(FEAT_DATA_PATH, index=False)
    print(f"Features saved to {FEAT_DATA_PATH} | New Shape: {df.shape}")


if __name__ == "__main__":
    run_features()
