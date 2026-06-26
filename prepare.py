import os
import pandas as pd
import numpy as np

RAW_PATH = "data/kc_house_data.csv"
OUT_PATH = "data/processed.csv"
os.makedirs("models", exist_ok=True)

SEATTLE_LAT, SEATTLE_LON = 47.6062, -122.3321
TARGET = "price"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    df["month_sold"] = pd.to_datetime(df["date"]).dt.month
    df = df.drop(columns=["id", "date"])
    df["zipcode"] = df["zipcode"].astype(str).str.strip()

    # Remove a 33-bedroom anomaly and a handful of >$5M sales that are too sparse to model.
    before = len(df)
    df = df[(df["bedrooms"] >= 1) & (df["bedrooms"] <= 10) & (df["price"] < 5_000_000)]
    print(f"Removed {before - len(df)} outliers ({len(df):,} remaining)")

    current_year = 2015

    df["house_age"] = current_year - df["yr_built"]

    # yr_renovated = 0 means never renovated; fall back to house age in that case.
    df["years_since_reno"] = np.where(
        df["yr_renovated"] > 0,
        current_year - df["yr_renovated"],
        df["house_age"],
    )
    df["recently_renovated"] = (
        (df["yr_renovated"] > 0) & (current_year - df["yr_renovated"] <= 10)
    ).astype(int)

    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # Relative size vs. the 15 nearest neighbors — captures whether the home is over- or under-built for the block.
    df["sqft_vs_neighbors"] = df["sqft_living"] - df["sqft_living15"]

    # Euclidean distance to downtown Seattle as a continuous location signal.
    df["dist_to_seattle"] = np.sqrt(
        (df["lat"] - SEATTLE_LAT) ** 2 + (df["long"] - SEATTLE_LON) ** 2
    )

    # King County grade 10+ marks "Very Good" through "Mansion" — a luxury tier flag.
    df["top_grade"] = (df["grade"] >= 10).astype(int)

    # High-grade finishes matter more in bigger homes — the interaction captures exponential luxury pricing.
    df["grade_x_sqft"] = df["grade"] * df["sqft_living"]

    # What fraction of total living space is underground; buyers discount basement space vs above-grade.
    df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"].replace(0, 1)

    # Save per-zip metadata so the app can look up location and neighbor defaults.
    # zip_median_price is intentionally NOT computed here — it is computed inside
    # train.py from training rows only to prevent target leakage.
    zip_meta = (
        df.groupby("zipcode")
        .agg(
            lat=("lat", "mean"),
            lon=("long", "mean"),
            sqft_living15_med=("sqft_living15", "median"),
            sqft_lot15_med=("sqft_lot15", "median"),
        )
        .reset_index()
    )
    zip_meta.to_json("models/zip_meta.json", orient="records")

    # Save human-readable columns (with raw zip and coordinates) for the comps panel.
    comps_display = df[
        ["zipcode", "lat", "long", "sqft_living", "grade", "condition",
         "view", "waterfront", "yr_built", "yr_renovated",
         "bedrooms", "bathrooms", "recently_renovated", TARGET]
    ].copy().rename(columns={TARGET: "sale_price"})
    comps_display.to_csv("models/comps_display.csv", index=False)

    # Keep zipcode so train.py can compute zip_median_price from training rows only.
    df = df.drop(columns=["yr_built", "yr_renovated"])

    df.drop(columns=[TARGET]).to_csv("models/comps_features.csv", index=False)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved clean data: {df.shape[0]:,} rows, {df.shape[1]} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
