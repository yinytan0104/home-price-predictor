"""
prepare.py
----------
Cleans raw King County sales data, removes statistical outliers, and engineers
10 domain-driven features that encode structural quality, age, location, and
relative size — the same factors appraisers and real estate agents use to
justify a price adjustment.

Outputs
-------
data/processed.csv         feature matrix consumed by train.py
models/comps_display.csv   human-readable sale records for the app's comps panel
models/comps_features.csv  numeric-only rows for NearestNeighbors search
models/zip_meta.json       per-zip centroid coordinates and neighbor sqft defaults

Note: zip_median_price is NOT computed here. It is computed inside train.py
from training rows only to prevent target leakage into the test set.
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "data/kc_house_data.csv"
OUT_PATH = "data/processed.csv"
os.makedirs("models", exist_ok=True)

SEATTLE_LAT, SEATTLE_LON = 47.6062, -122.3321
TARGET = "price"


def main():
    """
    Load the raw CSV, remove outliers, engineer 10 features, and write four
    output files used downstream by train.py and app.py.

    Outlier removal targets two specific distortions: a 33-bedroom data-entry
    error and a thin cluster of ultra-luxury sales above $5M where we have
    too few training examples to model reliably. Removing them prevents the
    model from learning a skewed relationship between extreme inputs and price.

    Feature engineering philosophy: every feature encodes a pricing signal that
    a human appraiser would consider — age, renovation recency, size relative to
    the block, proximity to downtown, and construction quality tier.
    """
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    df["month_sold"] = pd.to_datetime(df["date"]).dt.month
    df = df.drop(columns=["id", "date"])
    df["zipcode"] = df["zipcode"].astype(str).str.strip()

    # Drop a 33-bedroom data-entry error and the sparse ultra-luxury tier (>$5M).
    # Both distort the learned price-per-feature relationship for typical homes.
    before = len(df)
    df = df[(df["bedrooms"] >= 1) & (df["bedrooms"] <= 10) & (df["price"] < 5_000_000)]
    print(f"Removed {before - len(df)} outliers ({len(df):,} remaining)")

    # The dataset covers May 2014–May 2015, so 2015 is the correct reference year
    # for computing age and renovation recency.
    current_year = 2015

    df["house_age"] = current_year - df["yr_built"]

    # yr_renovated = 0 means the home has never been renovated. In that case,
    # years_since_reno falls back to house_age so the feature stays interpretable.
    df["years_since_reno"] = np.where(
        df["yr_renovated"] > 0,
        current_year - df["yr_renovated"],
        df["house_age"],
    )
    # Buyers pay a premium for recent work. A 10-year cutoff captures kitchen/bath
    # renovations that are still perceived as "modern" on the resale market.
    df["recently_renovated"] = (
        (df["yr_renovated"] > 0) & (current_year - df["yr_renovated"] <= 10)
    ).astype(int)

    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # sqft_living15 is the average living area of the 15 nearest homes — a census-style
    # context figure included in the dataset. The difference captures over- or under-built
    # relative to the immediate block, which affects appraised land-to-improvement ratio.
    df["sqft_vs_neighbors"] = df["sqft_living"] - df["sqft_living15"]

    # Straight-line degree distance to downtown Seattle. Location is the dominant
    # pricing signal in real estate; this turns lat/long into a single continuous
    # gradient rather than leaving coordinates as raw axes.
    df["dist_to_seattle"] = np.sqrt(
        (df["lat"] - SEATTLE_LAT) ** 2 + (df["long"] - SEATTLE_LON) ** 2
    )

    # King County's grading scale runs 1–13. Grade 10 ("Very Good") and above marks
    # a distinct luxury tier with custom finishes and higher design specifications.
    df["top_grade"] = (df["grade"] >= 10).astype(int)

    # Grade × sqft creates an interaction term: high-quality finishes compound in
    # value as square footage increases. A grade-10 home at 4,000 sq ft commands
    # a far larger premium than a grade-7 home at the same size — linear features
    # can't capture that multiplicative relationship on their own.
    df["grade_x_sqft"] = df["grade"] * df["sqft_living"]

    # Buyers discount below-grade space relative to above-grade living area.
    # Expressing this as a ratio normalizes across home sizes.
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
