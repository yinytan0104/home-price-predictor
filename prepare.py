"""
prepare.py
-----------
Step 1 of the pipeline: take the raw, messy Ames Housing data and turn it into a
clean table the model can learn from.

Run it with:   python prepare.py

It reads  data/ames.csv  and writes  data/processed.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/ames.csv"
OUT_PATH = "data/processed.csv"

# These are the "comp-style" features a broker actually looks at when pricing a home.
# Start here, then add or remove fields in the YOUR TURN section below.
NUMERIC_FEATURES = [
    "Gr Liv Area",       # above-ground living area (sq ft)
    "Total Bsmt SF",     # basement square footage
    "Overall Qual",      # overall material/finish quality, 1-10
    "Year Built",
    "Year Remod/Add",    # year of last remodel
    "Garage Cars",       # garage capacity
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",     # total rooms above grade
    "Lot Area",
    "Fireplaces",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
]
CATEGORICAL_FEATURES = ["Neighborhood"]
TARGET = "SalePrice"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---- 1. Keep only the columns we need -------------------------------------
    # Include source columns for derived features; they are dropped after use below.
    extra_source = ["Sale Condition", "Condition 1", "Condition 2", "Mo Sold"]
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + extra_source + [TARGET]
    df = df[cols].copy()

    # ---- 1b. Filter to arm's-length market sales only -------------------------
    # Distressed, family, and partial sales don't reflect true market value and
    # would skew the model the same way they'd skew a broker's comps.
    before_filter = len(df)
    df = df[df["Sale Condition"] == "Normal"].copy()
    print(f"Removed {before_filter - len(df)} non-Normal sales ({len(df)} remaining)")
    df.drop(columns=["Sale Condition"], inplace=True)

    # ---- 2. Clean missing values ----------------------------------------------
    # A missing basement / garage value almost always means "no basement / garage",
    # so 0 is the honest fill here, not the average.
    for col in ["Total Bsmt SF", "Garage Cars"]:
        df[col] = df[col].fillna(0)
    # Drop any remaining rows with missing values (very few).
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values")

    # ---- 3. Remove outliers ---------------------------------------------------
    # The Ames dataset author recommends removing a handful of giant homes that
    # sold cheaply (partial sales / family transfers). They distort any model.
    df = df[df["Gr Liv Area"] < 4000].copy()

    # ---- 4. Feature engineering -----------------------------------------------
    # Turn raw years into the things that actually drive value.
    current_year = 2010  # the data was collected 2006-2010; use the dataset's era
    df["House Age"] = current_year - df["Year Built"]
    df["Years Since Remodel"] = current_year - df["Year Remod/Add"]
    df["Total Bathrooms"] = df["Full Bath"]  # simple start; expand below

    # ============================ YOUR TURN ===================================
    # This is where your broker brain beats a generic Kaggle submission.
    # Ideas to add (each is one line of pandas):
    #   - Quality x Size interaction:   df["Qual x Area"] = df["Overall Qual"] * df["Gr Liv Area"]
    #   - Has a second story, basement finished %, porch/deck area, etc.
    #   - Group rare neighborhoods together, or rank neighborhoods by median price.
    # Add 2-3 features you can EXPLAIN in an interview ("I added this because...").
    # ==========================================================================

    # Quality amplifies value more in larger homes, so the product captures that non-linear relationship.
    df["Qual x Area"] = df["Overall Qual"] * df["Gr Liv Area"]

    # Buyers care about total usable space, not just above-ground area; finished basement adds real value.
    df["Total Fin SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]

    # Recent renovations command a premium; 15 years keeps updated kitchens/baths in, dated ones out.
    df["Recently Remodeled"] = (df["Years Since Remodel"] <= 15).astype(int)

    # Half-baths add value but not as much as full baths; the 0.5 weight reflects how brokers comp them.
    df["Total Bathrooms"] = df["Full Bath"] + 0.5 * df["Half Bath"]

    # Buyers in Ames treat any outdoor living area as a plus; summing all porch/deck types into one signal avoids fragmentation.
    df["Total Outdoor SF"] = (
        df["Wood Deck SF"] + df["Open Porch SF"] + df["Enclosed Porch"]
        + df["3Ssn Porch"] + df["Screen Porch"]
    )

    # Proximity to arterial roads, feeder roads, or railroads is a consistent price drag;
    # checking both Condition columns catches corner-lot homes exposed to two nuisances.
    _nuisance = {"Artery", "Feedr", "RRAn", "RRAe", "RRNn", "RRNe"}
    df["Near Nuisance"] = (
        df["Condition 1"].isin(_nuisance) | df["Condition 2"].isin(_nuisance)
    ).astype(int)

    # Spring/early-summer listings attract more buyers and more competition, which pushes prices up.
    df["Peak Season"] = df["Mo Sold"].isin([4, 5, 6, 7]).astype(int)

    # Drop raw source columns now that the derived features are built.
    df.drop(columns=["Condition 1", "Condition 2", "Mo Sold"], inplace=True)

    # ---- 5. Encode the neighborhood (text -> numeric columns) -----------------
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # ---- 6. Save --------------------------------------------------------------
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved clean data: {df.shape[0]} rows, {df.shape[1]} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
