import pandas as pd
import numpy as np

RAW_PATH = "data/ames.csv"
OUT_PATH = "data/processed.csv"

NUMERIC_FEATURES = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "Overall Qual",
    "Year Built",
    "Year Remod/Add",
    "Garage Cars",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",
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

    extra_source = ["Sale Condition", "Condition 1", "Condition 2", "Mo Sold"]
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + extra_source + [TARGET]
    df = df[cols].copy()

    # Exclude non-arm's-length transactions (distressed sales, family transfers,
    # partial sales) — they don't reflect true market value and skew the model.
    before_filter = len(df)
    df = df[df["Sale Condition"] == "Normal"].copy()
    print(f"Removed {before_filter - len(df)} non-Normal sales ({len(df)} remaining)")
    df.drop(columns=["Sale Condition"], inplace=True)

    for col in ["Total Bsmt SF", "Garage Cars"]:
        df[col] = df[col].fillna(0)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values")

    # Remove a handful of very large homes that sold at anomalously low prices
    # (documented partial sales in the Ames dataset).
    df = df[df["Gr Liv Area"] < 4000].copy()

    current_year = 2010
    df["House Age"] = current_year - df["Year Built"]
    df["Years Since Remodel"] = current_year - df["Year Remod/Add"]

    # Quality amplifies value more in larger homes; the interaction captures that.
    df["Qual x Area"] = df["Overall Qual"] * df["Gr Liv Area"]

    # Buyers price on total livable space, not just above-ground area.
    df["Total Fin SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]

    # Updated kitchens and baths command a premium; 15 years is the broker threshold.
    df["Recently Remodeled"] = (df["Years Since Remodel"] <= 15).astype(int)

    # Half-baths contribute real value but are conventionally weighted at 0.5.
    df["Total Bathrooms"] = df["Full Bath"] + 0.5 * df["Half Bath"]

    # Any outdoor living space adds value; one combined signal beats five sparse columns.
    df["Total Outdoor SF"] = (
        df["Wood Deck SF"] + df["Open Porch SF"] + df["Enclosed Porch"]
        + df["3Ssn Porch"] + df["Screen Porch"]
    )

    # Arterial roads, feeder roads, and railroads are consistent price drags.
    _nuisance = {"Artery", "Feedr", "RRAn", "RRAe", "RRNn", "RRNe"}
    df["Near Nuisance"] = (
        df["Condition 1"].isin(_nuisance) | df["Condition 2"].isin(_nuisance)
    ).astype(int)

    # Spring and early-summer listings see more competition and higher closing prices.
    df["Peak Season"] = df["Mo Sold"].isin([4, 5, 6, 7]).astype(int)

    df.drop(columns=["Condition 1", "Condition 2", "Mo Sold"], inplace=True)

    # Encode neighborhood as its median sale price — one informative number instead
    # of 27+ sparse one-hot columns. Captures the price tier each area commands.
    neighborhood_median_map = df.groupby("Neighborhood")["SalePrice"].median()
    df["Neighborhood Median Price"] = df["Neighborhood"].map(neighborhood_median_map)
    df.drop(columns=["Neighborhood"], inplace=True)

    # Save the mapping so the app can translate a neighborhood name to a median price.
    neighborhood_median_map.to_json("models/neighborhood_medians.json")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved clean data: {df.shape[0]} rows, {df.shape[1]} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
