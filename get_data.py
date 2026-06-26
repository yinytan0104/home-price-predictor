"""
get_data.py
-----------
Downloads the King County House Sales dataset (21,613 records) from OpenML
and saves it as a local CSV for the rest of the pipeline to read.

The dataset covers residential sales in King County, Washington (Seattle metro)
from May 2014 to May 2015, with 21 columns including price, square footage,
location coordinates, building grade, and neighborhood context features.

Source: https://www.openml.org/d/42092 (CC0 public domain)
Run once before prepare.py.
"""

import os
import pandas as pd

URL = "https://data.openml.org/datasets/0004/42092/dataset_42092.pq"
os.makedirs("data", exist_ok=True)
OUT = "data/kc_house_data.csv"

print("Downloading King County House Sales dataset (21,613 records)...")
df = pd.read_parquet(URL)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} records -> {OUT}")
