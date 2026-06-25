import os
import pandas as pd

URL = "https://data.openml.org/datasets/0004/42092/dataset_42092.pq"
os.makedirs("data", exist_ok=True)
OUT = "data/kc_house_data.csv"

print("Downloading King County House Sales dataset (21,613 records)...")
df = pd.read_parquet(URL)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} records -> {OUT}")
