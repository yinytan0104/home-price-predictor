"""
get_data.py
------------
Downloads the Ames Housing dataset into data/ames.csv so anyone can reproduce
this project from scratch.

Run it with:   python get_data.py
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
os.makedirs("data", exist_ok=True)
OUT = "data/ames.csv"

print("Downloading Ames Housing dataset...")
urllib.request.urlretrieve(URL, OUT)
print(f"Saved to {OUT}")
