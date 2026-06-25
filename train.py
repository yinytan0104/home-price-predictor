"""
train.py
---------
Step 2: train several models, compare them honestly on a held-out test set,
and save the best one for the app to use.

Run it with:   python train.py   (after prepare.py)

It reads  data/processed.csv  and writes  models/model.joblib  and  models/metrics.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

DATA_PATH = "data/processed.csv"
os.makedirs("models", exist_ok=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<22} RMSE ${rmse:>10,.0f}   MAE ${mae:>10,.0f}   R2 {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # Same split every run so your numbers are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} homes | Test: {len(X_test)} homes\n")

    # Three models, simplest to strongest. Comparing them IS the data-science story.
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)
        fitted[name] = model

    # Pick the model with the best (lowest) RMSE on the test set.
    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"\nBest model: {best_name}")

    # Save the winning model + the exact feature order the app must reproduce.
    joblib.dump(
        {"model": fitted[best_name], "features": list(X.columns)},
        "models/model.joblib",
    )
    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print("Saved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
