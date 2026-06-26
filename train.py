"""
train.py
--------
Trains four regression models on the processed King County dataset and saves
the best-performing one for the Streamlit app to serve.

Design decisions
----------------
Log-transform the target: home prices in King County are right-skewed (median
~$450K, max ~$4.5M). Predicting log(1 + price) compresses the tail, produces
more symmetric residuals, and prevents the model from over-fitting to a handful
of high-value sales. All error metrics are back-transformed to dollars so they
remain interpretable.

Leak-free zip encoding: zip_median_price (the most powerful location feature)
is computed from training rows only, then mapped onto the test set. Using the
full dataset before the split would let test-set prices bleed into a feature
the model sees at evaluation time — inflating apparent R² without reflecting
real-world performance.

CV on training data only: cross_val_score is called on X_train/y_train so the
held-out test set remains a clean, untouched estimate of generalization error.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

DATA_PATH = "data/processed.csv"
os.makedirs("models", exist_ok=True)


def evaluate(name, model, X_test, y_test_log):
    """
    Compute RMSE, MAE, and R² in original dollar space after back-transforming
    log-scale predictions.

    np.expm1 is the exact inverse of np.log1p: expm1(x) = e^x - 1, which
    correctly reverses log1p(x) = log(1 + x). Using np.exp instead would
    introduce a systematic ~$1 upward bias across all predictions.
    """
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test = np.expm1(y_test_log)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<22} RMSE ${rmse:>10,.0f}   MAE ${mae:>10,.0f}   R2 {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    """
    Run the full training pipeline: log-transform target, 80/20 split, compute
    leak-free zip encoding, benchmark four models, run 5-fold CV on the winner,
    and save the model bundle.

    Feature list is captured from list(X_train.columns) — not list(X.columns) —
    because X_train is mutated after the split: zipcode is replaced with
    zip_median_price. The saved feature order must exactly match the vector
    that build_row() in app.py assembles at prediction time.
    """
    df = pd.read_csv(DATA_PATH)
    # Log-transform corrects for the right-skewed distribution of sale prices.
    # The model learns to predict log(1 + price); all reported metrics are
    # converted back to dollars via np.expm1 before printing or saving.
    y = np.log1p(df["price"])
    X = df.drop(columns=["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} homes | Test: {len(X_test)} homes\n")

    # Zip-code median price encodes neighborhood tier as one number across 70 zip codes.
    # Computing it from training rows only is critical: if we used all rows, every test
    # home's price would have contributed to its own zip median — a form of target leakage
    # that makes the model look better on paper than it would perform on new data.
    zip_median = (
        X_train[["zipcode"]]
        .assign(price=np.expm1(y_train).values)
        .groupby("zipcode")["price"]
        .median()
    )
    # Fallback for zip codes that appear only in the test set (not seen during training).
    fallback = zip_median.median()
    X_train["zip_median_price"] = X_train["zipcode"].map(zip_median)
    X_test["zip_median_price"] = X_test["zipcode"].map(zip_median).fillna(fallback)
    X_train = X_train.drop(columns=["zipcode"])
    X_test = X_test.drop(columns=["zipcode"])

    # Save the training-set medians for the app to use at prediction time.
    zip_median.to_json("models/zip_medians.json")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500,        # 500 trees; learning_rate=0.05 means each adds a small correction
            learning_rate=0.05,      # slow learning rate + many trees outperforms fast + few
            max_depth=4,             # shallow trees prevent individual features from dominating
            subsample=0.8,           # row subsampling adds variance reduction (stochastic GB)
            min_samples_leaf=10,     # regularizes leaf nodes to avoid fitting noise in sparse zips
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,    # randomly sample 80% of features per tree (column subsampling)
            min_child_weight=10,     # XGBoost equivalent of min_samples_leaf
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)
        fitted[name] = model

    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"\nBest model: {best_name}")

    # 5-fold cross-validation on training data only — keeping test set fully held out.
    print(f"\n5-fold cross-validation ({best_name}):")
    cv_scores = cross_val_score(
        fitted[best_name], X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    cv_rmse_log = -cv_scores.mean()
    # Approximate dollar-space RMSE from log-space score using the mean log price.
    mean_log_price = y_train.mean()
    cv_rmse_dollars = np.expm1(mean_log_price + cv_rmse_log) - np.expm1(mean_log_price)
    cv_rmse_dollars = abs(cv_rmse_dollars)
    print(f"  CV RMSE (log): {cv_rmse_log:.4f} ± {-cv_scores.std():.4f}")
    print(f"  CV RMSE (approx $): ${cv_rmse_dollars:,.0f}")

    joblib.dump(
        {"model": fitted[best_name], "features": list(X_train.columns), "log_transform": True},
        "models/model.joblib",
    )
    with open("models/metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "results": results,
                "log_transform": True,
                "cv_rmse_log": cv_rmse_log,
            },
            f,
            indent=2,
        )
    print("\nSaved models/model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
