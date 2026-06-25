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
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test = np.expm1(y_test_log)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<22} RMSE ${rmse:>10,.0f}   MAE ${mae:>10,.0f}   R2 {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = pd.read_csv(DATA_PATH)
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} homes | Test: {len(X_test)} homes\n")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
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

    # 5-fold cross-validation on the best model for a more robust accuracy estimate.
    print(f"\n5-fold cross-validation ({best_name}):")
    cv_scores = cross_val_score(
        fitted[best_name], X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    cv_rmse_log = -cv_scores.mean()
    # Approximate dollar-space RMSE from log-space score using the mean log price.
    mean_log_price = y.mean()
    cv_rmse_dollars = np.expm1(mean_log_price + cv_rmse_log) - np.expm1(mean_log_price)
    cv_rmse_dollars = abs(cv_rmse_dollars)
    print(f"  CV RMSE (log): {cv_rmse_log:.4f} ± {-cv_scores.std():.4f}")
    print(f"  CV RMSE (approx $): ${cv_rmse_dollars:,.0f}")

    joblib.dump(
        {"model": fitted[best_name], "features": list(X.columns), "log_transform": True},
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
