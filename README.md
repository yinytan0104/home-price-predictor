# King County Home Price Predictor

A machine learning pipeline that predicts residential sale prices across the Seattle metro area, with an interactive Streamlit app that explains each prediction using SHAP values and shows comparable recent sales on a map.

Built on **21,591 King County home sales** (May 2014 – May 2015), covering Seattle and surrounding cities across 70 zip codes.

**Live app:** [home-price-predictor-tyy.streamlit.app](https://home-price-predictor-tyy.streamlit.app)

---

## Model performance

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $175,700 | 0.748 |
| Random Forest | $123,125 | 0.876 |
| **Gradient Boosting** *(selected)* | **$104,996** | **0.910** |
| XGBoost | $106,761 | 0.907 |

Evaluated on a held-out test set (20% of data, ~4,300 homes). 5-fold cross-validation on the training set confirms results are stable across splits — the test set is never touched during model selection.

RMSE is reported in original dollars after back-transforming log-scale predictions. The log transform corrects for the right-skewed distribution of home prices (median ~$450K, max ~$4.5M): predicting log(1 + price) prevents the model from over-weighting a small number of high-value sales.

---

## Engineering highlights

- **Leak-free neighborhood encoding:** `zip_median_price` (the most predictive feature) is computed from training rows only, then mapped onto the test set. Using all rows before the split would inflate apparent R² without reflecting real generalization — a common data leakage error in public notebooks on this dataset.
- **SHAP interpretability:** each prediction is explained at the feature level, with SHAP values converted from log-space to dollar impact so users see "grade adds $48,000" rather than an abstract log coefficient.
- **Comparable homes via k-NN:** NearestNeighbors on a StandardScaler-normalized feature matrix finds the 5 most structurally similar recent sales, displayed in a table and plotted on an interactive map.

---

## Features engineered

| Feature | Rationale |
|---|---|
| House Age | Years since construction — older homes trade at a discount on average |
| Years Since Renovation | Captures recency of updates relative to house age |
| Recently Renovated | Flag for homes updated in the last 10 years (perceived as "modern") |
| Has Basement | Binary; basement homes command a consistent premium in King County |
| Sqft vs Neighbors | Living area relative to the 15 nearest homes — over- or under-built for the block |
| Distance to Seattle | Euclidean distance from lat/long to downtown — continuous location gradient |
| Top Grade | Flag for King County grade ≥ 10 (Very Good through Mansion) |
| Grade × Sqft | Interaction term — high-quality finishes scale exponentially in larger homes |
| Basement Ratio | Fraction of living space underground; buyers discount basement vs above-grade |
| Zip Median Price | Encodes neighborhood price tier as one number across 70 zip codes |

---

## Project structure

| File | Purpose |
|---|---|
| `get_data.py` | Downloads 21,613-record King County dataset from OpenML |
| `prepare.py` | Cleans data, removes outliers, engineers 10 features, saves comps for app |
| `train.py` | Trains four models, evaluates on held-out test set, runs 5-fold CV, saves best |
| `app.py` | Streamlit app — price estimate, SHAP breakdown, comparable homes table and map |

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python get_data.py
python prepare.py
python train.py

streamlit run app.py
```

The app opens at `http://localhost:8501`.
