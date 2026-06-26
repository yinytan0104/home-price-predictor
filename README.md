# King County Home Price Predictor

A machine learning pipeline that predicts residential sale prices across the Seattle metro area, with an interactive Streamlit app that explains each prediction using SHAP values and shows comparable recent sales on a map.

Built on **21,591 King County home sales** (May 2014 – May 2015), covering Seattle and surrounding cities across 70 zip codes.

**Live app:** [home-price-predictor-tyy.streamlit.app](https://home-price-predictor-tyy.streamlit.app)

---

## Model performance

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $174,741 | 0.750 |
| Random Forest | $123,020 | 0.876 |
| **Gradient Boosting** *(selected)* | **$105,930** | **0.908** |
| XGBoost | $106,108 | 0.908 |

Evaluated on a held-out test set (20% of data, 4,319 homes). 5-fold cross-validation confirms results are stable across splits.

RMSE is reported in original dollars after back-transforming log-scale predictions. The log transform corrects for the right-skewed distribution of home prices (median $450K, max $4.5M).

---

## Features engineered

| Feature | Rationale |
|---|---|
| House Age | Years since construction — older homes trade at a discount |
| Years Since Renovation | Captures recency of updates relative to house age |
| Recently Renovated | Flag for homes updated in the last 10 years |
| Has Basement | Binary; basement homes command a premium in King County |
| Sqft vs Neighbors | Living area relative to 15 nearest neighbors — over- or under-built for the block |
| Distance to Seattle | Euclidean distance from lat/long to downtown — continuous location gradient |
| Top Grade | Flag for King County grade ≥ 10 (Very Good through Mansion) |
| Grade × Sqft | Interaction term — high-grade finishes scale exponentially in larger homes |
| Basement Ratio | Fraction of living space that is underground; buyers discount basement vs above-grade |
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
