# Home Price Predictor

A machine learning pipeline that predicts residential sale prices from property features, with an interactive Streamlit app that explains each prediction using SHAP values.

Built on the **Ames Housing dataset** — 2,413 arm's-length market sales filtered from 2,930 records to remove distressed sales, family transfers, and partial transactions that don't reflect true market value.

---

## Model performance

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $23,831 | 0.885 |
| Random Forest | $20,355 | 0.916 |
| Gradient Boosting | $19,068 | 0.926 |
| **XGBoost** *(selected)* | **$19,068** | **0.926** |

Evaluated on a held-out test set (20% of data, 483 homes). 5-fold cross-validation confirms **RMSE ~$19,400**, validating that results are not an artifact of the train/test split.

RMSE is in original dollars after back-transforming log-scale predictions. The log transform accounts for the right-skewed distribution of sale prices.

Starting from a naive linear baseline (RMSE $29,773, R² 0.875), the final model represents a **36% reduction in prediction error**.

---

## Features engineered

Beyond the raw property attributes, the pipeline derives:

| Feature | Rationale |
|---|---|
| Qual × Area | Quality matters more in larger homes — the interaction captures what neither column captures alone |
| Total Finished SF | Buyers price on total livable space, not just above-ground area |
| Total Bathrooms | Full baths + 0.5 × half baths, matching standard appraisal convention |
| Total Outdoor SF | Any outdoor living space adds value; one combined signal beats five sparse columns |
| Recently Remodeled | Updated kitchens and baths command a premium; 15-year threshold matches broker practice |
| Near Nuisance | Proximity to arterial roads, feeder roads, or railroads is a consistent price drag |
| Peak Season | Spring/early-summer listings attract more competition and close higher |
| Neighborhood Median Price | Encodes each neighborhood's price tier as one number — more signal than 27 one-hot columns |

---

## Project structure

| File | Purpose |
|---|---|
| `get_data.py` | Downloads the Ames Housing dataset |
| `prepare.py` | Cleans data, filters to normal sales, engineers features |
| `train.py` | Trains four models, evaluates on held-out test set, runs 5-fold CV, saves best |
| `app.py` | Streamlit app — enter a home's details, get a price and top SHAP drivers |

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
