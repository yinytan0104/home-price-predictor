# Home Price Predictor

A machine learning pipeline that predicts residential sale prices from property features, with an interactive Streamlit app that explains each prediction using SHAP values.

Built on the **Ames Housing dataset** — 2,413 arm's-length market sales, filtered from the full 2,930-record dataset to remove distressed sales and family transfers that don't reflect true market value.

---

## Model performance

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $20,055 | 0.919 |
| Random Forest | $20,699 | 0.913 |
| **Gradient Boosting** *(selected)* | **$19,222** | **0.925** |

Evaluated on a held-out test set (20% of data, 483 homes). RMSE is in original dollars after back-transforming log-scale predictions.

---

## Features engineered

Beyond the raw property attributes, the pipeline derives:

- **Qual × Area** — interaction between overall quality rating and above-ground living area; quality matters more in larger homes
- **Total Finished SF** — above-ground living area plus basement square footage
- **Total Bathrooms** — full baths + 0.5 × half baths, matching standard appraisal convention
- **Total Outdoor SF** — sum of all porch and deck square footage
- **Recently Remodeled** — flag for homes remodeled within the last 15 years
- **Near Nuisance** — flag for proximity to arterial roads, feeder roads, or railroads
- **Peak Season** — flag for listings in April–July, when buyer competition is highest

---

## Project structure

| File | Purpose |
|---|---|
| `get_data.py` | Downloads the Ames Housing dataset |
| `prepare.py` | Cleans data, filters to normal sales, engineers features |
| `train.py` | Trains three models, evaluates on held-out test set, saves best |
| `app.py` | Streamlit app — enter a home's details, get a price and top drivers |

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
