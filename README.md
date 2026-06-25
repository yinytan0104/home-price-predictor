# 🏠 Home Price Predictor

A regression model that predicts home sale prices from property features, served
as an interactive web app with explanations for *why* it predicted each price.

Built on the **Ames Housing dataset** (2,930 real home sales, 80+ features).

**What it demonstrates:** data cleaning & validation, feature engineering,
regression modeling, model evaluation (RMSE / R²), model interpretability (SHAP),
and a deployed, shareable app (Streamlit).

---

## What's in here

| File | What it does |
|------|--------------|
| `get_data.py` | Downloads the dataset into `data/ames.csv` |
| `prepare.py`  | Cleans the messy data + engineers comp-style features → `data/processed.csv` |
| `train.py`    | Trains 3 models, compares them, saves the best → `models/` |
| `app.py`      | The live Streamlit app: enter a home, get a price + top drivers |

---

## How to run it (about 10 minutes)

You need **Python 3.10+**. In a terminal, from this folder:

```bash
# 1. (recommended) create an isolated environment so installs don't clash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt

# 3. run the pipeline, in order
python get_data.py    # download the data
python prepare.py     # clean + engineer features
python train.py       # train and evaluate models

# 4. launch the app
streamlit run app.py
```

Step 4 opens the app in your browser at http://localhost:8501.

### Baseline numbers you should see from `train.py`
```
Linear Regression      RMSE ~$29,800   R2 0.875
Random Forest          RMSE ~$24,300   R2 0.917
Gradient Boosting      RMSE ~$23,100   R2 0.925   <- best
```
Your numbers will shift once you add your own features (that's the point).

---

## Make it yours (do this before putting it on your resume)

Open `prepare.py` and find the **YOUR TURN** block. Add 2-3 features you can
explain in an interview, e.g.:
- `Qual x Area` = Overall Qual × Gr Liv Area (quality matters more in bigger homes)
- Total porch/deck square footage
- Rank neighborhoods by their median sale price

Then re-run `prepare.py` → `train.py` and watch the RMSE / R² move. Being able
to say *"I added X because, as a broker, I know it drives price, and it improved
R² from 0.925 to 0.93"* is exactly the story that wins this interview.

---

## Resume bullet this produces (fill in YOUR numbers)

> Built and deployed a home-price regression model (Gradient Boosting, R² 0.92,
> RMSE ~$23K) on 2,900+ property records, engineering comp-style features and
> using SHAP for per-prediction interpretability, served via an interactive
> Streamlit app.

---

## Optional next steps (extra credit)
- Deploy it publicly for free on **Streamlit Community Cloud** so you have a live
  link to share (push this repo to GitHub, then connect it at share.streamlit.io).
- Add a map or a "comparable homes" panel.
- Log-transform the target (`SalePrice`) and compare — a common trick that often
  improves accuracy on skewed price data.
