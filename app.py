"""
app.py
-------
Step 3: the live app. Someone enters a home's features and gets a predicted
price plus the top factors driving that prediction (via SHAP).

Run it with:   streamlit run app.py
Then open the URL it prints (usually http://localhost:8501).
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

# ---- Load the trained model and metrics -------------------------------------
bundle = joblib.load("models/model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]          # exact column order used in training
with open("models/metrics.json") as f:
    metrics = json.load(f)

# Pull the neighborhood option names back out of the one-hot columns.
NEIGHBORHOODS = ["(baseline)"] + [
    c.replace("Neighborhood_", "") for c in FEATURES if c.startswith("Neighborhood_")
]

st.set_page_config(page_title="Home Price Predictor", page_icon="🏠")
st.title("🏠 Home Price Predictor")
best = metrics["best_model"]
r2 = metrics["results"][best]["r2"]
rmse = metrics["results"][best]["rmse"]
st.caption(f"Model: {best}  •  R² {r2:.3f}  •  typical error ±${rmse:,.0f}")

# ---- Inputs (sidebar) -------------------------------------------------------
st.sidebar.header("Enter the home's details")
gr_liv = st.sidebar.slider("Above-ground living area (sq ft)", 400, 4000, 1500, 50)
bsmt = st.sidebar.slider("Basement area (sq ft)", 0, 2500, 800, 50)
qual = st.sidebar.slider("Overall quality (1-10)", 1, 10, 6)
year_built = st.sidebar.slider("Year built", 1880, 2010, 1990)
year_remod = st.sidebar.slider("Year remodeled", 1950, 2010, 1995)
garage = st.sidebar.slider("Garage capacity (cars)", 0, 4, 2)
full_bath = st.sidebar.slider("Full bathrooms", 0, 4, 2)
beds = st.sidebar.slider("Bedrooms", 0, 8, 3)
rooms = st.sidebar.slider("Total rooms above grade", 2, 14, 6)
lot = st.sidebar.slider("Lot area (sq ft)", 1000, 30000, 9000, 500)
fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)
neighborhood = st.sidebar.selectbox("Neighborhood", NEIGHBORHOODS)


def build_row():
    """Assemble one input row in the exact shape the model expects."""
    row = {f: 0 for f in FEATURES}            # start at all zeros
    row["Gr Liv Area"] = gr_liv
    row["Total Bsmt SF"] = bsmt
    row["Overall Qual"] = qual
    row["Year Built"] = year_built
    row["Year Remod/Add"] = year_remod
    row["Garage Cars"] = garage
    row["Full Bath"] = full_bath
    row["Bedroom AbvGr"] = beds
    row["TotRms AbvGrd"] = rooms
    row["Lot Area"] = lot
    row["Fireplaces"] = fireplaces
    # engineered features must match prepare.py
    row["House Age"] = 2010 - year_built
    row["Years Since Remodel"] = 2010 - year_remod
    row["Total Bathrooms"] = full_bath
    # neighborhood one-hot
    col = f"Neighborhood_{neighborhood}"
    if col in row:
        row[col] = 1
    return pd.DataFrame([row])[FEATURES]


if st.sidebar.button("Predict price", type="primary"):
    X_one = build_row()
    price = model.predict(X_one)[0]
    st.metric("Estimated sale price", f"${price:,.0f}")

    # ---- Explain the prediction with SHAP -----------------------------------
    st.subheader("What's driving this estimate")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_one)[0]

    impact = (
        pd.DataFrame({"feature": FEATURES, "impact": shap_values})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(8)
    )
    for _, r in impact.iterrows():
        arrow = "▲ pushes up" if r["impact"] > 0 else "▼ pulls down"
        st.write(f"**{r['feature']}** — {arrow} by ${abs(r['impact']):,.0f}")
else:
    st.info("Set the home's details in the sidebar, then click **Predict price**.")
