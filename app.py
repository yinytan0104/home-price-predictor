import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

bundle = joblib.load("models/model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
LOG_TRANSFORM = bundle.get("log_transform", False)

with open("models/metrics.json") as f:
    metrics = json.load(f)

with open("models/neighborhood_medians.json") as f:
    neighborhood_medians = json.load(f)

NEIGHBORHOODS = sorted(neighborhood_medians.keys())

st.set_page_config(page_title="Home Price Predictor", page_icon="🏠")
st.title("🏠 Home Price Predictor")
best = metrics["best_model"]
r2 = metrics["results"][best]["r2"]
rmse = metrics["results"][best]["rmse"]
st.caption(f"Model: {best}  •  R² {r2:.3f}  •  typical error ±${rmse:,.0f}")

st.sidebar.header("Enter the home's details")
gr_liv = st.sidebar.slider("Above-ground living area (sq ft)", 400, 4000, 1500, 50)
bsmt = st.sidebar.slider("Basement area (sq ft)", 0, 2500, 800, 50)
qual = st.sidebar.slider("Overall quality (1–10)", 1, 10, 6)
year_built = st.sidebar.slider("Year built", 1880, 2010, 1990)
year_remod = st.sidebar.slider("Year remodeled", 1950, 2010, 1995)
garage = st.sidebar.slider("Garage capacity (cars)", 0, 4, 2)
full_bath = st.sidebar.slider("Full bathrooms", 0, 4, 2)
half_bath = st.sidebar.slider("Half bathrooms", 0, 2, 0)
beds = st.sidebar.slider("Bedrooms", 0, 8, 3)
rooms = st.sidebar.slider("Total rooms above grade", 2, 14, 6)
lot = st.sidebar.slider("Lot area (sq ft)", 1000, 30000, 9000, 500)
fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)
outdoor_sf = st.sidebar.slider("Total outdoor space (sq ft)", 0, 1500, 200, 25)
near_nuisance = st.sidebar.checkbox("Near arterial road or railroad")
peak_season = st.sidebar.checkbox("Listed April – July (peak season)")
neighborhood = st.sidebar.selectbox("Neighborhood", NEIGHBORHOODS)


def build_row():
    current_year = 2010
    house_age = current_year - year_built
    years_since_remodel = current_year - year_remod

    row = {f: 0 for f in FEATURES}
    row["Gr Liv Area"] = gr_liv
    row["Total Bsmt SF"] = bsmt
    row["Overall Qual"] = qual
    row["Year Built"] = year_built
    row["Year Remod/Add"] = year_remod
    row["Garage Cars"] = garage
    row["Full Bath"] = full_bath
    row["Half Bath"] = half_bath
    row["Bedroom AbvGr"] = beds
    row["TotRms AbvGrd"] = rooms
    row["Lot Area"] = lot
    row["Fireplaces"] = fireplaces
    row["House Age"] = house_age
    row["Years Since Remodel"] = years_since_remodel
    row["Qual x Area"] = qual * gr_liv
    row["Total Fin SF"] = gr_liv + bsmt
    row["Recently Remodeled"] = int(years_since_remodel <= 15)
    row["Total Bathrooms"] = full_bath + 0.5 * half_bath
    row["Total Outdoor SF"] = outdoor_sf
    row["Near Nuisance"] = int(near_nuisance)
    row["Peak Season"] = int(peak_season)
    row["Neighborhood Median Price"] = neighborhood_medians.get(neighborhood, np.median(list(neighborhood_medians.values())))
    return pd.DataFrame([row])[FEATURES]


if st.sidebar.button("Predict price", type="primary"):
    X_one = build_row()
    raw_pred = model.predict(X_one)[0]
    price = np.expm1(raw_pred) if LOG_TRANSFORM else raw_pred
    st.metric("Estimated sale price", f"${price:,.0f}")

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
