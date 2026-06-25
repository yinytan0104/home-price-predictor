import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

SEATTLE_LAT, SEATTLE_LON = 47.6062, -122.3321

bundle = joblib.load("models/model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]
LOG_TRANSFORM = bundle.get("log_transform", False)

with open("models/metrics.json") as f:
    metrics = json.load(f)
with open("models/zip_medians.json") as f:
    zip_medians = json.load(f)

zip_meta_df = pd.DataFrame(json.load(open("models/zip_meta.json")))
zip_meta = zip_meta_df.set_index("zipcode").to_dict(orient="index")
ZIPCODES = sorted(zip_meta.keys())


@st.cache_resource
def load_comps():
    features = pd.read_csv("models/comps_features.csv")
    display = pd.read_csv("models/comps_display.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn.fit(X_scaled)
    return nn, scaler, list(features.columns), display


st.set_page_config(page_title="KC Home Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 King County Home Price Predictor")
best = metrics["best_model"]
r2 = metrics["results"][best]["r2"]
rmse = metrics["results"][best]["rmse"]
st.caption(f"Model: {best}  •  R² {r2:.3f}  •  typical error ±${rmse:,.0f}  •  trained on 21,591 King County sales (2014–2015)")

st.sidebar.header("Home details")

zipcode = st.sidebar.selectbox("Zip code", ZIPCODES, index=ZIPCODES.index("98103") if "98103" in ZIPCODES else 0)
sqft_living = st.sidebar.slider("Living area (sq ft)", 300, 8000, 2000, 50)
sqft_basement = st.sidebar.slider("Basement area (sq ft)", 0, 2500, 0, 50)
sqft_lot = st.sidebar.slider("Lot size (sq ft)", 500, 100000, 7500, 500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.select_slider(
    "Bathrooms",
    options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
             3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.5, 8.0],
    value=2.0,
)
floors = st.sidebar.select_slider("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=1.0)
grade = st.sidebar.slider("Building grade (1–13)", 1, 13, 7,
                           help="King County scale: 7 = average, 10 = very good, 13 = mansion")
condition = st.sidebar.slider("Condition (1–5)", 1, 5, 3)
view = st.sidebar.slider("View quality (0–4)", 0, 4, 0)
waterfront = st.sidebar.checkbox("Waterfront property")
yr_built = st.sidebar.slider("Year built", 1900, 2015, 1990)
renovated = st.sidebar.checkbox("Has been renovated")
yr_renovated = 0
if renovated:
    yr_renovated = st.sidebar.slider("Year renovated", 1950, 2015, 2005)
month_sold = st.sidebar.slider("Month of sale (1–12)", 1, 12, 6)


def build_row():
    meta = zip_meta.get(zipcode, {})
    lat = meta.get("lat", SEATTLE_LAT)
    lon = meta.get("lon", SEATTLE_LON)
    sqft_living15 = meta.get("sqft_living15_med", 1900)
    sqft_lot15 = meta.get("sqft_lot15_med", 7500)

    house_age = 2015 - yr_built
    years_since_reno = (2015 - yr_renovated) if yr_renovated > 0 else house_age
    recently_renovated = int(yr_renovated > 0 and (2015 - yr_renovated) <= 10)
    sqft_above = max(0, sqft_living - sqft_basement)

    row = {f: 0 for f in FEATURES}
    row["bedrooms"] = bedrooms
    row["bathrooms"] = bathrooms
    row["sqft_living"] = sqft_living
    row["sqft_lot"] = sqft_lot
    row["floors"] = floors
    row["waterfront"] = int(waterfront)
    row["view"] = view
    row["condition"] = condition
    row["grade"] = grade
    row["sqft_above"] = sqft_above
    row["sqft_basement"] = sqft_basement
    row["lat"] = lat
    row["long"] = lon
    row["sqft_living15"] = sqft_living15
    row["sqft_lot15"] = sqft_lot15
    row["month_sold"] = month_sold
    row["house_age"] = house_age
    row["years_since_reno"] = years_since_reno
    row["recently_renovated"] = recently_renovated
    row["has_basement"] = int(sqft_basement > 0)
    row["sqft_vs_neighbors"] = sqft_living - sqft_living15
    row["dist_to_seattle"] = np.sqrt((lat - SEATTLE_LAT) ** 2 + (lon - SEATTLE_LON) ** 2)
    row["top_grade"] = int(grade >= 10)
    row["grade_x_sqft"] = grade * sqft_living
    row["basement_ratio"] = sqft_basement / max(sqft_living, 1)
    row["zip_median_price"] = zip_medians.get(zipcode, np.median(list(zip_medians.values())))

    return pd.DataFrame([row])[FEATURES]


if st.sidebar.button("Predict price", type="primary"):
    X_one = build_row()
    raw_pred = model.predict(X_one)[0]
    price = np.expm1(raw_pred) if LOG_TRANSFORM else raw_pred

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Estimated sale price", f"${price:,.0f}")

        st.subheader("What's driving this estimate")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_one)[0]
        baseline_price = np.expm1(explainer.expected_value)
        dollar_impacts = np.expm1(explainer.expected_value + shap_values) - baseline_price

        impact = (
            pd.DataFrame({"feature": FEATURES, "impact": dollar_impacts})
            .assign(abs_impact=lambda d: d["impact"].abs())
            .sort_values("abs_impact", ascending=False)
            .head(8)
        )
        for _, r in impact.iterrows():
            arrow = "▲ pushes up" if r["impact"] > 0 else "▼ pulls down"
            st.write(f"**{r['feature']}** — {arrow} by ${abs(r['impact']):,.0f}")

    with col2:
        st.subheader("Comparable homes")
        nn, scaler, feat_cols, comps_display = load_comps()
        X_comps = X_one.reindex(columns=feat_cols, fill_value=0)
        _, indices = nn.kneighbors(scaler.transform(X_comps))
        comps = comps_display.iloc[indices[0]].copy().reset_index(drop=True)

        comps["Sale Price"] = comps["price"].apply(lambda x: f"${x:,.0f}")
        comps["Renovated"] = comps["yr_renovated"].apply(lambda x: "Yes" if x > 0 else "No")
        comps["Waterfront"] = comps["waterfront"].map({1: "Yes", 0: "No"})
        display_cols = comps[["zipcode", "sqft_living", "grade", "yr_built",
                               "bedrooms", "bathrooms", "Waterfront", "view",
                               "Renovated", "Sale Price"]]
        display_cols.columns = ["Zip", "Living SF", "Grade", "Yr Built",
                                 "Beds", "Baths", "Waterfront", "View",
                                 "Renovated", "Sale Price"]
        st.dataframe(display_cols, use_container_width=True, hide_index=True)

        st.subheader("Map")
        map_df = comps[["lat", "long"]].rename(columns={"long": "lon"})
        input_loc = pd.DataFrame([{"lat": X_one["lat"].values[0], "lon": X_one["long"].values[0]}])
        st.map(pd.concat([map_df, input_loc], ignore_index=True), zoom=10)

else:
    st.info("Set the home details in the sidebar and click **Predict price**.")
