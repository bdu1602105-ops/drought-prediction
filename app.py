"""Ethiopia Agricultural Drought Risk Prediction & Recommendation (Streamlit)."""
import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ethiopia Drought Risk", page_icon="🌾", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_drought_model.joblib")
SHEET_NAME = "Enhanced_Dataset"

ADVICE = {
    "No Drought": "Conditions are good. Plant on time and look after your crops as usual. Keep checking the rain and soil moisture, and save some water in case dry days come.",
    "Moderate Drought": "Some drought is coming. Plant crops that need less water and ripen early. Cover the soil with straw or mulch to keep it moist. Save rain water and use it carefully. Check your fields every week.",
    "Severe Drought": "Serious drought. Plant only crops that survive dry weather and avoid crops that need a lot of water. Store water and animal feed now. Move animals early before they get weak. Ask your local agriculture office for seed, food or livestock support.",
}
CROPS = {  # (normal crops, drought-tolerant crops) per agro-ecological zone
    "Bereha": (["Sorghum", "Pearl millet", "Cowpea", "Sesame"], ["Pearl millet", "Sorghum", "Cowpea", "Mung bean"]),
    "Kolla": (["Sorghum", "Maize", "Haricot bean", "Sesame", "Groundnut"], ["Sorghum", "Pearl millet", "Mung bean", "Cowpea", "Sesame"]),
    "Woina Dega": (["Teff", "Maize", "Wheat", "Chickpea", "Faba bean"], ["Teff", "Chickpea", "Sorghum", "Lentil", "Field pea"]),
    "Dega": (["Barley", "Wheat", "Faba bean", "Field pea", "Potato"], ["Barley", "Field pea", "Lentil", "Faba bean", "Linseed"]),
}
LEGUMES = {"Faba bean", "Field pea", "Chickpea", "Lentil", "Haricot bean", "Cowpea", "Mung bean", "Groundnut"}
COLORS = {"No Drought": "#2e7d32", "Moderate Drought": "#ef6c00", "Severe Drought": "#c62828"}
PROXY_NUM = ["Evapotranspiration_mm", "Rainfall_Anomaly_pct", "SPI_3_Month", "SPI_6_Month",
             "Soil_Moisture_pct", "Consecutive_Dry_Days", "Water_Availability_Index", "Grain_Yield_kg_ha"]


def recommend_crops(row, category):
    stressed = category != "No Drought" or row["Water_Availability_Index"] < 40 or row["Soil_Moisture_pct"] < 20
    lst = list(CROPS[row["Agro_Ecological_Zone"]][1 if stressed else 0])
    if row["Soil_Type"] == "Leptosols":
        lst = [c for c in lst if c not in ("Potato", "Maize", "Faba bean")] or lst
    if row["Soil_Type"] == "Vertisols" and row["Rainfall_mm"] > 900:
        lst = [c for c in lst if c not in ("Haricot bean", "Potato")] or lst
    prev = str(row["Previous_Crop"]).lower()
    lst = [c for c in lst if c.lower() != prev] or lst
    if prev in ("wheat", "barley", "teff", "maize"):
        lst.sort(key=lambda c: 0 if c in LEGUMES else 1)
    elif prev in ("pulses", "fallow"):
        lst.sort(key=lambda c: 1 if c in LEGUMES else 0)
    return lst[:3]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    files = glob.glob(os.path.join(HERE, "*.xlsx"))
    if not files:
        return None
    return pd.read_excel(files[0], sheet_name=SHEET_NAME)


def mode(s):
    return s.mode().iloc[0]


def estimate_indicators(df, location, rainfall):
    """Estimate the indicators that depend on rainfall from the 12 most similar
    historical rows (same location if possible, ranked by closeness of rainfall)."""
    pool = df[df["Location"] == location]
    if len(pool) < 12:
        pool = df
    near = pool.iloc[(pool["Rainfall_mm"] - rainfall).abs().argsort()[:12]]
    est = {c: float(near[c].median()) for c in PROXY_NUM}
    est["Climate_Condition"] = mode(near["Climate_Condition"])
    return est


# ----------------------------------------------------------------- UI
st.title("🌾 Ethiopia Agricultural Drought Risk Prediction")
st.caption("Predicts the drought risk category for a location and gives crop and farm-management advice.")

try:
    bundle = load_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model file. Check that scikit-learn / xgboost versions match requirements.txt.\n\n{e}")
    st.stop()

model, le, feature_cols = bundle["model"], bundle["label_encoder"], bundle["feature_columns"]
data = load_data()
if data is None:
    st.error("Dataset (.xlsx) not found next to app.py.")
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Location")
    location = st.selectbox("Location", sorted(data["Location"].unique()),
                            index=sorted(data["Location"].unique()).index("Bahir Dar") if "Bahir Dar" in data["Location"].values else 0)
    loc_rows = data[data["Location"] == location]
    region = mode(loc_rows["Region"])
    zone = mode(loc_rows["Agro_Ecological_Zone"])
    altitude = int(loc_rows["Altitude_m"].median())
    c1, c2, c3 = st.columns(3)
    c1.metric("Region", region)
    c2.metric("Zone", zone)
    c3.metric("Altitude (m)", altitude)

    soils = sorted(data["Soil_Type"].unique())
    soil = st.selectbox("Soil type", soils, index=soils.index(mode(loc_rows["Soil_Type"])))
    crops_prev = sorted(data["Previous_Crop"].unique())
    prev_crop = st.selectbox("Previous crop", crops_prev)

    st.subheader("2. Weather")
    rain = st.slider("Rainfall (mm)", 100, 2000, int(loc_rows["Rainfall_mm"].median()), step=5)
    max_t = st.slider("Max temperature (°C)", 10.0, 45.0, float(round(loc_rows["Max_Temp_C"].median(), 1)), step=0.1)
    min_t = st.slider("Min temperature (°C)", 0.0, 30.0, float(round(loc_rows["Min_Temp_C"].median(), 1)), step=0.1)
    hum = st.slider("Humidity (%)", 15.0, 95.0, float(round(loc_rows["Humidity_pct"].median(), 1)), step=0.1)

est = estimate_indicators(data, location, rain)
with st.expander("Advanced: other drought indicators (auto-estimated from rainfall, editable)"):
    k = f"{location}_{rain}"
    a1, a2 = st.columns(2)
    spi3 = a1.number_input("SPI 3-month", -3.0, 3.5, round(est["SPI_3_Month"], 2), 0.01, key=f"spi3_{k}")
    spi6 = a2.number_input("SPI 6-month", -3.0, 3.5, round(est["SPI_6_Month"], 2), 0.01, key=f"spi6_{k}")
    anom = a1.number_input("Rainfall anomaly (%)", -100.0, 200.0, round(est["Rainfall_Anomaly_pct"], 1), 0.5, key=f"an_{k}")
    soil_m = a2.number_input("Soil moisture (%)", 0.0, 60.0, round(est["Soil_Moisture_pct"], 1), 0.5, key=f"sm_{k}")
    dry = a1.number_input("Consecutive dry days", 0, 60, int(round(est["Consecutive_Dry_Days"])), 1, key=f"dd_{k}")
    wai = a2.number_input("Water availability index", 0.0, 100.0, round(est["Water_Availability_Index"], 1), 0.5, key=f"wai_{k}")
    et = a1.number_input("Evapotranspiration (mm)", 1.0, 12.0, round(est["Evapotranspiration_mm"], 2), 0.01, key=f"et_{k}")
    yld = a2.number_input("Expected grain yield (kg/ha)", 0, 6000, int(round(est["Grain_Yield_kg_ha"])), 10, key=f"y_{k}")
    clim_opts = ["Dry", "Normal", "Wet"]
    climate = st.selectbox("Climate condition", clim_opts, index=clim_opts.index(est["Climate_Condition"]), key=f"cl_{k}")

with right:
    st.subheader("3. Result")
    if min_t >= max_t:
        st.warning("Minimum temperature must be lower than maximum temperature.")
        st.stop()

    row = {
        "Region": region, "Location": location, "Altitude_m": altitude, "Agro_Ecological_Zone": zone,
        "Soil_Type": soil, "Previous_Crop": prev_crop, "Rainfall_mm": rain, "Max_Temp_C": max_t,
        "Min_Temp_C": min_t, "Temperature_Range_C": round(max_t - min_t, 1), "Humidity_pct": hum,
        "Evapotranspiration_mm": et, "Climate_Condition": climate, "Rainfall_Anomaly_pct": anom,
        "SPI_3_Month": spi3, "SPI_6_Month": spi6, "Soil_Moisture_pct": soil_m,
        "Consecutive_Dry_Days": dry, "Water_Availability_Index": wai, "Grain_Yield_kg_ha": yld,
    }
    X = pd.DataFrame([row])[feature_cols]
    proba = model.predict_proba(X)[0]
    category = le.inverse_transform([int(np.argmax(proba))])[0]
    conf = float(proba.max())

    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{COLORS[category]}1a;"
        f"border-left:8px solid {COLORS[category]}'>"
        f"<div style='font-size:14px;color:#555'>Predicted drought risk</div>"
        f"<div style='font-size:32px;font-weight:700;color:{COLORS[category]}'>{category}</div>"
        f"<div style='font-size:14px;color:#555'>Confidence: {conf:.1%}</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Class probabilities**")
    st.bar_chart(pd.Series(proba, index=le.classes_, name="Probability"))

    st.markdown("**🌱 Recommended crops**")
    st.success(", ".join(recommend_crops(pd.Series(row), category)))
    st.markdown("**📋 Recommended actions**")
    st.info(ADVICE[category])

st.divider()
st.caption(
    "⚠️ Educational prototype. The training dataset uses estimated proxy indicators (SPI, soil moisture, "
    "dry days, etc.) and a provisional rule-based drought label, not official or measured values. "
    "Do not use for real farming or policy decisions without validation against official data."
)
