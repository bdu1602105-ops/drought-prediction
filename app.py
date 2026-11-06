
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Ethiopia Drought Risk", page_icon="🌾", layout="wide")
st.title("🌾 Agricultural Drought Risk Prediction & Recommendation System")
st.caption("Ethiopia | Machine-learning decision-support prototype")

MODEL_PATH = Path("artifacts/best_drought_model.joblib")
if not MODEL_PATH.exists():
    st.error("Model file not found. Run the Colab notebook, download best_drought_model.joblib, "
             "and place it inside the artifacts/ folder.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoder = bundle["label_encoder"]
features = bundle["feature_columns"]
st.sidebar.header("Project")
st.sidebar.write("Best model:", bundle.get("best_model_name", "Saved model"))

st.subheader("Drought risk prediction")
st.write("Enter the agricultural and climate observations. Use the same units and categories as the training dataset.")
with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            if feature in ["Rainfall_mm", "Max_Temp_C", "Min_Temp_C", "Humidity_pct",
                           "Evapotranspiration_mm", "Soil_Moisture_pct",
                           "Rainfall_Anomaly_pct", "SPI_3_Month", "NDVI"]:
                values[feature] = st.number_input(feature, value=0.0)
            else:
                values[feature] = st.text_input(feature, value="")
    submitted = st.form_submit_button("Predict drought risk", type="primary")

if submitted:
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in features}], columns=features)
    pred_encoded = model.predict(row)
    risk = encoder.inverse_transform(np.asarray(pred_encoded, dtype=int))[0]
    st.subheader("Prediction")
    if "severe" in str(risk).lower() or "high" in str(risk).lower():
        st.error(f"Predicted risk: {risk}")
        crops = "Drought-tolerant crops such as sorghum, millet or cowpea, subject to local suitability."
        actions = "Conserve soil moisture, use mulch and water harvesting, prioritize available irrigation, and consult extension staff."
    elif "moderate" in str(risk).lower() or "medium" in str(risk).lower():
        st.warning(f"Predicted risk: {risk}")
        crops = "Consider locally suitable drought-tolerant varieties."
        actions = "Monitor rainfall and soil moisture, plant on time, and prepare supplemental irrigation if available."
    else:
        st.success(f"Predicted risk: {risk}")
        crops = "Choose crops based on local soil, season and extension advice."
        actions = "Continue routine monitoring and good soil- and water-conservation practices."
    st.markdown("**Suggested crops**")
    st.write(crops)
    st.markdown("**Suggested actions**")
    st.write(actions)

with st.expander("Model evaluation from Colab"):
    st.json(bundle.get("metrics", []))
st.info("This is a decision-support prototype. Validate predictions and crop recommendations with local "
        "agricultural experts. Performance depends on the quality and representativeness of the training labels.")
