import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 1. Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ethiopia Drought Risk",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agricultural Drought Risk Prediction & Recommendation System")
st.caption("Ethiopia | Machine-learning decision-support prototype")

# ---------------------------------------------------------
# 2. Find and load the trained model
#    Supports BOTH locations:
#    - best_drought_model.joblib
#    - artifacts/best_drought_model.joblib
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "best_drought_model.joblib"

possible_model_paths = [
    BASE_DIR / MODEL_FILENAME,
    BASE_DIR / "artifacts" / MODEL_FILENAME,
]

MODEL_PATH = next(
    (path for path in possible_model_paths if path.is_file()),
    None
)

if MODEL_PATH is None:
    st.error(
        "Model file not found. Upload best_drought_model.joblib to the "
        "same GitHub folder as app.py, or place it inside an artifacts/ folder."
    )
    st.write("Searched locations:")
    for path in possible_model_paths:
        st.code(str(path))
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        st.error(
            "The model file was loaded, but it is not the expected dictionary "
            "bundle. Re-export it from the training notebook with model, "
            "label_encoder, and feature_columns."
        )
        st.stop()

    required_keys = ["model", "label_encoder", "feature_columns"]
    missing_keys = [key for key in required_keys if key not in bundle]

    if missing_keys:
        st.error(f"The saved model bundle is missing required keys: {missing_keys}")
        st.stop()

    model = bundle["model"]
    encoder = bundle["label_encoder"]
    features = bundle["feature_columns"]

    if not features:
        st.error("The saved model bundle contains no feature columns.")
        st.stop()

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ---------------------------------------------------------
# 3. Sidebar: model information
# ---------------------------------------------------------
st.sidebar.header("Project")
st.sidebar.write("Best model:", bundle.get("best_model_name", "Saved model"))
st.sidebar.caption(f"Loaded model file: {MODEL_PATH.name}")

# ---------------------------------------------------------
# 4. Prediction form
# ---------------------------------------------------------
st.subheader("Drought risk prediction")
st.write(
    "Enter the agricultural and climate observations. "
    "Use the same units and categories as the training dataset."
)

numeric_features = {
    "Rainfall_mm",
    "Max_Temp_C",
    "Min_Temp_C",
    "Humidity_pct",
    "Evapotranspiration_mm",
    "Soil_Moisture_pct",
    "Rainfall_Anomaly_pct",
    "SPI_3_Month",
    "NDVI",
}

with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)

    for i, feature in enumerate(features):
        with cols[i % 2]:
            if feature in numeric_features:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.4f"
                )
            else:
                values[feature] = st.text_input(feature, value="")

    submitted = st.form_submit_button(
        "Predict drought risk",
        type="primary"
    )

# ---------------------------------------------------------
# 5. Predict risk and show recommendations
# ---------------------------------------------------------
if submitted:
    try:
        row = pd.DataFrame(
            [{feature: values[feature] for feature in features}],
            columns=features
        )

        pred_encoded = model.predict(row)
        risk = encoder.inverse_transform(
            np.asarray(pred_encoded, dtype=int)
        )[0]

        st.subheader("Prediction")

        risk_text = str(risk).lower()

        if "severe" in risk_text or "high" in risk_text:
            st.error(f"Predicted drought risk: {risk}")
            crops = (
                "Drought-tolerant crops such as sorghum, millet or cowpea, "
                "subject to local suitability."
            )
            actions = (
                "Conserve soil moisture, use mulch and water harvesting, "
                "prioritize available irrigation, and consult extension staff."
            )

        elif "moderate" in risk_text or "medium" in risk_text:
            st.warning(f"Predicted drought risk: {risk}")
            crops = "Consider locally suitable drought-tolerant varieties."
            actions = (
                "Monitor rainfall and soil moisture, plant on time, and "
                "prepare supplemental irrigation if available."
            )

        else:
            st.success(f"Predicted drought risk: {risk}")
            crops = (
                "Choose crops based on local soil, season and extension advice."
            )
            actions = (
                "Continue routine monitoring and good soil- and "
                "water-conservation practices."
            )

        st.markdown("**Suggested crops**")
        st.write(crops)

        st.markdown("**Suggested actions**")
        st.write(actions)

        with st.expander("View input data"):
            st.dataframe(row, use_container_width=True)

    except Exception as error:
        st.error(
            "Prediction failed. Check that the input values, feature names, "
            "data types, and preprocessing match those used during model training."
        )
        st.exception(error)

# ---------------------------------------------------------
# 6. Display model evaluation information (if saved)
# ---------------------------------------------------------
with st.expander("Model evaluation from Colab"):
    metrics = bundle.get("metrics")
    if metrics is not None:
        st.json(metrics)
    else:
        st.info("No evaluation metrics were saved in the model bundle.")

# ---------------------------------------------------------
# 7. Responsible-use notice
# ---------------------------------------------------------
st.info(
    "This is a decision-support prototype. Validate predictions and crop "
    "recommendations with local agricultural experts. Performance depends "
    "on the quality and representativeness of the training labels."
)
