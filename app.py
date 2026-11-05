import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Ethiopia Agricultural Drought Risk",
    page_icon="🌾",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "best_drought_model.joblib"

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.title("🌾 Agricultural Drought Risk Prediction")
st.caption("Machine-learning-based drought risk assessment and agricultural recommendations for Ethiopia")

bundle = load_bundle()

if bundle is None:
    st.error("The trained model file is missing: best_drought_model.joblib")
    st.markdown(
        "The app code is running, but the trained model has not been uploaded to GitHub yet. "
        "Run **drought.py** with the correct Excel dataset. The repaired training script now "
        "automatically copies the finished model beside `app.py`."
    )
    st.code("app.py\ndrought.py\nbest_drought_model.joblib\nrequirements.txt\nREADME.md\n.gitignore", language="text")
    st.info("After training finishes, commit and push best_drought_model.joblib to the same GitHub repository as app.py, then redeploy/reboot the Streamlit app.")
    st.stop()

model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]

# Recover the numeric and categorical feature names from the fitted preprocessing pipeline.
preprocessor = model.named_steps["prep"]
numeric_columns = list(preprocessor.transformers_[0][2])
categorical_columns = list(preprocessor.transformers_[1][2])

st.sidebar.header("About")
st.sidebar.write(
    "Enter the agricultural and climate conditions for one location. "
    "The application uses the trained model to estimate the drought-risk category."
)
st.sidebar.info("The result is a model estimate, not an official drought warning.")

st.subheader("1. Enter location and agricultural conditions")
with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)

    for index, col in enumerate(feature_columns):
        container = left if index % 2 == 0 else right
        with container:
            if col in numeric_columns:
                # Use broad, neutral defaults; replace with locally appropriate ranges if known.
                defaults = {
                    "Rainfall_mm": 500.0,
                    "Temperature_C": 25.0,
                    "Humidity_pct": 50.0,
                    "Soil_Moisture_pct": 30.0,
                    "Water_Availability_Index": 50.0,
                    "NDVI": 0.5,
                    "SPI_3_Month": 0.0,
                    "SPI_6_Month": 0.0,
                    "Rainfall_Anomaly_pct": 0.0,
                }
                default = defaults.get(col, 0.0)
                values[col] = st.number_input(
                    col.replace("_", " "),
                    value=float(default),
                    help="Enter a value in the same unit and range used in the training dataset."
                )
            else:
                options = {
                    "Agro_Ecological_Zone": ["Bereha", "Kolla", "Woina Dega", "Dega"],
                    "Soil_Type": ["Leptosols", "Vertisols", "Nitisols", "Cambisols", "Other"],
                    "Previous_Crop": [
                        "Wheat", "Barley", "Teff", "Maize", "Sorghum",
                        "Faba bean", "Chickpea", "Lentil", "Pulses", "Fallow", "Other"
                    ],
                    "Season": ["Meher", "Belg", "Other"],
                }
                if col in options:
                    values[col] = st.selectbox(col.replace("_", " "), options[col])
                else:
                    values[col] = st.text_input(
                        col.replace("_", " "),
                        placeholder="Enter value as represented in the training data"
                    )

    submitted = st.form_submit_button("Predict Drought Risk", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{col: values[col] for col in feature_columns}])
    try:
        prediction_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([int(prediction_encoded)])[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities.max())

        st.subheader("2. Prediction result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted drought risk", prediction)
        with col2:
            st.metric("Highest model probability", f"{confidence:.1%}")

        st.caption(
            "Probability is the model's estimated class probability; it is not a guarantee "
            "that the prediction is correct."
        )

        probability_table = pd.DataFrame({
            "Drought category": label_encoder.classes_,
            "Model probability": probabilities
        }).sort_values("Model probability", ascending=False)
        st.bar_chart(probability_table.set_index("Drought category"))

        advice = {
            "No Drought": (
                "Conditions are currently classified as no drought. Plant according to the "
                "local agricultural calendar, monitor soil moisture and rainfall, and conserve water."
            ),
            "Moderate Drought": (
                "Consider locally suitable, shorter-duration or drought-tolerant crops. "
                "Use mulching, conserve rainwater, and monitor fields regularly."
            ),
            "Severe Drought": (
                "Seek advice from the local agricultural extension office. Prioritize water "
                "conservation, locally validated drought-tolerant crops, and livestock feed planning."
            ),
        }
        st.subheader("3. Agricultural action recommendation")
        st.info(advice.get(prediction, "Consult a local agricultural expert for location-specific advice."))

        crops_by_zone = {
            "Bereha": ["Sorghum", "Pearl millet", "Cowpea", "Sesame"],
            "Kolla": ["Sorghum", "Pearl millet", "Mung bean", "Cowpea", "Sesame"],
            "Woina Dega": ["Teff", "Chickpea", "Sorghum", "Lentil", "Field pea"],
            "Dega": ["Barley", "Field pea", "Lentil", "Faba bean", "Linseed"],
        }
        zone = values.get("Agro_Ecological_Zone")
        if zone in crops_by_zone:
            st.subheader("4. Potential crop options")
            st.write(", ".join(crops_by_zone[zone][:3]))
            st.caption(
                "These are preliminary options only. Confirm suitability using local soil, "
                "season, seed availability, forecast rainfall and agricultural extension advice."
            )

        st.download_button(
            "Download prediction result (CSV)",
            data=pd.DataFrame([{
                **values,
                "Predicted_Drought_Risk": prediction,
                "Prediction_Probability": confidence
            }]).to_csv(index=False).encode("utf-8"),
            file_name="drought_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.exception(error)
        st.warning(
            "Check that all inputs use the same column names, data types, units and category "
            "spellings as the training dataset. The model and scikit-learn versions must also match."
        )

st.divider()
st.caption("Research prototype. Validate predictions and recommendations with reliable local data and agricultural experts.")
