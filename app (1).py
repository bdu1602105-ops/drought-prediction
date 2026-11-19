# -*- coding: utf-8 -*-
"""
Professional Streamlit dashboard
Agricultural Drought Risk Prediction and Recommendation System in Ethiopia

Run locally:
    streamlit run app.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Agricultural Drought Risk Prediction and Recommendation System in Ethiopia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "drought_model_outputs"

# Your uploaded GitHub screenshot shows the model beside app.py.
# The second path is retained as a fallback if you later move it into the folder.
MODEL_CANDIDATES = [
    ROOT / "best_drought_model.joblib",
    OUTPUT_DIR / "best_drought_model.joblib",
    ROOT / "best_drought_model (1).joblib",
]

DATA_CANDIDATES = [
    ROOT / "Ethiopia_Drought_Risk_Dataset(2).xlsx",
    ROOT / "Ethiopia_Drought_Risk_Dataset.xlsx",
    ROOT / "Ethiopia_Agricultural_Drought_Risk_Dataset.xlsx",
    OUTPUT_DIR / "Ethiopia_Drought_Risk_Dataset(2).xlsx",
]

METRICS_CANDIDATES = [OUTPUT_DIR / "model_comparison.csv", ROOT / "model_comparison.csv"]
IMPORTANCE_CANDIDATES = [OUTPUT_DIR / "feature_importance.csv", ROOT / "feature_importance.csv"]
METADATA_CANDIDATES = [OUTPUT_DIR / "model_metadata.json", ROOT / "model_metadata.json"]
TARGET = "Drought_Risk_Category"
CROP_COL = "Recommended_Crops"
ACTION_COL = "Recommended_Actions"

# ---------------------------------------------------------------------
# 2. PROFESSIONAL THEME
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp {background:#f4f7f4;}
    [data-testid="stHeader"] {background:rgba(244,247,244,.92);}
    [data-testid="stSidebar"] {background:#163b2e;}
    [data-testid="stSidebar"] * {color:#f1f7f2 !important;}
    .hero {background:linear-gradient(115deg,#173d30,#236448 62%,#4d8a57);padding:1.8rem 2rem;border-radius:20px;color:white;margin:.2rem 0 1.3rem;box-shadow:0 10px 28px rgba(20,55,40,.12);}
    .hero h1 {font-family:Manrope,sans-serif;color:white;font-size:clamp(1.55rem,3vw,2.35rem);font-weight:800;margin:0 0 .5rem;}
    .hero p {color:#e3efe6;margin:0;font-size:1rem;}
    .section-title {font-family:Manrope,sans-serif;color:#173d30;font-size:1.25rem;font-weight:800;margin:.8rem 0;}
    .metric {background:white;border:1px solid #e1e9e2;border-radius:15px;padding:1rem 1.1rem;min-height:112px;box-shadow:0 3px 14px rgba(25,55,38,.04);}
    .metric-label {color:#64756b;font-size:.82rem;font-weight:600;}
    .metric-value {color:#173d30;font-family:Manrope,sans-serif;font-size:1.7rem;font-weight:800;margin-top:.3rem;}
    .result {background:white;border:1px solid #e1e9e2;border-radius:16px;padding:1.25rem;}
    div[data-testid="stForm"] {background:white;border:1px solid #e1e9e2;border-radius:16px;padding:1.1rem 1.2rem;}
    .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:#236448;border:0;border-radius:10px;font-weight:700;min-height:2.8rem;}
    .footer {color:#78867d;text-align:center;font-size:.78rem;padding:1.4rem 0 .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 3. FILE DISCOVERY AND LOADING
# ---------------------------------------------------------------------
def first_existing(paths):
    return next((p for p in paths if p.exists() and p.is_file()), None)

@st.cache_resource(show_spinner="Loading the trained drought model...")
def load_model(model_path, modified_time):
    # The training script provided by the user saves best_model directly
    # with joblib.dump(best_model, model_path). Therefore it is a raw sklearn
    # Pipeline, not necessarily a dictionary/model bundle.
    return joblib.load(model_path)

@st.cache_data(show_spinner=False)
def load_table(path_string, modified_time):
    path = Path(path_string)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)

def read_json(path):
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_csv_optional(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

model_path = first_existing(MODEL_CANDIDATES)
data_path = first_existing(DATA_CANDIDATES)
metadata_path = first_existing(METADATA_CANDIDATES)
metrics_path = first_existing(METRICS_CANDIDATES)
importance_path = first_existing(IMPORTANCE_CANDIDATES)

model = None
model_error = None
if model_path:
    try:
        model = load_model(str(model_path), model_path.stat().st_mtime)
    except Exception as exc:
        model_error = str(exc)

dataset = None
if data_path:
    try:
        dataset = load_table(str(data_path), data_path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Dataset could not be read: {exc}")

metadata = read_json(metadata_path)
metrics_df = read_csv_optional(metrics_path)
importance_df = read_csv_optional(importance_path)

# Accept either the raw fitted Pipeline from the user's training script,
# or a dictionary bundle if a future training script exports one.
if isinstance(model, dict):
    model_bundle = model
    estimator = model_bundle.get("model") or model_bundle.get("pipeline")
    feature_columns = model_bundle.get("feature_columns") or metadata.get("feature_columns", [])
    model_label_encoder = model_bundle.get("label_encoder")
    model_name = model_bundle.get("model_name", metadata.get("best_model_name", "Saved classifier"))
else:
    model_bundle = {}
    estimator = model
    model_label_encoder = None
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns and estimator is not None:
        feature_columns = list(getattr(estimator, "feature_names_in_", []))
    model_name = metadata.get("best_model_name", "Saved trained pipeline")

if estimator is not None and not feature_columns:
    feature_columns = list(getattr(estimator, "feature_names_in_", []))

# ---------------------------------------------------------------------
# 4. REUSABLE HELPERS
# ---------------------------------------------------------------------
def pretty_name(value):
    return str(value).replace("_", " ").strip().title()

def show_metric(label, value, note=""):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div style="color:#718078;font-size:.78rem">{note}</div></div>',
        unsafe_allow_html=True,
    )

def risk_style(label):
    text = str(label).lower()
    if "severe" in text:
        return "#b42318", "#fff0ee"
    if "moderate" in text:
        return "#a66a12", "#fff6e5"
    return "#23734d", "#eaf6ee"

def feature_is_numeric(feature, data):
    if data is not None and feature in data.columns:
        return pd.api.types.is_numeric_dtype(data[feature])
    # Fallback when the workbook is not uploaded. These names are numeric
    # in the agricultural dataset described in the project.
    numeric_hints = ("rain", "temp", "humidity", "evapo", "moisture", "spi", "ndvi", "anomaly", "wind", "solar")
    return any(hint in feature.lower() for hint in numeric_hints) or feature.lower().endswith(("_mm", "_c", "_pct"))

def create_prediction_inputs(features, data):
    result = {}
    for feature in features:
        label = pretty_name(feature)
        series = data[feature] if data is not None and feature in data.columns else None
        if feature_is_numeric(feature, data):
            numeric = pd.to_numeric(series, errors="coerce").dropna() if series is not None else pd.Series(dtype=float)
            default = float(numeric.median()) if not numeric.empty else 0.0
            low = float(numeric.min()) if not numeric.empty else None
            high = float(numeric.max()) if not numeric.empty else None
            help_text = f"Dataset median: {default:.2f}"
            if low is not None and high is not None:
                help_text += f" · observed range: {low:.2f}–{high:.2f}"
            result[feature] = st.number_input(label, value=default, step=0.1, format="%.3f", help=help_text)
        else:
            options = []
            if series is not None:
                options = sorted(series.dropna().astype(str).str.strip().unique().tolist())
            if options:
                result[feature] = st.selectbox(label, options=options)
            else:
                result[feature] = st.text_input(label, placeholder=f"Enter {label.lower()}")
    return result

def decode_output(raw_prediction):
    value = raw_prediction[0]
    # XGBoost was trained on LabelEncoder integer labels in the supplied
    # training script. The metadata contains the original sorted class names.
    classes = metadata.get("classes", [])
    # The supplied training code saves XGBoost as a raw pipeline and saves
    # the original labels in metadata. If that optional file was not uploaded,
    # use the known alphabetical order of the project's three target labels.
    if not classes:
        classes = ["Moderate Drought", "No Drought", "Severe Drought"]
    if model_label_encoder is not None:
        return str(model_label_encoder.inverse_transform(np.asarray([int(value)]))[0])
    if isinstance(value, (int, np.integer)) and classes:
        index = int(value)
        if 0 <= index < len(classes):
            return str(classes[index])
    return str(value)

def get_probability_frame(input_frame):
    if estimator is None or not hasattr(estimator, "predict_proba"):
        return None
    try:
        probabilities = np.asarray(estimator.predict_proba(input_frame))[0]
        classes = getattr(estimator, "classes_", np.arange(len(probabilities)))
        class_names = []
        saved_classes = metadata.get("classes", []) or ["Moderate Drought", "No Drought", "Severe Drought"]
        for cls in classes:
            if isinstance(cls, (int, np.integer)) and saved_classes and 0 <= int(cls) < len(saved_classes):
                class_names.append(str(saved_classes[int(cls)]))
            else:
                class_names.append(str(cls))
        return pd.DataFrame({"Risk category": class_names, "Probability": probabilities}).sort_values("Probability", ascending=False)
    except Exception:
        return None

def recommendation_lookup(data, predicted_risk, inputs):
    if data is None or TARGET not in data.columns:
        return None, None, 0
    mask = data[TARGET].astype(str).str.casefold() == str(predicted_risk).casefold()
    subset = data.loc[mask].copy()
    for col in ("Region", "Agro_Ecological_Zone", "Soil_Type", "Previous_Crop"):
        if col in subset.columns and col in inputs and str(inputs[col]).strip():
            narrowed = subset[subset[col].astype(str).str.casefold() == str(inputs[col]).casefold()]
            if not narrowed.empty:
                subset = narrowed
    def mode(col):
        if col not in subset.columns:
            return None
        vals = subset[col].dropna().astype(str).str.strip()
        vals = vals[vals.ne("")]
        return vals.mode().iloc[0] if not vals.empty else None
    return mode(CROP_COL), mode(ACTION_COL), len(subset)

# ---------------------------------------------------------------------
# 5. SIDEBAR AND HEADER
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🌾 Drought Intelligence")
    st.caption("ETHIOPIA · AGRICULTURAL DECISION SUPPORT")
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Drought Prediction", "Regional Analysis", "Model Insights", "About"])
    st.divider()
    if estimator is not None:
        st.success("Trained model loaded")
        st.caption(model_name)
    else:
        st.warning("Model not loaded")
    st.caption("Decision-support prototype · Validate outputs with local observations")

st.markdown(
    '<div class="hero"><h1>🌾 Agricultural Drought Risk Prediction and Recommendation System in Ethiopia</h1>'
    '<p>Assess agricultural drought-risk categories and explore context-aware recommendations.</p></div>',
    unsafe_allow_html=True,
)

if model_error:
    st.error("The model file was found but could not be loaded. This is often caused by a scikit-learn version mismatch between training and deployment.")
    st.code(model_error)
    st.info("The uploaded model was serialized with scikit-learn 1.6.1. Use scikit-learn==1.6.1 in requirements.txt, then reboot the Streamlit app.")
elif estimator is None:
    st.warning("Trained model not found. Put best_drought_model.joblib in the same GitHub folder as app.py, or in drought_model_outputs/.")

# ---------------------------------------------------------------------
# 6. OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="section-title">National overview</div>', unsafe_allow_html=True)
    if dataset is not None:
        total = len(dataset)
        region_count = dataset["Region"].nunique() if "Region" in dataset else "—"
        location_count = dataset["Location"].nunique() if "Location" in dataset else "—"
        risk_count = dataset[TARGET].nunique() if TARGET in dataset else "—"
        a, b, c, d = st.columns(4)
        with a: show_metric("Dataset records", f"{total:,}", "Rows in the supplied workbook")
        with b: show_metric("Regions", region_count, "Distinct regions")
        with c: show_metric("Locations", location_count, "Distinct locations")
        with d: show_metric("Risk categories", risk_count, "Supplied target classes")
        if TARGET in dataset.columns:
            left, right = st.columns(2)
            with left:
                st.markdown("**Drought-risk distribution**")
                counts = dataset[TARGET].astype(str).value_counts().rename_axis("Risk category").reset_index(name="Records")
                st.plotly_chart(px.pie(counts, names="Risk category", values="Records", hole=.55), use_container_width=True)
            with right:
                if "Region" in dataset.columns:
                    st.markdown("**Records by region**")
                    regional = dataset["Region"].astype(str).value_counts().rename_axis("Region").reset_index(name="Records")
                    st.plotly_chart(px.bar(regional, x="Records", y="Region", orientation="h"), use_container_width=True)
        with st.expander("Preview source dataset"):
            st.dataframe(dataset.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("The dataset workbook is optional for model prediction. Upload the Excel workbook to enable overview statistics and dataset-based recommendations.")
    x, y, z = st.columns(3)
    with x: show_metric("Prediction model", "Ready" if estimator is not None else "Unavailable", model_name)
    with y: show_metric("Model input features", len(feature_columns) if feature_columns else "—", "Features expected by the saved pipeline")
    with z: show_metric("Recommendation source", "Dataset lookup", "Not a separately trained recommender")

# ---------------------------------------------------------------------
# 7. PREDICTION INTERFACE
# ---------------------------------------------------------------------
elif page == "Drought Prediction":
    st.markdown('<div class="section-title">Drought-risk assessment</div>', unsafe_allow_html=True)
    st.write("Enter the climate, soil and farming information. The app sends the inputs to the saved preprocessing-and-classification pipeline.")
    if estimator is None:
        st.error("Prediction is unavailable until the trained model loads successfully.")
    elif not feature_columns:
        st.error("The saved model does not expose its input feature names. Please re-export the fitted pipeline together with feature_columns metadata.")
    else:
        with st.form("drought_prediction_form"):
            st.markdown("**Enter assessment conditions**")
            cols = st.columns(2)
            input_values = {}
            for index, feature in enumerate(feature_columns):
                with cols[index % 2]:
                    input_values[feature] = None
            # Use separate grouped display while retaining exact feature names.
            groups = {
                "Location and farming context": [],
                "Weather and climate": [],
                "Soil conditions": [],
                "Other model inputs": [],
            }
            for feature in feature_columns:
                low = feature.lower()
                if any(k in low for k in ("region", "location", "zone", "agro", "soil_type", "previous_crop")):
                    groups["Location and farming context"].append(feature)
                elif any(k in low for k in ("rain", "temp", "humidity", "climate", "evapo", "precip", "spi", "ndvi")):
                    groups["Weather and climate"].append(feature)
                elif "soil" in low or "moisture" in low:
                    groups["Soil conditions"].append(feature)
                else:
                    groups["Other model inputs"].append(feature)
            for group, features in groups.items():
                if features:
                    st.markdown(f"**{group}**")
                    group_cols = st.columns(2)
                    for i, feature in enumerate(features):
                        with group_cols[i % 2]:
                            input_values[feature] = create_prediction_inputs([feature], dataset)[feature]
            submitted = st.form_submit_button("🔍 Predict drought risk", type="primary", use_container_width=True)
        if submitted:
            input_frame = pd.DataFrame([input_values], columns=feature_columns)
            try:
                raw_prediction = estimator.predict(input_frame)
                predicted_risk = decode_output(raw_prediction)
                probs = get_probability_frame(input_frame)
                color, pale = risk_style(predicted_risk)
                st.markdown("---")
                st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
                r1, r2 = st.columns([1.2, 1])
                with r1:
                    st.markdown(
                        f'<div class="result"><div style="color:#66766d;font-size:.8rem;font-weight:700">PREDICTED DROUGHT-RISK CATEGORY</div>'
                        f'<div style="font-size:2rem;font-weight:800;color:{color};padding:.4rem 0">{predicted_risk}</div>'
                        f'<span style="background:{pale};color:{color};padding:.35rem .7rem;border-radius:30px;font-weight:700">Model output</span></div>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    if probs is not None and not probs.empty:
                        show_metric("Highest class probability", f"{probs.iloc[0]['Probability']:.1%}", "Model probability; not a guarantee")
                    else:
                        show_metric("Probability", "Unavailable", "Classifier has no usable predict_proba output")
                if probs is not None and not probs.empty:
                    st.markdown("**Probability by drought-risk category**")
                    chart = px.bar(probs.sort_values("Probability"), x="Probability", y="Risk category", orientation="h", text=probs.sort_values("Probability")["Probability"].map(lambda v: f"{v:.1%}"))
                    chart.update_xaxes(tickformat=".0%", range=[0, 1])
                    chart.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10, l=5, r=5))
                    st.plotly_chart(chart, use_container_width=True)
                st.markdown('<div class="section-title">Agricultural recommendations</div>', unsafe_allow_html=True)
                crops, actions, n_matches = recommendation_lookup(dataset, predicted_risk, input_values)
                rec1, rec2 = st.columns(2)
                with rec1:
                    st.markdown("**🌱 Recommended crops**")
                    st.info(crops or "No crop recommendation available. Add the source Excel workbook to enable recommendation lookup.")
                with rec2:
                    st.markdown("**💧 Recommended agricultural actions**")
                    st.info(actions or "No action recommendation available. Add the source Excel workbook to enable recommendation lookup.")
                if dataset is not None:
                    st.caption(f"Recommendations retrieved from {n_matches:,} matching dataset record(s). They are dataset lookups, not outputs of a separately trained recommendation model.")
                report = input_frame.copy()
                report["Predicted_Drought_Risk"] = predicted_risk
                if probs is not None:
                    for _, row in probs.iterrows():
                        report["Probability_" + str(row["Risk category"]).replace(" ", "_")] = row["Probability"]
                report["Recommended_Crops"] = crops or ""
                report["Recommended_Actions"] = actions or ""
                st.download_button("⬇ Download prediction report (CSV)", report.to_csv(index=False).encode("utf-8-sig"), "drought_risk_prediction.csv", "text/csv")
                with st.expander("Review model input values"):
                    st.dataframe(input_frame, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error("Prediction could not be completed. Check that the deployed scikit-learn version matches the version used to train the model and that all input features match the saved pipeline.")
                st.exception(exc)

# ---------------------------------------------------------------------
# 8. REGIONAL ANALYSIS
# ---------------------------------------------------------------------
elif page == "Regional Analysis":
    st.markdown('<div class="section-title">Regional drought-risk analysis</div>', unsafe_allow_html=True)
    if dataset is None:
        st.info("Upload the source Excel dataset beside app.py to enable regional analysis.")
    elif "Region" not in dataset.columns or TARGET not in dataset.columns:
        st.warning("The workbook needs Region and Drought_Risk_Category columns.")
    else:
        regions = ["All regions"] + sorted(dataset["Region"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Select region", regions)
        filtered = dataset if selected == "All regions" else dataset[dataset["Region"].astype(str) == selected]
        left, right = st.columns(2)
        with left:
            counts = filtered[TARGET].astype(str).value_counts().rename_axis("Risk category").reset_index(name="Records")
            st.plotly_chart(px.bar(counts, x="Risk category", y="Records", color="Risk category"), use_container_width=True)
        with right:
            composition = filtered[TARGET].astype(str).value_counts(normalize=True).mul(100).rename_axis("Risk category").reset_index(name="Percent")
            st.plotly_chart(px.pie(composition, names="Risk category", values="Percent", hole=.55), use_container_width=True)
        if "Location" in filtered.columns:
            st.markdown("**Location-level summary**")
            st.dataframe(pd.crosstab(filtered["Location"], filtered[TARGET]), use_container_width=True)

# ---------------------------------------------------------------------
# 9. MODEL INSIGHTS
# ---------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model performance and explainability</div>', unsafe_allow_html=True)
    st.write(f"**Loaded model:** {model_name if estimator is not None else 'Not loaded'}")
    if metrics_df is not None and not metrics_df.empty:
        st.markdown("**Model comparison**")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        metric_col = next((c for c in metrics_df.columns if c.lower() in {"macro f1", "macro_f1"}), None)
        name_col = next((c for c in metrics_df.columns if c.lower() == "model"), None)
        if metric_col and name_col:
            st.plotly_chart(px.bar(metrics_df, x=name_col, y=metric_col, title="Model comparison · Macro F1"), use_container_width=True)
    else:
        st.info("Model comparison CSV is optional. Upload model_comparison.csv to show training evaluation results.")
    if importance_df is not None and not importance_df.empty:
        feature_col = next((c for c in ("Feature", "feature", "Feature Name") if c in importance_df.columns), importance_df.columns[0])
        score_col = next((c for c in ("Importance_Mean", "Importance", "importance", "Mean Importance") if c in importance_df.columns), importance_df.columns[-1])
        chart_data = importance_df[[feature_col, score_col]].copy()
        chart_data[score_col] = pd.to_numeric(chart_data[score_col], errors="coerce")
        chart_data = chart_data.dropna().sort_values(score_col).tail(15)
        st.plotly_chart(px.bar(chart_data, x=score_col, y=feature_col, orientation="h", title="Top feature importance"), use_container_width=True)
    else:
        st.info("Feature importance CSV is optional. Upload feature_importance.csv to display it.")
    if metadata:
        with st.expander("Model metadata"):
            st.json(metadata)
    if estimator is not None:
        st.caption(f"Input features expected by the model: {len(feature_columns)}")

# ---------------------------------------------------------------------
# 10. ABOUT AND LIMITATIONS
# ---------------------------------------------------------------------
elif page == "About":
    st.markdown('<div class="section-title">About this system</div>', unsafe_allow_html=True)
    st.write("This application uses a fitted machine-learning pipeline to classify drought-risk categories from agricultural and climate inputs.")
    st.markdown("**Main components**")
    st.markdown("- Saved preprocessing and classification pipeline\n- Drought-risk prediction with class probabilities when available\n- Dataset-based crop and agricultural action recommendations\n- Regional analysis and model evaluation outputs")
    st.warning("Scientific limitation: the supplied dataset methodology describes some indicators as estimates/proxies and drought labels as provisional or rule-based. This application predicts those supplied labels; it is not a verified operational drought early-warning system until independently validated against observed drought records and future-period data.")
    st.markdown("**Required deployment files**")
    st.code("app.py\nrequirements.txt\nbest_drought_model.joblib\nEthiopia_Drought_Risk_Dataset(2).xlsx (optional for recommendations and regional analysis)\nmodel_metadata.json (recommended)\nmodel_comparison.csv (optional)\nfeature_importance.csv (optional)", language="text")

st.markdown('<div class="footer">Agricultural Drought Risk Prediction and Recommendation System in Ethiopia · Machine-learning decision support</div>', unsafe_allow_html=True)
