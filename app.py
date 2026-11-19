# -*- coding: utf-8 -*-
"""
Professional Streamlit dashboard
Agricultural Drought Risk Prediction and Recommendation System in Ethiopia

Run:
    streamlit run app.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION AND THEME
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Ethiopia Drought Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "drought_model_outputs"
MODEL_PATH = OUTPUT_DIR / "best_drought_model.joblib"
DATA_PATH = ROOT / "Ethiopia_Drought_Risk_Dataset(2).xlsx"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

TARGET = "Drought_Risk_Category"
CROP_COL = "Recommended_Crops"
ACTION_COL = "Recommended_Actions"

# ---------------------------------------------------------------------
# 2. CUSTOM DASHBOARD STYLING
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f7f4; }
      [data-testid="stHeader"] { background: rgba(245,247,244,.92); }
      [data-testid="stSidebar"] { background: #142b24; }
      [data-testid="stSidebar"] * { color: #edf5ef !important; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .45rem .2rem; }
      .hero {
        background: linear-gradient(115deg, #173d30 0%, #236448 60%, #4d8a57 100%);
        padding: 1.7rem 2rem; border-radius: 18px; color: white;
        margin: .2rem 0 1.3rem 0; box-shadow: 0 8px 24px rgba(20,55,40,.12);
      }
      .hero h1 { color: white; font-family: Manrope,sans-serif; font-size: 2rem; margin: 0 0 .35rem 0; }
      .hero p { color: #e1eee5; margin: 0; font-size: .98rem; }
      .section-title { color: #173d30; font-family: Manrope,sans-serif; font-weight: 800; font-size: 1.2rem; margin: .5rem 0 .8rem 0; }
      .small-muted { color: #64756b; font-size: .85rem; }
      .metric-card {
        background: white; border: 1px solid #e3eae4; border-radius: 14px;
        padding: 1rem 1.1rem; min-height: 105px; box-shadow: 0 3px 12px rgba(25,55,38,.035);
      }
      .metric-label { color: #64756b; font-size: .82rem; font-weight: 600; }
      .metric-value { color: #173d30; font-family: Manrope,sans-serif; font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
      .result-card { background: white; border: 1px solid #e1e9e2; border-radius: 16px; padding: 1.2rem 1.4rem; }
      .risk-pill { display:inline-block; padding:.35rem .75rem; border-radius:999px; font-weight:700; font-size:.85rem; }
      .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
        background: #236448; border: 0; border-radius: 10px; min-height: 2.8rem; font-weight: 700;
      }
      .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover { background:#173d30; }
      div[data-testid="stForm"] { background:white; padding:1.15rem 1.25rem; border:1px solid #e3eae4; border-radius:15px; }
      div[data-testid="stTabs"] button { font-weight: 700; }
      .footer { color:#758278; font-size:.78rem; text-align:center; padding:1.3rem 0 .4rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 3. DATA AND MODEL LOADING
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_bundle(path_string, modified_time):
    return joblib.load(path_string)

@st.cache_data
def load_dataset(path_string, modified_time):
    path = Path(path_string)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)

def load_optional_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def load_optional_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

dataset = None
if DATA_PATH.exists():
    try:
        dataset = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Dataset could not be opened: {exc}")

bundle = None
model_error = None
if MODEL_PATH.exists():
    try:
        bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        model_error = str(exc)

metrics_df = load_optional_csv(METRICS_PATH)
importance_df = load_optional_csv(IMPORTANCE_PATH)
metadata = load_optional_json(METADATA_PATH)

# ---------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def pretty_name(value):
    return str(value).replace("_", " ").strip().title()

def risk_color(risk):
    value = str(risk).casefold()
    if "severe" in value:
        return "#b42318", "#fff0ee"
    if "moderate" in value:
        return "#b7791f", "#fff7e6"
    return "#23734d", "#eaf6ee"

def metric_card(label, value, footnote=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="small-muted">{footnote}</div></div>""",
        unsafe_allow_html=True,
    )

def decode_predictions(raw_prediction, model_bundle):
    encoder = model_bundle.get("label_encoder")
    if encoder is not None:
        return encoder.inverse_transform(np.asarray(raw_prediction, dtype=int)).astype(str)
    return np.asarray(raw_prediction).astype(str)

def get_probabilities(model, input_frame, model_bundle):
    if not hasattr(model, "predict_proba"):
        return None
    raw_probabilities = model.predict_proba(input_frame)[0]
    encoder = model_bundle.get("label_encoder")
    if encoder is not None:
        labels = encoder.classes_.astype(str)
    else:
        labels = getattr(model, "classes_", np.arange(len(raw_probabilities))).astype(str)
    return pd.DataFrame({"Risk category": labels, "Probability": raw_probabilities}).sort_values(
        "Probability", ascending=False
    )

def make_recommendations(data, predicted_risk, input_values):
    if data is None or TARGET not in data.columns:
        return None, None, 0
    subset = data[data[TARGET].astype(str).str.casefold() == str(predicted_risk).casefold()].copy()
    # Use matching context only where the dataset has matching records.
    for column in ["Region", "Agro_Ecological_Zone", "Soil_Type"]:
        if column in subset.columns and column in input_values:
            selected = input_values[column]
            if selected is not None and str(selected).strip():
                narrowed = subset[subset[column].astype(str) == str(selected)]
                if not narrowed.empty:
                    subset = narrowed
    def mode_value(column):
        if column not in subset.columns:
            return None
        values = subset[column].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        return values.mode().iloc[0] if not values.empty else None
    return mode_value(CROP_COL), mode_value(ACTION_COL), len(subset)

def grouped_features(columns):
    groups = {
        "Location & farming context": [],
        "Weather & climate": [],
        "Soil conditions": [],
        "Other model inputs": [],
    }
    for col in columns:
        low = col.casefold()
        if any(k in low for k in ["region", "location", "zone", "agro", "soil_type", "previous_crop"]):
            groups["Location & farming context"].append(col)
        elif any(k in low for k in ["rain", "temp", "humidity", "climate", "evapo", "wind", "precip", "spi", "ndvi"]):
            groups["Weather & climate"].append(col)
        elif "soil" in low or "moisture" in low:
            groups["Soil conditions"].append(col)
        else:
            groups["Other model inputs"].append(col)
    return {k: v for k, v in groups.items() if v}

def create_input_form(features, data):
    values = {}
    group_items = grouped_features(features)
    for group_name, group_columns in group_items.items():
        st.markdown(f"**{group_name}**")
        cols = st.columns(2)
        for index, feature in enumerate(group_columns):
            target_col = cols[index % 2]
            series = data[feature] if data is not None and feature in data.columns else None
            label = pretty_name(feature)
            if series is not None and pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                default = float(numeric.median()) if not numeric.empty else 0.0
                if feature.casefold().endswith("_pct"):
                    step = 0.1
                elif "rainfall" in feature.casefold() or "evapo" in feature.casefold():
                    step = 1.0
                else:
                    step = 0.1
                values[feature] = target_col.number_input(
                    label, value=default, step=step, format="%.2f",
                    help=f"Dataset median: {default:.2f}" if not numeric.empty else "Enter a numeric value.",
                )
            elif series is not None:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if options:
                    values[feature] = target_col.selectbox(label, options, key=f"input_{feature}")
                else:
                    values[feature] = target_col.text_input(label, key=f"input_{feature}")
            else:
                values[feature] = target_col.text_input(label, key=f"input_{feature}")
        st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    return values

# ---------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION AND BRANDING
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 Drought Intelligence")
    st.caption("ETHIOPIA · AGRICULTURAL DECISION SUPPORT")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Drought Prediction", "Regional Analysis", "Model Insights", "About"],
        label_visibility="visible",
    )
    st.divider()
    if bundle:
        st.markdown("**Active model**")
        st.success(str(bundle.get("model_name", "Trained classifier")))
        st.caption(f"Target: {bundle.get('target_column', TARGET)}")
    else:
        st.warning("Trained model not found")
    st.markdown("---")
    st.caption("Decision-support prototype · Verify outputs with local observations")

# ---------------------------------------------------------------------
# 6. HERO HEADER
# ---------------------------------------------------------------------
st.markdown(
    """<div class="hero">
      <h1>🌾 Agricultural Drought Risk Intelligence</h1>
      <p>Predict drought-risk categories and explore agricultural recommendations across Ethiopia.</p>
    </div>""",
    unsafe_allow_html=True,
)

if model_error:
    st.error(f"Model loading error: {model_error}")
if bundle is None:
    st.warning(
        "The trained model is not available. Add "
        "`drought_model_outputs/best_drought_model.joblib` to activate prediction."
    )

# ---------------------------------------------------------------------
# 7. OVERVIEW PAGE
# ---------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="section-title">National overview</div>', unsafe_allow_html=True)
    if dataset is not None:
        total_records = len(dataset)
        regions = dataset["Region"].nunique() if "Region" in dataset.columns else "—"
        locations = dataset["Location"].nunique() if "Location" in dataset.columns else "—"
        categories = dataset[TARGET].nunique() if TARGET in dataset.columns else "—"
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Dataset records", f"{total_records:,}", "Rows in supplied workbook")
        with m2: metric_card("Regions represented", f"{regions}", "Unique region values")
        with m3: metric_card("Locations", f"{locations}", "Unique location values")
        with m4: metric_card("Risk categories", f"{categories}", "Existing target classes")
        st.markdown("")
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("**Drought-risk category distribution**")
            if TARGET in dataset.columns:
                counts = dataset[TARGET].astype(str).value_counts().rename_axis("Risk category").reset_index(name="Records")
                fig = px.pie(counts, names="Risk category", values="Records", hole=0.58)
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title="")
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("**Records by region**")
            if "Region" in dataset.columns:
                regional = dataset["Region"].astype(str).value_counts().reset_index()
                regional.columns = ["Region", "Records"]
                fig = px.bar(regional, x="Records", y="Region", orientation="h", text="Records")
                fig.update_layout(margin=dict(t=10, b=10, l=5, r=10), yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Dataset preview**")
        st.dataframe(dataset.head(8), use_container_width=True, hide_index=True)
    else:
        st.info("Place the Excel workbook beside app.py to enable dataset summaries and recommendations.")

    st.markdown('<div class="section-title">System workflow</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**01 · Enter conditions**")
        st.write("Provide location, farming context, rainfall, temperature, humidity and other model inputs.")
    with c2:
        st.markdown("**02 · Predict risk**")
        st.write("The saved preprocessing pipeline and trained classifier estimate a drought-risk category.")
    with c3:
        st.markdown("**03 · Review guidance**")
        st.write("Explore crop and farm-management suggestions retrieved from the supplied dataset.")

# ---------------------------------------------------------------------
# 8. PREDICTION PAGE
# ---------------------------------------------------------------------
elif page == "Drought Prediction":
    st.markdown('<div class="section-title">Drought-risk assessment</div>', unsafe_allow_html=True)
    st.write("Enter the available agricultural and climate conditions, then run the trained model.")
    if bundle is None:
        st.error("Prediction is disabled until the trained model bundle is uploaded.")
    else:
        feature_columns = bundle.get("feature_columns", [])
        if not feature_columns:
            st.error("The model bundle does not contain feature_columns.")
        else:
            with st.form("drought_prediction_form"):
                input_values = create_input_form(feature_columns, dataset)
                submitted = st.form_submit_button("🔍  Predict drought risk", type="primary", use_container_width=True)
            if submitted:
                input_frame = pd.DataFrame([input_values], columns=feature_columns)
                try:
                    model = bundle["model"]
                    raw_prediction = model.predict(input_frame)
                    predicted_risk = decode_predictions(raw_prediction, bundle)[0]
                    probabilities = get_probabilities(model, input_frame, bundle)
                    color, pale = risk_color(predicted_risk)

                    st.markdown("---")
                    st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
                    result_col, confidence_col = st.columns([1.25, 1])
                    with result_col:
                        st.markdown(
                            f"""<div class="result-card">
                            <div class="small-muted">PREDICTED DROUGHT-RISK CATEGORY</div>
                            <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;color:{color};padding:.35rem 0">{predicted_risk}</div>
                            <span class="risk-pill" style="background:{pale};color:{color}">Model prediction</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with confidence_col:
                        if probabilities is not None:
                            top_probability = float(probabilities.iloc[0]["Probability"])
                            metric_card("Highest class probability", f"{top_probability:.1%}", "Model probability for the predicted class")
                        else:
                            metric_card("Prediction confidence", "Not available", "Classifier does not expose predict_proba")

                    if probabilities is not None:
                        st.markdown("**Probability by risk category**")
                        prob_fig = px.bar(
                            probabilities.sort_values("Probability"),
                            x="Probability", y="Risk category", orientation="h",
                            text=probabilities.sort_values("Probability")["Probability"].map(lambda x: f"{x:.1%}"),
                        )
                        prob_fig.update_xaxes(tickformat=".0%", range=[0, 1])
                        prob_fig.update_layout(height=260, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
                        st.plotly_chart(prob_fig, use_container_width=True)

                    st.markdown('<div class="section-title">Agricultural recommendations</div>', unsafe_allow_html=True)
                    crops, actions, matched_records = make_recommendations(dataset, predicted_risk, input_values)
                    rec1, rec2 = st.columns(2)
                    with rec1:
                        st.markdown("**🌱 Recommended crops**")
                        st.info(crops if crops else "No crop recommendation found for this risk category and context.")
                    with rec2:
                        st.markdown("**💧 Recommended farm actions**")
                        st.info(actions if actions else "No action recommendation found for this risk category and context.")
                    st.caption(
                        f"Recommendation lookup used {matched_records:,} matching dataset record(s). "
                        "These recommendations are retrieved from the workbook, not produced by a separately trained recommendation model."
                    )

                    report = input_frame.copy()
                    report["Predicted_Drought_Risk"] = predicted_risk
                    if probabilities is not None:
                        for _, row in probabilities.iterrows():
                            safe_class = str(row["Risk category"]).replace(" ", "_")
                            report[f"Probability_{safe_class}"] = float(row["Probability"])
                    report["Recommended_Crops"] = crops or ""
                    report["Recommended_Actions"] = actions or ""
                    st.download_button(
                        "⬇ Download prediction report (CSV)",
                        data=report.to_csv(index=False).encode("utf-8-sig"),
                        file_name="drought_risk_prediction.csv",
                        mime="text/csv",
                    )
                    with st.expander("Review input values sent to the model"):
                        st.dataframe(input_frame, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.exception(exc)

# ---------------------------------------------------------------------
# 9. REGIONAL ANALYSIS PAGE
# ---------------------------------------------------------------------
elif page == "Regional Analysis":
    st.markdown('<div class="section-title">Regional drought-risk patterns</div>', unsafe_allow_html=True)
    if dataset is None:
        st.info("Upload the workbook beside app.py to activate regional analysis.")
    elif "Region" not in dataset.columns or TARGET not in dataset.columns:
        st.warning("The workbook needs Region and Drought_Risk_Category columns for this analysis.")
    else:
        regions = ["All regions"] + sorted(dataset["Region"].dropna().astype(str).unique().tolist())
        selected_region = st.selectbox("Select region", regions)
        filtered = dataset.copy()
        if selected_region != "All regions":
            filtered = filtered[filtered["Region"].astype(str) == selected_region]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Risk-category counts**")
            counts = filtered[TARGET].astype(str).value_counts().reset_index()
            counts.columns = ["Risk category", "Records"]
            fig = px.bar(counts, x="Risk category", y="Records", color="Risk category", text="Records")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Risk-category composition**")
            composition = filtered[TARGET].astype(str).value_counts(normalize=True).mul(100).reset_index()
            composition.columns = ["Risk category", "Percentage"]
            fig = px.pie(composition, names="Risk category", values="Percentage", hole=.55)
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        if "Location" in filtered.columns:
            st.markdown("**Location-level records**")
            location_summary = pd.crosstab(filtered["Location"], filtered[TARGET])
            st.dataframe(location_summary, use_container_width=True)

# ---------------------------------------------------------------------
# 10. MODEL INSIGHTS PAGE
# ---------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model performance and explainability</div>', unsafe_allow_html=True)
    if bundle:
        st.write(f"**Selected classifier:** {bundle.get('model_name', 'Saved model')}")
        st.write(f"**Prediction target:** {bundle.get('target_column', TARGET)}")
    if metrics_df is not None and not metrics_df.empty:
        st.markdown("**Model comparison from training outputs**")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison table not found. Add model_comparison.csv to drought_model_outputs.")
    if importance_df is not None and not importance_df.empty:
        st.markdown("**Feature importance**")
        feature_col = next((c for c in ["Feature", "feature", "Feature Name"] if c in importance_df.columns), importance_df.columns[0])
        score_col = next((c for c in ["Importance_Mean", "Importance", "importance", "Mean Importance", "Permutation Importance"] if c in importance_df.columns), importance_df.columns[-1])
        plot_data = importance_df[[feature_col, score_col]].copy()
        plot_data[score_col] = pd.to_numeric(plot_data[score_col], errors="coerce")
        plot_data = plot_data.dropna().sort_values(score_col).tail(15)
        fig = px.bar(plot_data, x=score_col, y=feature_col, orientation="h")
        fig.update_layout(height=max(350, 28 * len(plot_data)), margin=dict(t=10, b=10, l=5, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Interpret the importance values according to the method saved by the training script; they do not prove causation.")
    else:
        st.info("Feature-importance file not found. Add feature_importance.csv to drought_model_outputs.")
    if metadata:
        with st.expander("Training metadata"):
            st.json(metadata)

# ---------------------------------------------------------------------
# 11. ABOUT AND SCIENTIFIC LIMITATIONS
# ---------------------------------------------------------------------
elif page == "About":
    st.markdown('<div class="section-title">About this decision-support system</div>', unsafe_allow_html=True)
    st.write(
        "This application uses a previously trained classification pipeline to estimate "
        "the drought-risk category associated with the supplied agricultural and climate inputs."
    )
    st.markdown("**System components**")
    st.markdown(
        "- Data preparation and preprocessing saved with the trained model\n"
        "- Drought-risk classification using the selected Decision Tree, Random Forest or XGBoost model\n"
        "- Dataset-based crop and farm-action recommendations\n"
        "- Regional summaries and model evaluation outputs, when those files are available"
    )
    st.warning(
        "**Scientific limitation:** The supplied workbook's methodology notes describe some "
        "indicators as estimates/proxies and the drought-risk categories as provisional or "
        "rule-based. Consequently, this application predicts the labels provided in that "
        "dataset. It should not be presented as a verified operational drought early-warning "
        "system until validated against independent observed drought records and future-period data."
    )
    st.markdown("**Files required for full functionality**")
    st.code(
        "app.py\n"
        "requirements.txt\n"
        "Ethiopia_Drought_Risk_Dataset(2).xlsx\n"
        "drought_model_outputs/best_drought_model.joblib\n"
        "drought_model_outputs/model_comparison.csv        (optional)\n"
        "drought_model_outputs/feature_importance.csv      (optional)\n"
        "drought_model_outputs/model_metadata.json         (optional)",
        language="text",
    )

st.markdown(
    '<div class="footer">Agricultural Drought Risk Intelligence · Ethiopia · '
    'Machine-learning decision support</div>',
    unsafe_allow_html=True,
)
