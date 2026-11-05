# 🌾 Ethiopia Agricultural Drought Risk Prediction & Recommendation System

A Streamlit web app that predicts the drought risk category (**No Drought / Moderate Drought / Severe Drought**)
for a location in Ethiopia and recommends suitable crops and farm-management actions.

## How it works
1. Choose a location, soil type and previous crop.
2. Enter rainfall, temperature and humidity.
3. Other indicators (SPI, soil moisture, dry days, etc.) are estimated automatically from similar historical
   records and can be edited under **Advanced**.
4. The app shows the predicted risk, model confidence, class probabilities, recommended crops and actions.

## Model
- Three models were tuned with stratified 5-fold cross-validation (macro F1): Decision Tree, Random Forest, XGBoost.
- The best model (XGBoost) is saved in `best_drought_model.joblib`.
- Features: 14 numeric + 6 categorical columns. `Recommended_Crops` and `Recommended_Actions` are excluded to avoid data leakage.

## Repository files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit application |
| `best_drought_model.joblib` | Trained model, label encoder and feature list |
| `Ethiopia_Drought_Risk_Dataset.xlsx` | Dataset used for dropdown options and indicator estimates |
| `requirements.txt` | Python dependencies (scikit-learn and xgboost versions must match training) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push all files above to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select the repository and branch, set **Main file path** to `app.py`, then **Deploy**.

## Limitations
- Indicators such as SPI, rainfall anomaly, soil moisture, evapotranspiration, dry days and water availability
  in the dataset are **estimated proxies**, not official or measured values.
- The drought label is a **provisional rule-based class** derived from rainfall and these proxies, so the very
  high accuracy mostly reflects how the label was defined (rainfall is ~99% of feature importance).
- Use this project for learning and demonstration, not for real farming or policy decisions, until it is
  validated against official climate data (long-term SPI, satellite NDVI, measured soil moisture).
