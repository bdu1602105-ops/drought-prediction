# Ethiopia Agricultural Drought Risk Prediction

## 1) Train the model in Google Colab
1. Open `Ethiopia_Drought_Risk_Colab.ipynb` in Google Colab (upload the notebook or open it from GitHub).
2. Run the cells from top to bottom.
3. Upload `Ethiopia_Drought_Risk_Dataset.xlsx` when the notebook asks.
4. At the end, download `best_drought_model.joblib`.

## 2) Upload the deployment project to GitHub
Create a new GitHub repository and upload:
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `artifacts/best_drought_model.joblib` (downloaded from Colab)

The Colab notebook is a separate training artifact and can also be uploaded to the repository root if desired.

## 3) Deploy with Streamlit Community Cloud
1. Make sure `artifacts/best_drought_model.joblib` exists in the GitHub repository.
2. Go to https://share.streamlit.io/ and connect GitHub.
3. Select the repository, branch `main`, and main file path `app.py`.
4. Deploy.

## Dataset assumptions
The notebook expects a target column named `Drought_Risk_Category`. The known dataset schema includes:
`Region`, `Location`, `Agro_Ecological_Zone`, `Soil_Type`, `Previous_Crop`,
`Rainfall_mm`, `Max_Temp_C`, `Min_Temp_C`, `Humidity_pct`,
`Evapotranspiration_mm`, `Climate_Condition`, `Soil_Moisture_pct`,
`Recommended_Crops`, `Recommended_Actions`.
Recommendation columns are excluded from model inputs to avoid target leakage.

## Important scientific note
If drought labels are estimates or were created by rules, model metrics measure how well the model reproduces those labels; they do not prove real-world drought forecasting skill. For future-season forecasting, use time-indexed historical observations and define a forecast horizon. Recommendations are general guidance and should be validated for the local agro-ecological zone.
