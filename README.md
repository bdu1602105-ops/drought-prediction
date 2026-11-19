# Agricultural Drought Risk Prediction and Recommendation System in Ethiopia

A professional multi-page Streamlit dashboard for drought-risk prediction, agricultural recommendations, regional analysis, and model insights.

## 1. Repository files

```text
Ethiopia_Drought_Streamlit_GitHub/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── drought_model_outputs/
│   └── best_drought_model.joblib   # Add your trained file here
└── Ethiopia_Drought_Risk_Dataset(2).xlsx
```

## 2. Important: add your trained model

The trained model artifact is not included in this starter package because it must be produced by running your training notebook/script. In Google Colab, run the improved training code to create:

`drought_model_outputs/best_drought_model.joblib`

Download that exact file and upload it into the `drought_model_outputs/` folder in this GitHub repository. Do not rename it unless you also update `MODEL_PATH` in `app.py`.

The saved bundle is expected to contain:
- `model`: fitted preprocessing + classifier pipeline
- `model_name`: selected model name
- `label_encoder`: XGBoost label encoder, or `None`
- `feature_columns`: exact ordered input columns
- `target_column`: target column name

## 3. (Optional) Train or retrain the model

The repository also includes `train_model.py`, the improved training script. Run it from the repository root after installing the dependencies:

```bash
python train_model.py
```

It reads `Ethiopia_Drought_Risk_Dataset(2).xlsx` and saves the model bundle under `drought_model_outputs/best_drought_model.joblib`. For the first training run, allow time for the hyperparameter searches to finish.

## 4. Run locally

Install Python 3.10 or newer, then run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5. Upload to GitHub

1. Create a new GitHub repository.
2. Upload all files and folders in this project.
3. Add `drought_model_outputs/best_drought_model.joblib`.
4. Confirm that `app.py`, `requirements.txt`, the model artifact, and the dataset workbook are committed.


### Dependency compatibility

The `requirements.txt` file includes the packages used by both the Streamlit
dashboard and `train_model.py`. It uses compatible version ranges rather than
pinning one exact patch release. For the most predictable deployment, use
Python 3.11 in the Streamlit Community Cloud app settings.

Streamlit Community Cloud installs these packages automatically from
`requirements.txt` during deployment. Do not install packages manually in
`app.py`.

## 6. Deploy with Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Sign in with GitHub.
3. Select **Create app**.
4. Choose your repository and branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

If deployment fails, open **Manage app → Logs** and inspect the error.

## 7. Notes about the model

- The app reads the feature names and fitted model pipeline from the saved bundle, so its inputs follow the model's actual training columns.
- `Recommended_Crops` and `Recommended_Actions` are not model input features. Recommendations are looked up from the workbook after prediction.
- The dataset's `Column_Methodology` sheet describes methodology/estimated fields. If drought labels or input indicators are provisional or rule-derived, model performance measures agreement with those labels, not independently validated future drought forecasting.
- Do not use the app as the sole basis for high-impact agricultural decisions without validating it against independent observed drought data.

## 8. Troubleshooting

**“The trained model file is missing”**  
Upload `best_drought_model.joblib` to `drought_model_outputs/`.

**“Could not load the model artifact”**  
Check that the artifact was created by the same training script and that package versions are compatible.

**“Prediction failed”**  
Check that the saved bundle contains `model`, `feature_columns`, and (for XGBoost) `label_encoder`, and that the input feature names match the training features.
