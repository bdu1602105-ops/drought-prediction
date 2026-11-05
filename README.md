# Agricultural Drought Risk Prediction and Recommendation System in Ethiopia

A Streamlit research prototype that loads a trained scikit-learn pipeline and predicts one of three drought-risk categories:
- No Drought
- Moderate Drought
- Severe Drought

It also displays model probabilities and general agricultural action guidance.

## Repository files

- `app.py` — Streamlit user interface and prediction workflow.
- `best_drought_model.joblib` — trained model bundle (must be generated from your dataset).
- `requirements.txt` — Python dependencies.
- `drought.py` — your model-training script (keep this in the repository if you want training code available).
- `.gitignore` — files excluded from Git.

## Important: create the trained model first

The model file cannot be generated from the Python source alone. You need the actual Excel dataset used for training.

1. Put your dataset in the location/name expected by `drought.py`.
2. Run `drought.py` in an environment with its dependencies installed.
3. Confirm that it creates `drought_outputs/best_drought_model.joblib`.
4. Copy that file to the repository root and rename it `best_drought_model.joblib`.
5. Make sure the saved bundle contains `model`, `label_encoder`, and `feature_columns`, as in the training script.

The training script and deployed app must use compatible versions of scikit-learn and XGBoost. Ideally pin the exact versions used during training in `requirements.txt`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

## Important research limitations

- Predictions are only as reliable as the dataset, target labels, and validation method.
- If drought categories were derived from SPI or rainfall anomaly, assess data leakage before reporting model performance.
- Input values and category labels must match the training data's columns, units, and spelling.
- Crop suggestions are general options, not a substitute for locally validated agronomic recommendations.
- Do not present model probabilities as guaranteed accuracy or as official drought alerts.
