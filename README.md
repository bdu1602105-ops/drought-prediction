# Ethiopia Agricultural Drought Risk Prediction and Recommendation System

## Project files
- `app.py` — Streamlit dashboard.
- `drought.py` — model training and evaluation code.
- `best_drought_model.joblib` — model file supplied from Google Colab.
- `requirements.txt` — Python dependencies (scikit-learn is pinned to 1.6.1 to match the saved model).
- `.gitignore` — files excluded from Git.
- `Ethiopia_Drought_Risk_Dataset(1).xlsx` — supplied dataset for retraining (do not publish if it is private).

## Run locally
Use Python 3.11 or 3.12 where possible. From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Upload the project files to one GitHub repository.
2. Select `app.py` as the app's main file.
3. Deploy the app.
4. If the app was already deployed, reboot it after pushing changes.

## Model compatibility note
The uploaded model was serialized with scikit-learn 1.6.1. `requirements.txt` pins that version because loading scikit-learn pickle/joblib files with a different scikit-learn version can fail. The model must also have the expected bundle structure: `model`, `label_encoder`, and `feature_columns`.

## Important
The model artifact has been included in this package, but its loading could not be verified in this working environment because it has scikit-learn 1.6.1 serialization metadata and the available environment has scikit-learn 1.8.0. Install the pinned dependencies before testing. Validate model results and crop advice with Ethiopian agricultural experts before real-world decisions.
