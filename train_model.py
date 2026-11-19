# -*- coding: utf-8 -*-
"""Agricultural Drought Risk Prediction and Recommendation System in Ethiopia.

Improved from the user's uploaded drought (2).py.
IMPORTANT: this model learns the drought-risk labels already supplied in the
workbook. If those labels are provisional/rule-based estimates, test scores
measure agreement with those labels, not independently validated future drought
forecasting performance.
"""

# ============================================================
# 1. IMPORT IMPORTANT LIBRARIES
# ============================================================
import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print("Core libraries imported successfully.")

# ============================================================
# 2. DEFINE PROJECT SETTINGS AND VARIABLES
# ============================================================
DATA_FILE = "Ethiopia_Drought_Risk_Dataset(2).xlsx"  # Change if your file has another name.
TARGET_COLUMN = "Drought_Risk_Category"
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 12
MODEL_OUTPUT_DIR = "drought_model_outputs"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
EXCLUDED_FEATURE_COLUMNS = ["Recommended_Crops", "Recommended_Actions"]
print("Project settings defined. Dataset:", DATA_FILE)

# ============================================================
# 3. LOAD THE DATASET
# ============================================================
def locate_dataset(file_path):
    """Find the workbook or ask for upload when running in Google Colab."""
    if Path(file_path).exists():
        return file_path
    alternatives = [
        "Ethiopia_Drought_Risk_Dataset.xlsx",
        "Ethiopia_Drought_Risk_Dataset(1).xlsx",
        "Ethiopia_Drought_Risk_Dataset(2).xlsx",
        "Ethiopia_Agricultural_Drought_Risk_Dataset.xlsx",
    ]
    for name in alternatives:
        if Path(name).exists():
            print("Using dataset:", name)
            return name
    try:
        from google.colab import files
        print("Upload the Ethiopia drought-risk Excel workbook.")
        uploaded = files.upload()
        excel_files = [n for n in uploaded if n.lower().endswith((".xlsx", ".xls"))]
        if not excel_files:
            raise FileNotFoundError("No Excel workbook was uploaded.")
        return excel_files[0]
    except ImportError as exc:
        raise FileNotFoundError(
            f"Cannot find {file_path}. Put the workbook beside this script or update DATA_FILE."
        ) from exc

DATA_FILE = locate_dataset(DATA_FILE)
df = pd.read_excel(DATA_FILE)
if df.empty:
    raise ValueError("The selected Excel worksheet contains no records.")
print("Loaded:", DATA_FILE, "| Shape:", df.shape)
print(df.head())

# ============================================================
# 4. CHECK AND CLEAN THE DATASET
# ============================================================
df.columns = df.columns.astype(str).str.strip()
print("Columns:", df.columns.tolist())
print("Missing values before cleaning:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()

# Strip text fields and convert numeric-looking columns only when conversion
# succeeds for at least 90% of non-empty values.
text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
for col in df.columns:
    if col == TARGET_COLUMN:
        continue
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        numeric_version = pd.to_numeric(df[col], errors="coerce")
        nonmissing = int(df[col].notna().sum())
        if nonmissing and numeric_version.notna().sum() / nonmissing >= 0.90:
            df[col] = numeric_version
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Required target '{TARGET_COLUMN}' not found. Columns: {df.columns.tolist()}")
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Normalize target label capitalization and common spelling variants.
normalization = {
    "no drought": "No Drought", "no-drought": "No Drought", "nodrought": "No Drought",
    "moderate drought": "Moderate Drought", "moderate-drought": "Moderate Drought",
    "severe drought": "Severe Drought", "severe-drought": "Severe Drought",
}
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("string").str.strip().str.casefold()
df[TARGET_COLUMN] = df[TARGET_COLUMN].map(lambda v: normalization.get(v, v.title()))
df = df.dropna(subset=[TARGET_COLUMN]).copy()
if df[TARGET_COLUMN].nunique() < 2:
    raise ValueError("At least two drought-risk classes are required.")
print("Cleaned shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())
print("Target counts:\n", df[TARGET_COLUMN].value_counts())
print("Target percentages:\n", (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))
print("CAUTION: target labels are those supplied in the workbook; verify their source and quality.")

# ============================================================
# 5. VALIDATE THE EXISTING DROUGHT RISK TARGET
# ============================================================
# The workbook already contains Drought_Risk_Category. This script uses that
# target rather than inventing labels or deriving them from the predictors.
print("Using existing target:", TARGET_COLUMN)

# ============================================================
# 6. SELECT INPUT FEATURES (X) AND TARGET (y)
# ============================================================
excluded = [TARGET_COLUMN] + EXCLUDED_FEATURE_COLUMNS
feature_columns = [c for c in df.columns if c not in excluded]
X = df[feature_columns].copy()
y = df[TARGET_COLUMN].astype(str).copy()
empty_features = X.columns[X.isna().all()].tolist()
if empty_features:
    print("Dropping all-empty columns:", empty_features)
    X = X.drop(columns=empty_features)
constant_features = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if constant_features:
    print("Dropping constant columns:", constant_features)
    X = X.drop(columns=constant_features)
if X.empty or X.shape[1] == 0:
    raise ValueError("No usable predictor columns remain.")
all_missing_rows = X.isna().all(axis=1)
if all_missing_rows.any():
    X = X.loc[~all_missing_rows].copy()
    y = y.loc[X.index].copy()
    df = df.loc[X.index].copy()
feature_columns = X.columns.tolist()
print("Features:", feature_columns)
print("X:", X.shape, "| y:", y.shape)
# Never include the target or recommendation output columns in model inputs.

# ============================================================
# 7. PREPROCESS NUMERICAL AND CATEGORICAL DATA
# ============================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
transformers = []
if numeric_features:
    transformers.append(("numeric", numeric_pipe, numeric_features))
if categorical_features:
    transformers.append(("categorical", categorical_pipe, categorical_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ============================================================
# 8. SPLIT THE DATASET INTO TRAINING AND TESTING
# ============================================================
from sklearn.model_selection import train_test_split
if y.value_counts().min() < CV_FOLDS:
    raise ValueError(
        f"Smallest class has {y.value_counts().min()} records, fewer than CV_FOLDS={CV_FOLDS}. "
        "Reduce CV_FOLDS so every class can appear in each validation fold."
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Train classes:\n", y_train.value_counts())
print("Test classes:\n", y_test.value_counts())
if y_train.value_counts().min() < CV_FOLDS:
    raise ValueError(
        f"After the train/test split, the smallest training class has "
        f"{y_train.value_counts().min()} records, fewer than CV_FOLDS={CV_FOLDS}. "
        "Reduce CV_FOLDS or increase the dataset size."
    )

# ============================================================
# 9. DEFINE THE DECISION TREE MODEL
# ============================================================
from sklearn.tree import DecisionTreeClassifier
decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
])
print("Decision Tree pipeline defined.")

# ============================================================
# 10. DEFINE THE RANDOM FOREST MODEL
# ============================================================
from sklearn.ensemble import RandomForestClassifier
random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )),
])
print("Random Forest pipeline defined.")

# ============================================================
# 11. INSTALL AND DEFINE THE XGBOOST MODEL
# ============================================================
# If missing, install before running: pip install xgboost
# In a separate Colab cell: %pip install -q xgboost
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Install XGBoost first: pip install xgboost") from exc
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
unknown_labels = set(y_test.unique()) - set(label_encoder.classes_)
if unknown_labels:
    raise ValueError(f"Test labels missing from training labels: {unknown_labels}")
xgboost_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        objective="multi:softprob", num_class=len(label_encoder.classes_),
        eval_metric="mlogloss", tree_method="hist", random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
print("XGBoost pipeline defined. Classes:", list(label_encoder.classes_))

# ============================================================
# 12. TRAIN AND TUNE THE THREE MODELS
# ============================================================
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
search_spaces = {
    "Decision Tree": (decision_tree_pipeline, {
        "model__max_depth": [None, 4, 6, 10, 15, 20],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__criterion": ["gini", "entropy", "log_loss"],
    }, y_train),
    "Random Forest": (random_forest_pipeline, {
        "model__n_estimators": randint(100, 351),
        "model__max_depth": [None, 8, 12, 18, 25],
        "model__min_samples_split": randint(2, 15),
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": ["sqrt", "log2"],
    }, y_train),
    "XGBoost": (xgboost_pipeline, {
        "model__n_estimators": randint(100, 351),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": randint(1, 8),
    }, y_train_encoded),
}
search_results, best_models = {}, {}
for model_name, (pipeline, parameter_space, y_fit) in search_spaces.items():
    print(f"\n--- Tuning {model_name} ---")
    search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=parameter_space,
        n_iter=N_ITER_SEARCH, scoring="f1_macro", cv=cv_strategy,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1, refit=True,
        error_score="raise",
    )
    fit_parameters = {}
    if model_name == "XGBoost":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_encoded)
        fit_parameters["model__sample_weight"] = sample_weights
    search.fit(X_train, y_fit, **fit_parameters)
    search_results[model_name] = search
    best_models[model_name] = search.best_estimator_
    print("Best CV Macro F1:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)
print("All models tuned.")

# ============================================================
# 13. EVALUATE MODEL PERFORMANCE ON THE HELD-OUT TEST SET
# ============================================================
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
# IMPORTANT FIX: select the final model using training CV Macro F1 only.
# The held-out test set is not used to select/tune the final model.
cv_scores = {name: float(search.best_score_) for name, search in search_results.items()}
best_model_name = max(cv_scores, key=cv_scores.get)
best_model = best_models[best_model_name]
print("Selected using training CV Macro F1:", best_model_name, cv_scores[best_model_name])

evaluation_rows, predictions = [], {}
for model_name, model in best_models.items():
    if model_name == "XGBoost":
        encoded_prediction = model.predict(X_test).astype(int)
        y_pred = label_encoder.inverse_transform(encoded_prediction)
    else:
        y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
    evaluation_rows.append({
        "Model": model_name,
        "CV Macro F1": cv_scores[model_name],
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    })
    print(f"\n--- {model_name}: test classification report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

evaluation_df = pd.DataFrame(evaluation_rows).sort_values(
    "CV Macro F1", ascending=False
).reset_index(drop=True)
best_predictions = predictions[best_model_name]
print("Model comparison:\n", evaluation_df.round(4).to_string(index=False))
if "Severe Drought" in set(y_test):
    severe_recall = recall_score(
        (y_test == "Severe Drought").astype(int),
        (best_predictions == "Severe Drought").astype(int), zero_division=0,
    )
    print("Selected model Severe Drought recall:", round(severe_recall, 4))
print("Review Macro F1, balanced accuracy, and severe-class recall, not accuracy alone.")

# ============================================================
# 14. COMPARE THE THREE MODELS
# ============================================================
# evaluation_df contains held-out test metrics for comparison. The selected
# model remains the model with the highest training-CV Macro F1.
print("Final selected model:", best_model_name)

# ============================================================
# 15. CALCULATE PERMUTATION FEATURE IMPORTANCE
# ============================================================
from sklearn.inspection import permutation_importance
importance_result = permutation_importance(
    best_model, X_test, y_test, scoring="f1_macro", n_repeats=10,
    random_state=RANDOM_STATE, n_jobs=-1,
)
feature_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": importance_result.importances_mean,
    "Importance_Std": importance_result.importances_std,
}).sort_values("Importance_Mean", ascending=False)
print("Permutation feature importance:\n", feature_importance_df.to_string(index=False))

# ============================================================
# 16. CREATE DROUGHT RISK PREDICTIONS
# ============================================================
def predict_drought_risk(input_data):
    """Predict original drought-risk category labels for new records."""
    missing = [c for c in feature_columns if c not in input_data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    prepared = input_data[feature_columns].copy()
    if best_model_name == "XGBoost":
        encoded = best_model.predict(prepared).astype(int)
        return label_encoder.inverse_transform(encoded)
    return best_model.predict(prepared)

example_input = X_test.head(min(5, len(X_test))).copy()
example_predictions = predict_drought_risk(example_input)
example_results = example_input.copy()
example_results["Predicted_Drought_Risk"] = example_predictions
print("Example predictions:\n", example_results.to_string(index=False))

# ============================================================
# 17. GENERATE AGRICULTURAL RECOMMENDATIONS
# ============================================================
def generate_agricultural_recommendations(
    predicted_risk, agro_zone=None, soil_type=None, previous_crop=None, source_data=None
):
    """Look up the most frequent workbook recommendations for a matching risk.

    This is a dataset-based lookup, not a separately trained recommendation ML model.
    Local agricultural experts should validate the recommendations.
    """
    if source_data is None:
        source_data = df
    candidates = source_data[
        source_data[TARGET_COLUMN].astype(str) == str(predicted_risk)
    ].copy()
    for col, value in [
        ("Agro_Ecological_Zone", agro_zone),
        ("Soil_Type", soil_type),
        ("Previous_Crop", previous_crop),
    ]:
        if value is not None and col in candidates.columns:
            matches = candidates[
                candidates[col].astype(str).str.casefold() == str(value).casefold()
            ]
            if not matches.empty:
                candidates = matches
    result = {}
    for col in ["Recommended_Crops", "Recommended_Actions"]:
        if col in candidates and candidates[col].notna().any():
            result[col] = str(candidates[col].dropna().mode().iloc[0])
        else:
            result[col] = "No recommendation available in the source dataset."
    return result

if len(example_input):
    first_row = example_input.iloc[0]
    first_risk = str(example_predictions[0])
    example_recommendations = generate_agricultural_recommendations(
        first_risk,
        agro_zone=first_row.get("Agro_Ecological_Zone"),
        soil_type=first_row.get("Soil_Type"),
        previous_crop=first_row.get("Previous_Crop"),
    )
    print("Example recommendation for:", first_risk)
    print("Recommended crops:", example_recommendations["Recommended_Crops"])
    print("Recommended actions:", example_recommendations["Recommended_Actions"])

# ============================================================
# 18. CREATE GRAPHS AND VISUALIZATIONS
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.barplot(data=evaluation_df, x="Model", y="Macro F1")
plt.title("Model Comparison - Test Macro F1")
plt.ylim(0, 1)
plt.xlabel("Machine Learning Model")
plt.ylabel("Macro F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "model_comparison_macro_f1.png"), dpi=300)
plt.show()

top_features = feature_importance_df.head(15).sort_values("Importance_Mean", ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance_Mean"], xerr=top_features["Importance_Std"])
plt.title(f"Top Feature Importance - {best_model_name}")
plt.xlabel("Decrease in Macro F1 after shuffling feature")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "permutation_feature_importance.png"), dpi=300)
plt.show()

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_predictions, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("Actual supplied label")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
class_order = df[TARGET_COLUMN].value_counts().index
sns.countplot(data=df, x=TARGET_COLUMN, order=class_order)
plt.title("Drought Risk Category Distribution")
plt.xlabel("Drought Risk Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "drought_class_distribution.png"), dpi=300)
plt.show()

# ============================================================
# 19. SAVE THE BEST MODEL AND PROJECT OUTPUTS
# ============================================================
import joblib

model_path = os.path.join(MODEL_OUTPUT_DIR, "best_drought_model.joblib")
metadata_path = os.path.join(MODEL_OUTPUT_DIR, "model_metadata.json")
metrics_path = os.path.join(MODEL_OUTPUT_DIR, "model_comparison.csv")
importance_path = os.path.join(MODEL_OUTPUT_DIR, "feature_importance.csv")
predictions_path = os.path.join(MODEL_OUTPUT_DIR, "test_predictions.csv")

# Store the complete fitted preprocessing/model pipeline and encoder together.
model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "label_encoder": label_encoder if best_model_name == "XGBoost" else None,
    "feature_columns": feature_columns,
    "target_column": TARGET_COLUMN,
    "classes": sorted(y.unique().tolist()),
    "random_state": RANDOM_STATE,
}
joblib.dump(model_bundle, model_path)
evaluation_df.to_csv(metrics_path, index=False)
feature_importance_df.to_csv(importance_path, index=False)
test_prediction_df = X_test.copy()
test_prediction_df["Actual_Drought_Risk"] = y_test
test_prediction_df["Predicted_Drought_Risk"] = best_predictions
test_prediction_df.to_csv(predictions_path, index=False)

metadata = {
    "project_title": "Agricultural Drought Risk Prediction and Recommendation System in Ethiopia",
    "dataset_file": str(DATA_FILE),
    "target_column": TARGET_COLUMN,
    "feature_columns": feature_columns,
    "classes": sorted(y.unique().tolist()),
    "best_model_name": best_model_name,
    "selection_metric": "Training cross-validation Macro F1",
    "best_training_cv_macro_f1": cv_scores[best_model_name],
    "test_size": TEST_SIZE,
    "cv_folds": CV_FOLDS,
    "random_state": RANDOM_STATE,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "scientific_note": (
        "Trained against labels supplied in the workbook. If labels are provisional/rule-based, "
        "the model reproduces those labels and is not independently validated for future drought forecasting."
    ),
    "recommendation_note": (
        "Recommendations are retrieved from matching source-dataset rows; they are not a separately trained model."
    ),
}
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("\nTraining completed. Saved outputs:")
for output in [model_path, metadata_path, metrics_path, importance_path, predictions_path]:
    print(os.path.abspath(output))

# Automatically download the main outputs in Google Colab only.
try:
    from google.colab import files
    for output in [model_path, metadata_path, metrics_path, importance_path, predictions_path]:
        files.download(output)
except ImportError:
    print("Not running in Colab. All outputs are in:", os.path.abspath(MODEL_OUTPUT_DIR))

print("PROJECT TRAINING AND EVALUATION COMPLETED.")
