"""
model.py
--------
All machine learning logic: model definitions, training, evaluation,
cross-validation, feature importance, confusion matrix, error analysis.

Author     : Mridu Ghimire
Student ID : 77466817
Course     : BSc (Hons) Computing – Level 6 Production Project
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from data import get_feature_target, FEATURE_NAMES

# ─────────────────────────────────────────────
# 1. MODEL REGISTRY
# ─────────────────────────────────────────────

MODELS = {
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbours":   KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":            GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}

MODEL_SAVE_DIR = "saved_models"


# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Return stratified 80/20 train/test splits."""
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# ─────────────────────────────────────────────
# 3. SINGLE MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_single(model_name: str, df: pd.DataFrame):
    """
    Train a single named model on the dataset.
    Returns (trained_model, X_test, y_test).
    """
    from sklearn.base import clone
    X_train, X_test, y_train, y_test = get_train_test(df)
    model = clone(MODELS[model_name])
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Return accuracy, precision, recall, F1 as percentages."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred,
                           average="weighted", zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred,
                           average="weighted", zero_division=0) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred,
                           average="weighted", zero_division=0) * 100, 2),
    }


def cross_validate_model(model, X, y, cv: int = 5) -> float:
    """Return mean 5-fold stratified cross-validation accuracy (%)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return round(scores.mean() * 100, 2)


# ─────────────────────────────────────────────
# 4. TRAIN & COMPARE ALL MODELS
# ─────────────────────────────────────────────

def train_and_compare_all(df: pd.DataFrame):
    """
    Train all models, evaluate them, and return:
      - results_df   : DataFrame of metrics sorted by Accuracy
      - trained_models: dict {name: fitted_model}
      - best_name    : name of best model
      - best_model   : fitted best model
      - X_test, y_test: held-out test data
    """
    from sklearn.base import clone
    X_train, X_test, y_train, y_test = get_train_test(df)
    X, y = get_feature_target(df)

    results = []
    trained_models = {}

    for name, base_model in MODELS.items():
        model = clone(base_model)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["CV Accuracy"] = cross_validate_model(model, X, y)
        metrics["Model"] = name
        results.append(metrics)
        trained_models[name] = model

    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        [["Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy"]]
        .sort_values("Accuracy", ascending=False)
    )

    best_name  = results_df.index[0]
    best_model = trained_models[best_name]

    return results_df, trained_models, best_name, best_model, X_test, y_test


# ─────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────

def predict_top3(model, input_features: list) -> list:
    """
    Return top-3 (crop, confidence%) predictions for a single input row.
    input_features must be in FEATURE_NAMES order.
    """
    input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
    probs    = model.predict_proba(input_df)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    return [(model.classes_[i], round(probs[i] * 100, 2)) for i in top3_idx]


def batch_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on every row of an uploaded DataFrame.
    Returns the DataFrame with added 'Predicted_Crop' and 'Confidence_%' columns.
    """
    X = df[FEATURE_NAMES]
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confs = [round(probs[i].max() * 100, 2) for i in range(len(probs))]
    out = df.copy()
    out["Predicted_Crop"] = preds
    out["Confidence_%"]   = confs
    return out


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list = None) -> pd.Series:
    """
    Return feature importances as a sorted Series.
    Works for tree-based models (RF, DT, GB).
    Returns empty Series for others.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        return fi.sort_values(ascending=False)
    return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX & CLASSIFICATION REPORT
# ─────────────────────────────────────────────

def get_confusion_matrix(model, X_test, y_test):
    """Return (cm_array, class_labels)."""
    y_pred = model.predict(X_test)
    labels = sorted(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return cm, labels


def get_classification_report(model, X_test, y_test) -> str:
    """Return formatted sklearn classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0)


# ─────────────────────────────────────────────
# 8. ERROR ANALYSIS
# ─────────────────────────────────────────────

def get_error_analysis(model, X_test, y_test) -> pd.DataFrame:
    """
    Return a DataFrame of misclassified samples with columns:
    Actual, Predicted, Confidence, and all feature values.
    """
    y_pred = model.predict(X_test)
    probs  = model.predict_proba(X_test)
    confs  = [round(probs[i].max() * 100, 2) for i in range(len(probs))]

    mask = y_pred != y_test.values
    errors = X_test[mask].copy()
    errors["Actual"]     = y_test.values[mask]
    errors["Predicted"]  = y_pred[mask]
    errors["Confidence"] = [confs[i] for i, m in enumerate(mask) if m]
    errors = errors.reset_index(drop=True)
    return errors


def error_summary(model, X_test, y_test) -> dict:
    """
    Return a summary dict of error analysis:
      - total_errors, error_rate, most_confused_pairs, per_class_errors
    """
    y_pred = model.predict(X_test)
    mask   = y_pred != y_test.values
    total  = len(y_test)
    n_err  = int(mask.sum())

    # Most confused pairs
    actual_arr = y_test.values[mask]
    pred_arr   = y_pred[mask]
    pairs = pd.Series(
        [f"{a} → {p}" for a, p in zip(actual_arr, pred_arr)]
    ).value_counts().head(5)

    # Per-class error count
    per_class = (
        pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        .query("Actual != Predicted")
        .groupby("Actual")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "total_errors":        n_err,
        "total_samples":       total,
        "error_rate":          round(n_err / total * 100, 2),
        "most_confused_pairs": pairs,
        "per_class_errors":    per_class,
    }


# ─────────────────────────────────────────────
# 9. MODEL PERSISTENCE (joblib)
# ─────────────────────────────────────────────

def save_model(model, name: str):
    """Save a trained model to disk using joblib."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()
    path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}.pkl")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a saved model from disk. Returns None if not found."""
    safe_name = name.replace(" ", "_").lower()
    path = os.path.join(MODEL_SAVE_DIR, f"{safe_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
