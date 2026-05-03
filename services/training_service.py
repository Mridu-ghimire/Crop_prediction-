"""
services/training_service.py
-----------------------------
Handles all model training, evaluation, comparison, and persistence.

Author     : Mridu Ghimire
Student ID : 77466817
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

from config import (
    FEATURE_NAMES, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    MODELS_DIR, BACKUPS_DIR,
)
from model import MODELS
from data import get_feature_target
from utils.logger import log_training, log_error

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(BACKUPS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """Return stratified 80/20 train/test splits."""
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# ─────────────────────────────────────────────
# SINGLE MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(model_name: str, df: pd.DataFrame):
    """
    Train a single named model.
    Returns (trained_model, X_test, y_test, metrics_dict).
    """
    try:
        model = clone(MODELS[model_name])
        X_train, X_test, y_train, y_test = split_data(df)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        log_training(model_name, metrics["Accuracy"])
        return model, X_test, y_test, metrics
    except Exception as e:
        log_error(f"Training failed for {model_name}: {e}")
        raise


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

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


def cross_validate(model, df: pd.DataFrame) -> float:
    """Return mean CV accuracy (%) using StratifiedKFold."""
    X, y = get_feature_target(df)
    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return round(scores.mean() * 100, 2)


# ─────────────────────────────────────────────
# COMPARE ALL MODELS
# ─────────────────────────────────────────────

def compare_all_models(df: pd.DataFrame):
    """
    Train and evaluate all models.
    Returns (results_df, trained_models, best_name, best_model, X_test, y_test).
    """
    X_train, X_test, y_train, y_test = split_data(df)
    X, y = get_feature_target(df)

    results        = []
    trained_models = {}

    for name, base_model in MODELS.items():
        model = clone(base_model)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["CV Accuracy"] = cross_validate(model, df)
        metrics["Model"]       = name
        results.append(metrics)
        trained_models[name]   = model
        log_training(name, metrics["Accuracy"])

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
# MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model(model, name: str, backup: bool = True) -> str:
    """Save model to models/trained/. Optionally backup previous version."""
    safe = name.replace(" ", "_").lower()
    path = os.path.join(MODELS_DIR, f"{safe}.pkl")

    # Backup existing model
    if backup and os.path.exists(path):
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        bk_path = os.path.join(BACKUPS_DIR, f"{safe}_{ts}.pkl")
        joblib.dump(joblib.load(path), bk_path)

    joblib.dump(model, path)
    return path


def load_saved_model(name: str):
    """Load a saved model. Returns None if not found."""
    safe = name.replace(" ", "_").lower()
    path = os.path.join(MODELS_DIR, f"{safe}.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def list_saved_models() -> list:
    """Return list of saved model filenames."""
    if not os.path.exists(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]


def list_model_backups() -> list:
    """Return list of backup model filenames."""
    if not os.path.exists(BACKUPS_DIR):
        return []
    return sorted([f for f in os.listdir(BACKUPS_DIR) if f.endswith(".pkl")], reverse=True)
