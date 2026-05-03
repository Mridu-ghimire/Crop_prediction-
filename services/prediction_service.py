"""
services/prediction_service.py
-------------------------------
Handles all prediction logic: single prediction, batch prediction,
confidence scoring, and result formatting.

Author     : Mridu Ghimire
Student ID : 77466817
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from config import FEATURE_NAMES, FEATURE_DEFAULTS
from utils.logger  import log_prediction, log_error
from utils.helpers import append_prediction_log, validate_inputs


# ─────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────

def predict_single(model, input_values: dict, user: str = "user") -> dict:
    """
    Run a single prediction given a dict of feature values.

    Args:
        model:        A trained sklearn classifier with predict_proba.
        input_values: Dict {feature_name: value} for all FEATURE_NAMES.
        user:         Username for logging.

    Returns:
        Dict with keys: top_crop, confidence, top3, errors
    """
    # Validate inputs first
    errors = validate_inputs(input_values)
    if errors:
        return {"top_crop": None, "confidence": 0.0, "top3": [], "errors": errors}

    try:
        features = [input_values[f] for f in FEATURE_NAMES]
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        probs    = model.predict_proba(input_df)[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3     = [(model.classes_[i], round(probs[i] * 100, 2)) for i in top3_idx]

        top_crop, top_conf = top3[0]
        log_prediction(user, top_crop, top_conf, type(model).__name__)

        return {
            "top_crop":   top_crop,
            "confidence": top_conf,
            "top3":       top3,
            "errors":     [],
        }
    except Exception as e:
        log_error(f"Prediction failed: {e}")
        return {"top_crop": None, "confidence": 0.0, "top3": [], "errors": [str(e)]}


# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────

def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on every row of a DataFrame.

    Returns the DataFrame with added columns:
      - Predicted_Crop
      - Confidence_%
    """
    try:
        X     = df[FEATURE_NAMES]
        preds = model.predict(X)
        probs = model.predict_proba(X)
        confs = [round(probs[i].max() * 100, 2) for i in range(len(probs))]
        out   = df.copy()
        out["Predicted_Crop"] = preds
        out["Confidence_%"]   = confs
        return out
    except Exception as e:
        log_error(f"Batch prediction failed: {e}")
        raise


# ─────────────────────────────────────────────
# LOG PREDICTION
# ─────────────────────────────────────────────

def save_prediction_to_log(
    user: str,
    input_values: dict,
    top_crop: str,
    confidence: float,
    model_name: str,
    accuracy: float,
):
    """Persist a prediction result to the JSON log file."""
    entry = {
        "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User":        user,
        "N":           input_values.get("n"),
        "P":           input_values.get("p"),
        "K":           input_values.get("k"),
        "Temperature": input_values.get("temperature"),
        "Humidity":    input_values.get("humidity"),
        "pH":          input_values.get("ph"),
        "Rainfall":    input_values.get("rainfall"),
        "Top Crop":    top_crop.title(),
        "Confidence":  f"{confidence:.1f}%",
        "Model":       model_name,
        "Accuracy":    f"{accuracy:.1f}%",
    }
    append_prediction_log(entry)
    return entry
