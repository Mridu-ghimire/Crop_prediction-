"""
utils/helpers.py
----------------
General-purpose helper utilities for the Crop Prediction System.

Author     : Mridu Ghimire
Student ID : 77466817
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

try:
    from config import PRED_LOGS, FEATURE_NAMES, FEATURE_RANGES
except ImportError:
    PRED_LOGS     = os.path.join(os.path.dirname(__file__), "..", "logs", "prediction_logs.json")
    FEATURE_NAMES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
    FEATURE_RANGES = {}

os.makedirs(os.path.dirname(PRED_LOGS), exist_ok=True)


# ─────────────────────────────────────────────
# PASSWORD UTILITIES
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return SHA-256 hex digest of a password string."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    """Check if a plain password matches a stored hash."""
    return hash_password(plain) == hashed


# ─────────────────────────────────────────────
# PREDICTION LOG UTILITIES
# ─────────────────────────────────────────────

def load_prediction_logs() -> list:
    """Load all prediction logs from JSON file."""
    if not os.path.exists(PRED_LOGS):
        return []
    try:
        with open(PRED_LOGS, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def append_prediction_log(entry: dict):
    """Append a single prediction entry to the log file."""
    logs = load_prediction_logs()
    entry["logged_at"] = datetime.now().isoformat()
    logs.append(entry)
    with open(PRED_LOGS, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def clear_prediction_logs():
    """Delete all prediction logs."""
    if os.path.exists(PRED_LOGS):
        os.remove(PRED_LOGS)

def logs_to_dataframe() -> Optional[pd.DataFrame]:
    """Return prediction logs as a DataFrame, or None if empty."""
    logs = load_prediction_logs()
    if not logs:
        return None
    return pd.DataFrame(logs)


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

def validate_inputs(values: dict) -> list:
    """
    Validate user input values against known feature ranges.
    Returns list of error strings (empty = all valid).
    """
    errors = []
    for feat, val in values.items():
        if feat not in FEATURE_RANGES:
            continue
        lo, hi, label = FEATURE_RANGES[feat]
        if val < lo:
            errors.append(f"❌ **{feat.upper()}** ({label}): {val} is below minimum ({lo}).")
        elif val > hi:
            errors.append(f"❌ **{feat.upper()}** ({label}): {val} exceeds maximum ({hi}).")
    return errors


# ─────────────────────────────────────────────
# DATASET VALIDATION
# ─────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run a set of validation checks on an uploaded DataFrame.
    Returns dict of {check_name: (passed: bool, message: str)}.
    """
    checks = {}

    # Required columns
    required = set(FEATURE_NAMES)
    missing  = required - set(df.columns)
    checks["Required columns present"] = (
        len(missing) == 0,
        "All required columns found." if not missing
        else f"Missing: {', '.join(sorted(missing))}",
    )

    # Label column
    checks["Label column present"] = (
        "label" in df.columns,
        "label column found." if "label" in df.columns else "No 'label' column.",
    )

    # Missing values
    n_missing = int(df.isnull().sum().sum())
    checks["No missing values"] = (
        n_missing == 0,
        "No missing values." if n_missing == 0 else f"{n_missing} missing values found.",
    )

    # Sufficient rows
    checks["Sufficient rows (≥ 50)"] = (
        len(df) >= 50,
        f"{len(df)} rows found." if len(df) >= 50 else f"Only {len(df)} rows — need at least 50.",
    )

    # Numeric features
    non_numeric = [f for f in FEATURE_NAMES if f in df.columns
                   and not pd.api.types.is_numeric_dtype(df[f])]
    checks["All features are numeric"] = (
        len(non_numeric) == 0,
        "All features are numeric." if not non_numeric
        else f"Non-numeric: {', '.join(non_numeric)}",
    )

    return checks


# ─────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value:.{decimals}f}%"

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Return a formatted timestamp string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

def get_file_size_str(path: str) -> str:
    """Return human-readable file size string."""
    if not os.path.exists(path):
        return "N/A"
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
