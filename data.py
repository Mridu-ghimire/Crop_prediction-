"""
data.py
-------
Data loading, validation, preprocessing, and fertilizer recommendation logic.

Author     : Mridu Ghimire
Student ID : 77466817
Course     : BSc (Hons) Computing – Level 6 Production Project
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────

FEATURE_NAMES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

FEATURE_RANGES = {
    "n":           (0,    140,  "Nitrogen content in soil (kg/ha)"),
    "p":           (5,    145,  "Phosphorus content in soil (kg/ha)"),
    "k":           (5,    205,  "Potassium content in soil (kg/ha)"),
    "temperature": (8.0,  44.0, "Temperature (°C)"),
    "humidity":    (14.0, 100.0,"Relative humidity (%)"),
    "ph":          (3.5,  9.5,  "Soil pH level"),
    "rainfall":    (20.0, 300.0,"Annual rainfall (mm)"),
}

FEATURE_DEFAULTS = {
    "n": 90, "p": 40, "k": 40,
    "temperature": 25.0, "humidity": 70.0,
    "ph": 6.5, "rainfall": 200.0,
}

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str = "Crop_recommendation.csv") -> pd.DataFrame:
    """Load the crop recommendation CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Normalise label column name
    if "label" not in df.columns and "crop" in df.columns:
        df = df.rename(columns={"crop": "label"})
    return df


def load_uploaded_data(uploaded_file) -> tuple:
    """
    Parse a user-uploaded CSV file.
    Returns (df, error_message). error_message is None on success.
    """
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "label" not in df.columns and "crop" in df.columns:
            df = df.rename(columns={"crop": "label"})
        # Check required columns
        required = set(FEATURE_NAMES)
        missing = required - set(df.columns)
        if missing:
            return None, f"Missing columns: {', '.join(sorted(missing))}"
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# 2. INPUT VALIDATION
# ─────────────────────────────────────────────

def validate_inputs(values: dict) -> list:
    """
    Validate user-supplied input values against known ranges.
    Returns a list of error strings (empty list = all valid).
    """
    errors = []
    for feat, val in values.items():
        if feat not in FEATURE_RANGES:
            continue
        lo, hi, label = FEATURE_RANGES[feat]
        if val < lo:
            errors.append(f"❌ **{feat}** ({label}): value {val} is below minimum ({lo}).")
        elif val > hi:
            errors.append(f"❌ **{feat}** ({label}): value {val} exceeds maximum ({hi}).")
    return errors


# ─────────────────────────────────────────────
# 3. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────

def get_feature_target(df: pd.DataFrame):
    """Return (X, y) split from a labelled DataFrame."""
    X = df[FEATURE_NAMES]
    y = df["label"]
    return X, y


# ─────────────────────────────────────────────
# 4. DATASET STATISTICS
# ─────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return key statistics about the dataset."""
    return {
        "total_samples":      len(df),
        "num_features":       len(FEATURE_NAMES),
        "num_classes":        df["label"].nunique(),
        "class_distribution": df["label"].value_counts(),
        "missing_values":     int(df.isnull().sum().sum()),
        "describe":           df[FEATURE_NAMES].describe(),
        "feature_means":      df[FEATURE_NAMES].mean(),
    }


# ─────────────────────────────────────────────
# 5. FERTILIZER RECOMMENDATIONS
# ─────────────────────────────────────────────

FERTILIZER_DB = {
    # (crop, condition) → recommendation
    # Nitrogen
    "low_N":  "🌿 **Low Nitrogen:** Apply Urea (46-0-0) at 50–100 kg/ha, or use compost/green manure to boost N levels.",
    "high_N": "⚠️ **High Nitrogen:** Reduce N fertiliser. Excess N causes leafy growth at the expense of yield.",
    "ok_N":   "✅ **Nitrogen:** Level is optimal. Maintain with balanced NPK fertiliser.",

    # Phosphorus
    "low_P":  "🌿 **Low Phosphorus:** Apply Single Super Phosphate (SSP) or DAP at 25–50 kg/ha to improve root development.",
    "high_P": "⚠️ **High Phosphorus:** Avoid additional P fertiliser. High P can lock out zinc and iron.",
    "ok_P":   "✅ **Phosphorus:** Level is optimal.",

    # Potassium
    "low_K":  "🌿 **Low Potassium:** Apply Muriate of Potash (MOP) at 30–60 kg/ha to improve disease resistance.",
    "high_K": "⚠️ **High Potassium:** Reduce K application. Excess K can interfere with Mg and Ca uptake.",
    "ok_K":   "✅ **Potassium:** Level is optimal.",

    # pH
    "low_ph":  "🧪 **Acidic Soil (pH < 5.5):** Apply agricultural lime (CaCO₃) at 1–2 tonnes/ha to raise pH.",
    "high_ph": "🧪 **Alkaline Soil (pH > 7.5):** Apply elemental sulphur or acidifying fertilisers to lower pH.",
    "ok_ph":   "✅ **pH:** Soil pH is in the optimal range (5.5–7.5).",

    # Rainfall
    "low_rain":  "💧 **Low Rainfall:** Consider drip irrigation or mulching to conserve soil moisture.",
    "high_rain": "💧 **High Rainfall:** Ensure proper drainage to prevent waterlogging and root rot.",
    "ok_rain":   "✅ **Rainfall:** Adequate moisture for most crops.",
}


def get_fertilizer_recommendations(n, p, k, ph, rainfall) -> list:
    """
    Return a list of fertilizer/improvement tip strings based on input values.
    Uses dataset-derived thresholds for low/high classification.
    """
    tips = []

    # Nitrogen thresholds
    if n < 20:
        tips.append(FERTILIZER_DB["low_N"])
    elif n > 120:
        tips.append(FERTILIZER_DB["high_N"])
    else:
        tips.append(FERTILIZER_DB["ok_N"])

    # Phosphorus thresholds
    if p < 15:
        tips.append(FERTILIZER_DB["low_P"])
    elif p > 120:
        tips.append(FERTILIZER_DB["high_P"])
    else:
        tips.append(FERTILIZER_DB["ok_P"])

    # Potassium thresholds
    if k < 15:
        tips.append(FERTILIZER_DB["low_K"])
    elif k > 180:
        tips.append(FERTILIZER_DB["high_K"])
    else:
        tips.append(FERTILIZER_DB["ok_K"])

    # pH thresholds
    if ph < 5.5:
        tips.append(FERTILIZER_DB["low_ph"])
    elif ph > 7.5:
        tips.append(FERTILIZER_DB["high_ph"])
    else:
        tips.append(FERTILIZER_DB["ok_ph"])

    # Rainfall thresholds
    if rainfall < 50:
        tips.append(FERTILIZER_DB["low_rain"])
    elif rainfall > 250:
        tips.append(FERTILIZER_DB["high_rain"])
    else:
        tips.append(FERTILIZER_DB["ok_rain"])

    return tips


# ─────────────────────────────────────────────
# 6. CROP INFORMATION (Nepal-specific)
# ─────────────────────────────────────────────

CROP_INFO = {
    "rice": {
        "season": "Kharif (June–November)",
        "regions": "Terai, Inner Terai",
        "description": "Staple food crop of Nepal. Requires high humidity and rainfall.",
        "tips": "Ensure proper irrigation. Transplant seedlings after monsoon onset.",
        "ideal_N": (60, 100), "ideal_P": (30, 60), "ideal_K": (30, 50),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (150, 300),
    },
    "wheat": {
        "season": "Rabi (November–April)",
        "regions": "Terai, Mid-Hills",
        "description": "Second most important cereal crop in Nepal.",
        "tips": "Sow after rice harvest. Requires cool temperatures during grain filling.",
        "ideal_N": (60, 120), "ideal_P": (30, 60), "ideal_K": (30, 50),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (50, 150),
    },
    "maize": {
        "season": "Spring & Kharif",
        "regions": "Hills, Mid-Hills, Terai",
        "description": "Widely grown across altitudinal zones in Nepal.",
        "tips": "Intercrop with legumes for better soil health.",
        "ideal_N": (60, 100), "ideal_P": (30, 60), "ideal_K": (15, 40),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (50, 200),
    },
    "chickpea": {
        "season": "Rabi (October–March)", "regions": "Terai, Inner Terai",
        "description": "Important pulse crop providing protein-rich food.",
        "tips": "Grows well in well-drained loamy soils. Avoid waterlogging.",
        "ideal_N": (20, 60), "ideal_P": (40, 80), "ideal_K": (20, 50),
        "ideal_ph": (6.0, 8.0), "ideal_rain": (60, 150),
    },
    "kidneybeans": {
        "season": "Kharif", "regions": "Hills, Mid-Hills",
        "description": "Popular legume crop in hilly regions of Nepal.",
        "tips": "Requires moderate rainfall and well-drained soil.",
        "ideal_N": (20, 40), "ideal_P": (60, 100), "ideal_K": (15, 30),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (100, 200),
    },
    "pigeonpeas": {
        "season": "Kharif", "regions": "Terai",
        "description": "Drought-tolerant pulse crop suitable for Terai region.",
        "tips": "Can be intercropped with cereals.",
        "ideal_N": (15, 30), "ideal_P": (50, 80), "ideal_K": (15, 30),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (60, 150),
    },
    "mothbeans": {
        "season": "Kharif", "regions": "Terai",
        "description": "Drought-resistant legume suitable for dry areas.",
        "tips": "Grows well in sandy loam soils with low rainfall.",
        "ideal_N": (15, 30), "ideal_P": (40, 70), "ideal_K": (15, 30),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (30, 100),
    },
    "mungbean": {
        "season": "Spring & Summer", "regions": "Terai, Inner Terai",
        "description": "Short-duration pulse crop with high nutritional value.",
        "tips": "Ideal for crop rotation after wheat.",
        "ideal_N": (15, 30), "ideal_P": (40, 70), "ideal_K": (15, 30),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (60, 150),
    },
    "blackgram": {
        "season": "Kharif", "regions": "Terai",
        "description": "Important pulse crop in the Terai belt.",
        "tips": "Sensitive to waterlogging. Ensure good drainage.",
        "ideal_N": (15, 30), "ideal_P": (40, 70), "ideal_K": (15, 30),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (60, 150),
    },
    "lentil": {
        "season": "Rabi", "regions": "Terai, Mid-Hills",
        "description": "Widely consumed pulse crop in Nepal.",
        "tips": "Sow in October–November for best yields.",
        "ideal_N": (15, 30), "ideal_P": (40, 70), "ideal_K": (15, 30),
        "ideal_ph": (6.0, 8.0), "ideal_rain": (40, 100),
    },
    "pomegranate": {
        "season": "Perennial", "regions": "Mid-Hills, Inner Terai",
        "description": "Fruit crop with growing commercial potential in Nepal.",
        "tips": "Requires well-drained soil and moderate irrigation.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (30, 60),
        "ideal_ph": (5.5, 7.5), "ideal_rain": (50, 150),
    },
    "banana": {
        "season": "Perennial", "regions": "Terai, Inner Terai",
        "description": "Popular fruit crop in the lowland regions.",
        "tips": "Requires high humidity and regular watering.",
        "ideal_N": (80, 120), "ideal_P": (60, 100), "ideal_K": (40, 80),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (100, 200),
    },
    "mango": {
        "season": "Perennial", "regions": "Terai, Inner Terai",
        "description": "Major fruit crop with high market demand.",
        "tips": "Prune regularly and apply balanced fertiliser.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (30, 60),
        "ideal_ph": (5.5, 7.5), "ideal_rain": (50, 150),
    },
    "grapes": {
        "season": "Perennial", "regions": "Mid-Hills",
        "description": "Emerging fruit crop in Nepal's hill regions.",
        "tips": "Requires trellis support and well-drained soil.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (30, 60),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (50, 150),
    },
    "watermelon": {
        "season": "Summer", "regions": "Terai",
        "description": "Popular summer fruit with high water content.",
        "tips": "Requires sandy loam soil and warm temperatures.",
        "ideal_N": (80, 120), "ideal_P": (40, 80), "ideal_K": (40, 80),
        "ideal_ph": (6.0, 7.0), "ideal_rain": (40, 100),
    },
    "muskmelon": {
        "season": "Summer", "regions": "Terai",
        "description": "Sweet fruit crop grown in warm lowland areas.",
        "tips": "Needs full sun and well-drained soil.",
        "ideal_N": (80, 120), "ideal_P": (40, 80), "ideal_K": (40, 80),
        "ideal_ph": (6.0, 7.0), "ideal_rain": (40, 100),
    },
    "apple": {
        "season": "Perennial", "regions": "High Hills, Mountains",
        "description": "Important fruit crop in Nepal's high-altitude regions.",
        "tips": "Requires chilling hours and well-drained loamy soil.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (30, 60),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (100, 200),
    },
    "orange": {
        "season": "Perennial", "regions": "Mid-Hills",
        "description": "Major citrus crop in Nepal's mid-hill regions.",
        "tips": "Requires moderate rainfall and well-drained soil.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (10, 30),
        "ideal_ph": (5.5, 7.0), "ideal_rain": (100, 200),
    },
    "papaya": {
        "season": "Perennial", "regions": "Terai, Inner Terai",
        "description": "Fast-growing fruit crop with year-round production.",
        "tips": "Sensitive to frost. Requires warm temperatures.",
        "ideal_N": (40, 80), "ideal_P": (10, 30), "ideal_K": (40, 80),
        "ideal_ph": (6.0, 7.0), "ideal_rain": (100, 200),
    },
    "coconut": {
        "season": "Perennial", "regions": "Terai (limited)",
        "description": "Tropical crop grown in the southernmost Terai belt.",
        "tips": "Requires high temperatures and humidity.",
        "ideal_N": (15, 30), "ideal_P": (10, 30), "ideal_K": (30, 60),
        "ideal_ph": (5.5, 7.5), "ideal_rain": (100, 250),
    },
    "cotton": {
        "season": "Kharif", "regions": "Terai",
        "description": "Cash crop grown in the Terai region.",
        "tips": "Requires deep, well-drained soil and warm climate.",
        "ideal_N": (100, 140), "ideal_P": (15, 40), "ideal_K": (15, 40),
        "ideal_ph": (6.0, 8.0), "ideal_rain": (60, 150),
    },
    "jute": {
        "season": "Kharif", "regions": "Terai",
        "description": "Important fibre crop grown in the Terai belt.",
        "tips": "Requires high humidity and alluvial soil.",
        "ideal_N": (60, 100), "ideal_P": (40, 80), "ideal_K": (30, 60),
        "ideal_ph": (6.0, 7.5), "ideal_rain": (150, 250),
    },
    "coffee": {
        "season": "Perennial", "regions": "Mid-Hills",
        "description": "High-value cash crop with growing export potential.",
        "tips": "Grows best in shade with well-drained acidic soil.",
        "ideal_N": (80, 120), "ideal_P": (15, 40), "ideal_K": (15, 40),
        "ideal_ph": (4.5, 6.5), "ideal_rain": (150, 250),
    },
}

_DEFAULT_CROP_INFO = {
    "season": "Varies", "regions": "Varies by altitude",
    "description": "Crop information not available.",
    "tips": "Consult local agricultural extension services.",
    "ideal_N": (20, 100), "ideal_P": (20, 80), "ideal_K": (20, 80),
    "ideal_ph": (5.5, 7.5), "ideal_rain": (50, 200),
}


def get_crop_info(crop_name: str) -> dict:
    """Return Nepal-specific information for a given crop."""
    return CROP_INFO.get(crop_name.lower(), _DEFAULT_CROP_INFO)


# ─────────────────────────────────────────────
# 7. EXPLAINABILITY — "WHY THIS CROP?"
# ─────────────────────────────────────────────

def explain_prediction(crop: str, n, p, k, temperature, humidity, ph, rainfall) -> str:
    """
    Generate a 1–2 line human-readable explanation for why a crop was recommended.
    Compares user inputs against the crop's ideal ranges.
    """
    info = get_crop_info(crop)
    reasons = []

    def _in_range(val, lo, hi):
        return lo <= val <= hi

    if _in_range(ph, *info["ideal_ph"]):
        reasons.append(f"soil pH ({ph:.1f}) is ideal")
    if _in_range(rainfall, *info["ideal_rain"]):
        reasons.append(f"rainfall ({rainfall:.0f} mm) matches requirements")
    if _in_range(n, *info["ideal_N"]):
        reasons.append(f"nitrogen level ({n}) is suitable")
    if _in_range(temperature, 15, 40):
        reasons.append(f"temperature ({temperature:.1f}°C) is within growing range")

    if reasons:
        reason_str = ", ".join(reasons[:3])
        return (
            f"**{crop.title()}** is recommended because your {reason_str}. "
            f"{info['description']}"
        )
    else:
        return (
            f"**{crop.title()}** is the closest match to your soil and climate profile. "
            f"{info['description']}"
        )
