"""
config.py
---------
Central configuration for the Crop Prediction System.
All paths, constants, and settings in one place.

Author     : Mridu Ghimire
Student ID : 77466817
"""

import os

# ─────────────────────────────────────────────
# BASE DIRECTORIES
# ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
LOGS_DIR     = os.path.join(BASE_DIR, "logs")
MODELS_DIR   = os.path.join(BASE_DIR, "models", "trained")
BACKUPS_DIR  = os.path.join(BASE_DIR, "models", "backups")
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")
NOTEBOOKS_DIR= os.path.join(BASE_DIR, "notebooks")

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
DEFAULT_DATASET  = os.path.join(DATASET_DIR,  "Crop_recommendation.csv")
CUSTOM_DATASET   = os.path.join(DATASET_DIR,  "admin_dataset.csv")
SAMPLE_UPLOAD    = os.path.join(DATASET_DIR,  "sample_upload.csv")
USERS_DB         = os.path.join(DATABASE_DIR, "users.db")
PRED_LOGS        = os.path.join(LOGS_DIR,     "prediction_logs.json")
SYSTEM_LOG       = os.path.join(LOGS_DIR,     "system.log")
ENV_FILE         = os.path.join(BASE_DIR,     ".env")

# ─────────────────────────────────────────────
# ADMIN CREDENTIALS (change in .env for production)
# ─────────────────────────────────────────────
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# ─────────────────────────────────────────────
# ML SETTINGS
# ─────────────────────────────────────────────
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CV_FOLDS       = 5
N_ESTIMATORS   = 100

# ─────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────
FEATURE_NAMES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

FEATURE_DISPLAY = {
    "n":           "Nitrogen (N)",
    "p":           "Phosphorus (P)",
    "k":           "Potassium (K)",
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "ph":          "pH",
    "rainfall":    "Rainfall",
}

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
# APP SETTINGS
# ─────────────────────────────────────────────
APP_TITLE       = "Crop Prediction System"
APP_ICON        = "🌾"
ADMIN_TITLE     = "Admin Panel – Crop Prediction"
ADMIN_ICON      = "🛡️"
APP_AUTHOR      = "Mridu Ghimire"
STUDENT_ID      = "77466817"
COURSE          = "BSc (Hons) Computing – Level 6"

# ─────────────────────────────────────────────
# BACKWARD COMPATIBILITY — root-level fallbacks
# ─────────────────────────────────────────────
# If files still exist at root level, use them
if not os.path.exists(DEFAULT_DATASET) and os.path.exists(os.path.join(BASE_DIR, "Crop_recommendation.csv")):
    DEFAULT_DATASET = os.path.join(BASE_DIR, "Crop_recommendation.csv")

if not os.path.exists(USERS_DB) and os.path.exists(os.path.join(BASE_DIR, "users.db")):
    USERS_DB = os.path.join(BASE_DIR, "users.db")

if not os.path.exists(PRED_LOGS) and os.path.exists(os.path.join(BASE_DIR, "prediction_logs.json")):
    PRED_LOGS = os.path.join(BASE_DIR, "prediction_logs.json")

# ─────────────────────────────────────────────
# ENSURE DIRECTORIES EXIST
# ─────────────────────────────────────────────
for _dir in [DATASET_DIR, DATABASE_DIR, LOGS_DIR, MODELS_DIR, BACKUPS_DIR, ASSETS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(_dir, exist_ok=True)
