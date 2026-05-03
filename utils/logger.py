"""
utils/logger.py
---------------
Centralised logging for the Crop Prediction System.
Writes to logs/system.log and also outputs to console.

Author     : Mridu Ghimire
Student ID : 77466817
"""

import logging
import os
import sys
from datetime import datetime

# Import path from config
try:
    from config import SYSTEM_LOG
except ImportError:
    SYSTEM_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "system.log")

os.makedirs(os.path.dirname(SYSTEM_LOG), exist_ok=True)

# ─────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────
def get_logger(name: str = "CropPrediction") -> logging.Logger:
    """
    Return a configured logger that writes to both file and console.
    Call once per module: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(SYSTEM_LOG, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────
_logger = get_logger("CropPrediction")

def log_info(msg: str):    _logger.info(msg)
def log_warning(msg: str): _logger.warning(msg)
def log_error(msg: str):   _logger.error(msg)
def log_debug(msg: str):   _logger.debug(msg)

def log_prediction(user: str, crop: str, confidence: float, model: str):
    """Log a prediction event."""
    _logger.info(f"PREDICTION | user={user} | crop={crop} | conf={confidence:.1f}% | model={model}")

def log_training(model_name: str, accuracy: float):
    """Log a model training event."""
    _logger.info(f"TRAINING   | model={model_name} | accuracy={accuracy:.2f}%")

def log_upload(filename: str, rows: int, status: str):
    """Log a dataset upload event."""
    _logger.info(f"UPLOAD     | file={filename} | rows={rows} | status={status}")

def read_system_log(lines: int = 100) -> list:
    """Return the last N lines of the system log."""
    if not os.path.exists(SYSTEM_LOG):
        return []
    with open(SYSTEM_LOG, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    return [l.rstrip() for l in all_lines[-lines:]]
