"""
model_utils.py
--------------
All machine learning logic for the Predictive Crop Intelligence Framework.
Handles data loading, preprocessing, model training, evaluation, and comparison.

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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_data(path: str = "Crop_recommendation.csv") -> pd.DataFrame:
    """Load and return the crop recommendation dataset."""
    df = pd.read_csv(path)
    return df


def get_feature_target(df: pd.DataFrame):
    """Split dataframe into features (X) and target (y)."""
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Return train/test splits with stratification."""
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ─────────────────────────────────────────────
# 2. MODEL DEFINITIONS
# ─────────────────────────────────────────────

MODELS = {
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbours":   KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":            GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}


# ─────────────────────────────────────────────
# 3. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train):
    """Fit a model and return it."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict of evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
    }


def cross_validate_model(model, X, y, cv: int = 5) -> float:
    """Return mean cross-validation accuracy (%) using StratifiedKFold."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return round(scores.mean() * 100, 2)


def train_and_compare_all(df: pd.DataFrame):
    """
    Train all models, evaluate them, and return a comparison DataFrame.
    Also returns all trained models, the best model name, and the best model.
    """
    X_train, X_test, y_train, y_test = get_train_test(df)
    X, y = get_feature_target(df)

    results = []
    trained_models = {}

    for name, model in MODELS.items():
        trained = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test)
        cv_acc = cross_validate_model(trained, X, y)
        metrics["CV Accuracy"] = cv_acc
        metrics["Model"] = name
        results.append(metrics)
        trained_models[name] = trained

    results_df = pd.DataFrame(results).set_index("Model")
    results_df = results_df[["Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy"]]
    results_df = results_df.sort_values("Accuracy", ascending=False)

    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]

    return results_df, trained_models, best_model_name, best_model


def get_confusion_matrix(model, X_test, y_test):
    """Return confusion matrix and class labels."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return cm, list(model.classes_)


def get_classification_report(model, X_test, y_test) -> str:
    """Return a formatted classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0)


# ─────────────────────────────────────────────
# 4. PREDICTION
# ─────────────────────────────────────────────

def predict_top3(model, input_features: list, feature_names: list) -> list:
    """
    Given a trained model and input values, return the top-3 crop
    recommendations with their probability percentages.
    """
    input_df = pd.DataFrame([input_features], columns=feature_names)
    probs = model.predict_proba(input_df)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    return [(model.classes_[i], round(probs[i] * 100, 2)) for i in top3_idx]


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.Series:
    """Return feature importances as a sorted Series (tree-based models only)."""
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        return fi.sort_values(ascending=False)
    return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# 6. DATASET STATISTICS
# ─────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return key statistics about the dataset."""
    return {
        "total_samples":      len(df),
        "num_features":       df.shape[1] - 1,
        "num_classes":        df["label"].nunique(),
        "class_distribution": df["label"].value_counts(),
        "missing_values":     df.isnull().sum().sum(),
        "describe":           df.describe(),
    }


# ─────────────────────────────────────────────
# 7. CROP INFORMATION (Nepal-specific context)
# ─────────────────────────────────────────────

CROP_INFO = {
    "rice": {
        "season": "Kharif (June–November)",
        "regions": "Terai, Inner Terai",
        "description": "Staple food crop of Nepal. Requires high humidity and rainfall.",
        "tips": "Ensure proper irrigation. Transplant seedlings after monsoon onset.",
    },
    "wheat": {
        "season": "Rabi (November–April)",
        "regions": "Terai, Mid-Hills",
        "description": "Second most important cereal crop in Nepal.",
        "tips": "Sow after rice harvest. Requires cool temperatures during grain filling.",
    },
    "maize": {
        "season": "Spring & Kharif",
        "regions": "Hills, Mid-Hills, Terai",
        "description": "Widely grown across altitudinal zones in Nepal.",
        "tips": "Intercrop with legumes for better soil health.",
    },
    "chickpea": {
        "season": "Rabi (October–March)",
        "regions": "Terai, Inner Terai",
        "description": "Important pulse crop providing protein-rich food.",
        "tips": "Grows well in well-drained loamy soils. Avoid waterlogging.",
    },
    "kidneybeans": {
        "season": "Kharif",
        "regions": "Hills, Mid-Hills",
        "description": "Popular legume crop in hilly regions of Nepal.",
        "tips": "Requires moderate rainfall and well-drained soil.",
    },
    "pigeonpeas": {
        "season": "Kharif",
        "regions": "Terai",
        "description": "Drought-tolerant pulse crop suitable for Terai region.",
        "tips": "Can be intercropped with cereals.",
    },
    "mothbeans": {
        "season": "Kharif",
        "regions": "Terai",
        "description": "Drought-resistant legume suitable for dry areas.",
        "tips": "Grows well in sandy loam soils with low rainfall.",
    },
    "mungbean": {
        "season": "Spring & Summer",
        "regions": "Terai, Inner Terai",
        "description": "Short-duration pulse crop with high nutritional value.",
        "tips": "Ideal for crop rotation after wheat.",
    },
    "blackgram": {
        "season": "Kharif",
        "regions": "Terai",
        "description": "Important pulse crop in the Terai belt.",
        "tips": "Sensitive to waterlogging. Ensure good drainage.",
    },
    "lentil": {
        "season": "Rabi",
        "regions": "Terai, Mid-Hills",
        "description": "Widely consumed pulse crop in Nepal.",
        "tips": "Sow in October–November for best yields.",
    },
    "pomegranate": {
        "season": "Perennial",
        "regions": "Mid-Hills, Inner Terai",
        "description": "Fruit crop with growing commercial potential in Nepal.",
        "tips": "Requires well-drained soil and moderate irrigation.",
    },
    "banana": {
        "season": "Perennial",
        "regions": "Terai, Inner Terai",
        "description": "Popular fruit crop in the lowland regions.",
        "tips": "Requires high humidity and regular watering.",
    },
    "mango": {
        "season": "Perennial",
        "regions": "Terai, Inner Terai",
        "description": "Major fruit crop with high market demand.",
        "tips": "Prune regularly and apply balanced fertiliser.",
    },
    "grapes": {
        "season": "Perennial",
        "regions": "Mid-Hills",
        "description": "Emerging fruit crop in Nepal's hill regions.",
        "tips": "Requires trellis support and well-drained soil.",
    },
    "watermelon": {
        "season": "Summer",
        "regions": "Terai",
        "description": "Popular summer fruit with high water content.",
        "tips": "Requires sandy loam soil and warm temperatures.",
    },
    "muskmelon": {
        "season": "Summer",
        "regions": "Terai",
        "description": "Sweet fruit crop grown in warm lowland areas.",
        "tips": "Needs full sun and well-drained soil.",
    },
    "apple": {
        "season": "Perennial",
        "regions": "High Hills, Mountains",
        "description": "Important fruit crop in Nepal's high-altitude regions.",
        "tips": "Requires chilling hours and well-drained loamy soil.",
    },
    "orange": {
        "season": "Perennial",
        "regions": "Mid-Hills",
        "description": "Major citrus crop in Nepal's mid-hill regions.",
        "tips": "Requires moderate rainfall and well-drained soil.",
    },
    "papaya": {
        "season": "Perennial",
        "regions": "Terai, Inner Terai",
        "description": "Fast-growing fruit crop with year-round production.",
        "tips": "Sensitive to frost. Requires warm temperatures.",
    },
    "coconut": {
        "season": "Perennial",
        "regions": "Terai (limited)",
        "description": "Tropical crop grown in the southernmost Terai belt.",
        "tips": "Requires high temperatures and humidity.",
    },
    "cotton": {
        "season": "Kharif",
        "regions": "Terai",
        "description": "Cash crop grown in the Terai region.",
        "tips": "Requires deep, well-drained soil and warm climate.",
    },
    "jute": {
        "season": "Kharif",
        "regions": "Terai",
        "description": "Important fibre crop grown in the Terai belt.",
        "tips": "Requires high humidity and alluvial soil.",
    },
    "coffee": {
        "season": "Perennial",
        "regions": "Mid-Hills",
        "description": "High-value cash crop with growing export potential.",
        "tips": "Grows best in shade with well-drained acidic soil.",
    },
}

def get_crop_info(crop_name: str) -> dict:
    """Return Nepal-specific information for a given crop."""
    return CROP_INFO.get(crop_name.lower(), {
        "season": "Varies",
        "regions": "Varies by altitude",
        "description": "Crop information not available.",
        "tips": "Consult local agricultural extension services.",
    })
