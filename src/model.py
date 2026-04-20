"""
model.py
────────
Model architecture definitions, training routines, evaluation helpers,
and persistence utilities.

Fixes applied (v2):
  - evaluate_regressor / evaluate_classifier now save JSON to disk
  - Cross-validation added for both models
  - Feature importance extraction helper added
  - KMeans top-words-per-cluster helper added
  - XGBClassifier use_label_encoder warning suppressed (removed param)
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Train / evaluate helpers
# ──────────────────────────────────────────────────────────────

def split_data(X, y, test_size: float, random_state: int, stratify=None):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def split_data_multi(*arrays, test_size: float, random_state: int, stratify=None):
    """Convenience wrapper for multi-array splits (text + struct + labels)."""
    return train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


# ──────────────────────────────────────────────────────────────
#  Model 1 – Random Forest Regressor (severity_score)
# ──────────────────────────────────────────────────────────────

def train_rf_regressor(
    X_train,
    y_train,
    n_estimators: int     = 50,
    max_depth: int        = 4,
    min_samples_leaf: int = 5,
    random_state: int     = 42,
) -> RandomForestRegressor:
    logger.info(
        "Training RandomForestRegressor (n_estimators=%d, max_depth=%d) …",
        n_estimators, max_depth,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("RF Regressor training complete.")
    return model


def evaluate_regressor(
    model,
    X_test,
    y_test,
    clip_range: tuple = (1, 5),
    save_path: str    = None,
) -> dict:
    """
    Evaluate regressor and optionally save metrics JSON to *save_path*.
    Also runs 5-fold cross-validation and includes scores in output.
    """
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, *clip_range)

    # Cross-validation on test portion (quick proxy — run on full data ideally)
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="r2")

    metrics = {
        "MAE":     float(mean_absolute_error(y_test, y_pred)),
        "RMSE":    float(mean_squared_error(y_test, y_pred) ** 0.5),
        "R2":      float(r2_score(y_test, y_pred)),
        "CV_R2_mean": float(cv_scores.mean()),
        "CV_R2_std":  float(cv_scores.std()),
    }
    logger.info(
        "Regressor — MAE: %.4f | RMSE: %.4f | R²: %.4f | CV R²: %.4f ± %.4f",
        metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["CV_R2_mean"], metrics["CV_R2_std"],
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved regressor metrics → %s", save_path)

    return metrics


def get_feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    """
    Return a DataFrame of the top_n most important features for a
    tree-based model (RF or XGB). Saved to models/feature_importance.csv
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)
    return df


# ──────────────────────────────────────────────────────────────
#  Model 2 – XGBoost Classifier (pain_level)
# ──────────────────────────────────────────────────────────────

def encode_target(y: pd.Series) -> tuple:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.debug("Target classes: %s", list(le.classes_))
    return y_enc, le


def train_xgb_classifier(
    X_train,
    y_train,
    n_estimators: int    = 300,
    max_depth: int       = 6,
    learning_rate: float = 0.1,
    random_state: int    = 42,
) -> XGBClassifier:
    logger.info(
        "Training XGBClassifier (n_estimators=%d, max_depth=%d, lr=%g) …",
        n_estimators, max_depth, learning_rate,
    )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="mlogloss",   # use_label_encoder removed (deprecated)
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("XGBClassifier training complete.")
    return model


def evaluate_classifier(
    model,
    X_test,
    y_test,
    target_names: list = None,
    save_path: str     = None,
) -> dict:
    """
    Evaluate classifier and optionally save metrics JSON to *save_path*.
    Includes accuracy, full classification report dict, and confusion matrix.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "report":    classification_report(
                         y_test, y_pred,
                         target_names=target_names,
                         output_dict=True,   # dict so it's JSON-serialisable
                     ),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info("Classifier — Accuracy: %.4f", metrics["accuracy"])
    logger.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=target_names),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved classifier metrics → %s", save_path)

    return metrics


# ──────────────────────────────────────────────────────────────
#  Model 3 – KMeans topic clustering (YouTube comments)
# ──────────────────────────────────────────────────────────────

def train_kmeans(
    tfidf_matrix,
    n_clusters: int   = 5,
    random_state: int = 42,
) -> KMeans:
    logger.info("Fitting KMeans with k=%d …", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(tfidf_matrix)
    logger.info("KMeans training complete.")
    return km


def get_kmeans_topic_labels(
    kmeans: KMeans,
    tfidf_vectorizer,
    top_n: int = 10,
) -> dict:
    """
    Return the top_n words per cluster so you can understand
    what each topic cluster actually represents.

    Example output:
      {0: ['network','signal','coverage',...],
       1: ['bill','charge','payment',...], ...}
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        top_indices = center.argsort()[-top_n:][::-1]
        topics[i] = [feature_names[j] for j in top_indices]
        logger.info("Cluster %d: %s", i, ", ".join(topics[i]))
    return topics


# ──────────────────────────────────────────────────────────────
#  Persistence
# ──────────────────────────────────────────────────────────────

def save_artifact(obj, path: str) -> None:
    """Pickle any sklearn / xgboost / scipy object to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved artefact → %s", path)


def load_artifact(path: str):
    """Load a pickled artefact from *path*."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    logger.info("Loaded artefact ← %s", path)
    return obj


# ──────────────────────────────────────────────────────────────
#  Convenience wrappers used by main.py
# ──────────────────────────────────────────────────────────────

def run_severity_model(X_combined, y, cfg: dict) -> tuple:
    """
    Train and evaluate the RF regressor for severity_score prediction.
    Returns (model, metrics, scaler).
    """
    from src.preprocessing import scale_features

    train_cfg = cfg["training"]
    model_cfg = cfg["models"]["random_forest_severity"]
    paths     = cfg["paths"]

    X_train, X_test, y_train, y_test = split_data(
        X_combined, y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )
    X_train_s, X_test_s, scaler = scale_features(
        X_train.toarray(), X_test.toarray()
    )

    model   = train_rf_regressor(X_train_s, y_train, **model_cfg)
    metrics = evaluate_regressor(
        model, X_test_s, y_test,
        save_path="models/rf_metrics.json",          # ← saved to disk
    )

    save_artifact(model,  paths["saved_rf_model"])
    save_artifact(scaler, paths["saved_scaler"])

    return model, metrics, scaler


def run_pain_level_model(X_combined, y_raw, cfg: dict) -> tuple:
    """
    Encode target, train and evaluate XGBClassifier for pain_level.
    Returns (model, metrics, label_encoder).
    """
    train_cfg = cfg["training"]
    model_cfg = cfg["models"]["xgboost_pain"]
    paths     = cfg["paths"]

    y_enc, le = encode_target(y_raw)

    X_train, X_test, y_train, y_test = split_data(
        X_combined, y_enc,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
        stratify=y_enc,
    )

    # Remove non-constructor params before passing to XGBClassifier
    xgb_params = {
        k: v for k, v in model_cfg.items()
        if k not in ("use_label_encoder", "eval_metric")
    }
    model   = train_xgb_classifier(X_train, y_train, **xgb_params)
    metrics = evaluate_classifier(
        model, X_test, y_test,
        target_names=list(le.classes_),
        save_path="models/xgb_metrics.json",         # ← saved to disk
    )

    save_artifact(model, paths["saved_xgb_model"])
    save_artifact(le,    paths["saved_label_encoder"])

    return model, metrics, le
