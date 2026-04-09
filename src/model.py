"""
model.py
========
XGBoost-based model training, cross-validation, hyper-parameter tuning,
and evaluation for predicting ΔPM = (outdoor PM − indoor PM).

One model is trained per (PM bin, sensor) pair.  Results are persisted to
``output/models/`` using joblib.

Typical usage
-------------
>>> from src.model import train_model, evaluate_model, save_model
>>> result = train_model(X_train, y_train)
>>> metrics = evaluate_model(result["model"], X_test, y_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default XGBoost hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
}

# Hyper-parameter search space (used by tune_hyperparameters)
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    xgb_params: Optional[dict] = None,
) -> dict:
    """Train an XGBoost regressor.

    Parameters
    ----------
    X_train, y_train:
        Training feature matrix and target vector.
    X_val, y_val:
        Optional validation set for early stopping.  If not provided, 20 %
        of *X_train* is held out automatically.
    xgb_params:
        XGBoost hyper-parameter dict.  Defaults to :data:`DEFAULT_XGB_PARAMS`.

    Returns
    -------
    dict with keys:
        ``model``         – fitted :class:`xgboost.XGBRegressor`
        ``feature_names`` – list of feature column names
        ``val_rmse``      – validation RMSE (float)
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    if X_val is None or y_val is None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )
    else:
        X_tr, y_tr = X_train, y_train

    # early_stopping_rounds is a constructor argument in XGBoost >= 2.0.
    # Pop it from the shared params dict (which may come from DEFAULT_XGB_PARAMS
    # defined without it as a constructor key) and pass explicitly so the same
    # dict can be used with both old and new XGBoost versions.
    params["early_stopping_rounds"] = params.pop("early_stopping_rounds", 50)

    model = XGBRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_val = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    logger.info(
        "Training complete. Best iteration: %d | Val RMSE: %.4f",
        model.best_iteration,
        val_rmse,
    )

    return {
        "model": model,
        "feature_names": list(X_tr.columns),
        "val_rmse": val_rmse,
    }


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    cv_folds: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
) -> dict:
    """Randomised hyper-parameter search with time-series cross-validation.

    Parameters
    ----------
    X, y:
        Full feature matrix and target (will be split inside CV).
    n_iter:
        Number of random parameter settings sampled.
    cv_folds:
        Number of CV folds.
    scoring:
        scikit-learn scoring string.
    random_state:
        Random seed.

    Returns
    -------
    dict with keys:
        ``best_params`` – best parameter dict
        ``best_score``  – best cross-validation score
        ``cv_results``  – full CV results DataFrame
    """
    # Use TimeSeriesSplit to respect temporal ordering
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=scoring,
        cv=tscv,
        verbose=0,
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X, y)

    logger.info(
        "Hyper-parameter tuning complete. Best score: %.4f", search.best_score_
    )
    return {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained model on a held-out test set.

    Parameters
    ----------
    model:
        Fitted :class:`xgboost.XGBRegressor`.
    X_test, y_test:
        Test feature matrix and target.

    Returns
    -------
    dict with keys:
        ``rmse``, ``mae``, ``r2``, ``y_pred`` (np.ndarray)
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    logger.info("Test RMSE=%.4f  MAE=%.4f  R²=%.4f", rmse, mae, r2)
    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    xgb_params: Optional[dict] = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Time-series k-fold cross-validation.

    Parameters
    ----------
    X, y:
        Full data (will be split temporally).
    xgb_params:
        XGBoost hyper-parameter overrides.
    n_splits:
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns: fold, rmse, mae, r2.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records: list[dict] = []
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    # Remove early stopping for CV (no explicit eval set)
    params.pop("early_stopping_rounds", None)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, verbose=False)
        metrics = evaluate_model(model, X_te, y_te)
        records.append(
            {
                "fold": fold,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

    cv_df = pd.DataFrame(records)
    logger.info(
        "CV results:\n%s\nMean RMSE=%.4f  Mean R²=%.4f",
        cv_df.to_string(index=False),
        cv_df["rmse"].mean(),
        cv_df["r2"].mean(),
    )
    return cv_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    train_result: dict,
    output_dir: str | Path,
    pm_bin: str,
    sensor_id: int,
) -> Path:
    """Serialise a trained model to disk.

    Parameters
    ----------
    train_result:
        Output dict from :func:`train_model`.
    output_dir:
        Directory to write the model file.
    pm_bin:
        PM bin label (e.g. ``"pm2_5"``).
    sensor_id:
        Indoor sensor identifier.

    Returns
    -------
    Path
        Path to the saved model file (``.joblib``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"xgb_{pm_bin}_sensor{sensor_id:02d}.joblib"
    out_path = output_dir / filename
    joblib.dump(train_result, out_path)
    logger.info("Model saved to %s", out_path)
    return out_path


def load_model(filepath: str | Path) -> dict:
    """Load a serialised model from disk.

    Parameters
    ----------
    filepath:
        Path produced by :func:`save_model`.

    Returns
    -------
    dict
        Same structure as the output of :func:`train_model`.
    """
    result = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def run_all_models(
    merged_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    output_dir: str | Path,
    pm_bins: Optional[list[str]] = None,
    n_sensors: int = 20,
    test_fraction: float = 0.2,
    xgb_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Train and evaluate one XGBoost model per (PM bin, sensor) pair.

    Parameters
    ----------
    merged_df:
        Lag-corrected merged DataFrame.
    lag_df:
        Output of :func:`lag_analysis.run_lag_analysis` (unused for training
        itself but included for reference).
    output_dir:
        Root directory for saved models.
    pm_bins:
        PM bins to model (default: ``["pm1", "pm2_5", "pm10"]``).
    n_sensors:
        Number of indoor sensors.
    test_fraction:
        Fraction of data held out as a temporal test set.
    xgb_params:
        XGBoost hyper-parameter overrides.

    Returns
    -------
    pd.DataFrame
        Summary metrics (sensor_id, pm_bin, rmse, mae, r2, val_rmse).
    """
    from .feature_engineering import build_feature_matrix

    if pm_bins is None:
        pm_bins = ["pm1", "pm2_5", "pm10"]

    records: list[dict] = []
    output_dir = Path(output_dir)

    for pm_bin in pm_bins:
        for sensor_id in range(1, n_sensors + 1):
            try:
                X, y = build_feature_matrix(merged_df, pm_bin, sensor_id)
            except KeyError as exc:
                logger.warning("Skipping %s sensor%d: %s", pm_bin, sensor_id, exc)
                continue

            if len(X) < 30:
                logger.warning(
                    "Skipping %s sensor%d: too few samples (%d).",
                    pm_bin, sensor_id, len(X),
                )
                continue

            # Temporal train/test split (no shuffle)
            split_idx = int(len(X) * (1 - test_fraction))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            train_result = train_model(X_train, y_train, xgb_params=xgb_params)
            test_metrics = evaluate_model(train_result["model"], X_test, y_test)

            save_model(train_result, output_dir, pm_bin, sensor_id)

            records.append(
                {
                    "sensor_id": sensor_id,
                    "pm_bin": pm_bin,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "val_rmse": train_result["val_rmse"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                }
            )
            logger.info(
                "✓ %s sensor%d | RMSE=%.4f | R²=%.4f",
                pm_bin, sensor_id,
                test_metrics["rmse"],
                test_metrics["r2"],
            )

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Model summary written to %s", summary_path)
    return summary_df
