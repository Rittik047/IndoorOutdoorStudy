"""
lag_analysis.py
===============
Cross-correlation-based lag analysis between each indoor sensor and the
outdoor reference for each PM bin (PM1, PM2.5, PM10).

Workflow
--------
1. :func:`compute_lag` – for a single (sensor, PM-bin) pair, compute the
   cross-correlation and return the lag that maximises it.
2. :func:`run_lag_analysis` – iterate over all 20 sensors and 3 PM bins
   and return a summary DataFrame.
3. :func:`apply_lags` – shift the indoor (or outdoor) time series to
   compensate for the identified lags.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

logger = logging.getLogger(__name__)

# PM bins targeted by indoor sensors
INDOOR_PM_BINS: list[str] = ["pm1", "pm2_5", "pm10"]

# Mapping from indoor PM bin label to outdoor PM density column name
OUTDOOR_PM_COL: dict[str, str] = {
    "pm1":   "pm1_density",
    "pm2_5": "pm2_5_density",
    "pm10":  "pm10_density",
}

# Maximum lag to search in each direction (number of 10-minute intervals)
MAX_LAG_STEPS: int = 60   # ±60 × 10 min = ±10 hours


def _normalise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation; returns zeros if std ≈ 0."""
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def compute_lag(
    outdoor_series: pd.Series,
    indoor_series: pd.Series,
    max_lag_steps: int = MAX_LAG_STEPS,
) -> dict:
    """Compute the optimal lag between *outdoor_series* and *indoor_series*.

    The lag is defined as the shift (in number of samples) that must be
    applied to the *indoor* series to best match the *outdoor* series.
    A positive lag means the indoor sensor responds *after* the outdoor
    signal.

    Parameters
    ----------
    outdoor_series:
        Outdoor PM time series (10-minute resolution, aligned with indoor).
    indoor_series:
        Indoor PM time series (10-minute resolution) for a single sensor.
    max_lag_steps:
        Maximum lag magnitude in number of samples to consider.

    Returns
    -------
    dict with keys:
        ``lag_steps``   – optimal lag in samples (positive → indoor lags behind)
        ``lag_minutes`` – optimal lag in minutes
        ``max_corr``    – peak cross-correlation value (normalised, −1 to 1)
        ``corr_profile`` – full normalised cross-correlation array
        ``lags``        – corresponding lag array (samples)
    """
    # Drop rows where either series is NaN
    combined = pd.concat([outdoor_series, indoor_series], axis=1).dropna()
    if len(combined) < 10:
        logger.warning("Too few valid samples (%d) to compute lag.", len(combined))
        return {
            "lag_steps": 0,
            "lag_minutes": 0,
            "max_corr": np.nan,
            "corr_profile": np.array([]),
            "lags": np.array([]),
        }

    out_norm = _normalise(combined.iloc[:, 0].values)
    inn_norm = _normalise(combined.iloc[:, 1].values)

    full_corr = correlate(out_norm, inn_norm, mode="full")
    full_lags = correlation_lags(len(out_norm), len(inn_norm), mode="full")

    # Restrict to ±max_lag_steps
    mask = np.abs(full_lags) <= max_lag_steps
    corr = full_corr[mask] / len(combined)   # normalise to [-1, 1]
    lags = full_lags[mask]

    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])

    return {
        "lag_steps": best_lag,
        "lag_minutes": best_lag * 10,
        "max_corr": float(corr[best_idx]),
        "corr_profile": corr,
        "lags": lags,
    }


def run_lag_analysis(
    outdoor_avg: pd.DataFrame,
    indoor_combined: pd.DataFrame,
    n_sensors: int = 20,
    max_lag_steps: int = MAX_LAG_STEPS,
) -> pd.DataFrame:
    """Run lag analysis for all sensors × PM bins.

    Parameters
    ----------
    outdoor_avg:
        10-minute averaged outdoor DataFrame; must contain columns
        ``pm1_density``, ``pm2_5_density``, ``pm10_density``.
    indoor_combined:
        Combined indoor DataFrame with columns
        ``pm{bin}_sensor{i}`` (i = 1 … n_sensors).
    n_sensors:
        Number of indoor sensors.
    max_lag_steps:
        Passed through to :func:`compute_lag`.

    Returns
    -------
    pd.DataFrame
        Columns: sensor_id, pm_bin, lag_steps, lag_minutes, max_corr.
        One row per (sensor, PM-bin) pair.
    """
    records: list[dict] = []

    for pm_bin in INDOOR_PM_BINS:
        out_col = OUTDOOR_PM_COL[pm_bin]
        if out_col not in outdoor_avg.columns:
            logger.warning("Column %s not found in outdoor data; skipping.", out_col)
            continue

        outdoor_series = outdoor_avg[out_col]

        for sensor_id in range(1, n_sensors + 1):
            in_col = f"{pm_bin}_sensor{sensor_id}"
            if in_col not in indoor_combined.columns:
                logger.warning("Column %s not in indoor data; skipping.", in_col)
                continue

            indoor_series = indoor_combined[in_col]
            result = compute_lag(
                outdoor_series, indoor_series, max_lag_steps=max_lag_steps
            )

            records.append(
                {
                    "sensor_id": sensor_id,
                    "pm_bin": pm_bin,
                    "lag_steps": result["lag_steps"],
                    "lag_minutes": result["lag_minutes"],
                    "max_corr": result["max_corr"],
                }
            )
            logger.debug(
                "Sensor %d | %s | lag=%d steps (%d min) | corr=%.3f",
                sensor_id,
                pm_bin,
                result["lag_steps"],
                result["lag_minutes"],
                result["max_corr"] if not np.isnan(result["max_corr"]) else -999,
            )

    lag_df = pd.DataFrame(records)
    logger.info("Lag analysis complete: %d sensor×bin pairs", len(lag_df))
    return lag_df


def apply_lags(
    outdoor_avg: pd.DataFrame,
    indoor_combined: pd.DataFrame,
    lag_df: pd.DataFrame,
) -> pd.DataFrame:
    """Shift indoor PM series by their identified lags and return a merged DataFrame.

    For each (sensor, PM-bin) pair with a positive lag *L*, the indoor
    sensor column is shifted *forward* by L steps so that its timestamps
    align with the outdoor reference.  (Shifting the indoor series
    backwards in time is equivalent to shifting the outdoor series forward.)

    Parameters
    ----------
    outdoor_avg:
        10-minute outdoor DataFrame (predictor features + outdoor PM columns).
    indoor_combined:
        Combined indoor DataFrame.
    lag_df:
        Output of :func:`run_lag_analysis`.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with outdoor features and lag-corrected indoor PM
        columns.  Rows where the shift introduced NaN are dropped.
    """
    indoor_shifted = indoor_combined.copy()

    for _, row in lag_df.iterrows():
        pm_bin = row["pm_bin"]
        sensor_id = int(row["sensor_id"])
        lag_steps = int(row["lag_steps"])
        in_col = f"{pm_bin}_sensor{sensor_id}"

        if in_col not in indoor_shifted.columns:
            continue
        if lag_steps != 0:
            indoor_shifted[in_col] = indoor_shifted[in_col].shift(-lag_steps)

    merged = outdoor_avg.join(indoor_shifted, how="inner")
    merged = merged.dropna(
        subset=[
            c for c in indoor_shifted.columns if c in merged.columns
        ]
    )
    logger.info(
        "After lag correction: %d rows remain in merged DataFrame", len(merged)
    )
    return merged


def save_lag_results(
    lag_df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "lag_results.csv",
) -> Path:
    """Save lag analysis results to CSV.

    Parameters
    ----------
    lag_df:
        Output of :func:`run_lag_analysis`.
    output_dir:
        Destination directory.
    filename:
        Output filename (default: ``lag_results.csv``).

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    lag_df.to_csv(out_path, index=False)
    logger.info("Lag results saved to %s", out_path)
    return out_path


def load_lag_results(filepath: str | Path) -> pd.DataFrame:
    """Load previously saved lag results from CSV.

    Parameters
    ----------
    filepath:
        Path to the CSV produced by :func:`save_lag_results`.

    Returns
    -------
    pd.DataFrame
    """
    lag_df = pd.read_csv(filepath)
    logger.info("Lag results loaded from %s (%d rows)", filepath, len(lag_df))
    return lag_df
