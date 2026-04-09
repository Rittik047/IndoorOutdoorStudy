"""
visualization.py
================
Diagnostic and results plots for the IndoorOutdoorStudy pipeline.

Functions
---------
plot_lag_profile        – cross-correlation vs. lag for one (sensor, PM-bin).
plot_lag_heatmap        – lag or correlation heatmap for all sensors × PM bins.
plot_feature_importance – top-N feature importances from a trained XGBoost model.
plot_predictions        – scatter and time-series of observed vs. predicted ΔPM.
plot_pm_time_series     – outdoor and indoor PM on the same axis for quick QC.
plot_model_summary      – box/bar plots of RMSE and R² across all models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Common style settings
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

# PM bin → display label
PM_DISPLAY: dict[str, str] = {
    "pm1":   "PM₁",
    "pm2_5": "PM₂.₅",
    "pm10":  "PM₁₀",
}


def _save_or_show(fig: plt.Figure, output_path: Optional[str | Path]) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Plot saved to %s", output_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Lag analysis plots
# ---------------------------------------------------------------------------

def plot_lag_profile(
    lags: np.ndarray,
    corr: np.ndarray,
    best_lag: int,
    pm_bin: str,
    sensor_id: int,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the cross-correlation profile for one (sensor, PM-bin) pair.

    Parameters
    ----------
    lags:
        Lag array in number of 10-minute steps (from :func:`lag_analysis.compute_lag`).
    corr:
        Normalised cross-correlation values.
    best_lag:
        Optimal lag in steps (marked with a vertical dashed line).
    pm_bin, sensor_id:
        Labels for the plot title.
    output_path:
        If given, save figure to this path; otherwise display interactively.
    """
    lag_minutes = lags * 10
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lag_minutes, corr, lw=1.5, color="#1f77b4")
    ax.axvline(best_lag * 10, color="crimson", ls="--", lw=1.5,
               label=f"Best lag = {best_lag * 10} min")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Normalised cross-correlation")
    ax.set_title(
        f"Lag profile – {PM_DISPLAY.get(pm_bin, pm_bin)}, Indoor Sensor {sensor_id}"
    )
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


def plot_lag_heatmap(
    lag_df: pd.DataFrame,
    metric: str = "lag_minutes",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of lag or correlation across all sensors × PM bins.

    Parameters
    ----------
    lag_df:
        Output of :func:`lag_analysis.run_lag_analysis`.
    metric:
        Column in *lag_df* to plot (``"lag_minutes"`` or ``"max_corr"``).
    output_path:
        Optional save path.
    """
    pivot = lag_df.pivot(index="sensor_id", columns="pm_bin", values=metric)
    pivot.columns = [PM_DISPLAY.get(c, c) for c in pivot.columns]
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = "RdBu_r" if metric == "lag_minutes" else "viridis"
    center = 0 if metric == "lag_minutes" else None
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=".0f" if metric == "lag_minutes" else ".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Lag (min)" if metric == "lag_minutes" else "Corr."},
    )
    ax.set_xlabel("PM Bin")
    ax.set_ylabel("Indoor Sensor ID")
    ax.set_title(
        ("Optimal Lag (minutes)" if metric == "lag_minutes"
         else "Peak Cross-Correlation")
        + " – All Sensors × PM Bins"
    )
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Model diagnostics
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    pm_bin: str = "",
    sensor_id: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N XGBoost feature importances.

    Parameters
    ----------
    model:
        Fitted :class:`xgboost.XGBRegressor`.
    feature_names:
        List of feature column names (same order as training columns).
    top_n:
        Number of top features to display.
    pm_bin, sensor_id:
        Labels for the plot title.
    output_path:
        Optional save path.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colours = plt.cm.viridis(top_values / top_values.max())
    ax.barh(top_features, top_values, color=colours, edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)")
    title = "Feature Importance"
    if pm_bin:
        title += f" – {PM_DISPLAY.get(pm_bin, pm_bin)}"
    if sensor_id is not None:
        title += f" Sensor {sensor_id}"
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    pm_bin: str = "",
    sensor_id: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Scatter plot and time-series of observed vs. predicted ΔPM.

    Parameters
    ----------
    y_true:
        Observed target values (pd.Series with DatetimeIndex).
    y_pred:
        Predicted values.
    pm_bin, sensor_id:
        Labels for the plot title.
    output_path:
        Optional save path.
    """
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    title_suffix = ""
    if pm_bin:
        title_suffix += f" – {PM_DISPLAY.get(pm_bin, pm_bin)}"
    if sensor_id is not None:
        title_suffix += f" Sensor {sensor_id}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#1f77b4")
    lim = [
        min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1,
    ]
    ax.plot(lim, lim, "r--", lw=1.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed ΔPM (µg m⁻³)")
    ax.set_ylabel("Predicted ΔPM (µg m⁻³)")
    ax.set_title(f"Observed vs. Predicted{title_suffix}")
    ax.text(
        0.05, 0.93, f"R²={r2:.3f}  RMSE={rmse:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
    )

    # Time series
    ax2 = axes[1]
    ax2.plot(y_true.index, y_true.values, lw=1, label="Observed", color="#1f77b4")
    ax2.plot(y_true.index, y_pred, lw=1, alpha=0.85, label="Predicted", color="crimson")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax2.set_ylabel("ΔPM (µg m⁻³)")
    ax2.set_title(f"Time Series{title_suffix}")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


def plot_pm_time_series(
    outdoor_pm: pd.Series,
    indoor_pm: pd.Series,
    pm_bin: str = "",
    sensor_id: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot outdoor and indoor PM concentrations on the same axis.

    Parameters
    ----------
    outdoor_pm:
        Outdoor PM time series.
    indoor_pm:
        Indoor PM time series (single sensor).
    pm_bin, sensor_id:
        Labels.
    output_path:
        Optional save path.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(outdoor_pm.index, outdoor_pm.values, lw=1, label="Outdoor", color="#e5851f")
    ax.plot(indoor_pm.index, indoor_pm.values, lw=1, alpha=0.8,
            label=f"Indoor Sensor {sensor_id}", color="#1f77b4")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.set_ylabel(f"{PM_DISPLAY.get(pm_bin, pm_bin) if pm_bin else 'PM'} (µg m⁻³)")
    title = "PM Concentration"
    if pm_bin:
        title += f" – {PM_DISPLAY.get(pm_bin, pm_bin)}"
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


def plot_model_summary(
    summary_df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Box/bar plots of RMSE and R² across all (PM bin, sensor) models.

    Parameters
    ----------
    summary_df:
        Output of :func:`model.run_all_models`.
    output_dir:
        Directory where plots are saved.  If None, displayed interactively.
    """
    for metric, ylabel in [("test_rmse", "RMSE (µg m⁻³)"), ("test_r2", "R²")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        data_by_bin = [
            summary_df.loc[summary_df["pm_bin"] == b, metric].dropna().values
            for b in ["pm1", "pm2_5", "pm10"]
        ]
        bp = ax.boxplot(
            data_by_bin,
            labels=[PM_DISPLAY["pm1"], PM_DISPLAY["pm2_5"], PM_DISPLAY["pm10"]],
            patch_artist=True,
        )
        colours = ["#4e79a7", "#f28e2b", "#e15759"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Distribution Across All Sensors")
        fig.tight_layout()
        if output_dir is not None:
            out = Path(output_dir) / f"summary_{metric}.png"
            _save_or_show(fig, out)
        else:
            plt.show()
