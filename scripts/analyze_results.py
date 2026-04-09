"""
analyze_results.py
==================
Post-run analysis helper: load saved models and generate per-sensor,
per-PM-bin diagnostic plots.

Usage
-----
    python scripts/analyze_results.py \\
        --merged   output/merged_lag_corrected.csv \\
        --models   output/models/ \\
        --lag-csv  output/lag_results/lag_results.csv \\
        --plots    output/plots/

The script expects the models directory to contain files named:
    ``xgb_{pm_bin}_sensor{id:02d}.joblib``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_processing import SITE_TIMEZONE
from src.feature_engineering import build_feature_matrix
from src.lag_analysis import load_lag_results
from src.model import load_model, evaluate_model
from src.visualization import (
    plot_predictions,
    plot_feature_importance,
    plot_lag_heatmap,
    plot_lag_profile,
    plot_pm_time_series,
    plot_model_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_results")

INDOOR_PM_BINS = ["pm1", "pm2_5", "pm10"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IndoorOutdoorStudy – post-run analysis and plotting"
    )
    p.add_argument(
        "--merged",
        required=True,
        help="Path to the merged lag-corrected CSV.",
    )
    p.add_argument(
        "--models",
        required=True,
        help="Directory containing saved .joblib model files.",
    )
    p.add_argument(
        "--lag-csv",
        required=True,
        dest="lag_csv",
        help="Path to lag_results.csv.",
    )
    p.add_argument(
        "--plots",
        default="output/plots",
        help="Directory where plots are saved.",
    )
    p.add_argument(
        "--n-sensors",
        type=int,
        default=20,
        dest="n_sensors",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        dest="test_fraction",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)

    # Load merged DataFrame
    logger.info("Loading merged DataFrame from %s", args.merged)
    merged_df = pd.read_csv(args.merged, index_col=0, parse_dates=True)
    if merged_df.index.tz is None:
        merged_df = merged_df.tz_localize("UTC").tz_convert(SITE_TIMEZONE)
    else:
        merged_df = merged_df.tz_convert(SITE_TIMEZONE)

    # Load lag results and re-draw heatmaps
    lag_df = load_lag_results(args.lag_csv)
    plot_lag_heatmap(
        lag_df,
        metric="lag_minutes",
        output_path=plots_dir / "lag_heatmap_minutes.png",
    )
    plot_lag_heatmap(
        lag_df,
        metric="max_corr",
        output_path=plots_dir / "lag_heatmap_correlation.png",
    )

    # Collect model summary for the final box plot
    summary_records: list[dict] = []

    for pm_bin in INDOOR_PM_BINS:
        pm_plots = plots_dir / pm_bin
        pm_plots.mkdir(parents=True, exist_ok=True)

        for sensor_id in range(1, args.n_sensors + 1):
            model_path = models_dir / f"xgb_{pm_bin}_sensor{sensor_id:02d}.joblib"
            if not model_path.exists():
                logger.warning("Model file not found: %s", model_path)
                continue

            try:
                X, y = build_feature_matrix(merged_df, pm_bin, sensor_id)
            except KeyError as exc:
                logger.warning(
                    "Cannot build features for %s sensor%d: %s", pm_bin, sensor_id, exc
                )
                continue

            # Temporal test split
            split_idx = int(len(X) * (1 - args.test_fraction))
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]

            train_result = load_model(model_path)
            model = train_result["model"]
            feature_names = train_result["feature_names"]

            metrics = evaluate_model(model, X_test, y_test)
            y_pred = metrics["y_pred"]

            # Predictions plot
            plot_predictions(
                y_test,
                y_pred,
                pm_bin=pm_bin,
                sensor_id=sensor_id,
                output_path=pm_plots / f"predictions_sensor{sensor_id:02d}.png",
            )

            # Feature importance
            plot_feature_importance(
                model,
                feature_names,
                top_n=20,
                pm_bin=pm_bin,
                sensor_id=sensor_id,
                output_path=pm_plots / f"feature_importance_sensor{sensor_id:02d}.png",
            )

            # PM time series (outdoor vs indoor, before lag correction)
            outdoor_col = (
                "pm1_density" if pm_bin == "pm1"
                else "pm2_5_density" if pm_bin == "pm2_5"
                else "pm10_density"
            )
            indoor_col = f"{pm_bin}_sensor{sensor_id}"
            if outdoor_col in merged_df.columns and indoor_col in merged_df.columns:
                plot_pm_time_series(
                    outdoor_pm=merged_df[outdoor_col],
                    indoor_pm=merged_df[indoor_col],
                    pm_bin=pm_bin,
                    sensor_id=sensor_id,
                    output_path=pm_plots / f"timeseries_sensor{sensor_id:02d}.png",
                )

            summary_records.append(
                {
                    "sensor_id": sensor_id,
                    "pm_bin": pm_bin,
                    "test_rmse": metrics["rmse"],
                    "test_mae": metrics["mae"],
                    "test_r2": metrics["r2"],
                }
            )

    # Summary metrics box plot
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(plots_dir / "analysis_summary.csv", index=False)
        plot_model_summary(summary_df, output_dir=plots_dir)
        logger.info(
            "\nAnalysis summary:\n%s",
            summary_df.groupby("pm_bin")[["test_rmse", "test_r2"]].describe().to_string(),
        )

    logger.info("Analysis complete.  Plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
