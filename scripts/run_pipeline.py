"""
run_pipeline.py
===============
End-to-end orchestration script for the IndoorOutdoorStudy ML pipeline.

Usage
-----
    python scripts/run_pipeline.py \\
        --outdoor  data/raw/outdoor_sensor.csv \\
        --indoor-dir data/raw/indoor/ \\
        --output   output/ \\
        [--no-tune] [--pm-bins pm1 pm2_5 pm10] [--n-sensors 20]

Steps
-----
1. Load and time-average outdoor data (1 s → 10 min).
2. Load all 20 indoor sensors and align timestamps.
3. Run lag analysis; save results to ``data/lag_results/``.
4. Apply lag corrections and build merged DataFrame.
5. Train/evaluate one XGBoost model per (PM bin, sensor).
6. Save models and summary metrics.
7. Generate diagnostic plots.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import (
    load_outdoor_data,
    time_average_outdoor,
    load_all_indoor_sensors,
    align_indoor_outdoor,
)
from src.lag_analysis import (
    run_lag_analysis,
    apply_lags,
    save_lag_results,
)
from src.model import run_all_models
from src.visualization import (
    plot_lag_heatmap,
    plot_model_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IndoorOutdoorStudy – XGBoost ΔPM prediction pipeline"
    )
    p.add_argument(
        "--outdoor",
        required=True,
        help="Path to the outdoor sensor CSV (1-second resolution).",
    )
    p.add_argument(
        "--indoor-dir",
        required=True,
        dest="indoor_dir",
        help="Directory containing indoor sensor CSVs (one per sensor).",
    )
    p.add_argument(
        "--output",
        default="output",
        help="Root output directory (default: output/).",
    )
    p.add_argument(
        "--resample-freq",
        default="10min",
        dest="resample_freq",
        help="Target time resolution for outdoor resampling (default: 10min).",
    )
    p.add_argument(
        "--pm-bins",
        nargs="+",
        default=["pm1", "pm2_5", "pm10"],
        dest="pm_bins",
        help="PM bins to model (default: pm1 pm2_5 pm10).",
    )
    p.add_argument(
        "--n-sensors",
        type=int,
        default=20,
        dest="n_sensors",
        help="Number of indoor sensors (default: 20).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        dest="test_fraction",
        help="Fraction of data used as temporal test set (default: 0.2).",
    )
    p.add_argument(
        "--max-lag-steps",
        type=int,
        default=60,
        dest="max_lag_steps",
        help="Maximum ±lag in 10-min steps for cross-correlation (default: 60).",
    )
    p.add_argument(
        "--sensor-pattern",
        default="sensor_*.csv",
        dest="sensor_pattern",
        help="Glob pattern for indoor sensor files (default: sensor_*.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    lag_results_dir = output_root / "lag_results"
    models_dir = output_root / "models"
    plots_dir = output_root / "plots"

    for d in [lag_results_dir, models_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load and downsample outdoor data
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Load outdoor data ===")
    outdoor_raw = load_outdoor_data(args.outdoor)
    outdoor_avg = time_average_outdoor(outdoor_raw, freq=args.resample_freq)

    # ------------------------------------------------------------------
    # Step 2: Load indoor sensors
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Load indoor sensors ===")
    indoor_combined = load_all_indoor_sensors(
        args.indoor_dir, pattern=args.sensor_pattern
    )
    outdoor_aligned, indoor_aligned = align_indoor_outdoor(outdoor_avg, indoor_combined)

    # ------------------------------------------------------------------
    # Step 3: Lag analysis
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Lag analysis ===")
    lag_df = run_lag_analysis(
        outdoor_aligned,
        indoor_aligned,
        n_sensors=args.n_sensors,
        max_lag_steps=args.max_lag_steps,
    )
    save_lag_results(lag_df, lag_results_dir)

    # Lag heatmaps
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

    # ------------------------------------------------------------------
    # Step 4: Apply lags and build merged DataFrame
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Apply lag corrections ===")
    merged_df = apply_lags(outdoor_aligned, indoor_aligned, lag_df)
    merged_path = output_root / "merged_lag_corrected.csv"
    merged_df.to_csv(merged_path)
    logger.info("Merged DataFrame saved to %s (%d rows)", merged_path, len(merged_df))

    # ------------------------------------------------------------------
    # Step 5–6: Train and evaluate all models
    # ------------------------------------------------------------------
    logger.info("=== Step 5-6: Train XGBoost models ===")
    summary_df = run_all_models(
        merged_df=merged_df,
        lag_df=lag_df,
        output_dir=models_dir,
        pm_bins=args.pm_bins,
        n_sensors=args.n_sensors,
        test_fraction=args.test_fraction,
    )

    logger.info("\nModel summary:\n%s", summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 7: Summary plots
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Generate summary plots ===")
    plot_model_summary(summary_df, output_dir=plots_dir)

    logger.info("Pipeline complete.  All outputs written to %s", output_root)


if __name__ == "__main__":
    main()
