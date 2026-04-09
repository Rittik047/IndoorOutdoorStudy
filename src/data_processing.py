"""
data_processing.py
==================
Utilities for loading and pre-processing the outdoor (1-second) and indoor
(10-minute) sensor data used in the IndoorOutdoorStudy pipeline.

Outdoor sensor layout
---------------------
Sensor           | Columns
---------------- | ---------------------------------------------------------
Plantower IPS7100| pm{X}_density  (µg m⁻³), pm{X}_count (counts m⁻³)
                 | bins: 0.1, 0.3, 0.5, 1, 2.5, 5, 10
BME280           | bme280_temperature (°C), bme280_pressure (hPa),
                 |   bme280_humidity (%), bme280_dewpoint (°C)
RG-15 rain gauge | rainfall (mm)
AIRMAR           | airmar_wind_speed (m/s), airmar_wind_direction (°),
                 |   airmar_temperature (°C), airmar_pressure (hPa),
                 |   airmar_humidity (%), airmar_dewpoint (°C)
AS7265x          | as7265x_{wavelength}nm  – 18 channels (see CHANNEL_COLS)

Indoor sensors
--------------
20 sensors inside Richardson IQHQ building; each reports pm1, pm2_5, pm10
density (µg m⁻³) at 10-minute resolution.  Each sensor is represented by a
separate CSV (or by a per-sensor column prefix in a combined CSV).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

# IPS7100 PM size bins (µm)
PM_BINS: list[str] = ["0.1", "0.3", "0.5", "1", "2.5", "5", "10"]

# Derived column names: density and count for each bin
PM_DENSITY_COLS: list[str] = [f"pm{b.replace('.', '_')}_density" for b in PM_BINS]
PM_COUNT_COLS: list[str] = [f"pm{b.replace('.', '_')}_count" for b in PM_BINS]

# BME280
BME280_COLS: list[str] = [
    "bme280_temperature",
    "bme280_pressure",
    "bme280_humidity",
    "bme280_dewpoint",
]

# AIRMAR
AIRMAR_COLS: list[str] = [
    "airmar_wind_speed",
    "airmar_wind_direction",
    "airmar_temperature",
    "airmar_pressure",
    "airmar_humidity",
    "airmar_dewpoint",
]

# RG-15
RAIN_COLS: list[str] = ["rainfall"]

# AS7265x – 18 spectral channels (nm)
AS7265X_WAVELENGTHS: list[int] = [
    410, 435, 460, 485, 510, 535,   # die 1 (channels A–F)
    560, 585, 610, 645, 680, 705,   # die 2 (channels G–L)
    730, 760, 810, 860, 900, 940,   # die 3 (channels M–R)
]
AS7265X_COLS: list[str] = [f"as7265x_{w}nm" for w in AS7265X_WAVELENGTHS]

# All outdoor predictor columns (excluding timestamp and solar zenith)
OUTDOOR_FEATURE_COLS: list[str] = (
    PM_DENSITY_COLS
    + PM_COUNT_COLS
    + BME280_COLS
    + AIRMAR_COLS
    + RAIN_COLS
    + AS7265X_COLS
)

# PM bins predicted by indoor sensors
INDOOR_PM_BINS: list[str] = ["pm1", "pm2_5", "pm10"]

# Number of indoor sensors
N_INDOOR_SENSORS: int = 20

# Building location: Richardson IQHQ, Richardson TX
SITE_LATITUDE: float = 32.9483
SITE_LONGITUDE: float = -96.7299
SITE_TIMEZONE: str = "US/Central"

# Target time resolution after downsampling
RESAMPLE_FREQ: str = "10min"


# ---------------------------------------------------------------------------
# Outdoor data helpers
# ---------------------------------------------------------------------------

def load_outdoor_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw outdoor sensor data (1-second resolution).

    The CSV is expected to have a timestamp column (auto-detected) and the
    column names listed in :data:`OUTDOOR_FEATURE_COLS`.  Extra columns are
    kept and can be dropped later.

    Parameters
    ----------
    filepath:
        Path to the CSV file produced by the MINTS-AI outdoor node.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeTzAware (``US/Central``)
        Columns: all numeric columns present in the file.
    """
    filepath = Path(filepath)
    logger.info("Loading outdoor data from %s", filepath)

    df = pd.read_csv(filepath)

    # Detect timestamp column (common names used by MINTS nodes)
    ts_candidates = [c for c in df.columns if c.lower() in {
        "timestamp", "datetime", "time", "date_time", "datetime", "date"
    }]
    if not ts_candidates:
        raise ValueError(
            "No timestamp column found in outdoor CSV. "
            "Expected one of: timestamp, datetime, time, date_time."
        )
    ts_col = ts_candidates[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    df.index.name = "timestamp"

    # Localise / convert to site timezone
    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert(SITE_TIMEZONE)
    else:
        df = df.tz_convert(SITE_TIMEZONE)

    df = df.sort_index()
    logger.info("Outdoor data loaded: %d rows, %d columns", len(df), df.shape[1])
    return df


def time_average_outdoor(
    df: pd.DataFrame,
    freq: str = RESAMPLE_FREQ,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Downsample 1-second outdoor data to *freq* (default: 10 min).

    Parameters
    ----------
    df:
        Raw 1-second outdoor DataFrame (output of :func:`load_outdoor_data`).
    freq:
        Pandas offset alias for the target resolution (e.g. ``"10min"``).
    min_coverage:
        Fraction of non-NaN 1-second samples required within each *freq*
        window to retain the aggregated value.  Windows with fewer valid
        samples are set to NaN.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with mean values, same columns as input.
    """
    logger.info("Resampling outdoor data to %s resolution", freq)

    numeric_df = df.select_dtypes(include=[np.number])

    # Count non-NaN samples per window to enforce minimum coverage
    count_per_window = numeric_df.resample(freq).count()
    # Approximate total expected samples per window (e.g. 600 for 10 min @ 1 s)
    window_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9
    expected_count = window_seconds  # 1 sample/second

    avg_df = numeric_df.resample(freq).mean()

    # Mask windows with insufficient coverage
    coverage = count_per_window / expected_count
    avg_df[coverage < min_coverage] = np.nan

    logger.info(
        "Resampled: %d → %d rows (freq=%s)", len(df), len(avg_df), freq
    )
    return avg_df


# ---------------------------------------------------------------------------
# Indoor data helpers
# ---------------------------------------------------------------------------

def load_indoor_sensor(
    filepath: str | Path,
    sensor_id: Optional[int] = None,
) -> pd.DataFrame:
    """Load a single indoor sensor CSV (10-minute resolution).

    Parameters
    ----------
    filepath:
        Path to the indoor sensor CSV.
    sensor_id:
        Integer sensor identifier (1–20).  If provided, columns are renamed
        to ``pm1_sensor{id}``, ``pm2_5_sensor{id}``, ``pm10_sensor{id}``.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeTzAware (``US/Central``)
        Columns: pm1, pm2_5, pm10 (optionally suffixed with sensor_id).
    """
    filepath = Path(filepath)
    logger.info("Loading indoor sensor %s from %s", sensor_id, filepath)

    df = pd.read_csv(filepath)

    ts_candidates = [c for c in df.columns if c.lower() in {
        "timestamp", "datetime", "time", "date_time", "dateTime", "date"
    }]
    if not ts_candidates:
        raise ValueError(f"No timestamp column found in indoor CSV: {filepath}")
    ts_col = ts_candidates[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    df.index.name = "timestamp"

    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert(SITE_TIMEZONE)
    else:
        df = df.tz_convert(SITE_TIMEZONE)

    df = df.sort_index()

    # Keep only pm1, pm2_5, pm10 columns (flexible name matching)
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower().replace(".", "_").replace(" ", "_")
        # Match pm1 but not pm10 or pm1_0 (avoid partial matches like pm1_density)
        if "pm1" in cl and "pm10" not in cl and not cl.startswith("pm1_"):
            col_map[col] = "pm1"
        elif "pm2" in cl and ("2_5" in cl or "25" in cl):
            col_map[col] = "pm2_5"
        elif "pm10" in cl:
            col_map[col] = "pm10"
    df = df.rename(columns=col_map)[list(col_map.values())]

    if sensor_id is not None:
        df = df.rename(columns={b: f"{b}_sensor{sensor_id}" for b in df.columns})

    logger.info("Indoor sensor %s: %d rows", sensor_id, len(df))
    return df


def load_all_indoor_sensors(
    sensor_dir: str | Path,
    pattern: str = "sensor_*.csv",
) -> pd.DataFrame:
    """Load all indoor sensors from *sensor_dir* and merge on timestamp.

    Files are expected to match *pattern* and be named so that the sensor
    number can be extracted (e.g. ``sensor_01.csv``).

    Parameters
    ----------
    sensor_dir:
        Directory containing one CSV per indoor sensor.
    pattern:
        Glob pattern for sensor files.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns ``pm1_sensor{i}``, ``pm2_5_sensor{i}``,
        ``pm10_sensor{i}`` for i = 1 … 20.
    """
    sensor_dir = Path(sensor_dir)
    files = sorted(sensor_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No indoor sensor files matching '{pattern}' in {sensor_dir}"
        )

    dfs: list[pd.DataFrame] = []
    for i, fpath in enumerate(files, start=1):
        dfs.append(load_indoor_sensor(fpath, sensor_id=i))

    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()
    logger.info(
        "Loaded %d indoor sensors → combined shape %s", len(dfs), combined.shape
    )
    return combined


def align_indoor_outdoor(
    outdoor: pd.DataFrame,
    indoor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join outdoor and indoor DataFrames on timestamp.

    Parameters
    ----------
    outdoor:
        10-minute averaged outdoor DataFrame.
    indoor:
        10-minute indoor combined DataFrame.

    Returns
    -------
    outdoor_aligned, indoor_aligned : tuple[pd.DataFrame, pd.DataFrame]
        Temporally aligned sub-sets.
    """
    common_idx = outdoor.index.intersection(indoor.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping timestamps between outdoor and indoor data.")
    logger.info("Aligned on %d common timestamps", len(common_idx))
    return outdoor.loc[common_idx], indoor.loc[common_idx]
