"""
feature_engineering.py
=======================
Build the full predictor feature matrix used by the XGBoost models.

Key steps
---------
1. Calculate the Solar Zenith Angle (SZA) using pvlib for each 10-minute
   timestamp at the Richardson IQHQ site location.
2. Construct wind components (u, v) from AIRMAR speed and direction.
3. Assemble the final feature matrix for a given PM bin.

The predictor set mirrors the problem description:
  - All outdoor PM density and count columns (7 bins × 2 = 14 features)
  - BME280: temperature, pressure, humidity, dewpoint (4 features)
  - AIRMAR: wind speed, direction, temperature, pressure, humidity, dewpoint
             + derived wind_u, wind_v (8 features)
  - RG-15: rainfall (1 feature)
  - AS7265x: 18 spectral channels (18 features)
  - Solar zenith angle (1 feature)
  Total: 46 features
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pvlib
    _PVLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PVLIB_AVAILABLE = False

from .data_processing import (
    SITE_LATITUDE,
    SITE_LONGITUDE,
    SITE_TIMEZONE,
    OUTDOOR_FEATURE_COLS,
    AIRMAR_COLS,
    INDOOR_PM_BINS,
    AS7265X_WAVELENGTHS,
)
from .lag_analysis import OUTDOOR_PM_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solar position
# ---------------------------------------------------------------------------

def compute_solar_zenith_angle(
    timestamps: pd.DatetimeIndex,
    latitude: float = SITE_LATITUDE,
    longitude: float = SITE_LONGITUDE,
) -> pd.Series:
    """Calculate apparent solar zenith angle (degrees) for *timestamps*.

    Uses pvlib if available; falls back to a simplified analytical formula
    otherwise.

    Parameters
    ----------
    timestamps:
        Timezone-aware DatetimeIndex.
    latitude:
        Site latitude in decimal degrees (north positive).
    longitude:
        Site longitude in decimal degrees (east positive).

    Returns
    -------
    pd.Series
        Solar zenith angle in degrees, indexed by *timestamps*.
    """
    if _PVLIB_AVAILABLE:
        location = pvlib.location.Location(
            latitude=latitude,
            longitude=longitude,
            tz=SITE_TIMEZONE,
            altitude=189,          # Richardson TX elevation ~189 m
        )
        solpos = location.get_solarposition(timestamps)
        sza = solpos["apparent_zenith"].rename("solar_zenith_angle")
        logger.debug("Solar zenith angle computed with pvlib for %d timestamps.", len(timestamps))
    else:
        logger.warning(
            "pvlib not available – using simplified SZA formula. "
            "Install pvlib for better accuracy."
        )
        sza = _simple_solar_zenith(timestamps, latitude, longitude)

    return sza


def _simple_solar_zenith(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> pd.Series:
    """Simplified solar zenith angle calculation (fallback without pvlib).

    Based on the NOAA Solar Calculator equations.
    Accuracy is ~0.01° for dates near J2000.0.
    """
    # Convert to UTC for calculation
    if timestamps.tz is not None:
        utc_ts = timestamps.tz_convert("UTC")
    else:
        utc_ts = timestamps

    lat_rad = np.radians(latitude)
    results: list[float] = []

    for ts in utc_ts:
        jd = _julian_day(ts)
        jc = (jd - 2451545.0) / 36525.0

        # Geometric mean longitude / anomaly of the Sun
        geom_mean_lon = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
        geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
        ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
        anom_r = np.radians(geom_mean_anom)
        eq_ctr = (
            np.sin(anom_r) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
            + np.sin(2 * anom_r) * (0.019993 - 0.000101 * jc)
            + np.sin(3 * anom_r) * 0.000289
        )
        sun_true_lon = geom_mean_lon + eq_ctr
        sun_app_lon = sun_true_lon - 0.00569 - 0.00478 * np.sin(
            np.radians(125.04 - 1934.136 * jc)
        )

        # Obliquity and declination
        mean_obliq = (
            23 + (26 + (21.448 - jc * (46.8150 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
        )
        obliq_corr = mean_obliq + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))
        declination = np.degrees(
            np.arcsin(np.sin(np.radians(obliq_corr)) * np.sin(np.radians(sun_app_lon)))
        )

        # Hour angle
        var_y = np.tan(np.radians(obliq_corr / 2)) ** 2
        eq_time = (
            4
            * np.degrees(
                var_y * np.sin(2 * np.radians(geom_mean_lon))
                - 2 * ecc * np.sin(anom_r)
                + 4 * ecc * var_y * np.sin(anom_r) * np.cos(2 * np.radians(geom_mean_lon))
                - 0.5 * var_y**2 * np.sin(4 * np.radians(geom_mean_lon))
                - 1.25 * ecc**2 * np.sin(2 * anom_r)
            )
        )
        # True solar time (minutes)
        tst = (
            (ts.hour * 60 + ts.minute + ts.second / 60)
            + eq_time
            + 4 * longitude
        ) % 1440
        hour_angle = tst / 4 - 180 if tst / 4 < 0 else tst / 4 - 180

        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        cos_sza = (
            np.sin(lat_rad) * np.sin(dec_rad)
            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        )
        sza = np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))
        results.append(sza)

    return pd.Series(results, index=timestamps, name="solar_zenith_angle")


def _julian_day(ts: "pd.Timestamp") -> float:
    """Convert a UTC Timestamp to Julian Day Number."""
    y, m, d = ts.year, ts.month, ts.day
    h = ts.hour + ts.minute / 60 + ts.second / 3600
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + h / 24 + b - 1524.5


# ---------------------------------------------------------------------------
# Wind components
# ---------------------------------------------------------------------------

def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``wind_u`` and ``wind_v`` columns derived from AIRMAR data.

    Parameters
    ----------
    df:
        DataFrame containing ``airmar_wind_speed`` (m s⁻¹) and
        ``airmar_wind_direction`` (° from north, meteorological convention).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns added in-place.
    """
    if "airmar_wind_speed" not in df.columns or "airmar_wind_direction" not in df.columns:
        logger.debug("AIRMAR wind columns not found; skipping wind component addition.")
        return df

    wdir_rad = np.radians(df["airmar_wind_direction"])
    df = df.copy()
    df["wind_u"] = -df["airmar_wind_speed"] * np.sin(wdir_rad)  # eastward component
    df["wind_v"] = -df["airmar_wind_speed"] * np.cos(wdir_rad)  # northward component
    return df


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    merged_df: pd.DataFrame,
    pm_bin: str,
    sensor_id: int,
    add_sza: bool = True,
    add_wind_uv: bool = True,
    extra_lag_hours: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) pair for a given PM bin and indoor sensor.

    Parameters
    ----------
    merged_df:
        Lag-corrected merged DataFrame (output of
        :func:`lag_analysis.apply_lags`).  Must contain outdoor feature
        columns and ``{pm_bin}_sensor{sensor_id}``.
    pm_bin:
        One of ``"pm1"``, ``"pm2_5"``, ``"pm10"``.
    sensor_id:
        Integer sensor identifier (1–20).
    add_sza:
        If ``True``, add solar zenith angle as a feature.
    add_wind_uv:
        If ``True``, add u/v wind components derived from AIRMAR data.
    extra_lag_hours:
        List of additional time lags (in hours) to create as lagged outdoor
        PM features (e.g. ``[1, 2, 3]`` adds pm{bin}_density_lag1h, etc.).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (rows = timestamps, columns = features).
    y : pd.Series
        Target vector: ``outdoor_{pm_bin}_density − indoor_{pm_bin}_sensor{sensor_id}``
        (positive → outdoor > indoor, i.e. building is filtering).
    """
    if pm_bin not in INDOOR_PM_BINS:
        raise ValueError(f"pm_bin must be one of {INDOOR_PM_BINS}; got '{pm_bin}'.")

    outdoor_col = OUTDOOR_PM_COL[pm_bin]
    indoor_col = f"{pm_bin}_sensor{sensor_id}"

    for col in [outdoor_col, indoor_col]:
        if col not in merged_df.columns:
            raise KeyError(f"Required column '{col}' not found in merged_df.")

    # --- Target variable ---
    y = (merged_df[outdoor_col] - merged_df[indoor_col]).rename(
        f"delta_{pm_bin}_sensor{sensor_id}"
    )

    # --- Feature matrix ---
    feature_cols = [c for c in OUTDOOR_FEATURE_COLS if c in merged_df.columns]
    X = merged_df[feature_cols].copy()

    if add_wind_uv:
        X = add_wind_components(X)

    if add_sza:
        sza = compute_solar_zenith_angle(merged_df.index)
        X["solar_zenith_angle"] = sza.values

    # Lagged outdoor PM features: add historical outdoor PM values as additional
    # predictors (e.g. pm1_density 1 hour ago, 2 hours ago).  The values are
    # shifted *within the already-merged feature matrix* so that each row
    # includes past outdoor concentrations as context for the model.
    if extra_lag_hours:
        steps_per_hour = 6  # 10-min resolution → 6 steps/hour
        for h in extra_lag_hours:
            lag_col = f"{outdoor_col}_lag{h}h"
            X[lag_col] = X[outdoor_col].shift(h * steps_per_hour)

    # Drop rows where any feature or the target is NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    logger.info(
        "Feature matrix for %s sensor%d: %d samples, %d features",
        pm_bin, sensor_id, len(X), X.shape[1],
    )
    return X, y


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "pm0_1_density":        "PM0.1 density (µg m⁻³) – IPS7100",
    "pm0_3_density":        "PM0.3 density (µg m⁻³) – IPS7100",
    "pm0_5_density":        "PM0.5 density (µg m⁻³) – IPS7100",
    "pm1_density":          "PM1 density (µg m⁻³) – IPS7100",
    "pm2_5_density":        "PM2.5 density (µg m⁻³) – IPS7100",
    "pm5_density":          "PM5 density (µg m⁻³) – IPS7100",
    "pm10_density":         "PM10 density (µg m⁻³) – IPS7100",
    "pm0_1_count":          "PM0.1 particle count (counts m⁻³) – IPS7100",
    "pm0_3_count":          "PM0.3 particle count (counts m⁻³) – IPS7100",
    "pm0_5_count":          "PM0.5 particle count (counts m⁻³) – IPS7100",
    "pm1_count":            "PM1 particle count (counts m⁻³) – IPS7100",
    "pm2_5_count":          "PM2.5 particle count (counts m⁻³) – IPS7100",
    "pm5_count":            "PM5 particle count (counts m⁻³) – IPS7100",
    "pm10_count":           "PM10 particle count (counts m⁻³) – IPS7100",
    "bme280_temperature":   "Air temperature (°C) – BME280",
    "bme280_pressure":      "Atmospheric pressure (hPa) – BME280",
    "bme280_humidity":      "Relative humidity (%) – BME280",
    "bme280_dewpoint":      "Dew-point temperature (°C) – BME280",
    "airmar_wind_speed":    "Wind speed (m s⁻¹) – AIRMAR",
    "airmar_wind_direction":"Wind direction (° from N) – AIRMAR",
    "airmar_temperature":   "Air temperature (°C) – AIRMAR",
    "airmar_pressure":      "Atmospheric pressure (hPa) – AIRMAR",
    "airmar_humidity":      "Relative humidity (%) – AIRMAR",
    "airmar_dewpoint":      "Dew-point temperature (°C) – AIRMAR",
    "rainfall":             "Rainfall accumulation (mm) – RG-15",
    "wind_u":               "Eastward wind component (m s⁻¹) – derived",
    "wind_v":               "Northward wind component (m s⁻¹) – derived",
    "solar_zenith_angle":   "Solar zenith angle (°) – calculated from lat/lon/time",
    **{f"as7265x_{w}nm": f"Spectral irradiance at {w} nm (µW cm⁻² nm⁻¹) – AS7265x"
       for w in [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705,
                 730, 760, 810, 860, 900, 940]},
}
