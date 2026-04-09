# IndoorOutdoorStudy

Machine-learning pipeline that predicts the difference between **outdoor** and **indoor** Particulate Matter (PM) concentration at the Richardson IQHQ smart building in Richardson, Texas, using outdoor environmental sensor data as predictors and XGBoost as the ML model.

---

## Background

**Building:** Richardson IQHQ, Richardson TX (lat 32.9483 °N, lon 96.7299 °W).  
The building operates a Pathogen Defense System with HEPA filters in the HVAC.

**Outdoor sensor:** MINTS-AI node with:
| Sensor | Measurements |
|--------|-------------|
| Plantower IPS7100 | PM density (µg m⁻³) + count (counts m⁻³) for 7 bins: PM0.1, PM0.3, PM0.5, PM1, PM2.5, PM5, PM10 @ 1 s |
| Bosch BME280 | Temperature (°C), pressure (hPa), relative humidity (%), dew-point (°C) |
| Hydreon RG-15 | Rainfall accumulation (mm) |
| AIRMAR | Wind speed (m s⁻¹), wind direction (°), temperature, pressure, humidity, dew-point |
| AMS AS7265x | Solar irradiance in 18 spectral channels (410–940 nm) |

**Indoor sensors:** 20 sensors, each reporting PM1, PM2.5, PM10 density at 10-minute resolution.

**Target variable:** ΔPM = outdoor PM − indoor PM (µg m⁻³), modelled separately for PM1, PM2.5, PM10.

---

## Project Structure

```
IndoorOutdoorStudy/
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Load raw CSVs; time-average outdoor 1s → 10min
│   ├── lag_analysis.py       # Cross-correlation lag analysis & correction
│   ├── feature_engineering.py# Solar zenith angle, wind components, feature matrix
│   ├── model.py              # XGBoost training, CV, evaluation, persistence
│   └── visualization.py      # Lag, feature-importance, prediction plots
├── scripts/
│   ├── run_pipeline.py       # End-to-end orchestration script
│   └── analyze_results.py    # Load saved models and regenerate diagnostic plots
├── data/
│   ├── raw/                  # Place raw CSVs here (not tracked by git)
│   └── processed/            # Intermediate processed files (not tracked by git)
└── output/
    ├── models/               # Saved XGBoost models (*.joblib)
    ├── plots/                # Diagnostic plots (*.png)
    └── lag_results/          # lag_results.csv
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place data files as follows:

```
data/raw/outdoor_sensor.csv     ← 1-second outdoor sensor CSV
data/raw/indoor/
    sensor_01.csv               ← 10-min indoor sensor #1
    sensor_02.csv               ← 10-min indoor sensor #2
    ...
    sensor_20.csv               ← 10-min indoor sensor #20
```

Each CSV must contain a timestamp column (named `timestamp`, `datetime`, `time`, or `date_time`).

**Outdoor CSV column names** (case-insensitive, underscores for decimal points):

| Column | Description |
|--------|-------------|
| `pm{bin}_density` | PM density (µg m⁻³); bins: `0_1`, `0_3`, `0_5`, `1`, `2_5`, `5`, `10` |
| `pm{bin}_count` | Particle count (counts m⁻³) |
| `bme280_temperature` | Air temperature (°C) |
| `bme280_pressure` | Atmospheric pressure (hPa) |
| `bme280_humidity` | Relative humidity (%) |
| `bme280_dewpoint` | Dew-point temperature (°C) |
| `airmar_wind_speed` | Wind speed (m s⁻¹) |
| `airmar_wind_direction` | Wind direction (° from north) |
| `airmar_temperature` | AIRMAR temperature (°C) |
| `airmar_pressure` | AIRMAR pressure (hPa) |
| `airmar_humidity` | AIRMAR humidity (%) |
| `airmar_dewpoint` | AIRMAR dew-point (°C) |
| `rainfall` | Rainfall accumulation (mm) |
| `as7265x_{w}nm` | Spectral irradiance at wavelength *w* nm; *w* ∈ {410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940} |

**Indoor CSV column names** (per sensor file):

| Column | Description |
|--------|-------------|
| `pm1` | PM1 density (µg m⁻³) |
| `pm2_5` | PM2.5 density (µg m⁻³) |
| `pm10` | PM10 density (µg m⁻³) |

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py \
    --outdoor  data/raw/outdoor_sensor.csv \
    --indoor-dir data/raw/indoor/ \
    --output   output/
```

Additional options:

```
--resample-freq 10min    Target time resolution (default: 10min)
--pm-bins pm1 pm2_5 pm10 PM bins to model
--n-sensors 20           Number of indoor sensors
--test-fraction 0.2      Held-out temporal test fraction
--max-lag-steps 60       Max ±lag for cross-correlation (× 10 min)
--sensor-pattern "sensor_*.csv"
```

### 4. Analyse results / regenerate plots

```bash
python scripts/analyze_results.py \
    --merged  output/merged_lag_corrected.csv \
    --models  output/models/ \
    --lag-csv output/lag_results/lag_results.csv \
    --plots   output/plots/
```

---

## Pipeline Steps

```
1. Load outdoor CSV (1-second) ──► time-average to 10 min
2. Load all 20 indoor CSVs (10-min) ──► align timestamps
3. Cross-correlation lag analysis:
       for each indoor sensor i  (1 … 20)
         for each PM bin  b  (PM1, PM2.5, PM10)
           compute cross-corr(outdoor_b, indoor_b_i)
           find lag that maximises correlation
4. Apply lag corrections ──► merged, aligned DataFrame
5. Feature engineering:
       - All outdoor PM density & count columns (14)
       - BME280 temperature, pressure, humidity, dewpoint (4)
       - AIRMAR wind speed, direction, temperature, pressure, humidity, dewpoint (6)
       - Derived wind U/V components (2)
       - RG-15 rainfall (1)
       - AS7265x spectral channels 410–940 nm (18)
       - Solar zenith angle from lat/lon/time (1)
       ──► 46 predictors total
6. XGBoost model per (PM bin, sensor):
       target = outdoor_PM_b − indoor_PM_b_i  (ΔPM, µg m⁻³)
       temporal train/test split (80 % / 20 %)
       early stopping on validation RMSE
7. Save models, metrics, and diagnostic plots
```

---

## Outputs

| File / Directory | Contents |
|-----------------|----------|
| `output/models/xgb_{pm_bin}_sensor{id:02d}.joblib` | Serialised XGBoost model |
| `output/models/model_summary.csv` | RMSE, MAE, R² for every model |
| `output/lag_results/lag_results.csv` | Optimal lag and peak correlation per (sensor, PM bin) |
| `output/merged_lag_corrected.csv` | Merged, lag-corrected DataFrame used for training |
| `output/plots/lag_heatmap_minutes.png` | Heatmap of optimal lags (minutes) |
| `output/plots/lag_heatmap_correlation.png` | Heatmap of peak cross-correlations |
| `output/plots/{pm_bin}/predictions_sensor{id}.png` | Observed vs. predicted scatter + time series |
| `output/plots/{pm_bin}/feature_importance_sensor{id}.png` | Top-20 feature importances |
| `output/plots/{pm_bin}/timeseries_sensor{id}.png` | Outdoor vs. indoor PM time series |
| `output/plots/summary_test_rmse.png` | RMSE box plot across all models |
| `output/plots/summary_test_r2.png` | R² box plot across all models |

---

## Module Reference

### `src/data_processing`
- `load_outdoor_data(filepath)` – loads 1-second outdoor CSV, sets timezone-aware index.
- `time_average_outdoor(df, freq)` – downsamples to target resolution using mean; masks windows with < 50 % valid data.
- `load_indoor_sensor(filepath, sensor_id)` – loads a single indoor sensor CSV.
- `load_all_indoor_sensors(sensor_dir)` – loads all sensors and merges on timestamp.
- `align_indoor_outdoor(outdoor, indoor)` – inner-joins on common timestamps.

### `src/lag_analysis`
- `compute_lag(outdoor_series, indoor_series)` – returns optimal lag, peak correlation, and full cross-correlation profile.
- `run_lag_analysis(outdoor_avg, indoor_combined)` – iterates over all (sensor, PM-bin) pairs.
- `apply_lags(outdoor_avg, indoor_combined, lag_df)` – shifts indoor series and merges.
- `save_lag_results / load_lag_results` – CSV persistence.

### `src/feature_engineering`
- `compute_solar_zenith_angle(timestamps)` – uses pvlib (falls back to NOAA formula).
- `add_wind_components(df)` – derives eastward/northward wind U/V from speed and direction.
- `build_feature_matrix(merged_df, pm_bin, sensor_id)` – returns (X, y) ready for XGBoost.

### `src/model`
- `train_model(X_train, y_train)` – trains XGBoost with early stopping.
- `evaluate_model(model, X_test, y_test)` – returns RMSE, MAE, R², predictions.
- `cross_validate_model(X, y)` – time-series cross-validation.
- `tune_hyperparameters(X, y)` – randomised search over parameter grid.
- `run_all_models(merged_df, lag_df, output_dir)` – trains all 60 models (20 sensors × 3 PM bins).

### `src/visualization`
- `plot_lag_profile` – cross-correlation vs. lag curve.
- `plot_lag_heatmap` – heatmap of lags or correlations.
- `plot_feature_importance` – horizontal bar chart.
- `plot_predictions` – scatter + time-series of observed vs. predicted ΔPM.
- `plot_pm_time_series` – outdoor vs. indoor PM on one axis.
- `plot_model_summary` – box plots of RMSE and R² across all models.

---

## Citation / Acknowledgements

Outdoor sensor data collected by the **MINTS-AI Lab, UT Dallas**.  
Indoor sensor data from **Richardson IQHQ** smart building.
