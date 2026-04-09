# Predicting Indoor-Outdoor PM Differentials via XGBoost

This repository contains the data, source code, and analysis for a machine learning project focused on predicting **Indoor Air Quality (IAQ)** based on high-resolution outdoor environmental data. The study is conducted at the **Richardson IQHQ** in Richardson, Texas—a facility equipped with a specialized pathogen defense system and HEPA-filtered HVAC.

## Project Overview

The objective is to predict the difference between outdoor and indoor Particulate Matter (PM) concentrations (Delta PM = PM_outdoor - PM_indoor). By leveraging high-fidelity outdoor sensor arrays and machine learning (XGBoost), we aim to understand how outdoor pollutants penetrate or are mitigated by advanced building filtration systems.

## Data Source & Sensor Fusion

### 1. Outdoor Sensors (MINTS-AI Lab, UT Dallas)
* **Pierra IPS7100**: Measures PM density (micrograms/m<sup>3</sup>) and particle count (counts/m^3) for 7 size bins: PM0.1, PM0.3, PM0.5, PM1, PM2.5, PM5, and PM10.
* **BME280**: Air temperature, pressure, humidity, and dewpoint.
* **RG15**: Rainfall intensity.
* **AIRMAR**: Ultrasonic wind speed, direction, and secondary meteorological data.
* **AS7265x Multispectral Sensor**: Measures solar irradiance across 18 wavelength bins (410nm to 940nm).
* **Calculated Features**: Solar Zenith Angle (derived from lat/long and local time).

### 2. Indoor Sensors
* **Network**: 20 indoor sensors distributed throughout the IQHQ building.
* **Resolution**: 10-minute intervals.
* **Metrics**: PM1, PM2.5, and PM10 concentrations.

## Methodology

### Data Processing Pipeline
1.  **Downsampling**: Outdoor data (1-second resolution) was time-averaged to a **10-minute resolution** to match the indoor sensor frequency.
2.  **Lag Analysis**: To account for the time it takes for outdoor air to affect indoor environments, a sensor-by-sensor lag analysis was performed for each PM bin (PM1, PM2.5, PM10).
3.  **Feature Engineering**: Integration of spectral solar data, meteorological conditions, and calculated solar angles as predictors.

### Machine Learning
* **Model**: XGBoost (Extreme Gradient Boosting) in Python.
* **Target Variables**: 
    * Delta PM_1
    * Delta PM_2.5
    * Delta PM_10

## Repository Structure

* `/data`: Raw and processed datasets (10min averaged and lag-adjusted).
* `/src`: Python scripts for data cleaning, lag analysis, and XGBoost training.
* `/output`: Resulting dataframes and performance metrics.
* `/plots`: Visualizations including lag correlations, feature importance, and predicted vs. actual PM levels.

## Requirements

* Python 3.x
* `xgboost`
* `pandas`
* `numpy`
* `matplotlib` / `seaborn`
* `scipy` (for lag analysis)

## Usage

1.  **Preprocessing**: Run `data_alignment.py` to downsample outdoor data and apply the specific time-lags for the 20 indoor sensors.
2.  **Training**: Execute `train_xgboost.py` to generate the predictive models for each PM bin.
3.  **Visualization**: Use `plot_results.py` to generate the analysis charts found in the `/plots` directory.

## Acknowledgments
Special thanks to the **MINTS-AI Lab at UT Dallas** for providing the outdoor sensor infrastructure and the management at **Richardson IQHQ** for facility access.
