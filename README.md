# Predicting Indoor-Outdoor PM Differentials via XGBoost

This repository contains the data, source code, and analysis for a machine learning project focused on predicting **Indoor Air Quality (IAQ)** based on high-resolution outdoor environmental data. The study is conducted at the **Richardson IQHQ** in Richardson, Texas—a facility equipped with a specialized pathogen defense system and HEPA-filtered HVAC.

## Project Overview

The objective is to predict the difference between outdoor and indoor Particulate Matter (PM) concentrations ($\Delta$ PM = PM<sub>Outdoor</sub> - PM<sub>Indoor</sub>). By leveraging high-fidelity outdoor sensor arrays and machine learning (XGBoost), we aim to understand how outdoor pollutants penetrate or are mitigated by advanced building filtration systems.

## Data Source & Sensor Fusion

### 1. Outdoor Sensors (MINTS-AI Lab, UT Dallas)
* **Pierra IPS7100**: Measures PM density ($\mu$ g/m<sup>3</sup>) and particle count (counts/Litre) for 7 size bins: PM<sub>0.1</sub> , PM<sub>0.3</sub> , PM<sub>0.3</sub> , PM<sub>1,0</sub> , PM<sub>2.5</sub> , PM<sub>5.0</sub> , and PM<sub>10.0</sub> .
* **BME280**: Air temperature, pressure, humidity, and dewpoint.
* **RG15**: Rainfall intensity.
* **AIRMAR**: Ultrasonic wind speed, direction, and secondary meteorological data.
* **AS7265x Multispectral Sensor**: Measures solar irradiance across 18 wavelength bins (410nm to 940nm).
* **Calculated Features**: Solar Zenith Angle (derived from lat/long and local time).

### 2. Indoor Sensors
* **Network**: 20 indoor sensors distributed throughout the IQHQ building.
* **Resolution**: 10-minute intervals.
* **Metrics**: PM<sub>1.0</sub>, PM<sub>2.5</sub>, and PM<sub>10.0</sub> concentrations.

## Methodology

### Data Processing Pipeline
1.  **Downsampling**: Outdoor data (1-second resolution) was time-averaged to a **10-minute resolution** to match the indoor sensor frequency.
2.  **Lag Analysis**: To account for the time it takes for outdoor air to affect indoor environments, a sensor-by-sensor lag analysis was performed for each PM bin (PM<sub>2.5</sub>, PM<sub>2.5</sub>, PM<sub>10.0</sub>).
3.  **Feature Engineering**: Integration of spectral solar data, meteorological conditions, and calculated solar angles as predictors.

### Machine Learning
* **Model**: XGBoost (Extreme Gradient Boosting) in Python.
* **Target Variables**: 
    * $\Delta$ PM<sub>1.0</sub>
    * $\Delta$ PM<sub>2.5</sub>
    * $\Delta$ PM<sub>10.0</sub>

## Repository Structure

```
IndoorOutdoorStudy/
├── Code/
│   └── googleColab_GPUoptimized/
│       └── Indoor_Outdoor.ipynb          # Main GPU-optimized Google Colab notebook
├── Data/
│   └── MintsData/
│       └── Indoor/
│           ├── 001e064a1520/             # ⚠️ Outdoor sensor data (MINTS-AI Lab node)
│           │   └── combinedValo01Data.csv
│           ├── 70b3d540f40ce420/         # Indoor sensor 01
│           │   └── 70b3d540f40ce420_combined.csv
│           ├── 70b3d540f40ce421/         # Indoor sensor 02
│           ├── 70b3d540f40ce422/         # Indoor sensor 03
│           ├── 70b3d540f40ce423/         # Indoor sensor 04
│           ├── 70b3d540f40ce424/         # Indoor sensor 06
│           ├── 70b3d540f40ce425/         # Indoor sensor 07
│           ├── 70b3d540f40ce426/         # Indoor sensor 08
│           ├── 70b3d540f40ce427/         # Indoor sensor 09
│           ├── 70b3d540f40ce429/         # Indoor sensor 10
│           ├── 70b3d540f40ce42d/         # Indoor sensor 11
│           ├── 70b3d540f40ce42f/         # Indoor sensor 12
│           ├── 70b3d540f40ce430/         # Indoor sensor 13
│           ├── 70b3d540f40ce433/         # Indoor sensor 14
│           ├── 70b3d540f40ce434/         # Indoor sensor 15
│           ├── 70b3d540f40ce435/         # Indoor sensor 16
│           ├── 70b3d540f40ce436/         # Indoor sensor 17
│           ├── 70b3d540f40ce438/         # Indoor sensor 18
│           ├── 70b3d540f40ce43a/         # Indoor sensor 19
│           ├── 70b3d540f40ce43b/         # Indoor sensor 20
│           ├── 70b3d540f40ce43c/         # Indoor sensor 21
│           └── valo_cross_correlation_results.csv
└── Results/
    ├── PM1.0/
    │   ├── allParameters_PM1/            # Heatmaps & CSVs using all outdoor features
    │   └── top10_PM1/                    # Per-sensor plots using top-10 features
    ├── PM2.5/
    │   ├── allParameters_PM2.5/
    │   └── top10_PM2.5/
    └── PM10.0/
        ├── allParameters_PM10.0/
        └── top10_PM10.0/
```

> **Note on data placement**: The folder `001e064a1520` resides inside `Data/MintsData/Indoor/` for organizational convenience, but it contains **outdoor** sensor data collected by the MINTS-AI Lab node. All `70b3d540f40ce4xx` folders contain **indoor** sensor data.

## Requirements

* Python 3.x (Google Colab environment recommended)
* `xgboost`
* `pandas`
* `numpy`
* `matplotlib` / `seaborn`
* `scipy` (for lag/cross-correlation analysis)

## Usage

1.  **Open the notebook**: Load `Code/googleColab_GPUoptimized/Indoor_Outdoor.ipynb` in Google Colab (GPU runtime recommended).
2.  **Point to data**: Ensure `Data/MintsData/Indoor/` is accessible (e.g., mount Google Drive or upload the folder).
3.  **Run all cells**: The notebook handles data loading, downsampling, lag analysis, XGBoost training, and result export in sequence.
4.  **Outputs**: Model metrics, feature-importance CSVs, and visualization plots are saved to the corresponding `Results/PM*/` subfolders.

## Acknowledgments
Special thanks to the **MINTS-AI Lab at UT Dallas** for providing the outdoor sensor infrastructure and the management at **Richardson IQHQ** for facility access.
