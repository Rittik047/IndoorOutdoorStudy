"""
IndoorOutdoorStudy – Source Package
====================================
Machine-learning pipeline that predicts the difference between outdoor and
indoor Particulate Matter (PM) concentration (ΔPM = outdoor − indoor) using
XGBoost.

Sub-modules
-----------
data_processing   : Load raw sensor CSVs and time-average outdoor data.
lag_analysis      : Cross-correlation lag analysis and lag correction.
feature_engineering : Solar-zenith-angle calculation and feature assembly.
model             : XGBoost training, hyper-parameter tuning, evaluation.
visualization     : Standard diagnostic and results plots.
"""
