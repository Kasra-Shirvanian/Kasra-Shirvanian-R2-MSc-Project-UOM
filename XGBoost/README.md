Overview

This script analyzes the output of XGBoost-based neural spike prediction, comparing actual vs. predicted spike counts. It also evaluates model accuracy by comparing true data with shuffled control predictions.

1️⃣ What This Script Does 

Loads Required Data

Neural spike data (spike_data.mat)
XGBoost input data (data_for_prediction.mat)
XGBoost prediction results (data_for_prediction_results.mat)
Computes Prediction Accuracy

Computes correlation (C0) between actual spike counts and XGBoost predictions.
Computes shuffled baseline correlation (Csh) to evaluate if the model outperforms random predictions.
Identifies Significant Predictions

A neuron is considered significantly predictable if its correlation exceeds the shuffled baseline by 5 standard deviations.
Stores indices of significant neurons.
Plots Prediction vs. Shuffle Control

Scatter plot comparing true vs. shuffled correlations.
Highlights neurons with significant predictability.
Plots Example Predicted vs. True Spike Counts

Selects a significant neuron.
Plots true vs. predicted spike count trends over time.

2️⃣ How to Run the Script

Open MATLAB.
Modify the filenames to match your dataset.

analyze_xgb_predictions()
The script will generate plots to visualize prediction accuracy.

3️⃣ Output Visualizations

Scatter Plot: XGBoost Prediction vs. Shuffle Control

X-axis: Correlation of shuffled data (Corr_shuf).
Y-axis: Correlation of actual data (Corr_data).
Blue dots: All neurons.
Red circles: Neurons with significantly higher prediction accuracy than chance.
Diagonal Line: Indicates where shuffled and actual data correlations are equal.
Time Series Plot: True vs. Predicted Spikes for a Significant Neuron

X-axis: Frames (time).
Y-axis: Spike count.
Compares XGBoost prediction vs. actual neural spikes.
