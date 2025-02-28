Separating the script for CPU and GPU lets the user to run two parallel operation,s leading to 2X efficiency 

#Overview

These scripts implement XGBoost to predict neural spike activity from shape parameters synchronized with behavioral events and electrophysiology data. The model is trained using K-fold cross-validation, and predictions are made for each shuffle in the dataset. The scripts save model outputs, including predictions and feature importance.

We provide two implementations:

CPU-Based XGBoost (xgboost_cpu.py) – Runs on standard processors.
GPU-Based XGBoost (xgboost_gpu.py) – Utilizes CUDA acceleration for faster computation.

1️⃣ CPU-Based XGBoost (xgboost_cpu.py)

#Key Features:

Loads shape parameters (X) and neural spike data (Y) from .mat files.
Performs K-fold cross-validation to train and evaluate the model.
Uses XGBoost regression (count:poisson) to predict neural activity.
Stores feature importance for later interpretation.
Saves the results in a .mat file.

#Execution:
To run the script, simply execute:

python xgboost_cpu.py
Or call the main() function in a Python environment.

#Input Data:

.mat files containing:
X: Shape parameters/features.
Y: Neural spike data.
which_fold: Cross-validation split indices.
K: Number of folds for cross-validation.

#Output:

Predicted neural activity (Ypred)
Feature importance (importance)
XGBoost model parameters (xgb_params)
Saved results file: {filename}_results.mat

2️⃣ GPU-Based XGBoost (xgboost_gpu.py)

#Key Features:

Same functionality as the CPU version but optimized for GPUs.
Uses CUDA acceleration (device: 'cuda') for improved speed.
Suitable for large-scale datasets requiring high computational efficiency.

#Execution:

Run the script using:
python xgboost_gpu.py
Or call the main() function.

Differences from CPU Version:
Uses tree_method: 'hist', which is GPU-friendly.
Adds 'device': 'cuda' to leverage GPU acceleration.

#Output:

Predicted neural activity (Ypred)
Feature importance (importance)
XGBoost model parameters (xgb_params)
Saved results file: {filename}_results.mat
