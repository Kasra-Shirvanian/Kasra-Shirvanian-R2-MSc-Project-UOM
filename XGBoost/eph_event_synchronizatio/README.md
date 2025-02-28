Overview

This script generates input data for XGBoost by synchronizing neural spike data with behavioral events recorded under different lighting conditions. The output is structured for predicting neural activity based on shape parameters and movement data.

1️⃣ What This Script Does

Loads Data:

Event timestamps (evt_rec.mat)
Shape parameters (SSM120_data.mat)
Neural spike data (spike_data.mat)
Filters Neural Spikes

Removes spikes that occur before or after the recorded events.
Ensures only relevant spikes are considered.
Generates Output (Y) for XGBoost

Counts neural spikes occurring within event windows for each neuron.
Stores this data in a matrix (Y) where rows = events, columns = neurons.
Filters Low-Firing Neurons

Neurons with fewer than 100 spikes are removed to ensure model robustness.
Generates Input (X) for XGBoost

Shape parameters (b')
Distance metric (dist')
Event index (1:Nevt')
Adds Shuffle Controls

Creates randomized versions of X for control experiments.
Ensures results are not driven by chance.
Creates Cross-Validation Blocks

Splits events into 5-fold cross-validation (K=5).
Assigns blocks of 150 events per fold for balanced training.
Saves Processed Data

The output is saved as:
Copy
Edit
{mouseID}_{eventID}_data_for_prediction.mat
Contains:
X (Shape parameters & movement data)
Y (Neural spike count per event)
K (Number of cross-validation folds)
which_fold (Fold assignments for cross-validation)
clu_val (Neuron cluster identifiers)

2️⃣ How to Run the Script

Open MATLAB.
Run:
generate_xgb_data()
The script will process and save a MATLAB .mat file formatted for XGBoost training.

3️⃣ Output File Description

The saved .mat file will contain:

Variable	Description
X	Shape parameters & movement data (input features)
Y	Neural spike counts per event (prediction target)
K	Number of cross-validation folds (default = 5)
which_fold	Cross-validation fold assignments
clu_val	Neuron cluster IDs
