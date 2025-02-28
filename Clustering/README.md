# Kasra-Shirvanian-R2-MSc-Project-UOM
MATLAB scripts for 3D UPPER, kmeans Clustering implantation as well as general visualizations, Python scripts for XGB, synchronized data generation and  CPU and GPU based Neural spike prediction. 

Clustering Scripts:

1-find_optimal_k.m
Overview
This script performs K-means clustering on shape parameter data from 120 mice and determines the optimal number of clusters using the Elbow Method and correctness evaluation. It separates the dataset into training and test sets at the mouse level, applies PCA for dimensionality reduction, and evaluates clustering performance across different values of K (1 to 20). The results are saved for each K to allow further analysis.

Key Features
Reads and processes shape parameter data from 120 mice.
Splits data into training and test sets at the mouse level to prevent data leakage.
Standardizes features based on training set statistics.
Applies PCA to reduce dimensionality (explains 90% variance).
Performs K-means clustering for different values of K (1 to 20).
Uses the Elbow Method to determine the best K by plotting within-cluster sum of squares (WCSS).
Evaluates clustering correctness by comparing cluster biases with the actual time spent in the dark.
Saves results for each K, including cluster centroids and assignments for later analysis.
Input Files
Shape parameter data (SSM*_data.mat) for each mouse.
Preference data (preference_Q*.mat) to determine time spent in dark/light chambers.
Metadata file (MICELISTQ1-Q120-pref.xlsx) containing mouse chamber preferences.
Outputs
Figures:
Elbow Plot: WCSS vs. K to identify optimal clustering.
Correctness Plot: Correctness scores for different K values in training and test sets.
Saved Data (Clustering_Dataset_K*.mat) for each K:
Cluster assignments (train & test)
PCA-transformed feature data
Cluster centroids
Training and test mice
Correctness scores
How to Run
Run the script in MATLAB:
find_optimal_k();
This will process the data, perform clustering, generate plots, and save clustering results.

2-cluster_with_optimal_k.m
Overview
This script loads the clustering results for a user-specified optimal K (chosen based on previous analysis) and visualizes cluster distributions relative to chamber preference (i.e., average time spent in the dark). It uses previously saved data from the find_optimal_k.m script and examines how well the identified clusters align with behavioral chamber preferences.

Key Features
Loads previously saved clustering results from Clustering_Dataset_K*.mat for the user-specified K.
Compares the distribution of clusters based on time spent in dark vs. light chambers.
Performs significance testing by shuffling data and generating random cluster distributions.
Generates visual plots:
Difference plot showing how each cluster relates to chamber preference.
Cluster preference bar charts displaying the percentage of frames in dark per cluster.
Inputs
Precomputed clustering results (Clustering_Dataset_K*.mat) from find_optimal_k.m, including:
Cluster assignments (group_train, group_test)
Behavioral preference data (Y_train, Y_test)
PCA-transformed scores (score_train, score_test)
Cluster centroids (C)
Outputs
Figures:
Difference Plot: Displays the relative distribution of clusters based on light vs. dark preference.
Significance Testing Plot: Uses shuffled data to assess the robustness of the clustering.
Cluster Preference Bar Charts: Shows the percentage of time each cluster spends in the dark compared to the average threshold.
Final Clustering Data (Final_Clustering_Dataset.mat), which includes all key clustering results.
How to Run
Run the script in MATLAB:
cluster_with_optimal_k();
When prompted, enter the optimal K determined from previous analysis.

3-plot_shape_parameters.m
Overview
This script visualizes shape parameters extracted from behavioral data, comparing how they differ between dark and light chamber conditions. It computes mean, variance, and distributions for each shape parameter across training and test mice and generates relevant figures.

Key Features
Loads shape parameter data from preprocessed .mat files for each mouse.
Separates shape parameters into dark and light chamber conditions based on the experiment design.
Computes mean and variance of shape parameters for both conditions.
Generates visualization plots:
Mean and variance plot for shape parameters in dark vs. light chambers.
Histograms of the distributions for each shape parameter under both conditions.
Inputs
filepath: Path to the directory containing the shape parameter data.
Nmice: Number of mice included in the dataset.
Data files: The script loads shape parameter data from:
SSM*_data.mat: Contains shape parameter matrix b for each mouse.
MICELISTQ1-Q120-pref.xlsx: Determines whether a mouse was in the dark or light chamber.
Outputs
Figures:
Error bar plot comparing mean ± variance of shape parameters between dark and light chamber conditions.
Histograms showing the probability distribution of shape parameters for dark and light conditions.
How to Run
Run the function in MATLAB by calling:
plot_shape_parameters('K:\qian_behav_cluster\preference_SSM_Q120', 20);
This example loads data from 20 mice in the specified directory.
