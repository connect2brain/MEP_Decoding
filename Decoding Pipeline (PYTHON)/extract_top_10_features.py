#%%
"""
Extraction of top 10 predictive features 
======================================================
For each participant, the most influential EEG features are identified using normalized model coefficients (LR, SVM), Gini importance scores (RF), and ROC-AUC values. 
The top 10 features are selected based on consistent high rankings across metrics, stability across folds, and ROC-AUC > 0.60.
"""

import os
import pickle
import re
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from src.helper_functions import load_ml_sensor_features_df, load_ml_features_df 

# Load the configuration file to get paths
with open("config.json", "r") as file:
    config = json.load(file)

# Access paths from the configuration
Model_savepath = config["paths"]["Model_savepath"]
Feature_savepath = config["paths"]["Feature_savepath"]
#output_directory = ''
output_directory = config["paths"]["Model_savepath"]

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get list of all model files in Model_savepath
model_files = [f for f in os.listdir(Model_savepath) if f.endswith('_results.pkl')]

# Regular expression to parse filenames (e.g., 'Sub_0_64ch_shuffled_results.pkl')
pattern = r'Sub_(\d+)_(.*)_results\.pkl'

# Loop through each model file
for model_file in model_files:
    # Parse the filename to get subject number and condition suffix
    match = re.match(pattern, model_file)
    if match:
        subnum = int(match.group(1))
        condition_suffix = match.group(2)
        print(f"Processing subject {subnum} with condition '{condition_suffix}'")
        
        # Load the data from the pickle file
        file_path = os.path.join(Model_savepath, model_file)
        with open(file_path, 'rb') as f:
            Rep_mRMR2_SVM_LogReg_RF = pickle.load(f)
        
        # Extract final_features from the loaded data
        # Rep_mRMR2_SVM_LogReg_RF = [accuracy_arr, model_par, final_models, final_features, final_y_preds]
        final_features = Rep_mRMR2_SVM_LogReg_RF[3]
        
        # Initialize a dictionary to store the top features for each fold and model
        top_features_per_fold_model = defaultdict(lambda: defaultdict(set))

        # Process each fold's data
        for fold_idx, model_data in enumerate(final_features):
            # model_data is a list of models for this fold
            for model_info in model_data:
                # model_info is a tuple: (model_name, features, importances)
                model_name = model_info[0]
                features = model_info[1]
                importances = model_info[2]
                # Rank features based on importances in the current fold
                sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                top_50_features = [feature for feature, _ in sorted_features[:50]]
                # Store the top 50 features for this fold and model
                top_features_per_fold_model[fold_idx][model_name].update(top_50_features)

        # Find features that are in top 50 across all folds and models
        num_folds = len(top_features_per_fold_model)
        models = set()
        for fold_features in top_features_per_fold_model.values():
            models.update(fold_features.keys())
        number_of_models = len(models)
        total_combinations = num_folds * number_of_models

        # Count occurrences of features
        feature_occurrences = defaultdict(int)
        for fold_features in top_features_per_fold_model.values():
            for model_name, features_set in fold_features.items():
                for feature in features_set:
                    feature_occurrences[feature] += 1

        # Select features that appear in top 50 for all models and all folds
        consistent_features = [feature for feature, count in feature_occurrences.items() if count == total_combinations]

        if not consistent_features:
            print(f"No consistent features found for subject {subnum} under condition '{condition_suffix}'. Skipping.")
            continue

        # Load the features dataframe for this subject
        try:
            # Decide which function to use to load the features based on the condition
            if 'sensor_source' in condition_suffix:
                EEG_features_df = load_ml_features_df(Feature_savepath, subnum)
            else:
                EEG_features_df = load_ml_sensor_features_df(Feature_savepath, subnum)
        except FileNotFoundError:
            print(f"Feature file for subject {subnum} not found. Skipping.")
            continue

        # Process the dataframe (e.g., outlier detection, standardization)
        # Define the target and non-feature columns
        Target = "Condition"
        non_feature_cols = ["Trial_index"]

        # Exclude the target and any non-feature columns from standardization
        feature_cols = [col for col in EEG_features_df.columns if col != Target and col not in non_feature_cols]

        # Outlier detection using IQR
        threshold = 1.5
        Q1 = EEG_features_df[feature_cols].quantile(0.25)
        Q3 = EEG_features_df[feature_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify and replace outliers
        for feature in feature_cols:
            EEG_features_df[feature] = np.where(
                EEG_features_df[feature] < lower_bound[feature],
                lower_bound[feature],
                np.where(
                    EEG_features_df[feature] > upper_bound[feature],
                    upper_bound[feature],
                    EEG_features_df[feature]
                )
            )

        # Standardization
        scaler = RobustScaler()
        EEG_features_df[feature_cols] = scaler.fit_transform(EEG_features_df[feature_cols])

        # Prepare data for ROC-AUC computation
        X = EEG_features_df[consistent_features]
        y = EEG_features_df[Target]

        # Compute ROC-AUC values for each feature
        feature_auc_scores = {}
        for feature in consistent_features:
            feature_values = X[feature].values
            try:
                auc = roc_auc_score(y, feature_values)
                feature_auc_scores[feature] = auc
            except ValueError:
                # Handle cases where AUC cannot be computed
                print(f"Cannot compute AUC for feature '{feature}' in subject {subnum} under condition '{condition_suffix}'.")
                continue

        # Select features with ROC-AUC > 0.60
        selected_features_auc = {feature: auc for feature, auc in feature_auc_scores.items() if auc > 0.60}

        if not selected_features_auc:
            print(f"No features with ROC-AUC > 0.60 found for subject {subnum} under condition '{condition_suffix}'. Skipping.")
            continue

        # Select top 10 features with highest AUC values
        top_10_features = sorted(selected_features_auc.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_feature_names = [feature for feature, _ in top_10_features]

        # Save the filtered dataframe for this subject and condition
        filtered_EEG_df = EEG_features_df[top_10_feature_names + [Target]]  # Include the target variable

        # Construct output filename with condition suffix
        output_filename = f"Top10_features_df_Subject_{subnum}_{condition_suffix}.pkl"
        pickle_file_path = os.path.join(output_directory, output_filename)
        filtered_EEG_df.to_pickle(pickle_file_path)
        print(f"Saved selected features for subject {subnum} under condition '{condition_suffix}' to {pickle_file_path}")

    else:
        print(f"Filename '{model_file}' does not match expected format. Skipping.")

print("Processing complete.")

# %%
