#%%
"""
Analysis of Classification Performance Variability
=====================================================
This script assesses the impact of consistent feature selection, TMS coil displacement, and MEP amplitude distributions on classification accuracy.
"""

# Imports
import os
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from scipy.stats import spearmanr, mannwhitneyu, levene
from diptest import diptest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


# Define constants and configurations
MODEL_SAVEPATH = "C:/Users/Lisa Haxel/Documents/REFTEP_revision/models/"
PLOT_DIRECTORY = "C:/Users/Lisa Haxel/Documents/REFTEP_revision/publication_figures/"
TABLE_DIRECTORY = "C:/Users/Lisa Haxel/Documents/REFTEP_revision/tables/"

# Load performance metrics df
accuracy_df = pd.read_csv(os.path.join(TABLE_DIRECTORY, 'accuracy_df.csv'))
#%% Feature Robustness and Decoding Accuracy
# Hypothesis: Subjects with higher feature stability scores are associated with higher decoding accuracy

# Initialize variables
feature_ranks_per_subject = {}
subjects = [sub for sub in range(50)] 

for subnum in subjects:
    save_filename = f"Sub_{subnum}_63ch_results.pkl"
    save_path = os.path.join(
        '', save_filename
    )
    try:
        with open(save_path, "rb") as filehandle:
            data = pickle.load(filehandle)
            feature_ranks = defaultdict(dict)  # {feature: {(fold_idx, model_name): rank}}

            # Collect importance scores per model type
            model_importances = defaultdict(list)
            for fold_idx, model_data in enumerate(data[3]):
                for model_info in model_data:
                    model_name, _, importances = model_info
                    model_importances[model_name].extend(importances)

            # Fit scalers for each model type
            scalers = {
                model_name: StandardScaler().fit(
                    np.array(importances).reshape(-1, 1)
                )
                for model_name, importances in model_importances.items()
            }

            # Normalize importances and compute ranks
            for fold_idx, model_data in enumerate(data[3]):
                for model_info in model_data:
                    model_name, features, importances = model_info
                    importances = np.array(importances).reshape(-1, 1)
                    normalized = scalers[model_name].transform(importances).flatten()
                    ranks = (-normalized).argsort().argsort() + 1  # Ranks start from 1

                    for feature, rank in zip(features, ranks):
                        feature_key = feature.strip().lower()
                        feature_ranks[feature_key][(fold_idx, model_name)] = rank

            # Store feature ranks for the subject
            feature_ranks_per_subject[subnum] = feature_ranks

    except FileNotFoundError:
        print(f"File for subject {subnum} not found. Skipping.")
        continue

# Compute feature stability scores
feature_stability_scores = {}

for subnum, feature_ranks in feature_ranks_per_subject.items():
    # Get all features and fold-model combinations
    features = list(feature_ranks.keys())
    fold_models = list({
        key for ranks in feature_ranks.values() for key in ranks.keys()
    })

    # Build DataFrame of ranks, assigning worst rank if missing
    worst_rank = len(features) + 1
    rank_df = pd.DataFrame(worst_rank, index=features, columns=fold_models)

    # Fill in the actual ranks where available
    for feature, ranks in feature_ranks.items():
        for fold_model, rank in ranks.items():
            rank_df.at[feature, fold_model] = rank

    # Compute mean variance of ranks as stability score
    rank_variance = rank_df.var(axis=1, ddof=1)
    feature_stability_scores[subnum] = rank_variance.mean()

# Create DataFrame for feature stability scores
feature_stability_df = pd.DataFrame({
    'Subject': list(feature_stability_scores.keys()),
    'Feature Stability Score': list(feature_stability_scores.values())
})

# Scale the stability scores between 0 and 1
scaler = MinMaxScaler()
feature_stability_df['Scaled Variance'] = scaler.fit_transform(
    feature_stability_df[['Feature Stability Score']]
)

# Merge with accuracy scores
merged_df = pd.merge(feature_stability_df, accuracy_df, on='Subject')

# Spearman's Correlation
corr_coef, corr_p_value = spearmanr(
    merged_df['Accuracy Score'], merged_df['Feature Stability Score']
)
print(f"Spearman's Correlation Coefficient: {corr_coef}")
print(f"P-value: {corr_p_value}")

# Divide Participants into High and Low Performers
median_accuracy = merged_df['Accuracy Score'].median()
merged_df['Performance Group'] = np.where(
    merged_df['Accuracy Score'] >= median_accuracy, 'High', 'Low'
)

# Mann-Whitney U Test
high_group = merged_df[merged_df['Performance Group'] == 'High']['Feature Stability Score']
low_group = merged_df[merged_df['Performance Group'] == 'Low']['Feature Stability Score']

# Since lower variance indicates higher stability, high performers should have lower scores
statistic, p_value = mannwhitneyu(high_group, low_group, alternative='less')
print(f"Mann-Whitney U Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_df,
    x='Accuracy Score',
    y='Scaled Variance',
    hue='Performance Group',
    palette={'High': 'blue', 'Low': 'orange'}
)
plt.title('Accuracy Score vs. Normalized Feature Stability Score\n(Lower Score = Higher Stability)')
plt.xlabel('Accuracy Score')
plt.ylabel('Normalized Feature Stability Score (0-1)')
plt.legend(title='Performance Group')
plt.grid(False)
plt.tight_layout()
plt.savefig(
    '',
    dpi=300
)
plt.show()

#%% Hypothesis: Higher Coil Deviation is Associated with Lower Decoding Accuracy

# Load coil location data 
coil_location_df = pd.read_csv(
    ""
)

# Define Subject Mapping
subject_mapping = {
    0: "REFTEP_018", 1: "REFTEP_001", 2: "REFTEP_002", 3: "REFTEP_003", 4: "REFTEP_004",
    5: "REFTEP_005", 6: "REFTEP_007", 7: "REFTEP_008", 8: "REFTEP_009", 9: "SCREEN3_004",
    10: "SCREEN3_022", 11: "SCREEN3_048", 12: "SCREEN3_056", 13: "SCREEN3_058", 
    14: "SCREEN3_066", 15: "SCREEN3_074", 16: "SCREEN3_080", 17: "SCREEN3_082", 
    18: "SCREEN3_091", 19: "REFTEP_019", 20: "REFTEP_020", 21: "REFTEP_021", 
    22: "REFTEP_022", 23: "REFTEP_023", 24: "REFTEP_024", 25: "REFTEP_025", 
    26: "REFTEP_026", 27: "REFTEP_027", 28: "REFTEP_028", 29: "REFTEP_029", 
    30: "REFTEP_015", 31: "REFTEP_031", 32: "REFTEP_032", 33: "REFTEP_016", 
    34: "REFTEP_034", 35: "REFTEP_035", 36: "REFTEP_036", 37: "REFTEP_017", 
    38: "REFTEP_038", 39: "REFTEP_039", 40: "REFTEP_040", 41: "REFTEP_041", 
    42: "REFTEP_042", 43: "REFTEP_043", 44: "REFTEP_044", 45: "REFTEP_045", 
    46: "REFTEP_046", 47: "REFTEP_014", 48: "REFTEP_048", 49: "REFTEP_049", 
    50: "REFTEP_050"
}

# Load selected trial indices for each subject
final_trials_df = pd.read_csv(
    ''
)

# Map Subjects
final_trials_df['Subject_Mapped'] = final_trials_df['Subject'].map(subject_mapping)
accuracy_df['Subject_Mapped'] = accuracy_df['Subject'].map(subject_mapping)

# Ensure String Types and Strip Whitespace
coil_location_df['subject_reftep'] = coil_location_df['subject_reftep'].astype(str).str.strip()
final_trials_df['Subject_Mapped'] = final_trials_df['Subject_Mapped'].astype(str).str.strip()

def merge_trial_data(coil_df, trials_df):
    """
    Merge coil location and final trials data, keeping only matching trials.
    """
    merged_df = pd.merge(
        coil_df,
        trials_df,
        how='inner',
        left_on=['subject_reftep', 'trial_index'],
        right_on=['Subject_Mapped', 'TrialIndex']
    )
    # Drop Redundant Columns
    merged_df = merged_df.drop(columns=['Subject', 'Subject_Mapped', 'TrialIndex'])
    return merged_df

# Merge DataFrames
merged_df = merge_trial_data(coil_location_df, final_trials_df)

## Calculate Displacement Metrics and Create Result DataFrame
# Initialize Lists to Store Results
results = []

SCD = 20  # Skin-cortex distance in mm (assumed typical value)

def angular_deviation(vec1, vec2):
    """
    Calculate the angular deviation between two vectors in degrees.
    """
    cos_theta = np.clip(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
        -1.0, 1.0
    )
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

# Iterate Over Subjects
for subject, subject_df in merged_df.groupby('subject_reftep'):
    # Calculate Mean Target Metrics
    mean_target = {
        'pos': subject_df[['target_pos_x', 'target_pos_y', 'target_pos_z']].mean().values,
        'dir': subject_df[['target_dir_x', 'target_dir_y', 'target_dir_z']].mean().values,
        'ori': subject_df[['target_ori_x', 'target_ori_y', 'target_ori_z']].mean().values,
        'tilt': subject_df[['target_tilt_x', 'target_tilt_y', 'target_tilt_z']].mean().values,
    }
    
    # Iterate Over Trials
    for idx, row in subject_df.iterrows():
        # Extract Trial and Target Metrics
        trial_metrics = {
            'pos': row[['coil_pos_x', 'coil_pos_y', 'coil_pos_z']].values,
            'dir': row[['coil_dir_x', 'coil_dir_y', 'coil_dir_z']].values,
            'ori': row[['coil_ori_x', 'coil_ori_y', 'coil_ori_z']].values,
            'tilt': row[['coil_tilt_x', 'coil_tilt_y', 'coil_tilt_z']].values,
        }
        target_metrics = {
            'pos': row[['target_pos_x', 'target_pos_y', 'target_pos_z']].values,
            'dir': row[['target_dir_x', 'target_dir_y', 'target_dir_z']].values,
            'ori': row[['target_ori_x', 'target_ori_y', 'target_ori_z']].values,
            'tilt': row[['target_tilt_x', 'target_tilt_y', 'target_tilt_z']].values,
        }
        
        # Calculate Differences from Target and Mean Target
        diff_target = {key: np.linalg.norm(trial_metrics[key] - target_metrics[key]) for key in trial_metrics}
        diff_mean_target = {key: np.linalg.norm(trial_metrics[key] - mean_target[key]) for key in trial_metrics}
        
        # Calculate Angular Deviations
        delta_yaw = angular_deviation(trial_metrics['dir'], target_metrics['dir'])
        delta_pitch = angular_deviation(trial_metrics['ori'], target_metrics['ori'])
        delta_roll = angular_deviation(trial_metrics['tilt'], target_metrics['tilt'])
        
        # Calculate Rotational Components
        cdm_rot_yaw = SCD * np.abs(np.sin(np.radians(delta_yaw)))
        cdm_rot_pitch_roll = SCD * np.sqrt(delta_pitch ** 2 + delta_roll ** 2)
        cdm_rot = cdm_rot_yaw + cdm_rot_pitch_roll
        
        # Combined Displacement Metric (CDM)
        cdm = diff_target['pos'] + cdm_rot
        
        # Append Results
        results.append({
            'subject_reftep': row['subject_reftep'],
            'trial_index': row['trial_index'],
            'MEPClass': row['MEPClass'],
            'MEPSize': row['MEPSize'],
            'pcd_pos': row['pcd_pos'],
            **{f'diff_target_{k}': v for k, v in diff_target.items()},
            **{f'diff_mean_target_{k}': v for k, v in diff_mean_target.items()},
            'cdm_rot': cdm_rot,
            'cdm': cdm
        })

# Create DataFrame from Results
result_df = pd.DataFrame(results)

# Save the Result DataFrame
result_df.to_csv(
    "",
    index=False
)

def calculate_correlations(result_df, metrics):
    """
    Calculate within-subject correlations between metrics and MEPSize.
    """
    subject_correlations = {}
    included_subjects = result_df['subject_reftep'].unique()
    
    for subject in included_subjects:
        subject_data = result_df[result_df['subject_reftep'] == subject]
        subject_correlations[subject] = {}
        for metric in metrics:
            corr, p_val = spearmanr(
                subject_data[metric], subject_data['MEPSize'], nan_policy='omit'
            )
            subject_correlations[subject][metric] = {'correlation': corr, 'p_value': p_val}
    
    return subject_correlations

# Define Metrics to Analyze
metrics = ['diff_target_pos', 'diff_target_dir', 'diff_target_ori', 'diff_target_tilt', 'cdm_rot', 'cdm']

# Calculate Correlations
subject_correlations = calculate_correlations(result_df, metrics)

# Analyze Correlations for 'cdm'
positive_subjects = []
negative_subjects = []

for subject, corr_info in subject_correlations.items():
    corr = corr_info['cdm']['correlation']
    p_val = corr_info['cdm']['p_value']
    if corr > 0:
        positive_subjects.append({'subject': subject, 'correlation': corr, 'p_value': p_val})
    else:
        negative_subjects.append({'subject': subject, 'correlation': corr, 'p_value': p_val})

# Calculate Median Correlations and Significant Subjects
def summarize_correlations(group):
    correlations = [item['correlation'] for item in group]
    p_values = [item['p_value'] for item in group]
    median_corr = np.median(correlations)
    num_sig = sum(p < 0.05 for p in p_values)
    return median_corr, num_sig, len(correlations)

median_pos_corr, num_sig_pos, num_pos = summarize_correlations(positive_subjects)
median_neg_corr, num_sig_neg, num_neg = summarize_correlations(negative_subjects)

print("\n=== Median Within-Subject Correlation between MEP Size and CDM ===")
print("\nPositive Correlations Group:")
print(f"Number of subjects: {num_pos}")
print(f"Median correlation: {median_pos_corr:.3f}")
print(f"Significant subjects (p < 0.05): {num_sig_pos}/{num_pos}")

print("\nNegative Correlations Group:")
print(f"Number of subjects: {num_neg}")
print(f"Median correlation: {median_neg_corr:.3f}")
print(f"Significant subjects (p < 0.05): {num_sig_neg}/{num_neg}")

# Prepare Data for Plotting
average_correlations_pos = [
    item['correlation'] for item in positive_subjects if item['p_value'] < 0.05
]
average_correlations_neg = [
    item['correlation'] for item in negative_subjects if item['p_value'] < 0.05
]
average_correlations = average_correlations_pos + average_correlations_neg
groups = ['Positive'] * len(average_correlations_pos) + ['Negative'] * len(average_correlations_neg)
colors = ['blue' if group == 'Positive' else 'red' for group in groups]

# Create Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    range(len(average_correlations)),
    average_correlations,
    c=colors,
    alpha=0.7,
    s=80
)
plt.xlabel('Subject Index')
plt.ylabel('Average Correlation')
plt.title('Average Correlation per Significant Subject (p < 0.05) Colored by Group')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.xticks(
    range(len(average_correlations)),
    labels=range(1, len(average_correlations) + 1),
    rotation=90
)
# Add Legend
positive_patch = mpatches.Patch(color='blue', label='Positive Correlation Group')
negative_patch = mpatches.Patch(color='red', label='Negative Correlation Group')
plt.legend(handles=[positive_patch, negative_patch])
plt.tight_layout()
# Save the Plot
plt.savefig(
    '',
    dpi=300
)
plt.show()

#%% Hypothesis: MEP Distribution Differs Between High and Low Accuracy Groups

# Merge DataFrames to Align MEP Sizes with Accuracy Scores
merged_df = pd.merge(
    result_df,
    accuracy_df[['Subject_Mapped', 'Accuracy Score']],
    left_on='subject_reftep',
    right_on='Subject_Mapped'
)

# Remove Outliers (MEP Sizes > 5000)
merged_df = merged_df[merged_df['MEPSize'] <= 5000]

# Z-score MEP Sizes
merged_df['MEPSize_z'] = (
    merged_df['MEPSize'] - merged_df['MEPSize'].mean()
) / merged_df['MEPSize'].std()

# Divide Subjects into High and Low Accuracy Groups
median_accuracy = merged_df['Accuracy Score'].median()
high_accuracy_meps = merged_df[merged_df['Accuracy Score'] > median_accuracy]['MEPSize_z']
low_accuracy_meps = merged_df[merged_df['Accuracy Score'] <= median_accuracy]['MEPSize_z']

# Plot Kernel Density Estimates (KDE) for Both Groups
plt.figure(figsize=(14, 7))
sns.kdeplot(high_accuracy_meps, color='blue', label='High Accuracy', bw_adjust=0.5, linewidth=2)
sns.kdeplot(low_accuracy_meps, color='red', label='Low Accuracy', bw_adjust=0.5, linewidth=2)
plt.xlabel('Z-scored MEP Size')
plt.ylabel('Density')
plt.title('KDE of MEP Size for High vs Low Decoding Accuracy Subjects')
plt.legend()
plt.grid(True)
# Save the plot
plt.savefig(
    '',
    dpi=300
)
plt.show()

# Perform Mann-Whitney U Test
u_stat, p_value = mannwhitneyu(
    high_accuracy_meps, low_accuracy_meps, alternative='two-sided'
)
print(f'U-statistic: {u_stat}, p-value: {p_value}')

if p_value < 0.05:
    print("There is a significant difference in MEP size between high and low accuracy groups.")
else:
    print("No significant difference in MEP size between high and low accuracy groups.")

# Calculate Cohen's d
mean_diff = high_accuracy_meps.mean() - low_accuracy_meps.mean()
pooled_std = np.sqrt(
    (high_accuracy_meps.std(ddof=1)**2 + low_accuracy_meps.std(ddof=1)**2) / 2
)
cohens_d = mean_diff / pooled_std
print(f"Cohen's d: {cohens_d}")

# Dip Test for Multimodality
high_dip, high_pval = diptest(high_accuracy_meps)
low_dip, low_pval = diptest(low_accuracy_meps)
print(f'High Accuracy Group: Dip Statistic: {high_dip}, p-value: {high_pval}')
print(f'Low Accuracy Group: Dip Statistic: {low_dip}, p-value: {low_pval}')

# Levene's Test for Equality of Variances
levene_stat, levene_pval = levene(high_accuracy_meps, low_accuracy_meps)
print(f"Levene's test statistic: {levene_stat}, p-value: {levene_pval}")

if levene_pval < 0.05:
    print("The variances of the two groups are significantly different.")
else:
    print("No significant difference in variance between the two groups.")

# Fit Gaussian Mixture Models
gmm_high = GaussianMixture(n_components=2).fit(high_accuracy_meps.values.reshape(-1, 1))
gmm_low = GaussianMixture(n_components=2).fit(low_accuracy_meps.values.reshape(-1, 1))

# Extract Means of the Components
high_means = gmm_high.means_.flatten()
low_means = gmm_low.means_.flatten()
print(f"High Accuracy Modes: {high_means}")
print(f"Low Accuracy Modes: {low_means}")
# %%
