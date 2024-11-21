#%%
"""
Analysis of Model Performance and Statistical Testing
=====================================================
This script computes statistics, performs statistical tests, and generates plots to evaluate and visualize model performance across different feature set configurations, models, and experimental protocol groups.
"""
# Imports
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
)
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, kruskal
import scikit_posthocs as sp

# Define constants and configurations
MODEL_SAVEPATH = ""
PLOT_DIRECTORY = ""
TABLE_DIRECTORY = ""


FEATURE_CONFIGS = [
    "64ch_hjorth",
    "64ch",
    "126ch",
    "126ch_sensor_source",
    "64ch_hjorth_shuffled",
    "64ch_shuffled",
    "126ch_shuffled",
    "126ch_sensor_source_shuffled",

]

TIME_FEATURE_CONFIGS = [
    "126ch_sensor_source_time_based",
    "64ch_hjorth_time_based",
    "64ch_time_based",
    "126ch_time_based",
    
]

FEATURE_NAME_MAPPING = {
    '64ch_hjorth': 'Hjorth-filtered (4 ROIs)',
    '64ch': 'Sensor (64 Channels)',
    '126ch': 'Sensor (126 Channels)',
    '126ch_sensor_source': 'Sensor and Source (126 Channels)',
    '64ch_hjorth_shuffled': 'Random Labels Hjorth-filtered (4 ROIs)',
    '64ch_shuffled': 'Random Labels Sensor (64 Channels)',
    '126ch_shuffled': 'Random Labels Sensor (126 Channels)',
    '126ch_sensor_source_shuffled': 'Random Labels Sensor and Source (126 Channels)',
    '126ch_sensor_source_time_based': 'Sensor and Source (126 Channels) Time',
    '64ch_hjorth_time_based': 'Hjorth-filtered (4 ROIs) Time',
    '64ch_time_based': 'Sensor (64 Channels) Time',
    '126ch_time_based': 'Sensor (126 Channels) Time',
}

METRICS = {
    "F1 Score": f1_score,
    "ROC AUC Score": roc_auc_score,
    "Precision Score": precision_score,
    "Recall Score": recall_score,
    "Accuracy Score": accuracy_score
}

SUBJECT_NUMBERS = [i for i in range(50)]  
MODEL_NAME_MAPPING = {0: 'SVM', 1: 'LogReg', 2: 'RF'}

def load_y_test(feature_config, save_dir=MODEL_SAVEPATH):
    """Load true y values and adjust labels for specific configurations."""
    y_tests_path = os.path.join(save_dir, 'y_test.pkl')
    with open(y_tests_path, 'rb') as f:
        y_tests = np.array(pickle.load(f))

    if feature_config in ["126ch_sensor_source", "126ch_sensor_source_shuffled"]:
        # Adjust labels if needed
        y_tests = np.where(y_tests == 0, 1, 0)
    return y_tests

def collect_predictions():
    """Collect predictions, compute metrics, and aggregate results."""
    all_results = []

    for feature_config in FEATURE_CONFIGS + TIME_FEATURE_CONFIGS:
        is_time_feature = "time" in feature_config
        true_y_tests = None if is_time_feature else load_y_test(feature_config)

        for subnum in SUBJECT_NUMBERS:
            save_filename = f"Sub_{subnum}_{feature_config}_results.pkl"
            save_path = os.path.join(MODEL_SAVEPATH, save_filename)

            try:
                with open(save_path, "rb") as filehandle:
                    data = pickle.load(filehandle)
                    
                    if is_time_feature:
                        # For time feature configurations, extract accuracies directly
                        accuracy_arr = data[0]  # Assuming accuracy array is at index 0
                        average_accuracy = np.mean(accuracy_arr)
                        all_results.append({
                            'Subject': subnum,
                            'Feature Configuration': feature_config,
                            'Accuracy Score': average_accuracy,
                            'Shuffled': 'shuffled' in feature_config,
                            # Add other metrics as NaN since they're not applicable
                            'F1 Score': np.nan,
                            'ROC AUC Score': np.nan,
                            'Precision Score': np.nan,
                            'Recall Score': np.nan,
                            'Model': 'N/A'
                        })
                    else:
                        # For other configurations, extract predictions and compute metrics
                        final_y_preds = data[4]  

                        for rep_idx, rep in enumerate(final_y_preds):
                            for fold_idx, fold in enumerate(rep):
                                for model_idx, model in enumerate(fold):
                                    y_pred = model[1] 
                                    y_true = true_y_tests[fold_idx]

                                    if len(y_true) != len(y_pred):
                                        print(f"Mismatch in lengths for Subject {subnum}, Fold {fold_idx}, Model {model_idx}")
                                        continue

                                    # Compute metrics
                                    fold_metrics = {}
                                    for name, func in METRICS.items():
                                        try:
                                            fold_metrics[name] = func(y_true, y_pred)
                                        except ValueError:
                                            fold_metrics[name] = np.nan  
                                    fold_metrics.update({
                                        'Subject': subnum,
                                        'Rep': rep_idx,
                                        'Fold': fold_idx,
                                        'Model': MODEL_NAME_MAPPING.get(model_idx, f'Model_{model_idx}'),
                                        'Feature Configuration': feature_config,
                                        'Shuffled': 'shuffled' in feature_config
                                    })
                                    all_results.append(fold_metrics)

            except (FileNotFoundError, IOError, IndexError) as e:
                print(f"Error processing {save_filename}: {e}")
                continue

    return all_results

def compute_average_metrics(results_df):
    """Compute average metrics per subject and feature configuration."""
    # Select relevant metric columns
    metric_columns = ['F1 Score', 'ROC AUC Score', 'Precision Score', 'Recall Score', 'Accuracy Score']
    # Group by Subject and Feature Configuration and compute the mean
    average_metrics_df = results_df.groupby(['Subject', 'Feature Configuration'])[metric_columns].mean().reset_index()
    return average_metrics_df

def compute_mean_std_per_feature_config(average_metrics_df):
    """Compute mean and standard deviation per feature configuration over subjects."""
    # Select relevant metric columns
    metric_columns = ['F1 Score', 'ROC AUC Score', 'Precision Score', 'Recall Score', 'Accuracy Score']
    # Group by Feature Configuration and compute mean and std
    mean_std_df = average_metrics_df.groupby('Feature Configuration')[metric_columns].agg(['mean', 'std']).reset_index()
    return mean_std_df

# Define the cohort mapping function
def map_cohort(subnum):
    if 1 <= subnum <= 8:
        return 'Group 2'
    elif 9 <= subnum <= 18:
        return 'Group 3'
    elif 19 <= subnum <= 49 or subnum == 0:
        return 'Group 1'
    else:
        return 'Unknown'

# Map configurations to shorter labels
label_mapping = {
    'Hjorth-filtered (4 ROIs)': 'Hjorth-filtered\n(4 ROIs)',
    'Sensor (64 Channels)': 'Sensor\n(64 Channels)',
    'Sensor (126 Channels)': 'Sensor\n(126 Channels)',
    'Sensor and Source (126 Channels)': 'Sensor and Source\n(126 Channels)',
    'Random Labels Hjorth-filtered (4 ROIs)': 'Hjorth-filtered\n(4 ROIs)',
    'Random Labels Sensor (64 Channels)': 'Sensor\n(64 Channels)',
    'Random Labels Sensor (126 Channels)': 'Sensor\n(126 Channels)',
    'Random Labels Sensor and Source (126 Channels)': 'Sensor and Source\n(126 Channels)'
}
    
# Collect the data
all_results = collect_predictions()
results_df = pd.DataFrame(all_results)

# Replace 'Feature Configuration' names with human-readable names
results_df['Feature Configuration'] = results_df['Feature Configuration'].replace(FEATURE_NAME_MAPPING)

# Add a 'Shuffled' column based on whether 'Random Labels' is in the feature configuration name
results_df['Shuffled'] = results_df['Feature Configuration'].str.contains('Random Labels')

# Save the results to a CSV file
results_df.to_csv(os.path.join(TABLE_DIRECTORY, 'results_df.csv'), index=False)

# Compute average metrics per subject and feature configuration
average_metrics_df = compute_average_metrics(results_df)
average_metrics_df.to_csv(os.path.join(TABLE_DIRECTORY, 'average_metrics_df.csv'), index=False)

# Compute mean and standard deviation per feature configuration
mean_std_metrics_df = compute_mean_std_per_feature_config(average_metrics_df)
mean_std_metrics_df.to_csv(os.path.join(TABLE_DIRECTORY, 'mean_std_metrics_df.csv'), index=False)

# Add the cohort mapping to the DataFrame
results_df['Cohort'] = results_df['Subject'].apply(map_cohort)

# Separate non-shuffled and shuffled data
df_non_shuffled = results_df[~results_df['Shuffled']].copy()
df_shuffled = results_df[results_df['Shuffled']].copy()

# Filter data for time features only
df_time = df_non_shuffled[df_non_shuffled['Feature Configuration'].str.contains('Time', case=False)]
df_time_shuffled = df_shuffled[df_shuffled['Feature Configuration'].str.contains('Time', case=False)]

# Filter data for the 'Sensor (64 Channels)' configuration
df_cohort = df_non_shuffled[df_non_shuffled['Feature Configuration'] == 'Sensor (64 Channels)'].copy()

# Prepare data for plotting by excluding '_time' configurations
df_part1 = df_non_shuffled[~df_non_shuffled['Feature Configuration'].str.contains('Time', case=False)].copy()

# Compute median shuffled accuracy per feature configuration
median_shuffled = df_shuffled.groupby('Feature Configuration')['Accuracy Score'].median().reset_index()

# Compute median shuffled accuracy per cohort
median_shuffled_cohort = df_shuffled[df_shuffled['Feature Configuration'] == 'Random Labels Sensor (64 Channels)']
median_shuffled_cohort = median_shuffled_cohort.groupby('Cohort')['Accuracy Score'].median().reset_index()
median_shuffled_cohort['Feature Configuration'] = 'Sensor\n(64 Channels)'


df_part1['Feature Configuration'] = df_part1['Feature Configuration'].map(label_mapping)
median_shuffled['Feature Configuration'] = median_shuffled['Feature Configuration'].map(label_mapping)

# Compute and print accuracy statistics for feature configurations
avg_df_part1 = df_part1.groupby(['Feature Configuration', 'Subject'])['Accuracy Score'].mean().reset_index()
feature_config_stats = avg_df_part1.groupby('Feature Configuration')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
feature_config_stats['range'] = feature_config_stats['max'] - feature_config_stats['min']
print("Feature Configuration Accuracy Scores (Non-Shuffled):\n", feature_config_stats)

avg_df_shuffled = df_shuffled.groupby(['Feature Configuration', 'Subject'])['Accuracy Score'].mean().reset_index()
shuffled_config_stats = avg_df_shuffled.groupby('Feature Configuration')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
shuffled_config_stats['range'] = shuffled_config_stats['max'] - shuffled_config_stats['min']
print("Feature Configuration Accuracy Scores (Shuffled):\n", shuffled_config_stats)

# Compute and print accuracy statistics for models
sensor_64_data = df_non_shuffled[df_non_shuffled['Feature Configuration'] == 'Sensor (64 Channels)'].copy()
avg_df_models = sensor_64_data.groupby(['Model', 'Subject'])['Accuracy Score'].mean().reset_index()
model_stats = avg_df_models.groupby('Model')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
model_stats['range'] = model_stats['max'] - model_stats['min']
print("Model Accuracy Scores (Non-Shuffled) for Sensor (64 Channels):\n", model_stats)

sensor_64_data_shuffled = df_shuffled[df_shuffled['Feature Configuration'] == 'Random Labels Sensor (64 Channels)'].copy()
avg_df_shuffled_models = sensor_64_data_shuffled.groupby(['Model', 'Subject'])['Accuracy Score'].mean().reset_index()
shuffled_model_stats = avg_df_shuffled_models.groupby('Model')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
shuffled_model_stats['range'] = shuffled_model_stats['max'] - shuffled_model_stats['min']
print("Model Accuracy Scores (Shuffled) for Sensor (64 Channels):\n", shuffled_model_stats)

# Compute and print accuracy statistics for cohorts
avg_df_cohort = df_cohort.groupby(['Cohort', 'Subject'])['Accuracy Score'].mean().reset_index()
cohort_stats = avg_df_cohort.groupby('Cohort')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
cohort_stats['range'] = cohort_stats['max'] - cohort_stats['min']
print("Cohort Accuracy Scores (Non-Shuffled) for Sensor (64 Channels):\n", cohort_stats)

avg_df_shuffled_cohort = sensor_64_data_shuffled.groupby(['Cohort', 'Subject'])['Accuracy Score'].mean().reset_index()
shuffled_cohort_stats = avg_df_shuffled_cohort.groupby('Cohort')['Accuracy Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
shuffled_cohort_stats['range'] = shuffled_cohort_stats['max'] - shuffled_cohort_stats['min']
print("Cohort Accuracy Scores (Shuffled) for Sensor (64 Channels):\n", shuffled_cohort_stats)

# Perform Wilcoxon signed-rank tests for each participant against chance level
individual_results = []
for subject in sensor_64_data['Subject'].unique():
    subject_data = sensor_64_data[sensor_64_data['Subject'] == subject]['Accuracy Score']
    chance_level = df_shuffled[(df_shuffled['Feature Configuration'] == 'Random Labels Sensor (64 Channels)') & (df_shuffled['Subject'] == subject)]['Accuracy Score'].median()

    if len(subject_data) > 1:
        w_stat, p_value = wilcoxon(subject_data - chance_level)
        individual_results.append({
            'Subject': subject,
            'w-statistic': w_stat,
            'p-value': p_value,
            'Chance Level': chance_level,
            'p-value < 0.001': p_value < 0.001
        })
    else:
        individual_results.append({
            'Subject': subject,
            'w-statistic': None,
            'p-value': None,
            'Chance Level': chance_level,
            'p-value < 0.001': None
        })

test_results_df = pd.DataFrame(individual_results)
print("Wilcoxon signed-rank tests for Sensor (64 Channels) Configuration against Chance Level (Shuffled Median):\n", test_results_df)

# Kruskal-Wallis test for feature configurations
clean_df_part1 = df_part1.dropna(subset=['Feature Configuration', 'Accuracy Score'])
groups_feature = [group['Accuracy Score'].values for _, group in clean_df_part1.groupby('Feature Configuration')]
kw_feature = kruskal(*groups_feature)
print("Kruskal-Wallis for Feature Configurations (Non-Shuffled):\nH-statistic =", kw_feature.statistic, ", p-value =", kw_feature.pvalue)

# Post-hoc Dunn tests for feature configurations
if kw_feature.pvalue < 0.05:
    dunn_feature = sp.posthoc_dunn(clean_df_part1, val_col='Accuracy Score', group_col='Feature Configuration', p_adjust='bonferroni')
    print("Post-hoc Dunn tests for Feature Configurations (with Bonferroni correction):\n", dunn_feature)

# Kruskal-Wallis test for models
groups_model = [group['Accuracy Score'].values for _, group in sensor_64_data.groupby('Model')]
kw_model = kruskal(*groups_model)
print("Kruskal-Wallis for Models (Non-Shuffled, Sensor 64 Channels):\nH-statistic =", kw_model.statistic, ", p-value =", kw_model.pvalue)

# Post-hoc Dunn tests for models
if kw_model.pvalue < 0.05:
    dunn_model = sp.posthoc_dunn(sensor_64_data, val_col='Accuracy Score', group_col='Model', p_adjust='bonferroni')
    print("Post-hoc Dunn tests for Models (Sensor 64 Channels, with Bonferroni correction):\n", dunn_model)

# Kruskal-Wallis test for cohorts
groups_cohort = [group['Accuracy Score'].values for _, group in sensor_64_data.groupby('Cohort')]
kw_cohort = kruskal(*groups_cohort)
print("Kruskal-Wallis for Cohorts (Non-Shuffled, Sensor 64 Channels):\nH-statistic =", kw_cohort.statistic, ", p-value =", kw_cohort.pvalue)

# Post-hoc Dunn tests for cohorts
if kw_cohort.pvalue < 0.05:
    dunn_cohort = sp.posthoc_dunn(sensor_64_data, val_col='Accuracy Score', group_col='Cohort', p_adjust='bonferroni')
    print("Post-hoc Dunn tests for Cohorts (Sensor 64 Channels, with Bonferroni correction):\n", dunn_cohort)

# Set up the combined plot
fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.5), dpi=300, gridspec_kw={'width_ratios': [3, 3, 3], 'wspace': 0.25})

# Define a function to adjust boxplot aesthetics
def adjust_boxplot(ax):
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor('none')  # No fill
        patch.set_edgecolor((r, g, b, 1))  # Colored edge
        patch.set_linewidth(1.5)  # Thick lines
    sns.despine(ax=ax)
    ax.grid(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

# Part 1: Feature Configurations
ax1 = axes[0]
sns.boxplot(
    y='Feature Configuration',
    x='Accuracy Score',
    data=df_part1,
    palette=sns.color_palette("pastel", n_colors=4),
    fliersize=0,
    linewidth=1.5,
    ax=ax1,
    whis=1.5
)
adjust_boxplot(ax1)
ax1.set_ylabel("Feature Configurations", fontsize=8)
ax1.set_xlabel("Test Accuracy", fontsize=8)

# Add median shuffled accuracy lines
for i, config in enumerate(df_part1['Feature Configuration'].unique()):
    median_value = median_shuffled[median_shuffled['Feature Configuration'] == config]['Accuracy Score'].values
    if median_value.size > 0:
        y_pos = i
        ax1.vlines(x=median_value[0], ymin=y_pos - 0.3, ymax=y_pos + 0.3, color='red', linestyle='--', linewidth=1)
        ax1.plot([median_value[0]], [y_pos], 'ro', markersize=3)

# Part 2: Models
ax2 = axes[1]
sns.boxplot(
    y='Model',
    x='Accuracy Score',
    data=sensor_64_data,
    palette=sns.color_palette("pastel", n_colors=3),
    fliersize=0,
    linewidth=1.5,
    ax=ax2,
    whis=1.5
)
adjust_boxplot(ax2)
ax2.set_ylabel("Model", fontsize=8)
ax2.set_xlabel("Test Accuracy", fontsize=8)

# Add median shuffled accuracy lines
median_shuffled_model = df_shuffled.groupby('Model')['Accuracy Score'].median().reset_index()
for i, model in enumerate(sensor_64_data['Model'].unique()):
    median_value = median_shuffled_model[median_shuffled_model['Model'] == model]['Accuracy Score'].values
    if median_value.size > 0:
        y_pos = i
        ax2.vlines(x=median_value[0], ymin=y_pos - 0.3, ymax=y_pos + 0.3, color='red', linestyle='--', linewidth=1)
        ax2.plot([median_value[0]], [y_pos], 'ro', markersize=3)

# Part 3: Cohorts
ax3 = axes[2]
sns.boxplot(
    y='Cohort',
    x='Accuracy Score',
    data=df_cohort,
    palette=sns.color_palette("pastel", n_colors=3),
    fliersize=0,
    linewidth=1.5,
    ax=ax3,
    whis=1.5
)
adjust_boxplot(ax3)
ax3.set_ylabel("Cohort", fontsize=8)
ax3.set_xlabel("Test Accuracy", fontsize=8)

# Add median shuffled accuracy lines
for i, cohort in enumerate(df_cohort['Cohort'].unique()):
    median_value = median_shuffled_cohort[median_shuffled_cohort['Cohort'] == cohort]['Accuracy Score'].values
    if median_value.size > 0:
        y_pos = i
        ax3.vlines(x=median_value[0], ymin=y_pos - 0.3, ymax=y_pos + 0.3, color='red', linestyle='--', linewidth=1)
        ax3.plot([median_value[0]], [y_pos], 'ro', markersize=3)

# Ensure all subplots have the same x-axis limits
x_min = min(ax.get_xlim()[0] for ax in axes)
x_max = max(ax.get_xlim()[1] for ax in axes)
for ax in axes:
    ax.set_xlim([x_min, x_max])

# Adjust layout and save the figure
plt.subplots_adjust(left=0.175, right=0.99, top=0.965, bottom=0.12, wspace=0.25)
plt.savefig(os.path.join(PLOT_DIRECTORY, 'combined_figure_with_cohorts.pdf'), dpi=300)
plt.show()
plt.close()
