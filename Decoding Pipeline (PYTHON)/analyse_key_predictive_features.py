#%%
"""
Analysis of Key Predictive Features
=====================================================
This script 
- identifies key EEG features that contribute to classification; 
- evaluates the stability and consistency of the feature selection across feature set configurations; 
- clusters participants based on their EEG feature profiles; 
- assesses the similarity of feature distributions across experimental protocols
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import os
import re
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform
import scikit_posthocs as sp
from scipy.stats import ttest_ind

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# Define functions to classify features
def classify_feature_nature(feature):
    return 'Connectivity' if '-' in feature else 'Local'

def categorize_feature_by_type(feature_name):
    feature_name_lower = feature_name.lower()
    if "source" in feature_name_lower:
        return 'Source'
    elif "hjorth" in feature_name_lower:
        return 'Hjorth'
    else:
        return 'Sensor'

# Define dictionaries for region categorization
source_prefix_to_region = {
    '6ma': 'Frontal', 'BA9': 'Frontal', 'SMA': 'Frontal', '8C': 'Frontal', 'PMC': 'Frontal', 'M1': 'Frontal',
    '6d': 'Frontal', '9m': 'Frontal', 'BA46': 'Frontal', '8BM': 'Frontal', 'Frontal': 'Frontal','4': 'Central',
    'Central': 'Central', 'S1':'Parietal', 'BA1': 'Parietal', 'BA2': 'Parietal', 'BA3a': 'Parietal', '3a': 'Parietal',
    'Parietal': 'Parietal', '1': 'Parietal', '2': 'Parietal', '7PL': 'Temporal', '7PC': 'Temporal', '7Am': 'Temporal',
    'Temporal': 'Temporal', 'Occipital': 'Occipital'
}

channel_name_to_region = {
    **{ch: 'Frontal' for ch in ['FCC3H','FCC4H','Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'F1', 'F2', 'F5', 'F6',
                                'Fpz', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8', 'AFF2h', 'AFF6h', 'AFp1', 'AFp2', 'FC1',
                                'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCC1h', 'FCC2h', 'FCC3h', 'FCC4h', 'FCC5h',
                                'FCC6h', 'FFC1h', 'FFC2h', 'FFC3h', 'FFC4h', 'FFC5h', 'FFC6h']},
    **{ch: 'Central' for ch in ['C3', 'C4', 'Cz', 'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                                'CCP1h', 'CCP2h', 'CCP3h', 'CCP4h', 'CCP5h', 'CCP6h','CPz']},
    **{ch: 'Parietal' for ch in ['P3', 'P4', 'Pz', 'P1', 'P2', 'P5', 'P6', 'P7','P8','P9', 'PO3', 'PO4', 'PO7', 'PO8',
                                 'POz', 'PO9', 'PO10', 'PPO1h', 'PPO2h', 'PPO5h', 'PPO6h', 'PPO9h', 'PPO10h',
                                 'CPP1h', 'CPP2h', 'CPP3h', 'CPP4h', 'CPP5h', 'CPP6h']},
    **{ch: 'Temporal' for ch in ['T7', 'T8', 'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8', 'TP9', 'TP10', 'FTT7h',
                                 'FTT8h', 'FFT7h', 'FFT8h', 'FFT9h', 'FFT10h', 'TPP7h', 'TPP8h', 'TPP9h', 'TPP10h',
                                 'TTP7h', 'TTP8h']},
    **{ch: 'Occipital' for ch in ['O1', 'O2', 'Oz', 'OI1h', 'OI2h', 'O9', 'O10', 'POO1', 'POO2', 'POO9h', 'POO10h']}
}

# Merge the dictionaries
prefix_to_region = {**source_prefix_to_region, **channel_name_to_region}

def categorize_feature_by_region(feature_name):
    def find_region_for_part(feature_part):
        for prefix, region in prefix_to_region.items():
            if prefix in feature_part:
                return region
        return 'Other'

    feature_parts = re.split('_|-', feature_name)
    regions = [find_region_for_part(part) for part in feature_parts]
    filtered_regions = [region for region in regions if region != 'Other']

    # Reduce consecutive duplicates
    final_regions = []
    for i, region in enumerate(filtered_regions):
        if i == 0 or region != filtered_regions[i - 1]:
            final_regions.append(region)

    # Sort regions according to predefined order to ensure consistent ordering
    region_order = ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital', 'Other']
    # Remove duplicates by converting to a set, then sort
    unique_regions = sorted(set(final_regions), key=lambda x: region_order.index(x) if x in region_order else len(region_order))

    region_sequence = '-'.join(unique_regions) if unique_regions else 'Other'
    region_sequence = adjust_region_sequence(region_sequence)
    return region_sequence


def categorize_feature_by_hemisphere(feature_name):
    if any(num in feature_name for num in ['1', '3', '5', '7', '9']):
        return 'Left'
    elif any(num in feature_name for num in ['2', '4', '6', '8', '10']):
        return 'Right'
    else:
        return 'Mid'

def categorize_feature_by_frequencyband(feature_name):
    freq_bands = ["delta", "theta", "alpha", "low_beta", "high_beta", "low_gamma", "high_gamma"]
    for band in freq_bands:
        if band in feature_name:
            return band
    return 'Unknown'  # If no band is found

def categorize_connectivity_region(feature_name):
    def find_region_for_part(feature_part):
        for prefix, region in prefix_to_region.items():
            if prefix in feature_part:
                return region
        return 'Other'

    # Check if the feature is a connectivity feature
    if classify_feature_nature(feature_name) == 'Connectivity':
        # Split the feature name on known separators
        feature_parts = re.split('_|-', feature_name)
        regions = [find_region_for_part(part) for part in feature_parts]
        # Remove 'Other' and duplicates
        filtered_regions = []
        for region in regions:
            if region != 'Other' and (len(filtered_regions) == 0 or region != filtered_regions[-1]):
                filtered_regions.append(region)
        # Adjust region sequence
        region_sequence = '-'.join(filtered_regions) if filtered_regions else 'Other'
        region_sequence = adjust_region_sequence(region_sequence)
        return region_sequence
    else:
        return 'Not Connectivity'
    
def adjust_region_sequence(region_sequence):
    regions = region_sequence.split('-')
    # Remove duplicates
    regions = list(dict.fromkeys(regions))
    # Define the desired order
    region_order = ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital', 'Other']
    # Sort the regions according to the order
    regions.sort(key=lambda x: region_order.index(x) if x in region_order else len(region_order))
    return '-'.join(regions)


# Define cohort mappings
subnums = list(range(0, 50))

cohort_mapping = {}
for subnum in subnums:
    if 1 <= subnum <= 8:
        cohort_mapping[subnum] = 'Group 1 (REFTEP)'
    elif 9 <= subnum <= 18:
        cohort_mapping[subnum] = 'Group 2 (Screen 3)'
    elif 19 <= subnum <= 49 or subnum == 0:
        cohort_mapping[subnum] = 'Group 3 (REFTEP++)'
    else:
        cohort_mapping[subnum] = 'Unknown'


# Load top 10 features per subject
def load_top_features(top_features_directory, subnums, file_suffix):
    top_10_features_per_subj = {}
    for subnum in subnums:
        pickle_file_path = f"{top_features_directory}/Top10_features_df_Subject_{subnum}_{file_suffix}.pkl"
        if os.path.exists(pickle_file_path):
            filtered_EEG_df = pd.read_pickle(pickle_file_path)
            top_10_features = [col for col in filtered_EEG_df.columns if col != 'Condition']
            top_10_features_per_subj[subnum] = top_10_features
        else:
            print(f"Top 10 features file for subject {subnum} not found. Skipping.")
    return top_10_features_per_subj


def categorize_features_per_subject(top_10_features_per_subj):
    feature_categories_per_subject = {}
    for subnum, features in top_10_features_per_subj.items():
        categories = {
            'type': [],
            'region': [],
            'hemisphere': [],
            'band': [],
            'nature': [],
            'connectivity_region': []
        }
        for feature in features:
            band = categorize_feature_by_frequencyband(feature)
            if band is not None:
                categories['band'].append(band)
                categories['type'].append(categorize_feature_by_type(feature))
                categories['region'].append(categorize_feature_by_region(feature))
                categories['hemisphere'].append(categorize_feature_by_hemisphere(feature))
                categories['nature'].append(classify_feature_nature(feature))
                categories['connectivity_region'].append(categorize_connectivity_region(feature))
        feature_categories_per_subject[subnum] = categories
    return feature_categories_per_subject


def compute_feature_category_proportions_per_subject(feature_categories_per_subject):
    feature_category_proportions_per_subject = {}
    for subnum, categories in feature_categories_per_subject.items():
        feature_category_proportions = {}
        for cat_type, features in categories.items():
            # Exclude None or empty features
            features = [f for f in features if f is not None and f != 'Unknown' and f != 'Other']
            counts = pd.Series(features).value_counts()
            total_count = counts.sum()
            if total_count > 0:
                proportions = counts / total_count
                feature_category_proportions[cat_type] = proportions
        feature_category_proportions_per_subject[subnum] = feature_category_proportions

    return feature_category_proportions_per_subject

# Create bar plots for each category across all subjects
def plot_feature_distributions(feature_category_proportions_per_subject):

    category_types = ['type', 'nature', 'hemisphere', 'band', 'region']

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    sns.set_style("whitegrid", {'axes.grid': False})

    upper_row_categories = category_types[:3]
    lower_row_categories = category_types[3:]

    width = 7.16
    total_height = 5  # Adjusted as needed

    fig = plt.figure(figsize=(width, total_height))
    grid_spec = fig.add_gridspec(2, 6, height_ratios=[1, 1], width_ratios=[1]*6)

    # Plotting the upper row
    for idx, cat_type in enumerate(upper_row_categories):
        ax = fig.add_subplot(grid_spec[0, idx*2:(idx+1)*2])
        data = []
        for subnum, proportions in feature_category_proportions_per_subject.items():
            prop_series = proportions.get(cat_type, pd.Series(dtype=float))
            prop_series.name = subnum
            data.append(prop_series)
        df = pd.DataFrame(data)
        df.index.name = 'Subject'
        df.reset_index(inplace=True)
        df_melted = df.melt(id_vars='Subject', var_name=cat_type, value_name='Proportion')

        # Exclude invalid categories
        df_melted = df_melted[df_melted[cat_type].notnull()]
        df_melted = df_melted[~df_melted[cat_type].isin(['Unknown', 'Other'])]

        # Calculate and normalize mean proportions
        mean_props = df_melted.groupby(cat_type)['Proportion'].mean().reset_index()
        mean_props['Proportion'] = mean_props['Proportion'] / mean_props['Proportion'].sum()

        mean_props = mean_props.sort_values('Proportion', ascending=True)
        # print mean proportions for each category
        print(f"\n{cat_type.upper()}:")
        print(mean_props)
        sns.barplot(data=mean_props, x='Proportion', y=cat_type, ax=ax, errorbar=None)
        ax.set_title(f'{cat_type.replace("_", " ").capitalize()}')
        ax.grid(False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel('Proportion')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Plotting the lower row
    for idx, cat_type in enumerate(lower_row_categories):
        ax = fig.add_subplot(grid_spec[1, idx*3:(idx+1)*3])
        data = []
        for subnum, proportions in feature_category_proportions_per_subject.items():
            prop_series = proportions.get(cat_type, pd.Series(dtype=float))
            prop_series.name = subnum
            data.append(prop_series)
        df = pd.DataFrame(data)
        df.index.name = 'Subject'
        df.reset_index(inplace=True)
        df_melted = df.melt(id_vars='Subject', var_name=cat_type, value_name='Proportion')

        # Exclude invalid categories
        df_melted = df_melted[df_melted[cat_type].notnull()]
        df_melted = df_melted[~df_melted[cat_type].isin(['Unknown', 'Other'])]

        # Calculate and normalize mean proportions
        mean_props = df_melted.groupby(cat_type)['Proportion'].mean().reset_index()
        mean_props['Proportion'] = mean_props['Proportion'] / mean_props['Proportion'].sum()

        mean_props = mean_props.sort_values('Proportion', ascending=True)

        # print mean proportions for each category
        print(f"\n{cat_type.upper()}:")
        print(mean_props)
        
        sns.barplot(data=mean_props, x='Proportion', y=cat_type, ax=ax, errorbar=None)
        ax.set_title(f'{cat_type.replace("_", " ").capitalize()}')
        ax.grid(False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel('Proportion')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig('feature_proportions.pdf', dpi=300)
    plt.show()

# Function to create combined DataFrame for clustering
def create_combined_df(feature_category_proportions_per_subject):
    combined_data = []
    subject_order = []
    for subnum, cat_props in feature_category_proportions_per_subject.items():
        # Combine all category proportions into a single Series
        combined_series = pd.concat(cat_props.values(), keys=cat_props.keys())
        combined_series.name = subnum
        combined_data.append(combined_series)
        subject_order.append(subnum)
    # Create DataFrame
    combined_df = pd.DataFrame(combined_data)
    combined_df.index = subject_order
    combined_df.fillna(0, inplace=True)
    return combined_df


# -----------------------------
# Cross-Feature-Configuration Analyses
# -----------------------------

def compare_temporal_stability(top_features_directory, subnums):
    configurations = ['64ch', '64ch_time_based']
    configuration_features = {}
    for config in configurations:
        top_features = load_top_features(top_features_directory, subnums, config)
        configuration_features[config] = top_features

    # For each pair of configurations, compute normalized Euclidean distances and Pearson correlations
    for i in range(len(configurations)):
        for j in range(i + 1, len(configurations)):
            config1 = configurations[i]
            config2 = configurations[j]
            subjects_in_both = set(configuration_features[config1].keys()).intersection(configuration_features[config2].keys())
            correlations = []
            significant_counts = 0

            for subnum in subjects_in_both:
                features1 = configuration_features[config1][subnum]
                features2 = configuration_features[config2][subnum]

                # Categorize features
                categories1 = categorize_features(features1)
                categories2 = categorize_features(features2)

                # Compute normalized counts
                proportions1 = compute_proportions(categories1)
                proportions2 = compute_proportions(categories2)

                # Align DataFrames
                df1 = pd.DataFrame(proportions1).fillna(0)
                df2 = pd.DataFrame(proportions2).fillna(0)
                df1, df2 = df1.align(df2, fill_value=0)

                # Compute Pearson correlation
                corr, p_value = pearsonr(df1.values.flatten(), df2.values.flatten())
                correlations.append(corr)

                # Check if the correlation is statistically significant
                if p_value < 0.05:
                    significant_counts += 1

            # Compute average correlation
            avg_correlation = np.mean(correlations)
            print(f"\nComparison between {config1} and {config2}:")
            print(f"Average Pearson Correlation: {avg_correlation:.4f}")
            print(f"Number of Significant Correlations (p < 0.05): {significant_counts} out of {len(subjects_in_both)}")

def categorize_features(features):
    categories = {'Type': [], 'Region': [], 'Hemisphere': [], 'Band': [], 'Nature': []}
    for feature in features:
        categories['Type'].append(categorize_feature_by_type(feature)) 

        categories['Region'].append(categorize_feature_by_region(feature))
        categories['Hemisphere'].append(categorize_feature_by_hemisphere(feature))
        categories['Band'].append(categorize_feature_by_frequencyband(feature))
        categories['Nature'].append(classify_feature_nature(feature))
    return categories

def categorize_features_configs(features):
    categories = {'Type': [], 'Region': [], 'Hemisphere': [], 'Band': [], 'Nature': []}
    for feature in features:
        #categories['Type'].append(categorize_feature_by_type(feature)) 

        categories['Region'].append(categorize_feature_by_region(feature))
        categories['Hemisphere'].append(categorize_feature_by_hemisphere(feature))
        categories['Band'].append(categorize_feature_by_frequencyband(feature))
        categories['Nature'].append(classify_feature_nature(feature))
    return categories

def compute_proportions(categories):
    proportions = {}
    for cat_type, features in categories.items():
        counts = pd.Series(features).value_counts()
        props = counts / counts.sum() if counts.sum() > 0 else counts
        proportions[cat_type] = props
    return proportions

def compare_feature_consistency_across_configurations(top_features_directory, subnums):
    configurations = ['64ch', '126ch', '64ch_hjorth', '126ch_sensor_source']
    configuration_features = {}
    for config in configurations:
        top_features = load_top_features(top_features_directory, subnums, config)
        configuration_features[config] = top_features
    # For each pair of configurations, compute normalized Euclidean distances and Pearson correlations
    for i in range(len(configurations)):
        for j in range(i+1, len(configurations)):
            config1 = configurations[i]
            config2 = configurations[j]
            subjects_in_both = set(configuration_features[config1].keys()).intersection(configuration_features[config2].keys())
            distances = []
            correlations = []
            significant_counts = 0
            for subnum in subjects_in_both:
                features1 = configuration_features[config1][subnum]
                features2 = configuration_features[config2][subnum]
                # Categorize features
                categories1 = categorize_features_configs(features1)
                categories2 = categorize_features_configs(features2)
                # Compute normalized counts
                proportions1 = compute_proportions(categories1)
                proportions2 = compute_proportions(categories2)
                # Align DataFrames
                df1 = pd.DataFrame(proportions1).fillna(0)
                df2 = pd.DataFrame(proportions2).fillna(0)
                df1, df2 = df1.align(df2, fill_value=0)
                # Compute Pearson correlation
                corr, p_value = pearsonr(df1.values.flatten(), df2.values.flatten())
                correlations.append(corr)
            
                # Check if the correlation is statistically significant
                if p_value < 0.05:
                    significant_counts += 1
            # Compute average  correlation
            avg_correlation = np.mean(correlations)
            print(f"\nComparison between {config1} and {config2}:")
            print(f"Average Pearson Correlation: {avg_correlation:.4f}")
            print(f"Number of Significant Correlations (p < 0.05): {significant_counts} out of {len(subjects_in_both)}")


# -----------------------------
# Statistics and Visualization
# -----------------------------
# Set directories and parameters
top_features_directory = ""
subnums = list(range(0, 50))

# Data Loading and Preprocessing
print("Loading top features per subject...")
top_10_features_per_subj = load_top_features(top_features_directory, subnums, '64ch')
print("Categorizing features per subject...")
feature_categories_per_subject = categorize_features_per_subject(top_10_features_per_subj)
print("Computing feature category proportions per subject...")
feature_category_proportions_per_subject = compute_feature_category_proportions_per_subject(feature_categories_per_subject)
# Create combined DataFrame
combined_df = create_combined_df(feature_category_proportions_per_subject)
combined_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in combined_df.columns]
plot_feature_distributions(feature_category_proportions_per_subject)

# -----------------------------
# Cross-Feature-Configuration Analyses
# -----------------------------
print("\n--- Cross-Feature-Configuration Analyses ---")
print("Comparing temporal stability of top features across configurations...")
compare_temporal_stability(top_features_directory, subnums)

print("Comparing feature consistency across configurations...")
compare_feature_consistency_across_configurations(top_features_directory, subnums)

# -----------------------------
# Clustering Across Participants
# -----------------------------

# Data Loading and Preprocessing
print("Loading top features per subject...")
top_10_features_per_subj = load_top_features(top_features_directory, subnums, '64_ch')
print("Categorizing features per subject...")
feature_categories_per_subject = categorize_features_per_subject(top_10_features_per_subj)
print("Computing feature category proportions per subject...")
feature_category_proportions_per_subject = compute_feature_category_proportions_per_subject(feature_categories_per_subject)
# Create combined DataFrame
combined_df = create_combined_df(feature_category_proportions_per_subject)
combined_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in combined_df.columns]
plot_feature_distributions(feature_category_proportions_per_subject)

# Perform the clustering as before
distance_matrix = pdist(combined_df.drop(columns='Cluster', errors='ignore').values, metric='euclidean')
linkage_matrix = linkage(distance_matrix, method='ward')
cluster_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')
combined_df['Cluster'] = cluster_labels

# Analyze clusters
clustered_subjects = combined_df.groupby('Cluster').apply(lambda x: x.index.tolist())

# Calculate percentage of participants in each cluster
total_participants = len(combined_df)
cluster_percentages = combined_df['Cluster'].value_counts(normalize=True) * 100

print("\nCluster Percentages Across All Participants:")
for cluster_num in sorted(cluster_percentages.index):
    percent = cluster_percentages[cluster_num]
    print(f"Cluster {cluster_num}: {percent:.2f}% of participants")

# Compute cluster means
cluster_means = combined_df.groupby('Cluster').mean()

# Compute the difference in mean feature values between clusters
if len(cluster_means) == 2:
    cluster_mean_diff = cluster_means.loc[1] - cluster_means.loc[2]

    # Get the top features favored by each cluster
    top_features_cluster1 = cluster_mean_diff[cluster_mean_diff > 0].sort_values(ascending=False).head(10)
    top_features_cluster2 = (-cluster_mean_diff[cluster_mean_diff < 0]).sort_values(ascending=False).head(10)

    print("\nClusters Across Participants - Cluster 1 Favored Features:")
    print(top_features_cluster1)

    print("\nClusters Across Participants - Cluster 2 Favored Features:")
    print(top_features_cluster2)
else:
    print("More than two clusters detected, cannot compute mean differences between two clusters.")


cluster_1_data = combined_df[combined_df['Cluster'] == 1]
cluster_2_data = combined_df[combined_df['Cluster'] == 2]

p_values = {}
for feature in combined_df.columns.drop('Cluster'):
    _, p_value = ttest_ind(cluster_1_data[feature], cluster_2_data[feature], equal_var=False)
    p_values[feature] = p_value

# Print features that are statistically significant
significance_threshold = 0.05  # use alpha = 0.05
significant_features = {k: v for k, v in p_values.items() if v < significance_threshold}

print("Significant Features Between Clusters:")
for feature, p_val in significant_features.items():
    print(f"{feature}: p-value = {p_val:.5f}")

corrected_threshold = significance_threshold / len(p_values)
significant_features_bonferroni = {k: v for k, v in p_values.items() if v < corrected_threshold}
print("\nSignificant Features (Bonferroni Corrected) Between Clusters:")
for feature, p_val in significant_features_bonferroni.items():
    print(f"{feature}: p-value = {p_val:.5f}")


# Set up the combined plot with adjusted width for a column-wide layout
fig, (ax_barplot, ax_dendro) = plt.subplots(2, 1, figsize=(3.5, 2), dpi=300, gridspec_kw={'height_ratios': [1, 1]})

# -----------------------------
# Barplot Visualization
# -----------------------------
# Combine the top features from both clusters for visualization
top_features = pd.Index(top_features_cluster1.index.tolist() + top_features_cluster2.index.tolist()).unique()

# Reorder the top features so that different categories are next to each other
top_features = pd.Index([
    'region_Central', 'region_Parietal',
    'band_low_beta', 'band_alpha', 'band_theta', 'band_high_gamma', 'band_low_gamma',
    'type_Hjorth', 'type_Sensor',
    'nature_Local'
])
cluster_means_top_features = cluster_means[top_features]

# Plot the cluster means for the top features
cluster_means_top_features.T.plot(kind='bar', ax=ax_barplot, width=0.85)
ax_barplot.set_xlabel("Features", fontsize=2, fontname='Arial')
ax_barplot.set_ylabel("Mean Value", fontsize=2, fontname='Arial')
ax_barplot.set_xticklabels(ax_barplot.get_xticklabels(), rotation=45, ha='right', fontsize=2, fontname='Arial', wrap=True)
ax_barplot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_barplot.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=2, fontname='Arial')
ax_barplot.get_legend().remove()
# remove frame
for spine in ['top', 'right']:
    ax_barplot.spines[spine].set_visible(False)

# -----------------------------
# Dendrogram Visualization
# -----------------------------
# Get the maximum distance in the linkage matrix for color threshold
max_d = max(linkage_matrix[:, 2])
# Set color threshold to create 2 clusters
color_threshold = max_d * 0.7  

# Create the dendrogram with cluster coloring
dendro = dendrogram(
    linkage_matrix,
    labels=combined_df.index.astype(str),
    ax=ax_dendro,
    orientation='top',
    color_threshold=color_threshold,  # Creates the color split
    above_threshold_color='gray'      # Color for links above the threshold
)

ax_dendro.set_xticks([])
ax_dendro.set_yticks([])
# remove frame
for spine in ['top', 'right', 'left', 'bottom']:
    ax_dendro.spines[spine].set_visible(False)

# Adjust the layout to be tighter and prevent overlap
plt.tight_layout()
# save the plot
plt.savefig('cluster_analysis.pdf', dpi=300)
plt.show()

# -----------------------------
# Clustering Per Cohort
# -----------------------------
# Add the 'Cohort' column using the cohort_mapping dictionary
combined_df['Cohort'] = combined_df.index.map(cohort_mapping)

# List of cohorts
cohorts = combined_df['Cohort'].unique()

# Dictionary to store cluster data per cohort
clusters_per_cohort = {}

for cohort in cohorts:
    # Select data for the current cohort
    cohort_data = combined_df[combined_df['Cohort'] == cohort].drop(columns=['Cohort'], errors='ignore')
    if cohort_data.empty or len(cohort_data) < 2:
        continue  # Skip if there's not enough data

    # Perform hierarchical clustering using Ward's method
    distance_matrix = pdist(cohort_data.values, metric='euclidean')
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Determine clusters (using 2 clusters)
    num_clusters = 2
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Assign cluster labels to participants
    cohort_data['Cluster'] = cluster_labels
    combined_df.loc[cohort_data.index, 'Cluster'] = cluster_labels

    # Store cluster data
    clusters_per_cohort[cohort] = cohort_data

    # Calculate percentage of participants in each cluster
    total_participants = len(cohort_data)
    cluster_percentages = cohort_data['Cluster'].value_counts(normalize=True) * 100

    # Print cluster percentages
    print(f"\n{cohort} Cohort Clusters:")
    for cluster_num in range(1, num_clusters + 1):
        percent = cluster_percentages.get(cluster_num, 0)
        print(f"Cluster {cluster_num}: {percent:.2f}% of participants")

    # Analyze cluster characteristics
    cluster_features = cohort_data.groupby('Cluster').mean()

    # Compute the difference in mean feature values between clusters
    cluster_mean_diff = cluster_features.loc[1] - cluster_features.loc[2]

    # Get the top features favored by each cluster
    top_features_cluster1 = cluster_mean_diff[cluster_mean_diff > 0].sort_values(ascending=False).head(10)
    top_features_cluster2 = (-cluster_mean_diff[cluster_mean_diff < 0]).sort_values(ascending=False).head(10)

    print(f"\n{cohort} Cohort - Cluster 1 Favored Features:")
    print(top_features_cluster1)

    print(f"\n{cohort} Cohort - Cluster 2 Favored Features:")
    print(top_features_cluster2)

# -----------------------------
# Comparative Analysis Between Cohorts
# -----------------------------

print("\nComparative Analysis of Clusters Between Cohorts:")

cohorts_list = list(clusters_per_cohort.keys())

for i in range(len(cohorts_list)):
    for j in range(i + 1, len(cohorts_list)):
        cohort1 = cohorts_list[i]
        cohort2 = cohorts_list[j]
        print(f"\nComparing {cohort1} and {cohort2} Cohorts:")

        # Get cluster features for both cohorts
        cluster_features1 = clusters_per_cohort[cohort1].groupby('Cluster').mean()
        cluster_features2 = clusters_per_cohort[cohort2].groupby('Cluster').mean()

        # Compute mean differences
        cluster_mean_diff1 = cluster_features1.loc[1] - cluster_features1.loc[2]
        cluster_mean_diff2 = cluster_features2.loc[1] - cluster_features2.loc[2]

        # Get top features favored by each cluster in both cohorts
        top_features_cluster1_cohort1 = set(cluster_mean_diff1[cluster_mean_diff1 > 0].nlargest(10).index)
        top_features_cluster2_cohort1 = set((-cluster_mean_diff1[cluster_mean_diff1 < 0]).nlargest(10).index)

        top_features_cluster1_cohort2 = set(cluster_mean_diff2[cluster_mean_diff2 > 0].nlargest(10).index)
        top_features_cluster2_cohort2 = set((-cluster_mean_diff2[cluster_mean_diff2 < 0]).nlargest(10).index)

        # Compare top features between clusters
        def feature_overlap(features1, features2):
            overlap = features1.intersection(features2)
            overlap_percent = (len(overlap) / 10) * 100  # Since we have top 10 features
            return overlap_percent, overlap

        # Overlaps between clusters
        overlap_percent_c1_c1, overlap_features_c1_c1 = feature_overlap(top_features_cluster1_cohort1, top_features_cluster1_cohort2)
        overlap_percent_c1_c2, overlap_features_c1_c2 = feature_overlap(top_features_cluster1_cohort1, top_features_cluster2_cohort2)
        overlap_percent_c2_c1, overlap_features_c2_c1 = feature_overlap(top_features_cluster2_cohort1, top_features_cluster1_cohort2)
        overlap_percent_c2_c2, overlap_features_c2_c2 = feature_overlap(top_features_cluster2_cohort1, top_features_cluster2_cohort2)

        print(f"Overlap between {cohort1} Cluster 1 and {cohort2} Cluster 1: {overlap_percent_c1_c1:.2f}% - Features: {overlap_features_c1_c1}")
        print(f"Overlap between {cohort1} Cluster 1 and {cohort2} Cluster 2: {overlap_percent_c1_c2:.2f}% - Features: {overlap_features_c1_c2}")
        print(f"Overlap between {cohort1} Cluster 2 and {cohort2} Cluster 1: {overlap_percent_c2_c1:.2f}% - Features: {overlap_features_c2_c1}")
        print(f"Overlap between {cohort1} Cluster 2 and {cohort2} Cluster 2: {overlap_percent_c2_c2:.2f}% - Features: {overlap_features_c2_c2}")