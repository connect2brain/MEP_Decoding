#%%
"""
Permutation Testing Pipeline 
======================================================
This pipeline performs permutation testing on the trained classifiers to determine the significance of the classification accuracy.
Different conditions such as channel setups (126 vs 64), feature sets (Hjorth-filtered only, sensor & source, sensor) and time-based classification can be selected.
"""

# Load all relevant libraries
from src.Dependencies import *
from src.mRMR_feature_select import mRMR_feature_select
from src.helper_functions import *

# Load the configuration file
with open("config.json", "r") as file:
    config = json.load(file)

# Access Feature savepath 
Feature_savepath: str = config["paths"]["Feature_savepath"]

# Set seed for reproducibility
random.seed(789)

# Running the pipeline with various conditions
subjects = list(range(0, 50))
conditions = [
    {'condition': '126_channels', 'shuffle': False, 'time_based': False, 'hjorth_only': False, 'sensor_source': False},
    {'condition': '64_channels', 'shuffle': False, 'time_based': False, 'hjorth_only': False, 'sensor_source': False},
    {'condition': '126_channels', 'shuffle': False, 'time_based': True, 'hjorth_only': False, 'sensor_source': False},
    {'condition': '64_channels', 'shuffle': False, 'time_based': True, 'hjorth_only': False, 'sensor_source': False},
    {'condition': '64_channels', 'shuffle': False,  'time_based': False, 'hjorth_only': True, 'sensor_source': False},
    {'condition': '64_channels', 'shuffle': False,  'time_based': True, 'hjorth_only': True, 'sensor_source': False},
    {'condition': '126_channels', 'shuffle': False,  'time_based': False, 'hjorth_only': False, 'sensor_source': True},
    {'condition': '126_channels', 'shuffle': False, 'time_based': True, 'hjorth_only': False, 'sensor_source': True},
]

for cond in conditions:
    run_permutation_pipeline(subjects, Feature_savepath, **cond)

# %%
