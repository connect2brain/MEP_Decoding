# EEG Feature Extraction and Decoding Pipeline

This repository contains MATLAB and Python code for the prediction of motor cortex excitability states (high vs. low MEP amplitude) from pre-stimulus EEG features, as described in

[**Decoding Motor Excitability in TMS using EEG-Features: An Exploratory Machine Learning Approach**](https://ieeexplore.ieee.org/document/10795227)

The code is organized into two main components:
1. **EEG Feature Extraction (MATLAB)**
2. **Decoding Pipeline (Python)**

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Feature Extraction (MATLAB)](#feature-extraction-matlab)
  - [Decoding Pipeline (Python)](#decoding-pipeline-python)
- [Scripts Overview](#scripts-overview)
- [Contact Information](#contact-information)

## Prerequisites

### MATLAB
- MATLAB R2020a or later
- FieldTrip Toolbox (for EEG data processing)
- MATLAB Toolboxes:
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

### Python
- Python 3.8 or later
- Required Python packages listed in `requirements.txt`

## Installation

### MATLAB Setup
1. **Install MATLAB and the Required Toolboxes**
2. **Install FieldTrip**
   - Download [FieldTrip](https://www.fieldtriptoolbox.org/)
   - Add FieldTrip to your MATLAB path:
     ```matlab
     addpath('path_to_fieldtrip');
     ft_defaults;
     ```
3. **Clone the Repository**
   ```sh
   git clone https://github.com/lisahaxel/MEP_Decoding.git
   ```
4. **Navigate to the MATLAB Directory**
   ```sh
   cd 'MEP_Decoding/Feature extraction (MATLAB)'
   ```

### Python Setup
1. **Install Python 3.8 or Later**
2. **Clone the Repository** (if not done already)
   ```sh
   git clone https://github.com/lisahaxel/MEP_Decoding.git
   ```
3. **Navigate to the Python Directory**
   ```sh
   cd 'MEP_Decoding/Decoding Pipeline (Python)'
   ```
4. **Create a Virtual Environment (Recommended)**
   ```sh
   python -m venv venv
   ```
5. **Activate the Virtual Environment**
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
6. **Install Required Packages**
   ```sh
   pip install -r requirements.txt
   ```

## Directory Structure
```
MEP_Decoding/
├── Feature extraction (MATLAB)
│   ├── realign_electrodes_create_headmodels.m
│   ├── source_sensor_feature_extraction.m
│   ├── MATLAB utility functions/
│       ├── MVL_surrogate.m
│       ├── calculateAsmmetry.m
│       ├── calculateConnectivity.m
│       ├── categorize_channels.m
│       ├── categorize_channels_rl.m
│       ├── phastimate.m
│       ├── prepareLFM.m
│       ├── preprocessData.m
│       ├── rid_rihaczek4.m
│       ├── spatialFilter.m
│       └── tfMVL.m
│   ├── Glasser_parcellation.mat
│   └── RdBuReversed.mat
├── Decoding Pipeline (Python)
│   ├── requirements.txt
│   ├── config.json
│   ├── Machine_Learning.py
│   ├── Permutation_testing.py
│   ├── extract_top_10_features.py
│   ├── evaluate_performance.py
│   ├── analyse_key_predictive_features.py
│   ├── analyse_classification_performance_variability.py
│   └── src/
│       ├── Dependencies.py
│       ├── helper_functions.py
│       └── mRMR_feature_select.py
```

## Usage

### Feature Extraction (MATLAB)

1. **Create Individual Head Models**
   
   Run `realign_electrodes_create_headmodels.m` to create individual head models from BEM surface meshes.
   ```matlab
   % In MATLAB
   run('realign_electrodes_create_headmodels.m');
   ```

2. **Extract Sensor and Source Features**

   Run `source_sensor_feature_extraction.m` to extract EEG sensor and source features.
   ```matlab
   % In MATLAB
   run('source_sensor_feature_extraction.m');
   ```
   **Input Structure**: The script expects a fieldtrip `epochs` structure containing the preprocessed, labled and aligned EEG data:
   ```matlab
   % Structure fields:
   % epochs.trial         : [1000×400×ch double] - timepoints × trials × channels
   % epochs.time          : [1×1000 double] - time vector in seconds (-1.0050 to -0.0060)
   % epochs.dimord        : 'time_rpt_chan'
   % epochs.label         : {113×1 cell} - channel labels
   % epochs.cfg           : [struct] - preprocessing information
   % epochs.sampleinfo    : [400×2 double] - sample boundaries per trial
   % epochs.trialSorting  : [400×1 double] - original trial indices
   % epochs.trialLabels   : [400×1 double] - binary excitability labels (1: high MEPs[1-200], 0: low MEPs[201-400])
   % epochs.mepsize       : [400×1 double] - MEP amplitudes in mV
   % epochs.fsample       : 1000 - sampling rate in Hz
   ```

3. **Required Files**
   - `Glasser_parcellation.mat`: Parcellation atlas for source reconstruction.
   - `RdBuReversed.mat`: Colormap for plotting head models.

### Decoding Pipeline (Python)

1. **Configure Parameters**

   Edit `config.json` to set parameters for the machine learning pipeline and permutation testing.

2. **Run Machine Learning Analysis**

   Execute `Machine_Learning.py` to perform feature selection and classification.
   ```sh
   python Machine_Learning.py
   ```
   **Features**:
   - Trains Support Vector Machine, Logistic Regression, and Random Forest classifiers.
   - Allows selection of channel setups (126 vs. 64 channels).
   - Supports different feature sets (e.g., Hjorth-filtered only, sensor & source, sensor only).
   - Options for label shuffling and time-based classification.

3. **Perform Permutation Testing**

   Run `Permutation_testing.py` to determine the significance of classification accuracy for each participant.
   ```sh
   python Permutation_testing.py
   ```

4. **Extract Top Predictive Features**

   Use `extract_top_10_features.py` to identify the most important EEG features driving the classification for each participant.
   ```sh
   python extract_top_10_features.py
   ```
   **Methodology**:
   - Uses normalized model coefficients (LR, SVM), Gini importance scores (RF), and ROC-AUC values.
   - Selects top 10 features based on consistent high rankings, stability across folds, and ROC-AUC > 0.60.

5. **Evaluate Performance**

   Execute `evaluate_performance.py` to evaluate and compare classification performance between subjects and feature set configurations and generate performance plots.
   ```sh
   python evaluate_performance.py
   ```

6. **Analyze Key Predictive Features**

   Run `analyse_key_predictive_features.py` for a detailed analysis of predictive EEG features.
   ```sh
   python analyse_key_predictive_features.py
   ```
   **Analysis Includes**:
   - Identification of key features contributing to classification.
   - Stability and consistency assessment of feature selection across feature set configurations.
   - Clustering participants based on EEG feature profiles.
   - Comparison of feature distributions across protocols.

7. **Analyze Classification Performance Variability**

   Use `analyse_classification_performance_variability.py` to assess factors affecting classification accuracy.
   ```sh
   python analyse_classification_performance_variability.py
   ```
   **Factors Assessed**:
   - Consistent feature selection across cross-validation folds.
   - TMS coil displacement.
   - MEP amplitude distributions.

## Scripts Overview

### MATLAB Scripts
- **realign_electrodes_create_headmodels.m**: Creates individual head models from BEM surface meshes.
- **source_sensor_feature_extraction.m**: Extracts sensor and source features from EEG data.
- **MATLAB utility functions/**: Contains supporting utility functions.
- **Glasser_parcellation.mat**: Parcellation atlas for source reconstruction.
- **RdBuReversed.mat**: Colormap data for visualizations.

### Python Scripts
- **requirements.txt**: Lists Python dependencies.
- **config.json**: Configuration settings for the pipelines.
- **Machine_Learning.py**: Performs feature selection and classification.
- **Permutation_testing.py**: Conducts permutation testing for significance assessment.
- **extract_top_10_features.py**: Identifies and analyzes top predictive EEG features.
- **evaluate_performance.py**: Analyzes and visualizes model performance.
- **analyse_key_predictive_features.py**: Examines key EEG features and their stability.
- **analyse_classification_performance_variability.py**: Investigates factors influencing classification variability.

#### src/
- **Dependencies.py**: Loads necessary libraries for the pipelines.
- **helper_functions.py**: Contains helper functions for analysis scripts.
- **mRMR_feature_select.py**: Implements the mRMR feature selection algorithm.

## Contact Information
For questions or assistance, please contact:

**Lisa Haxel**  
Email: lisa.haxel@uni-tuebingen.de  
Affiliations: [Hertie Institute for Clinical Brain Research](https://www.hih-tuebingen.de/en/) & [Tübingen AI Center, University of Tübingen](https://tuebingen.ai)


Please cite our publication if you use this code in your research.
 
