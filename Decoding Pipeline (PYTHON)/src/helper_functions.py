#### Import relevent libraries ####
from src.Dependencies import *


####  Load feature dataframes ####
def load_ml_features_df(Feature_savepath: str, subnum: int) -> pd.DataFrame:
    """Load and concatenate feature dataframes for a given subject.
    
    Returns:
        pd.DataFrame: The concatenated features dataframe.
    """
    # Modified condition map to handle both cases
    condition_map = {
        "low": 0,
        "high": 1,
        "Low": 0,
        "High": 1
    }

    def standardize_conditions(df, column_name='Condition'):
        """Standardize the condition column to lowercase."""
        df[column_name] = df[column_name].str.lower()  # Convert to lowercase
        return df

    # Initialize dataframes
    irasa_source_df, con_source_df, pac_source_df, power_df, irasa_df, asymmetry_df, hjorth_power_df, hjorth_phase_df, M1_S1_power_asymmetry_df, hjorth_con_df, hjorth_pac_df, hjorth_fractal_df = [pd.DataFrame() for _ in range(12)]

    def read_mat_feature(Feature_savepath: str, subnum: int, feature_name: str) -> pd.DataFrame:
        """
        Read a .mat file and return its contents as a pandas DataFrame.
        """
        file_path = f"{Feature_savepath}{feature_name}_df_Subject_{subnum}.mat"
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True)

        # Find the key that contains the actual data, ignoring meta keys
        data_key = next(key for key in mat_data.keys() if not key.startswith('__'))

        # Extract the data
        data_cell = mat_data[data_key]

        # Process the cell array to convert it into a list of lists
        data = []
        for i in range(data_cell.shape[0]):
            row = []
            for j in range(data_cell.shape[1]):
                elem = data_cell[i, j]
                if isinstance(elem, np.ndarray):
                    elem = elem.item()
                row.append(elem)
            data.append(row)

        return pd.DataFrame(data[1:], columns=data[0])
    
    def check_consistency(dfs: List[pd.DataFrame]) -> None:
        """Check for consistency across multiple dataframes."""
        base_df = dfs[0]
        for df in dfs[1:]:
            non_matching_rows = pd.merge(base_df, df, on=['Trial_index', 'Condition'], how='outer', indicator=True)
            non_matching_rows = non_matching_rows[non_matching_rows['_merge'] != 'both']
            if not non_matching_rows.empty:
                raise ValueError("Inconsistency in Trial_index or Condition across dataframes.")
    
    def add_measurement_column(df: pd.DataFrame, measurement: str) -> pd.DataFrame:
        """Add a 'Measurement' column to the DataFrame."""
        df["Measurement"] = [measurement] * df.shape[0]
        return df

    def prepare_and_pivot(df: pd.DataFrame, measurement: str, *values_columns: str, additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare and pivot a DataFrame."""
        df = add_measurement_column(df, measurement)
        columns_to_pivot = ["Measurement"]
        if additional_columns:
            columns_to_pivot.extend(additional_columns)
            
        temp_df = df.pivot_table(index=["Trial_index", "Condition"], 
                                columns=columns_to_pivot, 
                                values=values_columns).reset_index()
        
        temp_df.columns = ["_".join(col) for col in temp_df.columns]
        return temp_df
    
    try:
        feature_names = ['Source', 'SourceCon', 'pac', 'Power', 'irasa', 'Asymmetry', 'Hjorth_Power', 'Hjorth_Phase', 'M1_S1_power_asymmetry', 'HjorthCon','hjorth_pac', 'hjorth_fractal']
        
        # Load and store dataframes
        feature_dfs = Parallel(n_jobs=-1)(delayed(read_mat_feature)(Feature_savepath, subnum, feature) for feature in feature_names)

        # Standardize 'Condition' values in each DataFrame
        feature_dfs = [standardize_conditions(df) for df in feature_dfs]
    
        # Assign to your individual feature dataframes here
        irasa_source_df, con_source_df, pac_source_df, power_df, irasa_df, asymmetry_df, hjorth_power_df, hjorth_phase_df, M1_S1_power_asymmetry_df, hjorth_con_df, hjorth_pac_df, hjorth_fractal_df = feature_dfs
        
        check_consistency(feature_dfs[:3])

        # Define the names of the columns that contain the phase values
        phase_df_variable_names = [
            'Phase_C3H', 'Phase_C4H', 'Phase_FCC3H', 'Phase_FCC4H', 
            'Sin_Phase_C3H', 'Cos_Phase_C3H', 
            'Sin_Phase_C4H', 'Cos_Phase_C4H', 
            'Sin_Phase_FCC3H', 'Cos_Phase_FCC3H', 
            'Sin_Phase_FCC4H', 'Cos_Phase_FCC4H'
        ]

        EEG_features_df = pd.concat([
            prepare_and_pivot(irasa_source_df, "Source", "Peak_Freq", "Peak_SNR_dB", "Mean_SNR", "AUC_SNR", "Abs_Power", "Rel_Power", "Fractal_Activity", "Fractal_Slope", "Fractal_Offset",  additional_columns=["ROI", "Freq_band"]),
            prepare_and_pivot(con_source_df, "SourceCon", "iPLV", "PLI", "wPLI", "Coh", "imCoh", "lagCoh", "oPEC",  additional_columns=["ROI_comb", "Freq_band"]),
            prepare_and_pivot(pac_source_df, "PAC", "PAC_value", additional_columns=["ROI", "Freq_combinations"]),
            prepare_and_pivot(power_df, "Power", "PSD", additional_columns=["Quant_status", "Freq_band", "Channel"]),
            prepare_and_pivot(irasa_df, "irasa", "Metric_Value",  additional_columns=["Channel", "Freq_band", "Quant_status", "Component"]),
            prepare_and_pivot(asymmetry_df, "Asymmetry", "Asymmetry_score", additional_columns=["Freq_band", "Brain_region"]),
            prepare_and_pivot(hjorth_power_df, "Hjorth_Power", "Power_C3", "Power_C4", "Power_FCC3H", "Power_FCC4H"),
            prepare_and_pivot(hjorth_phase_df, "Hjorth_Phase", *phase_df_variable_names),
            prepare_and_pivot(M1_S1_power_asymmetry_df, "M1_S1_power_asymmetry", "Power_Asymmetry_C", "Power_Asymmetry_FCC"),
            prepare_and_pivot(hjorth_con_df, "Hjorth_Con", "iPLV", "PLI", "wPLI", "Coh", "imCoh", "lagCoh", "oPEC",  additional_columns=["Signal_comb", "Freq_band"]),
            prepare_and_pivot(hjorth_pac_df, "Hjorth_PAC", "PAC_value", additional_columns=["Signal", "Freq_combinations"]),
            prepare_and_pivot(hjorth_fractal_df, "hjorth_fractal", "Exponent", "Offset", additional_columns=["Freq_band", "Brain_region", "Channel"])
        ],
          axis=1)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()    

    def rename_columns_by_pattern(df: pd.DataFrame, pattern: str, new_name: str) -> pd.DataFrame:
        """Rename columns based on a regex pattern."""
        regex_pattern = re.compile(pattern)
        rename_dict = {col: new_name for col in df.columns if regex_pattern.fullmatch(col)}
        df.rename(columns=rename_dict, inplace=True)
        return df

    def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns, keeping only the first occurrence."""
        _, idx_to_keep = np.unique(df.columns, return_index=True)
        return df.iloc[:, np.sort(idx_to_keep)]


    # Rename all variations of 'Condition' and 'Trial_index'
    EEG_features_df = rename_columns_by_pattern(EEG_features_df, r".*Condition.*", 'Condition')
    EEG_features_df = rename_columns_by_pattern(EEG_features_df, r".*Trial_index.*", 'Trial_index')

    # Remove duplicate 'Condition' and 'Trial_index' columns
    EEG_features_df = remove_duplicate_columns(EEG_features_df)

    # Reorder columns
    all_columns: List[str] = EEG_features_df.columns.tolist()
    index_condition_cols: List[str] = ['Trial_index', 'Condition']
    feature_columns: List[str] = [col for col in all_columns if col not in index_condition_cols]
    new_column_order: List[str] = index_condition_cols + feature_columns[::-1]
    EEG_features_df = EEG_features_df[new_column_order]

    # Replace infinite values with NaN
    EEG_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert 'Condition' column to integers using the predefined condition map
    EEG_features_df["Condition"] = EEG_features_df["Condition"].map(condition_map)

    return EEG_features_df

def load_ml_sensor_features_df(Feature_savepath: str, subnum: int) -> pd.DataFrame:
    """Load and concatenate feature dataframes for a given subject (excluding source dataframes).
    
    Returns:
        pd.DataFrame: The concatenated sensor-level features dataframe.
    """
    # Modified condition map to handle both cases
    condition_map = {
        "low": 0,
        "high": 1,
        "Low": 0,
        "High": 1
    }

    def standardize_conditions(df, column_name='Condition'):
        """Standardize the condition column to lowercase."""
        df[column_name] = df[column_name].str.lower()  # Convert to lowercase
        return df

    # Initialize dataframes
    power_df, irasa_df, asymmetry_df, hjorth_power_df, hjorth_phase_df, M1_S1_power_asymmetry_df, hjorth_con_df, hjorth_pac_df, hjorth_fractal_df = [pd.DataFrame() for _ in range(9)]

    def read_mat_feature(Feature_savepath: str, subnum: int, feature_name: str) -> pd.DataFrame:
        """
        Read a .mat file and return its contents as a pandas DataFrame.
        """
        file_path = f"{Feature_savepath}{feature_name}_df_Subject_{subnum}.mat"
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True)

        # Find the key that contains the actual data, ignoring meta keys
        data_key = next(key for key in mat_data.keys() if not key.startswith('__'))

        # Extract the data
        data_cell = mat_data[data_key]

        # Process the cell array to convert it into a list of lists
        data = []
        for i in range(data_cell.shape[0]):
            row = []
            for j in range(data_cell.shape[1]):
                elem = data_cell[i, j]
                if isinstance(elem, np.ndarray):
                    elem = elem.item()
                row.append(elem)
            data.append(row)

        return pd.DataFrame(data[1:], columns=data[0])
    
    def check_consistency(dfs: List[pd.DataFrame]) -> None:
        """Check for consistency across multiple dataframes."""
        base_df = dfs[0]
        for df in dfs[1:]:
            non_matching_rows = pd.merge(base_df, df, on=['Trial_index', 'Condition'], how='outer', indicator=True)
            non_matching_rows = non_matching_rows[non_matching_rows['_merge'] != 'both']
            if not non_matching_rows.empty:
                raise ValueError("Inconsistency in Trial_index or Condition across dataframes.")
    
    def add_measurement_column(df: pd.DataFrame, measurement: str) -> pd.DataFrame:
        """Add a 'Measurement' column to the DataFrame."""
        df["Measurement"] = [measurement] * df.shape[0]
        return df

    def prepare_and_pivot(df: pd.DataFrame, measurement: str, *values_columns: str, additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare and pivot a DataFrame."""
        df = add_measurement_column(df, measurement)
        columns_to_pivot = ["Measurement"]
        if additional_columns:
            columns_to_pivot.extend(additional_columns)
            
        temp_df = df.pivot_table(index=["Trial_index", "Condition"], 
                                columns=columns_to_pivot, 
                                values=values_columns).reset_index()
        
        temp_df.columns = ["_".join(col) for col in temp_df.columns]
        return temp_df
    
    try:
        feature_names = ['Power', 'irasa', 'Asymmetry', 'Hjorth_Power', 'Hjorth_Phase', 'M1_S1_power_asymmetry', 'HjorthCon', 'hjorth_pac', 'hjorth_fractal']
        
        # Load and store dataframes, excluding source dataframes
        feature_dfs = Parallel(n_jobs=-1)(delayed(read_mat_feature)(Feature_savepath, subnum, feature) for feature in feature_names)
        
        # Standardize 'Condition' values in each DataFrame
        feature_dfs = [standardize_conditions(df) for df in feature_dfs]
    
        # Assign to your individual feature dataframes here
        power_df, irasa_df, asymmetry_df, hjorth_power_df, hjorth_phase_df, M1_S1_power_asymmetry_df, hjorth_con_df, hjorth_pac_df, hjorth_fractal_df = feature_dfs
        
        check_consistency(feature_dfs[:1])  # Check consistency for pac_source_df

        # Define the names of the columns that contain the phase values
        phase_df_variable_names = [
            'Phase_C3H', 'Phase_C4H', 'Phase_FCC3H', 'Phase_FCC4H', 
            'Sin_Phase_C3H', 'Cos_Phase_C3H', 
            'Sin_Phase_C4H', 'Cos_Phase_C4H', 
            'Sin_Phase_FCC3H', 'Cos_Phase_FCC3H', 
            'Sin_Phase_FCC4H', 'Cos_Phase_FCC4H'
        ]

        EEG_features_df = pd.concat([
            prepare_and_pivot(power_df, "Power", "PSD", additional_columns=["Quant_status", "Freq_band", "Channel"]),
            prepare_and_pivot(irasa_df, "irasa", "Metric_Value",  additional_columns=["Channel", "Freq_band", "Quant_status", "Component"]),
            prepare_and_pivot(asymmetry_df, "Asymmetry", "Asymmetry_score", additional_columns=["Freq_band", "Brain_region"]),
            prepare_and_pivot(hjorth_power_df, "Hjorth_Power", "Power_C3", "Power_C4", "Power_FCC3H", "Power_FCC4H"),
            prepare_and_pivot(hjorth_phase_df, "Hjorth_Phase", *phase_df_variable_names),
            prepare_and_pivot(M1_S1_power_asymmetry_df, "M1_S1_power_asymmetry", "Power_Asymmetry_C", "Power_Asymmetry_FCC"),
            prepare_and_pivot(hjorth_con_df, "Hjorth_Con", "iPLV", "PLI", "wPLI", "Coh", "imCoh", "lagCoh", "oPEC",  additional_columns=["Signal_comb", "Freq_band"]),
            prepare_and_pivot(hjorth_pac_df, "Hjorth_PAC", "PAC_value", additional_columns=["Signal", "Freq_combinations"]),
            prepare_and_pivot(hjorth_fractal_df, "hjorth_fractal", "Exponent", "Offset", additional_columns=["Freq_band", "Brain_region", "Channel"])
        ], axis=1)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()    

    def rename_columns_by_pattern(df: pd.DataFrame, pattern: str, new_name: str) -> pd.DataFrame:
        """Rename columns based on a regex pattern."""
        regex_pattern = re.compile(pattern)
        rename_dict = {col: new_name for col in df.columns if regex_pattern.fullmatch(col)}
        df.rename(columns=rename_dict, inplace=True)
        return df

    def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns, keeping only the first occurrence."""
        _, idx_to_keep = np.unique(df.columns, return_index=True)
        return df.iloc[:, np.sort(idx_to_keep)]

    # Rename all variations of 'Condition' and 'Trial_index'
    EEG_features_df = rename_columns_by_pattern(EEG_features_df, r".*Condition.*", 'Condition')
    EEG_features_df = rename_columns_by_pattern(EEG_features_df, r".*Trial_index.*", 'Trial_index')

    # Remove duplicate 'Condition' and 'Trial_index' columns
    EEG_features_df = remove_duplicate_columns(EEG_features_df)

    # Reorder columns
    all_columns: List[str] = EEG_features_df.columns.tolist()
    index_condition_cols: List[str] = ['Trial_index', 'Condition']
    feature_columns: List[str] = [col for col in all_columns if col not in index_condition_cols]
    new_column_order: List[str] = index_condition_cols + feature_columns[::-1]
    EEG_features_df = EEG_features_df[new_column_order]

    # Replace infinite values with NaN
    EEG_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert 'Condition' column to integers using the predefined condition map
    EEG_features_df["Condition"] = EEG_features_df["Condition"].map(condition_map)

    return EEG_features_df

#### ML PIPELINE FUNCTIONS ####
def create_outer_cv_splits(data, target, n_repetitions, k_out, random_state_multiplier):
    """
    Generate outer cross-validation splits.
    
    :param data: DataFrame containing the features.
    :param target: Series or array containing the target variable.
    :param n_repetitions: Number of repetitions for the CV.
    :param k_out: Number of outer folds.
    :param random_state_multiplier: Multiplier for random state to ensure different shuffles.
    :return: Lists containing train-test indices and test indices for each repetition and fold.
    """
    rep_outer_cv = []
    rep_outer_cv_test = []
    for rep in range(n_repetitions):
        skf = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=(rep + 1) * random_state_multiplier)
        outer_cv = []
        outer_cv_test = []
        for train_index, test_index in skf.split(data, target):
            outer_cv.append([train_index, test_index])
            outer_cv_test.append(test_index)
        rep_outer_cv.append(outer_cv)
        rep_outer_cv_test.append(outer_cv_test)
    return rep_outer_cv, rep_outer_cv_test


def prepare_and_standardize_data(train_index, test_index, data, feature_cols, non_feature_cols, target_col):
    """
    Prepares and scales data given train and test indices.
    
    :param train_index: Indices for training data.
    :param test_index: Indices for testing data.
    :param data: Complete dataset containing features and target.
    :param feature_cols: List of feature column names.
    :param non_feature_cols: List of non-feature column names, unused in this function.
    :param target_col: Name of the target column.
    :return: Scaled training and testing dataframes, target series for training and testing, and the scaler.
    """
    # Split data
    X_train, X_test = data.iloc[train_index][feature_cols], data.iloc[test_index][feature_cols]
    y_train, y_test = data.iloc[train_index][target_col], data.iloc[test_index][target_col]

    # Robust scaling of features
    scaler = RobustScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    return X_train, X_test, y_train, y_test,scaler

def check_alignment_and_reset_indices(X_train, y_train, X_test, y_test):
    """
    Ensures alignment between feature and label datasets and resets their indices.
    
    :param X_train: Training features DataFrame.
    :param y_train: Training labels Series.
    :param X_test: Testing features DataFrame.
    :param y_test: Testing labels Series.
    :return: DataFrames and Series with reset indices if aligned, otherwise raises an error.
    """
    # Check lengths
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Mismatch in number of rows between features and labels.")

    # Reset indices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Check if indices match now
    if not (X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)):
        raise ValueError("Indices of features and labels do not match after reset.")

    return X_train, y_train, X_test, y_test


def perform_mRMR_feature_selection(X_train, y_train, feature_types, max_mRMR_features):
    """
    Perform mRMR feature selection for each feature type and return the selected features.
    
    :param X_train: Standardized training data.
    :param y_train: Training target variable.
    :param feature_types: List of feature types for mRMR selection.
    :param max_mRMR_features: Maximum number of features to select using mRMR.
    :return: List of selected features.
    """
    initial_selected_features = []

    for feature_type in feature_types:
        features_of_type = [col for col in X_train.columns if feature_type in col]

        temp_X_train = X_train[features_of_type]

        if temp_X_train.shape[1] <= max_mRMR_features:
            selected_features = temp_X_train.columns.tolist()
            # Handle scores for these features separately if needed
        else:
            # Capture the additional outputs from mRMR_feature_select
            mRMR_indices = mRMR_feature_select(
                temp_X_train.values, y_train.values,
                num_features_to_select=max_mRMR_features,
                K_MAX=500, n_jobs=-1, verbose=False
            )
            selected_features = temp_X_train.columns[mRMR_indices].tolist()


        initial_selected_features.extend(selected_features)

    # Remove duplicates if any
    selected_features = list(set(initial_selected_features))

    # Return selected features and their scores
    return selected_features


def check_alignment_and_reset_indices_train2(X_train, y_train):
    """
    Validates alignment between training features and labels, and resets their indices.
    
    :param X_train: Training features DataFrame.
    :param y_train: Training labels Series.
    :return: Aligned and index-reset training features and labels.
    """
    # Ensure the number of feature rows matches the number of labels
    if len(X_train) != len(y_train):
        raise ValueError("Mismatch in number of rows between features and labels.")

    # Reset indices to align features with labels, removing the old index
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    # Verify that features and labels have matching indices after reset
    if not X_train.index.equals(y_train.index):
        raise ValueError("Indices of features and labels do not match after reset.")

    return X_train, y_train


def initialize_hyperparameters_and_accuracy_arrays(SVM_exponent_range, SVM_C_base, LogReg_exponent_range, LogReg_C_base, RF_trees, RF_depth, n_feat_mRMR1, max_mRMR_features, SVM_kernels):
    """
    Initializes hyperparameters for SVM, Logistic Regression, and Random Forest, along with accuracy and feature arrays.
    
    :param SVM_exponent_range: Tuple specifying the range of exponents for SVM's C parameter.
    :param SVM_C_base: Base for SVM's C parameter calculation.
    :param LogReg_exponent_range: Tuple specifying the range of exponents for Logistic Regression's C parameter.
    :param LogReg_C_base: Base for Logistic Regression's C parameter calculation.
    :param RF_trees: List of tree counts for Random Forest.
    :param RF_depth: List of tree depths for Random Forest.
    :param n_feat_mRMR1: List of feature counts to evaluate.
    :param max_mRMR_features: Maximum number of mRMR features.
    :param SVM_kernels: List of SVM kernel types.
    :return: Initialized hyperparameters and arrays for accuracy, selected features, and feature importance.
    """
    # SVM hyperparameters
    svm_exponent = np.round(np.linspace(SVM_exponent_range[0], SVM_exponent_range[1], 10), 5)
    C_parameter_SVM = np.power(np.full(len(svm_exponent), SVM_C_base), svm_exponent)
    kernels: List[str] = SVM_kernels

    # Logistic Regression hyperparameters
    logreg_exponent = np.round(np.linspace(LogReg_exponent_range[0], LogReg_exponent_range[1], 10), 5)
    C_parameter_LogReg = np.power(np.full(len(logreg_exponent), LogReg_C_base), logreg_exponent)

    # Random Forest hyperparameters
    trees = np.array(RF_trees)
    depth = np.array(RF_depth)

    # Initialize accuracy and feature importance arrays
    val_acc_svm = np.zeros((len(n_feat_mRMR1), len(C_parameter_SVM), len(SVM_kernels)))
    val_feat_svm = np.full((len(n_feat_mRMR1), len(C_parameter_SVM), len(SVM_kernels), max_mRMR_features), np.nan)
    val_feat_import_svm = np.full((len(n_feat_mRMR1), len(C_parameter_SVM), len(SVM_kernels), max_mRMR_features), np.nan)

    val_acc_logreg = np.zeros((len(n_feat_mRMR1), len(C_parameter_LogReg)))
    val_feat_logreg = np.full((len(n_feat_mRMR1), len(C_parameter_LogReg), max_mRMR_features), np.nan)
    val_feat_import_logreg = np.full((len(n_feat_mRMR1), len(C_parameter_LogReg), max_mRMR_features), np.nan)

    val_acc_rf = np.zeros((len(n_feat_mRMR1), len(trees), len(depth)))
    val_feat_rf = np.full((len(n_feat_mRMR1), len(trees), len(depth), max_mRMR_features), np.nan)
    val_feat_import_rf = np.full((len(n_feat_mRMR1), len(trees), len(depth), max_mRMR_features), np.nan)

    return C_parameter_SVM, C_parameter_LogReg, trees, depth, val_acc_svm, val_feat_svm, val_feat_import_svm, val_acc_logreg, val_feat_logreg, val_feat_import_logreg, val_acc_rf, val_feat_rf, val_feat_import_rf, kernels



def rerun_mRMR_and_filter_dataset(X_train_filtered, y_train, initial_selected_features, n_features_to_select, mRMR_params):
    """
    Re-run mRMR on the selected subset and filter the dataset to include only the selected features
    
    :param X_train_filtered: The filtered training set.
    :param y_train: Training target variables.
    :param initial_selected_features: List of initially selected features.
    :param n_features_to_select: Number of features to select in this run.
    :param mRMR_params: Parameters for the mRMR feature selection, like K_MAX and n_jobs.
    :return: Filtered training set with the selected mRMR features
    """
    re_ranked_mRMR_indices = mRMR_feature_select(
        X_train_filtered.values, y_train.values,
        num_features_to_select=n_features_to_select,
        **mRMR_params
    )
    selected_mRMR_features_array = np.array(initial_selected_features)[re_ranked_mRMR_indices]
    selected_mRMR_features = selected_mRMR_features_array.tolist()

    X_train_mRMR_subset = X_train_filtered[selected_mRMR_features]

    # Return the filtered dataset, selected features, and their score dictionaries
    return X_train_mRMR_subset, selected_mRMR_features


def create_inner_folds(X_train, y_train, k_fold, rep, random_state_multiplier):
    """
    Generates inner cross-validation folds for nested cross-validation.
    
    :param X_train: Features of the training set.
    :param y_train: Labels of the training set.
    :param k_fold: Number of folds for the inner cross-validation.
    :param rep: Current repetition number in an outer loop, affecting the random state.
    :param random_state_multiplier: Multiplier for the random state to ensure variability across repetitions.
    :return: List of tuples containing train and test indices for each fold.
    """
    skf2 = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=(rep + 1) * random_state_multiplier)
    inner_cv_splits = [(train_index, test_index) for train_index, test_index in skf2.split(X_train, y_train)]
    return inner_cv_splits


def train_validate_svm_rfe_scaled(X_train, y_train, C_param, kernel, inner_cv_splits, min_features_to_select):
    """
    Trains and validates an SVM model using RFE with cross-validation, scaling features within the pipeline.
    
    :param X_train: Training features.
    :param y_train: Training labels.
    :param C_param: Regularization parameter C for SVM.
    :param kernel: Kernel type for SVM.
    :param inner_cv_splits: Inner cross-validation splits for model validation.
    :param min_features_to_select: Minimum number of features to be selected by RFE.
    :return: Cross-validated accuracy, names of selected features, and feature importances for linear kernel.
    """
    # Initialize and configure SVM model
    svm_classifier = SVC(C=C_param, kernel=kernel, tol=1e-3, cache_size=4000)
    scaler = RobustScaler()
    pipeline = Pipeline(steps=[('scaler', scaler), ('svm', svm_classifier)])

    # Specify the importance_getter for linear kernels
    if kernel == 'linear':
        importance_getter = lambda x: x.named_steps['svm'].coef_
    else:
        # For non-linear kernels, feature importance might not be available
        importance_getter = 'auto' 

    rfecv = RFECV(estimator=pipeline, n_jobs=-1, scoring="accuracy", cv=inner_cv_splits, 
                  min_features_to_select=min_features_to_select, importance_getter=importance_getter)
    rfecv.fit(X_train, y_train)

    acc = rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - min_features_to_select]
    selected_features = np.where(rfecv.support_)[0]
    selected_feature_names = X_train.columns[selected_features]

    feature_importances = None
    if kernel == 'linear':
        # Scaling and fitting on selected features for linear kernel
        X_train_selected = X_train.iloc[:, selected_features]
        scaler.fit(X_train_selected)
        X_train_scaled = scaler.transform(X_train_selected)
        svm_classifier.fit(X_train_scaled, y_train)
        feature_importances = np.abs(svm_classifier.coef_.ravel()) / np.sum(np.abs(svm_classifier.coef_.ravel()))

    return acc, selected_feature_names, feature_importances

def find_optimal_svm_parameters_scaled(X_train, y_train, C_parameters, kernels, inner_cv_splits, min_features_to_select):
    """
    Identifies optimal SVM parameters and feature selection using RFE and cross-validation with feature scaling.
    
    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param C_parameters: List of C parameter values to evaluate.
    :param kernels: List of kernel types to evaluate.
    :param inner_cv_splits: Inner cross-validation splits for model evaluation.
    :param min_features_to_select: Minimum number of features to retain in RFE.
    :return: Optimal C parameter, kernel, selected features, feature importances, and accuracy.
    """
    best_acc = 0
    best_C = None
    best_kernel = None
    best_features = None
    best_importances = None
    all_results = []

    for C_param in C_parameters:
        for kernel in kernels:
            # Call the updated function with scaling
            acc, selected_features, feature_importances = train_validate_svm_rfe_scaled(X_train, y_train, C_param, kernel, inner_cv_splits, min_features_to_select)
            all_results.append((acc, C_param, kernel, selected_features, feature_importances))

    # Sort results by accuracy, then by number of features (fewer is better), then by C (higher is better for regularization)
    sorted_results = sorted(all_results, key=lambda x: (-x[0], len(x[3]), -x[1]))

    best_acc, best_C, best_kernel, best_features, best_importances = sorted_results[0]

    return best_C, best_kernel, best_features, best_importances, best_acc

def train_evaluate_final_svm(X_train, y_train, X_test, y_test, C_param, kernel):
    """
    Trains an SVM model on the training set and evaluates its performance on both the training and test sets.
    
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param C_param: Optimal C parameter for regularization.
    :param kernel: Selected kernel type for the SVM.
    :return: Trained SVM model, training accuracy, and test accuracy.
    """
    # Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Fit the SVM model
    svm_model = SVC(C=C_param, kernel=kernel, tol=1e-3, cache_size=4000)
    svm_model.fit(X_train, y_train)

    # Predict and calculate accuracy for both training and test sets
    train_predictions = svm_model.predict(X_train)
    test_predictions = svm_model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    #train_accuracy = balanced_accuracy_score(y_train, train_predictions) -> for classification stability evaluation
    #test_accuracy = balanced_accuracy_score(y_test, test_predictions) -> for classification stability evaluation

    # Return the model along with accuracies
    return svm_model, train_accuracy, test_accuracy


def logistic_regression_sfs_scaled(X_train, y_train, C_values, k_features, inner_cv):
    """
    Performs feature selection for Logistic Regression using Sequential Feature Selection (SFS) within a cross-validated framework.
    
    :param X_train: Training features.
    :param y_train: Training labels.
    :param C_values: Range of regularization strengths to evaluate.
    :param k_features: Target numbers of features to select.
    :param inner_cv: Cross-validation splitting strategy.
    :return: CV scores for combinations of C values and feature counts, optimal features, normalized coefficients, best C value, best feature count, and highest validation accuracy.
    """
    # Initialize the array to store the cross-validation scores
    inner_cv_scores = np.zeros((len(C_values), len(k_features)))

    best_validation_accuracy = 0
    for C_idx, C in enumerate(C_values):
        # Initialize and configure logistic regression model
        LogReg = LogisticRegression(penalty="l2", solver='liblinear', C=C, max_iter=50000)
        scaler = RobustScaler()
        pipeline = Pipeline(steps=[('scaler', scaler), ('logreg', LogReg)])

        # Initialize and fit SFS with the pipeline
        sfs = SFS(pipeline, k_features=k_features, forward=True, scoring="accuracy", verbose=0, floating=False, cv=inner_cv, n_jobs=-1)
        sfs.fit(X_train, y_train)

        # Save CV scores for each SFS step and update best_validation_accuracy if needed
        for feat, k_feat in enumerate(k_features):
            mean_score = sfs.get_metric_dict()[k_feat]["avg_score"]
            inner_cv_scores[C_idx, feat] = mean_score

            if mean_score > best_validation_accuracy:
                best_validation_accuracy = mean_score

        # Find the best number of features for this C 
        best_K_idx = np.argmax(inner_cv_scores[C_idx])
        best_K = k_features[best_K_idx]

        # Save the features chosen by SFS based on index from pre mRMR
        sfs_feat_idx = list(sfs.subsets_[best_K]["feature_idx"])
        optimal_features = X_train.columns[sfs_feat_idx]

        # Refit LogReg Model on optimal features to extract coefficients for feature importance
        LogReg.fit(X_train[optimal_features], y_train)
        logreg_coef = np.abs(LogReg.coef_.ravel())
        logreg_coef_normalized = logreg_coef / np.sum(logreg_coef)

        # Return the optimal number of features and their indices, along with the best validation accuracy
        return inner_cv_scores, optimal_features, logreg_coef_normalized, C, best_K, best_validation_accuracy

def train_evaluate_logreg(X_train, y_train, X_test, y_test, C, optimal_features):
    """
    Trains a Logistic Regression model on selected features and evaluates its performance.
    
    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param X_test: Test feature set.
    :param y_test: Test labels.
    :param C: Regularization strength for the Logistic Regression.
    :param optimal_features: List of selected features for training the model.
    :return: Trained Logistic Regression model, training accuracy, and test accuracy.
    """
    # Initialize the logistic regression model
    LogReg = LogisticRegression(penalty='l2', C=C, solver='liblinear')

    # Train the model using only the optimal features
    LogReg.fit(X_train[optimal_features], y_train)

    # Predict and calculate accuracy for both training and test sets using the same features
    train_predictions = LogReg.predict(X_train[optimal_features])
    test_predictions = LogReg.predict(X_test[optimal_features])

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    #train_accuracy = balanced_accuracy_score(y_train, train_predictions) -> for classification stability evaluation
    #test_accuracy = balanced_accuracy_score(y_test, test_predictions) -> for classification stability evaluation

    return LogReg, train_accuracy, test_accuracy


def train_validate_rf_with_scaling(X, y, trees, depth, cv_splits, n_jobs=-1, verbose=False):
    """
    Trains and validates a Random Forest model with feature scaling, iterating over hyperparameter combinations.
    
    :param X: Feature set.
    :param y: Labels.
    :param trees: List of trees (n_estimators) to try in the Random Forest.
    :param depth: List of depths to try for each tree in the Random Forest.
    :param cv_splits: Predefined cross-validation training/test splits.
    :param n_jobs: Number of jobs to run in parallel for Random Forest training and cross-validation.
    :param verbose: Verbosity flag.
    :return: Dictionary with best model, its parameters, score, feature importances, and validation accuracies for all combinations.
    """
    best_score = 0
    best_params = None
    best_model = None
    feature_importances = None
    validation_accuracies = {}

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for n_estimators in trees:
        for max_depth in depth:
            # Create a pipeline with robust scaler and random forest
            rf_pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('random_forest', RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=n_jobs,
                    random_state=42
                ))
            ])

            scores = []
            for train_idx, test_idx in cv_splits:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                score = cross_val_score(rf_pipeline, X_train, y_train, scoring='accuracy', cv=3, n_jobs=n_jobs)
                scores.append(np.mean(score))

            mean_score = np.mean(scores)

            validation_accuracies[(n_estimators, max_depth)] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                rf_pipeline.fit(X, y)
                best_model = rf_pipeline
                # Feature importances are obtained from the 'random_forest' step of the pipeline
                feature_importances = rf_pipeline.named_steps['random_forest'].feature_importances_

    return {
        'model': best_model,
        'params': best_params,
        'score': best_score,
        'feature_importances': feature_importances,
        'validation_accuracies': validation_accuracies
    }


def evaluate_rf_model(X_train, y_train, X_test, y_test, selected_features, n_estimators, max_depth):
    """
    Train and evaluate a Random Forest model.

    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Testing feature set.
        y_test (pd.Series): Testing target vector.
        selected_features (list): List of selected feature names.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.

    Returns:
        dict: Dictionary containing the RF model, its parameters, training accuracy, testing accuracy, and predictions.
    """
    # Subset the data with the selected features
    X_train_rf = X_train[selected_features]
    X_test_rf = X_test[selected_features]

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
    rf_model.fit(X_train_rf, y_train)

    # Predict and calculate accuracies
    train_predictions = rf_model.predict(X_train_rf)
    test_predictions = rf_model.predict(X_test_rf)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    #train_accuracy = balanced_accuracy_score(y_train, train_predictions) -> for classification stability evaluation
    #test_accuracy = balanced_accuracy_score(y_test, test_predictions) -> for classification stability evaluation

    return {
        'model': rf_model,
        'params': {'n_estimators': n_estimators, 'max_depth': max_depth},
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }


#### EVALUTATE TEMPORAL CLASSIFICATION STABILITY ####
def create_temporal_outer_cv_split(data, n_train, n_repetitions=1):
    """
    Generate a single temporal outer cross-validation split.
    
    :param data: DataFrame containing the features.
    :param n_train: Number of trials to include in the training set.
    :param n_repetitions: Number of repetitions for the CV (default is 1).
    :return: Lists containing train-test indices and test indices for each repetition.
    """
    n_total = data.shape[0]

    # Prepare the split indices
    train_index = list(range(n_train))
    test_index = list(range(n_train, n_total))

    # Prepare the output variables
    Rep_Outer_CV = [[[train_index, test_index]] for _ in range(n_repetitions)]
    Rep_Outer_CV_test = [[test_index] for _ in range(n_repetitions)]

    return Rep_Outer_CV, Rep_Outer_CV_test

def create_balanced_temporal_cv_split(data, n_train_per_condition, n_test_per_condition):
    """
    Generate a single balanced temporal cross-validation split.
    
    :param data: DataFrame containing the features and 'Condition' column.
    :param n_train_per_condition: Number of trials per condition (1 and 0) to include in the training set.
    :param n_test_per_condition: Number of trials per condition (1 and 0) to include in the testing set.
    :return: Lists containing train-test indices and test indices for the split.
    """
    # Separating the data based on 'Condition' and getting the row numbers
    data_condition_1 = data[data['Condition'] == 1].index.tolist()
    data_condition_0 = data[data['Condition'] == 0].index.tolist()

    # Select the first n_train_per_condition indices for each condition for training
    train_index_condition_1 = data_condition_1[:n_train_per_condition]
    train_index_condition_0 = data_condition_0[:n_train_per_condition]
    train_index = train_index_condition_1 + train_index_condition_0

    # Select the next n_test_per_condition indices for each condition for testing
    test_index_condition_1 = data_condition_1[n_train_per_condition:n_train_per_condition + n_test_per_condition]
    test_index_condition_0 = data_condition_0[n_train_per_condition:n_train_per_condition + n_test_per_condition]
    test_index = test_index_condition_1 + test_index_condition_0

    # Prepare the output variables
    Rep_Outer_CV = [[[train_index, test_index]]]  # Nested list for compatibility
    Rep_Outer_CV_test = [[test_index]]  # Nested list for compatibility

    return Rep_Outer_CV, Rep_Outer_CV_test

def prepare_and_standardize_data_classification_stability(train_df, test_df, feature_cols, target_col):
    """
    Scales features and splits data for testing temporal classification stability.
    
    :param train_df: Training dataset with features and target.
    :param test_df: Testing dataset with features and target.
    :param feature_cols: List of feature column names.
    :param target_col: Name of the target column.
    :return: Scaled feature dataframes for training and testing, target series for training and testing, and the scaler.
    """
    # Splitting features
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Extracting target as pandas Series
    y_train = train_df[target_col].squeeze()
    y_test = test_df[target_col].squeeze()

    # Robust scaling of features
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


#### OPTIONAL PLOTTING FUNCTIONS ####
def plot_learning_curve(estimator, title, X, y, X_test, y_test, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(20, 5))  # Changed to 1, 4 to include a fourth plot for test data

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # New code to include test data performance
    test_data_scores = []
    for train_subset_size in train_sizes:
        subset_indices = np.random.choice(range(X.shape[0]), train_subset_size)
        X_train_subset = X.iloc[subset_indices]
        y_train_subset = y.iloc[subset_indices]
        estimator.fit(X_train_subset, y_train_subset)
        test_data_score = estimator.score(X_test, y_test)
        test_data_scores.append(test_data_score)
    
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, np.mean(fit_times, axis=1), 'o-')
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(np.mean(fit_times, axis=1), test_scores_mean, 'o-')
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    # Plot learning curve for test data
    axes[3].grid()
    axes[3].plot(train_sizes, test_data_scores, 'o-', color="b", label="Test score")
    axes[3].set_xlabel("Training examples")
    axes[3].set_ylabel("Score")
    axes[3].set_title("Test Data Learning Curve")
    axes[3].legend(loc="best")

    return plt

def plot_feature_distributions_by_condition(X_train, y_train, X_test, y_test, selected_feature_names):
    num_features = len(selected_feature_names)
    fig, axes = plt.subplots(num_features, 2, figsize=(15, 5 * num_features))

    # Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    for i, feature in enumerate(selected_feature_names):
        # Plot for low condition
        sns.histplot(X_train[feature][y_train == 0], ax=axes[i, 0], kde=True, color='blue', label='Train Low Condition')
        sns.histplot(X_test[feature][y_test == 0], ax=axes[i, 0], kde=True, color='red', alpha=0.6, label='Test Low Condition')
        axes[i, 0].set_title(f"Low Condition - {feature}")
        axes[i, 0].legend()

        # Plot for high condition
        sns.histplot(X_train[feature][y_train == 1], ax=axes[i, 1], kde=True, color='green', label='Train High Condition')
        sns.histplot(X_test[feature][y_test == 1], ax=axes[i, 1], kde=True, color='orange', alpha=0.6, label='Test High Condition')
        axes[i, 1].set_title(f"High Condition - {feature}")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()


# Extract feature names without condition tags
def remove_condition_tags(feature_names):
    return [feature.split(" (")[0] for feature in feature_names]


def compare_distributions_conditionally(X_train, y_train, X_test, y_test, feature_name):
    # Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Split based on condition
    X_train_high = X_train[y_train == 1][feature_name]
    X_train_low = X_train[y_train == 0][feature_name]
    X_test_high = X_test[y_test == 1][feature_name]
    X_test_low = X_test[y_test == 0][feature_name]

    # Visual comparison
    plt.figure(figsize=(12, 6))
    sns.kdeplot(X_train_high, color="blue", label='Train High')
    sns.kdeplot(X_train_low, color="green", label='Train Low')
    sns.kdeplot(X_test_high, color="red", label='Test High')
    sns.kdeplot(X_test_low, color="orange", label='Test Low')
    plt.title(f'Feature: {feature_name} Distribution by Condition')
    plt.legend()
    plt.show()

    # Statistical comparison
    stat_high, p_high = ks_2samp(X_train_high, X_test_high)
    stat_low, p_low = ks_2samp(X_train_low, X_test_low)
    print(f"Feature: {feature_name} - High Condition, KS Statistic: {stat_high:.4f}, P-Value: {p_high:.4f}")
    print(f"Feature: {feature_name} - Low Condition, KS Statistic: {stat_low:.4f}, P-Value: {p_low:.4f}")


