
"""
Machine Learning Analysis Pipeline
==================================
This pipeline is designed to perform feature selection and classification using SVM, Logistic Regression and Random Forest.

"""
# Load all relevant libraries
from src.Dependencies import * 
from src.mRMR_feature_select import mRMR_feature_select
from src.helper_functions import * 

# Load the configuration file
with open("config_MEP_Decoding.json", "r") as file:
    config = json.load(file)

# Access paths
input_directory: str = config["paths"]["input_directory"]
Feature_savepath: str = config["paths"]["Feature_savepath"]
Model_savepath: str = config["paths"]["Model_savepath"]

# Access feature names
EEG_features_name_list: List[str] = config["feature_names"]

# Access classifiers
CLF_models: List[str] = config["classifiers"]
n_models: int = len(CLF_models)

# Access cross-validation settings
n_repetitions: int = config["cross_validation"]["n_repetitions"]
k_out: int = config["cross_validation"]["k_out"] # outer CV
k_in: int = config["cross_validation"]["k_in"] # inner CV
random_state_multiplier: int = config["cross_validation"]["random_state_multiplier"]

# Access feature selection settings
n_feat_mRMR1: List[int] = config["feature_selection"]["n_feat_mRMR1"]
max_mRMR_features: int = config["feature_selection"]["max_mRMR_features"]
min_features_to_select: int = config["feature_selection"]["min_features_to_select"]

# Access SVM settings
SVM_exponent_range: List[float] = config["SVM"]["exponent_range"]
SVM_C_base: int = config["SVM"]["C_base"]
SVM_kernels: List[str] = config["SVM"]["kernels"]

# Access Logistic Regression settings
LogReg_exponent_range: List[float] = config["LogReg"]["exponent_range"]
LogReg_C_base: int = config["LogReg"]["C_base"]

# Access Random Forest settings
RF_trees = [int(x) for x in config["RF"]["trees"]["value"]]
RF_depth = [int(x) for x in config["RF"]["depth"]["value"]]

# Set seed for reprocucibility
random.seed(789)

# Load data
# Initialize subjects list
subjects: List[int] = list(range(1,30))  
#subjects = [1]

# Loop through subjects
for subnum in subjects:
    print(f"Starting Subject {subnum}")
    
    # Access loop random seed
    np.random.seed(config["misc"]["loop_random_seed"])

    # Construct the subject-specific savepath
    subject_savepath = Feature_savepath.format(subnum=subnum)

    # Load the features dataframe for the current subject
    #EEG_features_df: pd.DataFrame = load_ml_features_df(subject_savepath, subnum)
    EEG_features_df: pd.DataFrame = load_ml_sensor_features_df(subject_savepath, subnum) # for sensor-level features

    EEG_features_df_original = EEG_features_df.copy()

    # Shuffling the rows of the DataFrame
    EEG_features_df = EEG_features_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define the target and stratification variables
    Target: str = "Condition"  # The label we are trying to predict

    # Locate the column index for the target variable
    Target_col_idx: int = np.where(EEG_features_df.columns == Target)[0][0]

    # Stratification based on the 'Condition' column, which is also the target variable
    stratification_col: pd.Series = EEG_features_df[Target]

    # List of non-feature, non-target columns
    non_feature_cols: List[str] = ["Trial_index"]

    # Exclude the target and any non-feature columns from standardization
    feature_cols = [col for col in EEG_features_df.columns if col != Target and col not in non_feature_cols]

    # Set a threshold for outlier detection based on the IQR method
    threshold = 1.5  # Adjust the threshold as needed

    # Initialize a dictionary to store the percentage of trials replaced per feature
    percent_replaced_dict = {feature: {0: 0, 1: 0} for feature in feature_cols}

    # Calculate IQR for all features at once for the entire dataset
    Q1 = EEG_features_df[feature_cols].quantile(0.25)
    Q3 = EEG_features_df[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Identify outliers
    outliers_lower = EEG_features_df[feature_cols] < lower_bound
    outliers_upper = EEG_features_df[feature_cols] > upper_bound
    
    # Convert the Series lower_bound and upper_bound to DataFrames for compatibility
    lower_bound_df = pd.DataFrame([lower_bound] * len(EEG_features_df), index=EEG_features_df.index)
    upper_bound_df = pd.DataFrame([upper_bound] * len(EEG_features_df), index=EEG_features_df.index)
    
    # Replace outliers with the appropriate bounds
    for feature in feature_cols:
        EEG_features_df.loc[outliers_lower[feature], feature] = lower_bound[feature]
        EEG_features_df.loc[outliers_upper[feature], feature] = upper_bound[feature]
    
    # Calculate the percentage of outliers replaced for each feature
    total_outliers = outliers_lower.sum() + outliers_upper.sum()
    percent_replaced = (total_outliers / len(EEG_features_df)) * 100
    
    # Update the percentage replaced dictionary
    for feature in feature_cols:
        percent_replaced_dict[feature] = percent_replaced[feature]

    # Initialize arrays to store results
    accuracy_arr: np.ndarray = np.zeros((n_repetitions, k_out, n_models, 3))
    model_par: List[Any] = []
    final_models: List[Any] = []
    final_features: List[Any] = []
    final_y_preds: List[Any] = []

    # Prepare Stratified K-Folds for all repetitions
    Outer_CV = []
    Outer_CV_test = []

    # Prepare the StratifiedKFold splits beforehand
    Outer_CV, Outer_CV_test = create_outer_cv_splits(EEG_features_df, EEG_features_df[Target], 
                                                         n_repetitions, k_out, 
                                                         config["misc"]["random_state_multiplier"]) 

    # Check that none of the repetitions are the same
    Outer_CV_test_flat: List[Any] = [item for sublist in Outer_CV_test for item in sublist]
    Outer_CV_test_flat_str: List[str] = ["".join(map(str, ele)) for ele in Outer_CV_test_flat]

    def all_unique(x: List[str]) -> bool:
        return len(set(x)) == len(x)

    assert all_unique(Outer_CV_test_flat_str)

    # Capture the start time
    start_time = datetime.now()

    for rep in range(n_repetitions):
        # The outer fold CV has already been saved as lists
        print("---- Starting with Outer Fold")
        Outer_CV: List[List[int]] = Outer_CV[rep]
        
        # Pre-allocate memory
        model_par0: List[Any] = []
        final_models0: List[Any] = []
        final_features0: List[Any] = []
        final_y_preds0: List[Any] = []
        Outer_counter: int = 0

        # Loop through each outer fold
        for train_index, test_index in Outer_CV:

            # Part 1 - Data Preparation and Standardization

            # Call the data preparation and standardization function
            X_train, X_test, y_train, y_test,scaler = prepare_and_standardize_data(
                train_index, test_index, EEG_features_df, feature_cols, non_feature_cols, 'Condition'
            )
     
            # Check alignment and reset indices
            X_train, y_train, X_test, y_test = check_alignment_and_reset_indices(X_train, y_train, X_test, y_test)

            #  Call the mRMR feature selection function
            initial_selected_features = perform_mRMR_feature_selection(X_train, y_train, EEG_features_name_list, max_mRMR_features)

            # Filter the training set to include only the features selected in the initial step
            X_train_filtered = X_train[initial_selected_features]

            X_train_filtered, y_train = check_alignment_and_reset_indices_train2(X_train_filtered, y_train)

            # Print the number of features selected after filtering 
            print(f"{len(initial_selected_features)} features are left after first round of mRMR filtering")
            print("Shape of X_train_filtered:", X_train_filtered.shape)

            # Part 2 - Feature Selection and Model Training

            # Initialization
            C_parameter_SVM, C_parameter_LogReg, trees, depth, val_acc_svm, val_feat_svm, val_feat_import_svm, val_acc_logreg, val_feat_logreg, val_feat_import_logreg, val_acc_rf, val_feat_rf, val_feat_import_rf, kernels = initialize_hyperparameters_and_accuracy_arrays(SVM_exponent_range, SVM_C_base, LogReg_exponent_range, LogReg_C_base, RF_trees, RF_depth, n_feat_mRMR1, max_mRMR_features, SVM_kernels)

            # Initialize tracking variables for SVM
            best_val_accuracy_svm = 0
            best_hyperparameters_svm = {'C': None, 'kernel': None}
            best_features_svm = None
            best_svm_feature_importances = None

            # Initialize tracking variables for Logistic Regression
            best_val_accuracy_logreg = 0
            best_hyperparameters_logreg = {'C': None}
            best_features_logreg = None
            best_logreg_feature_importances = None

            # Initialize tracking variables for Random Forest
            best_val_accuracy_rf = 0
            best_hyperparameters_rf = {'trees': None, 'depth': None}
            best_features_rf = None
            best_rf_feature_importances = None

            # Loop through the number of features as determined by mRMR
            for num_features_index in range(len(n_feat_mRMR1)):
                # mRMR and dataset filtering
                X_train_mRMR_subset, selected_mRMR_features = rerun_mRMR_and_filter_dataset(
                    X_train_filtered, y_train, initial_selected_features, 
                    n_feat_mRMR1[num_features_index], 
                    {'K_MAX': 1000, 'n_jobs': -1, 'verbose': False}
                )
                print(f'Finished mRMR feature selection for set {num_features_index + 1}')

                # Inner Fold Split for Cross-Validation
                inner_cv_splits = create_inner_folds(X_train_mRMR_subset, y_train, k_in, rep, config["misc"]["random_state_multiplier"])

                # SVM Hyperparameter Optimization
                optimal_C, optimal_kernel, selected_feature_names, feature_importances, val_acc = find_optimal_svm_parameters_scaled(
                    X_train_mRMR_subset, y_train, C_parameter_SVM, kernels, inner_cv_splits, min_features_to_select
                )

                # Check if the current validation accuracy is the best
                if val_acc > best_val_accuracy_svm: #change to > for accuracy
                    best_val_accuracy_svm = val_acc
                    best_hyperparameters_svm = {'C': optimal_C, 'kernel': optimal_kernel}
                    best_features_svm = selected_feature_names
                    best_svm_feature_importances = feature_importances

                    # Find indices of C and kernel in the original arrays
                    C_index = np.where(C_parameter_SVM == optimal_C)[0][0]
                    kernel_index = kernels.index(optimal_kernel)

                    # Update accuracy array for best model
                    val_acc_svm[num_features_index, C_index, kernel_index] = val_acc

                    # Update feature indices array for best model
                    original_feature_indices = np.where(X_train.columns.isin(selected_feature_names))[0]
                    val_feat_svm[num_features_index, C_index, kernel_index, 0:len(original_feature_indices)] = original_feature_indices

                    # Update feature importances array for best model (if linear kernel)
                    if optimal_kernel == 'linear':
                        val_feat_import_svm[num_features_index, C_index, kernel_index, :len(feature_importances)] = feature_importances

                # Logistic Regression
                # Define the range of features to select from
                k_features = np.arange(1, n_feat_mRMR1[num_features_index] + 1, 1)
                k_features_tuple = (k_features[0], k_features[-1])

                # Perform logistic regression with SFS
                inner_cv_scores, optimal_features, logreg_coef_normalized, best_C, best_K, best_val_acc_logreg = logistic_regression_sfs_scaled(
                    X_train_mRMR_subset, y_train, C_parameter_LogReg, k_features_tuple, inner_cv_splits
                )
                # Check if the current validation accuracy is the best for LogReg
                if best_val_acc_logreg > best_val_accuracy_logreg: 
                    best_val_accuracy_logreg = best_val_acc_logreg
                    best_hyperparameters_logreg = {'C': best_C}
                    best_features_logreg = optimal_features
                    best_logreg_feature_importances = logreg_coef_normalized

                    # Update the initialized arrays
                    best_C_index = np.where(C_parameter_LogReg == best_C)[0][0]

                    # Update the validation accuracy array with the current model's accuracy
                    val_acc_logreg[num_features_index, best_C_index] = best_val_acc_logreg  # Use current model's accuracy

                    # Update the feature indices array with the indices of the optimal features
                    optimal_feature_indices = np.where(X_train.columns.isin(optimal_features))[0]
                    val_feat_logreg[num_features_index, best_C_index, :len(optimal_feature_indices)] = optimal_feature_indices

                    # Update the feature importances array with the normalized coefficients
                    val_feat_import_logreg[num_features_index, best_C_index, :len(logreg_coef_normalized)] = logreg_coef_normalized

                # Random Forest
                # Random Forest Training and Validation
                rf_results = train_validate_rf_with_scaling(
                    X_train_mRMR_subset, y_train, trees=trees, depth=depth, cv_splits=inner_cv_splits
                )

                # Extract the results
                rf_model = rf_results['model']
                best_rf_score = rf_results['score']
                best_rf_params = rf_results['params']
                rf_feature_importances = rf_results['feature_importances']
                t_chosen = best_rf_params['n_estimators']
                d_chosen = best_rf_params['max_depth']

                # Check if the current validation accuracy (score) is the best for RF
                if best_rf_score > best_val_accuracy_rf:
                    best_val_accuracy_rf = best_rf_score
                    best_rf_model = rf_model
                    best_rf_params = {'n_estimators': t_chosen, 'max_depth': d_chosen}
                    best_rf_feature_importances = rf_feature_importances
                    best_features_rf = selected_mRMR_features

                    # Update RF arrays
                    num_trees_index = np.where(trees == t_chosen)[0][0]
                    num_depth_index = np.where(depth == d_chosen)[0][0]
                    val_acc_rf[num_features_index, num_trees_index, num_depth_index] = best_rf_score
                    val_feat_import_rf[num_features_index, num_trees_index, num_depth_index, :len(rf_feature_importances)] = rf_feature_importances

            # Train and evaluate the final SVM model using the best configuration
            if best_hyperparameters_svm['C'] is not None and best_hyperparameters_svm['kernel'] is not None:
                svm_model, train_accuracy_svm, test_accuracy_svm = train_evaluate_final_svm(
                    X_train_mRMR_subset[best_features_svm], y_train, X_test[best_features_svm], y_test, best_hyperparameters_svm['C'], best_hyperparameters_svm['kernel']
                )
                print(f"Optimized SVM Model - Training Accuracy: {train_accuracy_svm:.2f}, Test Accuracy: {test_accuracy_svm:.2f}")
                print(f"Optimized Features: {len(best_features_svm)}")
            else:
                print("No optimal SVM model configuration found.")

            # Train and evaluate the final Logistic Regression model using the best configuration
            if best_hyperparameters_logreg['C'] is not None:
                # Train and evaluate the logistic regression model using the optimal features
                LogReg_model, train_accuracy_logreg, test_accuracy_logreg = train_evaluate_logreg(
                    X_train, y_train, X_test, y_test, best_C, best_features_logreg
                )
                # Print the accuracies
                print(f"Logistic Regression Training Accuracy: {train_accuracy_logreg:.2f}, Test Accuracy: {test_accuracy_logreg:.2f}")
                print(f"SFS+LogReg: Number of Features Chosen: {len(best_features_logreg)}")  
            else:
                print("No optimal Logistic Regression model configuration found.")

             # Train and evaluate the final Random Forest model using the best configuration
            if best_rf_params['n_estimators'] is not None and best_rf_params['max_depth'] is not None:
                rf_eval_results = evaluate_rf_model(
                    X_train_mRMR_subset, y_train, X_test, y_test, best_features_rf,
                    best_rf_params['n_estimators'], best_rf_params['max_depth']
                )

                train_acc_rf = rf_eval_results['train_accuracy']
                test_acc_rf = rf_eval_results['test_accuracy']
                print(f"Optimized Random Forest Model - Training Accuracy: {train_acc_rf:.2f}, Test Accuracy: {test_acc_rf:.2f}")
            else:
                print("No optimal Random Forest model configuration found.")
   
    
            # SVM: Save the results for further analysis
            SVM_model_par = [optimal_C, optimal_kernel, len(best_features_svm)]
            SVM_y_pred = [svm_model.predict(X_train_mRMR_subset[best_features_svm]), svm_model.predict(X_test[best_features_svm])]
            svm_feat_importances = best_svm_feature_importances
            final_features0.append(['SVM', best_features_svm, svm_feat_importances])
            accuracy_arr[rep, Outer_counter, 0, :] = [train_accuracy_svm, best_val_accuracy_svm, test_accuracy_svm]   

            # LogReg: Save the results for further analysis
            LogReg_model_par = [best_C, len(best_features_logreg)]
            LogReg_y_pred = [LogReg_model.predict(X_train_mRMR_subset[best_features_logreg]), LogReg_model.predict(X_test[best_features_logreg])]
            logreg_feat_importances = best_logreg_feature_importances
            final_features0.append(['LogReg', best_features_logreg, logreg_feat_importances])
            accuracy_arr[rep, Outer_counter, 1, :] = [train_accuracy_logreg, best_val_acc_logreg, test_accuracy_logreg]

            # RF: Store the results for further analysis 
            RF_model_par = [t_chosen, d_chosen, len(best_features_rf)]
            RF_y_pred = [rf_eval_results['train_predictions'], rf_eval_results['test_predictions']]
            accuracy_arr[rep, Outer_counter, 2, :] = [train_acc_rf, best_rf_score, test_acc_rf]   

            # Save all models and parameters
            model_par0.append([SVM_model_par, LogReg_model_par, RF_model_par])
            final_models0.append([svm_model, LogReg_model, rf_model])
            final_y_preds0.append([SVM_y_pred, LogReg_y_pred, RF_y_pred])
            final_features0.append(['RF', best_features_rf, best_rf_feature_importances])

            # Move counter
            Outer_counter += 1
            print("Finished outer fold {} out of {} for rep: {}".format(Outer_counter,k_out,rep+1))

        # Append results to lists
        model_par.append(model_par0)
        final_models.append(final_models0)
        final_features.append(final_features0)
        final_y_preds.append(final_y_preds0)

        # Aggregate all results into a single list
        Rep_mRMR2_SVM_LogReg_RF = [accuracy_arr, model_par, final_models, final_features, final_y_preds]

        # Save the results
        save_filename = f"Rep1_5x10CV_mRMR_SVM_LogReg_RF_Sub_{subnum}_results_10122023.pkl" 
        save_path = os.path.join(Model_savepath, save_filename)
        
        with open(save_path, "wb") as filehandle:
            pickle.dump(Rep_mRMR2_SVM_LogReg_RF, filehandle)
        
        # Capture the end time
        end_time = datetime.now()

        # Calculate the time difference
        time_difference = end_time - start_time
        print(f"Started: {start_time}\nEnded: {end_time}\nElapsed time: {time_difference}")
        
        # Print total progress
        print(f"Finished outer fold repetition {rep+1} out of {n_repetitions} for Subject {subnum}")  

