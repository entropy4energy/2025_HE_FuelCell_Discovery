"""
Purpose: Simplified random forest training script that saves only the best CV fold results.
"""

# ---------------- USER CONFIGURATIONS ----------------
# Model parameters and paths for easy customization
info_dict = {
    'metric_name': "wasserstein_dist_vs_pt",
    'target_key': "wasserstein_dist_vs_pt",  # Target key
    'xlabel': r"DFT Calculated cosine_similarity_vs_pt$",
    'ylabel': r"ML Predicted cosine_similarity_vs_pt$",
    'json_path': "Data/lib5_FCC_CS_training.json",  # Data path
    'base_output_dir': "Output/Training/",
    'remove_nan': True,  # Remove data points where target value is nan
    'param_grid': {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [10],
        'min_samples_leaf': [4]
    },
    'fixed_rf_params': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    },
    'cv_num': 5,
    'random_state': 42,
    'scoring_metrix': 'neg_root_mean_squared_error',
    'prop_list':  [
        "atomic_mass",
        "atomic_number",
        "chemical_scale_Pettifor",
        "conductivity_thermal",
        "density",
        "density_PT",
        "electron_affinity_PT",
        "electronegativity_Allen",
        "electronegativity_Ghosh",
        "electronegativity_Pauling",
        "enthalpy_atomization_WE",
        "enthalpy_fusion",
        "enthalpy_vaporization",
        "group",
        "hardness_chemical_Ghosh",
        "period",
        "polarizability",
        "radii_Ghosh08",
        "radii_Pyykko",
        "radii_Slatter",
        "radius_PT",
        "radius_covalent",
        "radius_covalent_PT",
        "specific_heat_PT",
        "temperature_boiling",
        "temperature_melting",
        "valence_PT",
        "valence_d",
        "valence_iupac",
        "valence_p",
        "valence_s",
        "valence_std",
        "variance_parameter_mass",
        "volume",
        "volume_molar",
        "work_function_Miedema"
    ],
    'n_jobs': -1,
    'verbose': 1,
    'normalize': 'quantile', #'Z-score', 'min-max', 'min-max-0.1-0.9', 'log-zscore', 'quantile', or 'none'
}

# Extract configuration variables
metric_name = info_dict['metric_name']
target_key = info_dict['target_key']
xlabel = info_dict['xlabel']
ylabel = info_dict['ylabel']
json_path = info_dict['json_path']
param_grid = info_dict['param_grid']
cv_num = info_dict['cv_num']
scoring_metrix = info_dict['scoring_metrix']
prop_list = info_dict['prop_list']
normalize = info_dict['normalize']
base_output_dir = info_dict['base_output_dir']
random_state = info_dict['random_state']

import sys
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pickle

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, json_loader

def save_results_to_csv(file_path, true_values, predicted_values, element_lists):
    """Save prediction results to CSV file"""
    max_len = max(len(e) for e in element_lists)
    data = {
        "True Values": true_values,
        "Predicted Values": predicted_values
    }
    for i in range(max_len):
        data[f"Element {i+1}"] = [e[i] if len(e) > i else None for e in element_lists]
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def save_model_and_params(model, params, output_dir, final_mae=None):
    """
    Save the trained model and parameters (no 'best' in filenames)
    Args:
        model: Trained RandomForestRegressor
        params: Model parameters (dict)
        output_dir: Directory to save files
        final_mae: Final MAE score (optional, for logging)
    """
    # Save the model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    # Save the parameters in JSON format
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    # Save the parameters in TXT format
    with open(os.path.join(output_dir, 'params.txt'), 'w') as f:
        f.write("Model Parameters:\n")
        f.write("=================\n\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        # Add model performance metrics if provided
        if final_mae is not None:
            f.write("\nModel Performance:\n")
            f.write("=================\n")
            f.write(f"Final MAE: {final_mae:.4f}\n")

def main():
    # 1. Load data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Collect unique elements from the dataset
    unique_elements = []
    for item in data:
        unique_elements.extend(item['Elements'])
    unique_elements = list(set(unique_elements))
    info_dict['unique_elements'] = unique_elements
    
    # 3. Create element property dataframe (feature table for all elements)
    ele_DF = ele_df(prop_list=prop_list, 
                    total_e_list=unique_elements,
                    normalize=normalize)
    
    # 4. Extract features and targets from the dataset using json_loader
    data_loader = json_loader(json_path, df_data=ele_DF)
    X, y, elements = data_loader.load_data_nocomp(
        target_key=target_key, split=False
    )
    
    # 5. Convert X and y to numpy arrays for sklearn compatibility
    X = np.array(X)
    y = np.array(y)
    
    # 6. Data cleaning: replace 0 with np.nan, remove samples with nan target
    if info_dict['remove_nan']:
        y = np.where(y == 0, np.nan, y)
        non_nan_mask = ~np.isnan(y)
        original_count = len(y)
        X = X[non_nan_mask]
        y = y[non_nan_mask]
        elements = [elements[i] for i in range(len(elements)) if non_nan_mask[i]]
        removed_count = original_count - len(y)
        print(f"Removed {removed_count} data points with nan values")
        print(f"Remaining data points: {len(y)}")
    
    # 7. Create output directory for results
    output_dir = os.path.join(base_output_dir, datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 8. Prepare RandomForest parameters
    rf_params = info_dict['fixed_rf_params']
    rf_params['random_state'] = info_dict['random_state']

    # 9. Cross-validation: train and evaluate model on each fold
    print("Starting cross-validation with fixed parameters...")
    cv_results = {
        'mae': [],
        'train_indices': [],
        'val_indices': [],
        'models': []
    }
    
    kf = KFold(n_splits=info_dict['cv_num'], shuffle=True, random_state=info_dict['random_state'])
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Processing fold {fold_idx + 1}/{info_dict['cv_num']}")
        # Store indices
        cv_results['train_indices'].append(train_idx)
        cv_results['val_indices'].append(val_idx)
        # Train on this fold with fixed parameters
        fold_model = RandomForestRegressor(**rf_params)
        fold_model.fit(X[train_idx], y[train_idx])
        # Get predictions for validation set
        fold_predictions = fold_model.predict(X[val_idx])
        fold_true = y[val_idx]
        # Calculate MAE
        fold_mae = mean_absolute_error(fold_true, fold_predictions)
        cv_results['mae'].append(fold_mae)
        cv_results['models'].append(fold_model)
        print(f"Fold {fold_idx + 1} - MAE: {fold_mae:.3f}")
    
    # 10. Select the best fold (lowest MAE)
    best_fold_idx = np.argmin(cv_results['mae'])
    train_idx = cv_results['train_indices'][best_fold_idx]
    val_idx = cv_results['val_indices'][best_fold_idx]
    best_model = cv_results['models'][best_fold_idx]
    print(f"Best fold (#{best_fold_idx + 1}) MAE: {cv_results['mae'][best_fold_idx]:.3f}")
    
    # 11. Save train/test predictions for the best fold
    train_predictions = best_model.predict(X[train_idx])
    train_true = y[train_idx]
    test_predictions = best_model.predict(X[val_idx])
    test_true = y[val_idx]
    save_results_to_csv(os.path.join(output_dir, "train_result.csv"),
                       train_true, train_predictions,
                       [elements[i] for i in train_idx])
    save_results_to_csv(os.path.join(output_dir, "test_result.csv"),
                       test_true, test_predictions,
                       [elements[i] for i in val_idx])
    
    # 12. Train final model on all data and save
    final_model = RandomForestRegressor(**rf_params)
    final_model.fit(X, y)
    final_predictions = final_model.predict(X)
    final_mae = mean_absolute_error(y, final_predictions)
    save_model_and_params(final_model, rf_params, output_dir,
                         final_mae=final_mae)
    print(f"Final model MAE: {final_mae:.3f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()