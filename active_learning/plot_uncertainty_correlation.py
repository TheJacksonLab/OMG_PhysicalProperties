import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pathlib import Path

############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################

############################## TARGET COLUMNS ##############################
target_cols_list = [
    ['asphericity_Boltzmann_average',
    'eccentricity_Boltzmann_average',
    'inertial_shape_factor_Boltzmann_average',
    'radius_of_gyration_Boltzmann_average',
    'spherocity_Boltzmann_average'],

    [
    'HOMO_minus_1_Boltzmann_average',
    'HOMO_Boltzmann_average',
    'LUMO_Boltzmann_average',
    'LUMO_plus_1_Boltzmann_average',
    'dipole_moment_Boltzmann_average',
    'quadrupole_moment_Boltzmann_average',
    'polarizability_Boltzmann_average',],

    ['s1_energy_Boltzmann_average',
    'dominant_transition_energy_Boltzmann_average',
    'dominant_transition_oscillator_strength_Boltzmann_average',
    't1_energy_Boltzmann_average',],

    ['chi_parameter_water_mean',
    'chi_parameter_ethanol_mean',
    'chi_parameter_chloroform_mean',]
]
############################## TARGET COLUMNS ##############################


if __name__ == '__main__':
    """
    This script plots uncertainty correleation with prediction errors.
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred.py
    4) uncertainty_sampling.py
    -> repeat to 1)
    """
    ################### MODIFY ###################
    # file path
    gnn_idx = 3
    target_columns_list = target_cols_list[gnn_idx]
    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 2  # 0 for initial training data
    # dir_name = '240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence'
    # dir_name = '240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence'
    # dir_name = '240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence'
    # dir_name = '240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence'

    # AL 1
    # dir_name = '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence'
    # dir_name = '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence'
    # dir_name = '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence'
    # dir_name = '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence'

    # AL 2
    # dir_name = '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence'
    # dir_name = '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence'
    # dir_name = '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence'
    dir_name = '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence'
    ################### MODIFY ###################

    csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/{dir_name}/fold_0')
    uncertainty_dir = os.path.join(csv_file_dir, 'uncertainty')

    if not os.path.exists(uncertainty_dir):
        os.mkdir(uncertainty_dir)

    # plot
    for target_columns in target_columns_list:
        plt.figure(figsize=(6, 6), dpi=300)
        plt.xlabel(f'Absolute Error', fontsize=14)
        plt.ylabel(f'Predictive Uncertainty', fontsize=14)

        # train .pred results
        train_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'train_pred.csv'))  # str
        train_target_true = train_pred_results[f'true_{target_columns}'].to_numpy()
        train_target_pred = train_pred_results[f'{target_columns}'].to_numpy()
        train_absolute_error = np.abs(train_target_pred - train_target_true)
        train_target_model_uncertainty = train_pred_results[f'std_{target_columns}'].to_numpy()  # Var[mean] (epistemic uncertainty)
        train_rank_correlation_object = spearmanr(a=train_absolute_error, b=train_target_model_uncertainty)
        train_rank_correlation_coefficient = train_rank_correlation_object.statistic
        train_rank_correlation_p_value = train_rank_correlation_object.pvalue  # if larger than 0.05 -> two arrays has no significant correlation.
        plt.plot(train_absolute_error, train_target_model_uncertainty, 'bo', markersize=4.0, label=f'Train $\\rho$={train_rank_correlation_coefficient:.2f}')

        # # valid .pred results
        # val_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'val_pred.csv'))  # str
        # val_target_true = val_pred_results[f'true_{target_columns}'].to_numpy()
        # val_target_pred = val_pred_results[f'{target_columns}'].to_numpy()
        # val_target_model_uncertainty = val_pred_results[f'std_{target_columns}'].to_numpy()  # Var[mean] (epistemic uncertainty)
        # val_absolute_error = np.abs(val_target_pred - val_target_true)
        # val_rank_correlation_object = spearmanr(a=val_absolute_error, b=val_target_model_uncertainty)
        # val_rank_correlation_coefficient = val_rank_correlation_object.statistic
        # val_rank_correlation_p_value = val_rank_correlation_object.pvalue  # if larger than 0.05 -> two arrays has no significant correlation.
        # plt.plot(val_absolute_error, val_target_model_uncertainty, 'co', markersize=4.0, label=f'Valid $\\rho$={val_rank_correlation_coefficient:.2f}')

        # test .pred results
        test_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'test_pred.csv'))  # str
        test_target_true = test_pred_results[f'true_{target_columns}'].to_numpy()
        test_target_pred = test_pred_results[f'{target_columns}'].to_numpy()
        test_target_model_uncertainty = test_pred_results[f'std_{target_columns}'].to_numpy()  # Var[mean] (epistemic uncertainty)
        test_absolute_error = np.abs(test_target_pred - test_target_true)
        test_rank_correlation_object = spearmanr(a=test_absolute_error, b=test_target_model_uncertainty)
        test_rank_correlation_coefficient = test_rank_correlation_object.statistic
        test_rank_correlation_p_value = test_rank_correlation_object.pvalue  # if larger than 0.05 -> two arrays has no significant correlation.
        plt.plot(test_absolute_error, test_target_model_uncertainty, 'ro', markersize=4.0, label=f'Test $\\rho$={test_rank_correlation_coefficient:.2f}')

        # plot
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(uncertainty_dir, f'{target_columns}_correlation_error_uncertainty.png'))
        plt.close()
