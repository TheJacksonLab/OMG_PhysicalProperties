import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, max_error
from scipy.stats import spearmanr


############################## TARGET COLUMNS ##############################
target_cols_list = [
    ['asphericity_Boltzmann_average',
    'eccentricity_Boltzmann_average',
    'inertial_shape_factor_Boltzmann_average',
    'radius_of_gyration_Boltzmann_average',
    'spherocity_Boltzmann_average'],

    ['HOMO_minus_1_Boltzmann_average',
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


############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################

if __name__ == '__main__':
    """
    This script saves prediction rmse errors of active learning cycles with fold. Please make sure you obtained prediction results until the fold "5".
    The saved arr has a shape of (num_batch, num_targets, num_fold) containing "normalized RMSE arr" with train std.
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred & plot_fold_pred_error.py
    4) uncertainty_sampling.py
    -> repeat to 1)
    """
    ################### MODIFY ###################
    gnn_idx = 3
    target_columns_list = target_cols_list[gnn_idx]
    dir_name_gnn_0_list = ['240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence', '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence', '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence', '240531-100412913979_OMG_train_batch_3_chemprop_train_gnn_0_evidence']
    dir_name_gnn_1_list = ['240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence', '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence', '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence', '240531-100828584294_OMG_train_batch_3_chemprop_train_gnn_1_evidence']
    dir_name_gnn_2_list = ['240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence', '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence', '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence', '240531-100906010104_OMG_train_batch_3_chemprop_train_gnn_2_evidence']
    dir_name_gnn_3_list = ['240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence', '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence', '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence', '240531-101615032820_OMG_train_batch_3_chemprop_train_gnn_3_evidence']
    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 3  # 0 for initial training data
    fold_num = 5  # also works for 1
    ################### MODIFY ###################

    # metric
    metric = 'rmse'
    dir_name_list = [dir_name_gnn_0_list, dir_name_gnn_1_list, dir_name_gnn_2_list, dir_name_gnn_3_list]
    ################### MODIFY ###################
    arr_save_dir = Path(os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/figure/arr'))
    arr_save_dir.mkdir(parents=True, exist_ok=True)
    std_normalization_list = list()  # for normalization
    normalized_rmse_list = list()
    for batch_idx in reversed(range(current_batch + 1)):  # get .csv files from predictions until now. Iterate first from current batch to get mean / std for normalization.
        unnormalized_rmse_score_list = list()  # to save. unnormalized rmse score
        for target_columns in target_columns_list:
            train_pred_list = list()
            test_pred_list = list()
            train_target_list = list()
            test_target_list = list()
            for fold_idx in range(fold_num):
                csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{batch_idx}_train/gnn_{gnn_idx}/{dir_name_list[gnn_idx][batch_idx]}/fold_{fold_idx}')
                train_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'train_pred.csv'))
                test_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'test_pred.csv'))

                # train
                train_target_true = train_pred_results[f'true_{target_columns}'].to_numpy()
                train_target_pred = train_pred_results[f'{target_columns}'].to_numpy()

                # test
                test_target_true = test_pred_results[f'true_{target_columns}'].to_numpy()
                test_target_pred = test_pred_results[f'{target_columns}'].to_numpy()

                # append
                train_pred_list.append(train_target_pred)
                test_pred_list.append(test_target_pred)
                train_target_list.append(train_target_true)
                test_target_list.append(test_target_true)

            # arr
            train_pred_arr = np.array(train_pred_list)
            test_pred_arr = np.array(test_pred_list)
            train_target_arr = np.array(train_target_list)
            test_target_arr = np.array(test_target_list)

            # for normalization. train_target_arr -> all have the same target arr (e.g.,  train_target_arr[0] = train_target_arr[1])
            if batch_idx == current_batch:  # the batch that has the largest train dataset
                std_normalization_list.append(train_target_arr[0].std(ddof=0))

            # get test RMSE
            sub_rmse_score_list = list()  # per target
            for fold_idx in range(fold_num):
                rmse_score = mean_squared_error(y_true=test_target_arr[fold_idx], y_pred=test_pred_arr[fold_idx], squared=False)
                sub_rmse_score_list.append(rmse_score)
            unnormalized_rmse_score_list.append(sub_rmse_score_list)

        # get arr for std arr to normalized RMSE
        std_normalization_arr = np.array(std_normalization_list)  # (number_of_targets,)
        std_normalization_arr = std_normalization_arr.reshape(-1, 1)  # (number_of_targets, 1)
        tiled_std_normalization_arr = np.tile(std_normalization_arr, reps=fold_num)  # (number_of_targets, fold_num)

        # normalize
        rmse_score_arr = np.array(unnormalized_rmse_score_list)  # (number_of_targets, fold_num)
        normalized_rmse_score_arr = rmse_score_arr / tiled_std_normalization_arr  # (number_of_targets, fold_num)

        # append
        normalized_rmse_list.append(normalized_rmse_score_arr.tolist())

    # results
    total_normalized_rmse_arr = np.array(normalized_rmse_list)  # (num_batch, num_targets, num_fold). Note that the axis=0 has a reverse order
    total_normalized_rmse_arr = np.flip(total_normalized_rmse_arr, axis=0)  # flipped

    # save
    np.save(os.path.join(arr_save_dir, 'normalized_rmse_arr.npy'), total_normalized_rmse_arr)
