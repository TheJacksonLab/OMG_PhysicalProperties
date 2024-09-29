import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, max_error
from scipy.stats import spearmanr
from matplotlib.ticker import FormatStrFormatter


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
    This script draws a plot of error of active sampled points to set active learning stopping criteria based on the training data.

    * run order
    1) save_data_for_stopping_criteria.py
    2) plot_for_stopping_criteria.py

    * To draw a test RMSE please run the following scripts:
    1) save_fold_pred_error.py
    2) plot_fold_pred_error.py
    """
    ################### MODIFY ###################
    current_batch = 2  # If "0", this function draws prediction errors of active sampled points for the AL "1".
    strategy = 'pareto_greedy'
    dir_name_gnn_0_list = ['240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence',
                           '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence',
                           '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence',
                           '240531-100412913979_OMG_train_batch_3_chemprop_train_gnn_0_evidence']
    dir_name_gnn_1_list = ['240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence',
                           '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence',
                           '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence',
                           '240531-100828584294_OMG_train_batch_3_chemprop_train_gnn_1_evidence']
    dir_name_gnn_2_list = ['240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence',
                           '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence',
                           '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence',
                           '240531-100906010104_OMG_train_batch_3_chemprop_train_gnn_2_evidence']
    dir_name_gnn_3_list = ['240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence',
                           '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence',
                           '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence',
                           '240531-101615032820_OMG_train_batch_3_chemprop_train_gnn_3_evidence']
    ################### MODIFY ###################

    # get mean & std to scale - use the largest data to calculate the std, mean
    dir_name_list = [dir_name_gnn_0_list, dir_name_gnn_1_list, dir_name_gnn_2_list, dir_name_gnn_3_list]
    num_gnn = 4
    std_list = list()
    mean_list = list()
    for gnn_idx in range(num_gnn):
        # csv dir
        csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch + 1}_train/gnn_{gnn_idx}/{dir_name_list[gnn_idx][current_batch + 1]}/fold_0')
        csv_file_path = os.path.join(csv_file_dir, 'train_pred.csv')
        df_csv = pd.read_csv(csv_file_path)
        print(df_csv.shape)
        target_cols = target_cols_list[gnn_idx]
        print(target_cols)
        sub_std_list, sub_mean_list = list(), list()
        for col in target_cols:
            # true value
            true_values = df_csv[f'true_{col}'].to_numpy()
            true_arr = np.array(true_values)
            sub_std_list.append(true_arr.std(ddof=0))
            sub_mean_list.append(true_arr.mean())
        # append
        std_list.append(sub_std_list)
        mean_list.append(sub_mean_list)

    # save info
    total_scaled_rmse_list, total_scaled_max_error_list = list(), list()
    for batch_idx in range(current_batch + 1):
        data_path = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point', f'current_batch_{batch_idx}_train', 'stopping_criteria_train_unscaled.csv')
        df = pd.read_csv(data_path)  # unscaled
        scaled_rmse_list = list()
        scaled_max_error_list = list()
        for gnn_idx in range(num_gnn):
            target_cols = target_cols_list[gnn_idx]
            df_gnn_true = df[[f'true_{col}' for col in target_cols]]
            df_gnn_pred = df[[f'pred_{col}' for col in target_cols]]

            # get error
            true_arr = df_gnn_true.to_numpy()
            pred_arr = df_gnn_pred.to_numpy()

            # scale
            unscaled_rmse = np.sqrt(((true_arr - pred_arr)**2).mean(axis=0))
            unscaled_max_error = np.abs(true_arr - pred_arr).max(axis=0)

            # scale
            std_arr = np.array(std_list[gnn_idx])
            scaled_rmse = unscaled_rmse / std_arr
            scaled_max_error = unscaled_max_error / std_arr

            # append
            scaled_rmse_list.append(scaled_rmse.tolist())
            scaled_max_error_list.append(scaled_max_error.tolist())

        # append
        total_scaled_rmse_list.append(scaled_rmse_list)
        total_scaled_max_error_list.append(scaled_max_error_list)

    # plot per gnn
    save_dir = Path(os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/stopping_criteria'))
    save_dir.mkdir(parents=True, exist_ok=True)
    for gnn_idx in range(num_gnn):
        plot_scaled_rmse_list = list()
        plot_scaled_max_error_list = list()
        for batch_idx in range(current_batch + 1):
            plot_scaled_rmse_list.append(np.mean(total_scaled_rmse_list[batch_idx][gnn_idx]))
            plot_scaled_max_error_list.append(np.mean(total_scaled_max_error_list[batch_idx][gnn_idx]))

        # plot
        # save_path = os.path.join(save_dir, f'gnn_{gnn_idx}.png')
        save_path = os.path.join(save_dir, f'gnn_{gnn_idx}_without_max.png')
        plt.figure(figsize=(6, 6))
        xlabel = range(1, current_batch + 1 + 1)
        plt.plot(xlabel, plot_scaled_rmse_list, 'co', label='Average RMSE')
        plt.plot(xlabel, plot_scaled_rmse_list, 'c', alpha=0.5)
        # plt.plot(xlabel, plot_scaled_max_error_list, 'mo', label='Average Max Error')
        plt.xlabel('Active learning cycle', fontsize=14)
        plt.ylabel('Error (normalized)', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(ticks=xlabel, labels=[f'Round {plot_batch_idx}' for plot_batch_idx in range(1, current_batch + 1 + 1)], fontsize=12)
        plt.legend(fontsize=10)
        plt.savefig(save_path)
        plt.close()

    # plot total
    plot_scaled_rmse_list, plot_scaled_max_error_list = list(), list()
    for batch_idx in range(current_batch + 1):
        sub_scaled_rmse_list, sub_scaled_max_error_list = list(), list()
        for gnn_idx in range(num_gnn):
            sub_scaled_rmse_list.extend(total_scaled_rmse_list[batch_idx][gnn_idx])
            sub_scaled_max_error_list.extend(total_scaled_max_error_list[batch_idx][gnn_idx])
        # append
        plot_scaled_rmse_list.append(np.mean(sub_scaled_rmse_list))
        plot_scaled_max_error_list.append(np.mean(sub_scaled_max_error_list))

    # save_path = os.path.join(save_dir, f'total.png')
    save_path = os.path.join(save_dir, f'total_without_max.png')
    plt.figure(figsize=(6, 6))
    xlabel = range(1, current_batch + 1 + 1)
    plt.scatter(xlabel, plot_scaled_rmse_list, color='#AD2F53', marker='o', label='Average RMSE')
    plt.plot(xlabel, plot_scaled_rmse_list, color='#AD2F53', alpha=0.5)
    # plt.plot(xlabel, plot_scaled_max_error_list, 'mo', label='Average Max Error')
    # plt.xlabel('Active learning cycle', fontsize=14)
    plt.ylabel('Average RMSE for sampled OMG CRUs (scaled)', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(ticks=xlabel, labels=[f'Round {plot_batch_idx}' for plot_batch_idx in range(1, current_batch + 1 + 1)],
               fontsize=18)
    # plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
