import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

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
    This script plots a rank correlation between uncertainty and actual error as active learning goes on.
    Run this script to see the rank correlation trend of uncertainty prediction.
    """
    ################### MODIFY ###################
    # file path
    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 3  # 0 for initial training data.
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

    # fig save dir
    fig_save_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train')

    # get mean & std to scale - use the largest data to calculate the std, mean
    num_gnn = 4
    dir_name_list = [dir_name_gnn_0_list, dir_name_gnn_1_list, dir_name_gnn_2_list, dir_name_gnn_3_list]
    total_train_uncertainty_rank_correlation_list, total_test_uncertainty_rank_correlation_list = list(), list()   # (gnn_idx, num_batch, num_targets). Note that num_targets are different for each property category.
    for gnn_idx in range(num_gnn):
        dir_name_gnn_batch_list = dir_name_list[gnn_idx]
        target_columns_list = target_cols_list[gnn_idx]
        gnn_train_uncertainty_rank_correlation_list, gnn_test_uncertainty_rank_correlation_list = list(), list()  # (num_batch, num_targets)
        for batch_idx in range(current_batch + 1):
            dir_name = dir_name_gnn_batch_list[batch_idx]
            csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{batch_idx}_train/gnn_{gnn_idx}/{dir_name}/fold_0')
            batch_train_uncertainty_rank_correlation_list, batch_test_uncertainty_rank_correlation_list = list(), list()  # to save
            train_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'train_pred.csv'))  # str
            test_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'test_pred.csv'))  # str
            for target_columns in target_columns_list:
                # train uncertainty rank correlation
                train_target_true = train_pred_results[f'true_{target_columns}'].to_numpy()
                train_target_pred = train_pred_results[f'{target_columns}'].to_numpy()
                train_absolute_error = np.abs(train_target_pred - train_target_true)
                train_target_model_uncertainty = train_pred_results[f'std_{target_columns}'].to_numpy()  # Var[mean] (epistemic uncertainty)

                train_rank_correlation_object = spearmanr(a=train_absolute_error, b=train_target_model_uncertainty)
                train_rank_correlation_coefficient = train_rank_correlation_object.statistic
                train_rank_correlation_p_value = train_rank_correlation_object.pvalue  # if larger than 0.05 -> two arrays has no significant correlation.

                batch_train_uncertainty_rank_correlation_list.append(train_rank_correlation_coefficient)  # append

                # test uncertainty rank correlation
                test_target_true = test_pred_results[f'true_{target_columns}'].to_numpy()
                test_target_pred = test_pred_results[f'{target_columns}'].to_numpy()
                test_absolute_error = np.abs(test_target_pred - test_target_true)
                test_target_model_uncertainty = test_pred_results[f'std_{target_columns}'].to_numpy()  # Var[mean] (epistemic uncertainty)

                test_rank_correlation_object = spearmanr(a=test_absolute_error, b=test_target_model_uncertainty)
                test_rank_correlation_coefficient = test_rank_correlation_object.statistic
                test_rank_correlation_p_value = test_rank_correlation_object.pvalue  # if larger than 0.05 -> two arrays has no significant correlation.

                batch_test_uncertainty_rank_correlation_list.append(test_rank_correlation_coefficient)  # append

            # append
            gnn_train_uncertainty_rank_correlation_list.append(batch_train_uncertainty_rank_correlation_list)  # batch_train_uncertainty_rank_correlation_list -> length of num_targets
            gnn_test_uncertainty_rank_correlation_list.append(batch_test_uncertainty_rank_correlation_list)  # batch_train_uncertainty_rank_correlation_list -> length of num_targets

        # append
        total_train_uncertainty_rank_correlation_list.append(gnn_train_uncertainty_rank_correlation_list)  # gnn_train_uncertainty_rank_correlation_list -> (num_batch, num_targets)
        total_test_uncertainty_rank_correlation_list.append(gnn_test_uncertainty_rank_correlation_list)  # gnn_train_uncertainty_rank_correlation_list -> (num_batch, num_targets)

    # plot per gnn
    num_gnn = 4
    num_targets = 19  # total num targets
    figure_title_list = ['3D geometry', 'Electronic properties', 'Optical properties', 'Flory-Huggins $\chi$']
    plot_total_train_uncertainty_rank_correlation_list, plot_total_test_uncertainty_rank_correlation_list = list(), list()  # (gnn_idx, num_batch). -> summed up for each target.
    for gnn_idx in range(num_gnn):  # plot per categories
        gnn_train_uncertainty_rank_correlation_arr = np.array(total_train_uncertainty_rank_correlation_list[gnn_idx])  # (num_batch, num_targets)
        gnn_test_uncertainty_rank_correlation_arr = np.array(total_test_uncertainty_rank_correlation_list[gnn_idx])  # (num_batch, num_targets)

        gnn_mean_train_uncertainty_rank_correlation = gnn_train_uncertainty_rank_correlation_arr.mean(axis=1)  # (num_batch,)
        gnn_mean_test_uncertainty_rank_correlation = gnn_test_uncertainty_rank_correlation_arr.mean(axis=1)  # (num_batch,)

        # append for total plot
        plot_total_train_uncertainty_rank_correlation_list.append(gnn_train_uncertainty_rank_correlation_arr.sum(axis=1))  # (num_batch,)
        plot_total_test_uncertainty_rank_correlation_list.append(gnn_test_uncertainty_rank_correlation_arr.sum(axis=1))  # (num_batch,)

        # plot
        fig_save_dir_gnn = Path(os.path.join(fig_save_dir, f'gnn_{gnn_idx}', 'figure', 'uncertainty_trend'))
        fig_save_dir_gnn.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        x_plot = range(0, current_batch + 1)

        # train
        plt.plot(x_plot, gnn_mean_train_uncertainty_rank_correlation, f'co', label='Train')
        plt.plot(x_plot, gnn_mean_train_uncertainty_rank_correlation, f'c', alpha=0.5)

        # test
        plt.plot(x_plot, gnn_mean_test_uncertainty_rank_correlation, f'mo', label='Test')
        plt.plot(x_plot, gnn_mean_test_uncertainty_rank_correlation, f'm', alpha=0.5)

        plt.xticks(ticks=x_plot, labels=['Initial train', 'AL1', 'AL2', 'AL3'], fontsize=14)
        plt.yticks(fontsize=14)
        ax = plt.gca()  # get current axis
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.ylabel('Averaged rank correlation of uncertainty with error', fontsize=14)  # Averaged rank correlation of uncertainty with error
        plt.legend(fontsize=12)
        plt.title(figure_title_list[gnn_idx], fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir_gnn, f'gnn_{gnn_idx}.png'))
        plt.close()

    # plot total
    plot_total_train_uncertainty_rank_correlation_arr = np.array(plot_total_train_uncertainty_rank_correlation_list)  # (gnn_idx, num_batch)
    plot_total_test_uncertainty_rank_correlation_arr = np.array(plot_total_test_uncertainty_rank_correlation_list)  # (gnn_idx, num_batch)

    plot_total_sum_train_uncertainty_rank_correlation_arr = plot_total_train_uncertainty_rank_correlation_arr.sum(axis=0)  # (num_batch,)
    plot_total_sum_test_uncertainty_rank_correlation_arr = plot_total_test_uncertainty_rank_correlation_arr.sum(axis=0)  # (num_batch,)

    plot_total_mean_train_uncertainty_rank_correlation_arr = plot_total_sum_train_uncertainty_rank_correlation_arr / num_targets  # / 19
    plot_total_mean_test_uncertainty_rank_correlation_arr = plot_total_sum_test_uncertainty_rank_correlation_arr / num_targets  # / 19

    # plot
    plt.figure(figsize=(6, 6))
    x_plot = range(0, current_batch + 1)

    # train
    plt.plot(x_plot, plot_total_mean_train_uncertainty_rank_correlation_arr, f'co', label='Train')
    plt.plot(x_plot, plot_total_mean_train_uncertainty_rank_correlation_arr, f'c', alpha=0.5)

    # test
    plt.plot(x_plot, plot_total_mean_test_uncertainty_rank_correlation_arr, f'mo', label='Test')
    plt.plot(x_plot, plot_total_mean_test_uncertainty_rank_correlation_arr, f'm', alpha=0.5)

    plt.xticks(ticks=x_plot, labels=['Initial train', 'AL1', 'AL2', 'AL3'], fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()  # get current axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylabel('Averaged rank correlation of uncertainty with error',
               fontsize=14)  # Averaged rank correlation of uncertainty with error
    plt.legend(fontsize=12)
    plt.title('Total averaged rank correlation of 19 properties', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'uncertainty_trend.png'))
    plt.close()

    # plot - test
    plt.figure(figsize=(6, 6))
    x_plot = range(0, current_batch + 1)

    # test
    plt.scatter(x_plot, plot_total_mean_test_uncertainty_rank_correlation_arr, color='#AD2F53')
    plt.plot(x_plot, plot_total_mean_test_uncertainty_rank_correlation_arr, color='#AD2F53')

    plt.xticks(ticks=x_plot, labels=['Initial train', 'AL1', 'AL2', 'AL3'], fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()  # get current axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylabel('Averaged rank correlation of uncertainty with error', fontsize=14)  # Averaged rank correlation of uncertainty with error
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'uncertainty_trend_test.png'))
    plt.close()
