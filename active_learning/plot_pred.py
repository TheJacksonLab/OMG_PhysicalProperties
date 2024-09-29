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

label_name_list_all = [
    ['$\Omega_A$', '$\epsilon$', '$S_I$', '$R_g$', '$\Omega_s$'],
    ['$E_{HOMO\minus1}}$', '$E_{HOMO}$', '$E_{LUMO}$', '$E_{LUMO\plus1}$', '$\mu$', '$q$', '$\\alpha$'],
    ['$E_{S_1}$', "$E^{\prime}_{singlet}$", "$f^{\prime}_{osc}$", '$E_{T_1}$'],
    ['$\chi_{water}$', '$\chi_{ethanol}$', '$\chi_{chloroform}$']
]
############################## TARGET COLUMNS ##############################


############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################

if __name__ == '__main__':
    """
    This script plots prediction results.
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred.py
    4) uncertainty_sampling.py
    -> repeat to 1)
    """
    ################### MODIFY ###################
    gnn_idx = 3
    target_columns_list = target_cols_list[gnn_idx]
    label_name_list = label_name_list_all[gnn_idx]
    # dir_name = '240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence'
    # dir_name = '240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence'  # ex) 240125-085635641199_pareto_greedy_train_batch_0_chemprop_train_gnn_1_evidence
    # dir_name = '240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence'
    # dir_name = '240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence'

    # dir_name = '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence'
    # dir_name = '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence'
    # dir_name = '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence'
    # dir_name = '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence'

    # dir_name = '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence'
    # dir_name = '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence'
    # dir_name = '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence'
    # dir_name = '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence'

    # dir_name = '240531-100412913979_OMG_train_batch_3_chemprop_train_gnn_0_evidence'
    # dir_name = '240531-100828584294_OMG_train_batch_3_chemprop_train_gnn_1_evidence'
    # dir_name = '240531-100906010104_OMG_train_batch_3_chemprop_train_gnn_2_evidence'
    dir_name = '240531-101615032820_OMG_train_batch_3_chemprop_train_gnn_3_evidence'

    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 3  # 0 for initial training data
    fold_num = 1  # also works for 1
    ################### MODIFY ###################

    # metric
    metric = 'r2'  # r2 or rmse
    ################### MODIFY ###################
    fig_save_dir = Path(os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/figure'))
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    for target_columns, label_name in zip(target_columns_list, label_name_list):
        train_pred_list = list()
        test_pred_list = list()
        train_target_list = list()
        test_target_list = list()
        for fold_idx in range(fold_num):
            csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/{dir_name}/fold_{fold_idx}')
            train_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'train_pred.csv'))
            test_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'test_pred.csv'))

            # train
            train_target_true = train_pred_results[f'true_{target_columns}'].to_numpy()  # same for different fold_idx
            train_target_pred = train_pred_results[f'{target_columns}'].to_numpy()

            # test
            test_target_true = test_pred_results[f'true_{target_columns}'].to_numpy()  # same for different fold_idx
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

        # mean and var
        train_pred_mean = train_pred_arr.mean(axis=0)
        train_pred_std = train_pred_arr.std(axis=0, ddof=0)

        test_pred_mean = test_pred_arr.mean(axis=0)
        test_pred_std = test_pred_arr.std(axis=0, ddof=0)

        train_target_mean = train_target_arr.mean(axis=0)
        train_target_std = train_target_arr.std(axis=0, ddof=0)

        test_target_mean = test_target_arr.mean(axis=0)
        test_target_std = test_target_arr.std(axis=0, ddof=0)

        # plot
        plt.figure(figsize=(6, 6), dpi=300)
        plt.xlabel(f'True {label_name}', fontsize=14)
        plt.ylabel(f'Prediction {label_name}', fontsize=14)
        # plt.xlabel(f'{target_columns} True', fontsize=14)
        # plt.ylabel(f'{target_columns} Prediction', fontsize=14)

        # train
        color = 'c'

        # absolute max error
        train_absolute_max_error = max_error(y_true=train_target_mean, y_pred=train_pred_mean)
        # print(f"Train maximum absolute error {target_columns}= {train_absolute_max_error:.2f}")

        # train rank correlation
        train_rank_correlation_object = spearmanr(a=train_target_mean, b=train_pred_mean)  # the same when a and b are swapped.
        train_rank_correlation_coefficient = train_rank_correlation_object.statistic
        if metric == 'r2':
            r2 = r2_score(y_true=train_target_mean, y_pred=train_pred_mean)
            print(f"Train r2 score {target_columns}= {r2:.2f}")
            plt.errorbar(x=train_target_mean, y=train_pred_mean, yerr=train_pred_std, color=color, ecolor=color,
                         linewidth=2.5, elinewidth=0.25, capsize=2.0, label='Train $R^2$:' + f' {r2:.2f} ($\\rho$={train_rank_correlation_coefficient:.2f})', fmt="o", markersize=3.0, alpha=1.0)
        elif metric == 'rmse':
            rmse_score = mean_squared_error(y_true=train_target_mean, y_pred=train_pred_mean, squared=False)
            print(f"Train rmse score {target_columns}= {rmse_score:.2f}")
            plt.errorbar(x=train_target_mean, y=train_pred_mean, yerr=train_pred_std, color=color, ecolor=color,
                         linewidth=2.5, elinewidth=0.25, capsize=2.0, label='Train RMSE:' + f' {rmse_score:.2f} ($\\rho$={train_rank_correlation_coefficient:.2f})', fmt="o", markersize=3.0, alpha=1.0)

        # test
        color = 'm'

        # absolute max error
        test_absolute_max_error = max_error(y_true=test_target_mean, y_pred=test_pred_mean)
        # print(f"Test maximum absolute error {target_columns}= {test_absolute_max_error:.2f}")

        # test rank correlation
        test_rank_correlation_object = spearmanr(a=test_target_mean, b=test_pred_mean)  # the same when a and b are swapped.
        test_rank_correlation_coefficient = test_rank_correlation_object.statistic
        if metric == 'r2':
            r2 = r2_score(y_true=test_target_mean, y_pred=test_pred_mean)
            print(f"Test r2 score {target_columns}= {r2:.2f}")
            plt.errorbar(x=test_target_mean, y=test_pred_mean, yerr=test_pred_std, color=color, ecolor=color,
                         linewidth=2.5, elinewidth=0.25, capsize=2.0, label='Test $R^2$:' + f' {r2:.2f} ($\\rho$={test_rank_correlation_coefficient:.2f})',
                         fmt="o", markersize=3.0, alpha=1.0)
        elif metric == 'rmse':
            rmse_score = mean_squared_error(y_true=test_target_mean, y_pred=test_pred_mean, squared=False)
            print(f"Test rmse score {target_columns}= {rmse_score:.2f}")
            plt.errorbar(x=test_target_mean, y=test_pred_mean, yerr=test_pred_std, color=color, ecolor=color,
                         linewidth=2.5, elinewidth=0.25, capsize=2.0, label='Test RMSE:' + f' {rmse_score:.2f} ($\\rho$={test_rank_correlation_coefficient:.2f})', fmt="o", markersize=3.0, alpha=1.0)

        # y=x line
        true_min_plot = np.min([np.min(train_target_mean), np.min(test_target_mean)])
        pred_min_plot = np.min([np.min(train_pred_mean), np.min(test_pred_mean)])
        min_plot = np.min([true_min_plot, pred_min_plot])
        true_max_plot = np.max([np.max(train_target_mean), np.max(test_target_mean)])
        pred_max_plot = np.max([np.max(train_pred_mean), np.max(test_pred_mean)])
        max_plot = np.max([true_max_plot, pred_max_plot])
        plt.plot([min_plot, max_plot], [min_plot, max_plot], color='grey', linestyle='--')

        # plot
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'{target_columns}_prediction_{metric}_label.png'))
        plt.close()
