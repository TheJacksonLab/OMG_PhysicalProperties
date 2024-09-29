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
    This script plots prediction rmse errors of active learning cycles with fold based on normalized RMSE arr from save_fold_pred_error.py
    The saved arr has a shape of (num_batch, num_targets, num_fold) containing "normalized RMSE arr" with train std.
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred & plot_fold_pred_error.py
    4) uncertainty_sampling.py
    -> repeat to 1)
    """
    ################### MODIFY ###################
    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 3  # 0 for initial training data
    fold_num = 5  # also works for 1
    ################### MODIFY ###################

    # load arr & plot
    num_gnn = 4
    num_targets = 19
    figure_title_list = ['3D geometry', 'Electronic properties', 'Optical properties', 'Flory-Huggins $\chi$']
    total_arr_list = list()
    for gnn_idx in range(num_gnn):  # plot per categories
        arr_save_dir = Path(os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/figure/arr'))
        normalized_rmse_arr = np.load(os.path.join(arr_save_dir, 'normalized_rmse_arr.npy'))  # (num_batch, num_targets, num_fold). Here, num_targets is not 19. (for each property categoy)

        # save to total
        total_arr_list.append(normalized_rmse_arr.sum(axis=1))  # note that each category has a different number of targets.

        # over targets
        mean_normalized_rmse_arr = normalized_rmse_arr.mean(axis=1)  # (num_batch, num_fold)

        # for plot
        plot_mean_normalized_rmse_arr = mean_normalized_rmse_arr.mean(axis=-1)
        plot_std_normalized_rmse_arr = mean_normalized_rmse_arr.std(axis=-1, ddof=0)

        # plot
        plt.figure(figsize=(6, 6), dpi=300)
        color = 'm'
        x_plot = range(0, current_batch + 1)
        plt.errorbar(x=x_plot, y=plot_mean_normalized_rmse_arr, yerr=plot_std_normalized_rmse_arr, color=color, ecolor=color,
                     linewidth=2.5, elinewidth=0.5, capsize=2.0,
                     label='Test RMSE', fmt="o",
                     markersize=3.0)
        plt.plot(x_plot, plot_mean_normalized_rmse_arr, f'{color}', alpha=0.5)
        plt.xticks(ticks=x_plot, labels=['Initial train', 'AL1', 'AL2', 'AL3'], fontsize=14)
        plt.yticks(fontsize=14)
        ax = plt.gca()  # get current axis
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.ylabel('Normalized RMSE (mean)', fontsize=14)
        plt.legend(fontsize=12)
        plt.title(figure_title_list[gnn_idx], fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(arr_save_dir, 'rmse.png'))

    # plot total mean
    total_arr = np.array(total_arr_list)  # (num_gnn, num_batch, num_fold)
    total_arr = total_arr.sum(axis=0)  # (num_batch, num_fold)
    total_normalized_mean_rmse_arr = total_arr / num_targets  # (num_batch, num_fold)

    # for plot
    plot_mean_normalized_rmse_arr = total_normalized_mean_rmse_arr.mean(axis=-1)
    plot_std_normalized_rmse_arr = total_normalized_mean_rmse_arr.std(axis=-1, ddof=0)

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    fig_save_dir = Path(os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train'))
    color = 'm'
    x_plot = range(0, current_batch + 1)
    plt.errorbar(x=x_plot, y=plot_mean_normalized_rmse_arr, yerr=plot_std_normalized_rmse_arr, color=color,
                 ecolor=color,
                 linewidth=2.5, elinewidth=0.5, capsize=2.0,
                 label='Test RMSE', fmt="o",
                 markersize=3.0)
    plt.plot(x_plot, plot_mean_normalized_rmse_arr, f'{color}', alpha=0.5)
    plt.xticks(ticks=x_plot, labels=['Initial train', 'AL1', 'AL2', 'AL3'], fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()  # get current axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylabel('Normalized RMSE (mean)', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Total RMSE of 19 properties', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'rmse.png'))
