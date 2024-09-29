import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr, spearmanr


property_columns_Boltzmann = \
        ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity',
         'molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi'] + \
        ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
        ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy']

property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

if __name__ == '__main__':
    ###### Load sampled data and start to plot! ######
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/diversity_sampled_predictions_AL3.csv')

    # plot columns
    plot_columns_list = [f'{column}_Boltzmann_average' for column in property_columns_Boltzmann] + [f'{column}_mean' for column in property_columns_mean]

    # number of target cols
    number_of_properties = len(plot_columns_list)

    # append
    scale = True  # doesn't affect the linear correlation and rank correlation
    plot_rank = False  # plot rank or plot linear correlation
    r_list = list()
    rank_correlation_list = list()
    name_list = list()
    polymerization_mechanism_idx_arr = df['polymerization_mechanism_idx'].to_numpy()
    step_growth_idx = np.where((polymerization_mechanism_idx_arr >= 1) & (polymerization_mechanism_idx_arr <= 7))[0]
    chain_growth_addition_idx = np.where((polymerization_mechanism_idx_arr >= 8) & (polymerization_mechanism_idx_arr <= 9))[0]
    ring_opening_idx = np.where((polymerization_mechanism_idx_arr >= 10) & (polymerization_mechanism_idx_arr <= 15))[0]
    metathesis_idx = np.where((polymerization_mechanism_idx_arr >= 16) & (polymerization_mechanism_idx_arr <= 17))[0]

    # plot idx
    plot_idx_list = list()  # used for a matrix plot. Component is [row_idx, col_idx]

    # save dir
    if plot_rank:
        save_dir = Path(os.path.join('./pair_correlation/AL3_diverse/rank_correlation'))
    else:
        save_dir = Path(os.path.join('./pair_correlation/AL3_diverse/linear_correlation'))
    save_dir.mkdir(parents=True, exist_ok=True)

    for row_idx in range(number_of_properties):
        if row_idx == number_of_properties - 1:
            break
        for col_idx in range(row_idx + 1, number_of_properties):
            x_target = plot_columns_list[col_idx]  # col_idx -> x
            y_target = plot_columns_list[row_idx]  # row_idx -> y

            # arr
            x_target_arr = df[f'{x_target}'].to_numpy()
            y_target_arr = df[f'{y_target}'].to_numpy()

            if scale:
                # get mean and std
                x_target_std = np.std(x_target_arr, ddof=0)
                x_target_mean = np.mean(x_target_arr)
                y_target_std = np.std(y_target_arr, ddof=0)
                y_target_mean = np.mean(y_target_arr)

                # scaling
                x_target_arr = (x_target_arr - x_target_mean) / x_target_std
                y_target_arr = (y_target_arr - y_target_mean) / y_target_std

            # pearson r - symmetric
            pearson_r = pearsonr(x_target_arr, y_target_arr)
            r = pearson_r.statistic

            # rank correlation
            rank_correlation_object = spearmanr(a=x_target_arr, b=y_target_arr)  # the same when a and b are swapped.
            rank_correlation_coefficient = rank_correlation_object.statistic

            # append
            r_list.append(r)
            rank_correlation_list.append(rank_correlation_coefficient)

            # estimate a correlation
            if plot_rank:
                correlation_value = rank_correlation_coefficient
            else:
                correlation_value = r
    print(len(r_list))

    # get values for linear correlations
    r_array = np.array(r_list)
    r_array_abs = np.abs(r_array)
    df_linear_correlation_abs = pd.DataFrame(data=r_array_abs, columns=['linear_correlation'])
    df_linear_correlation_abs = df_linear_correlation_abs.sort_values(by='linear_correlation', ascending=False)
    df_linear_correlation_abs = df_linear_correlation_abs[df_linear_correlation_abs['linear_correlation'] >= 0.55]
    print(df_linear_correlation_abs)
    print(df_linear_correlation_abs['linear_correlation'].describe())

    # plot hist
    intermediate_linear_threshold = 0.57
    high_linear_threshold = 0.80
    plt.hist(r_array_abs, color='m', bins=100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Pearson linear correlation $\\rho$', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.vlines(x=intermediate_linear_threshold, ymin=0, ymax=10)
    plt.vlines(x=high_linear_threshold, ymin=0, ymax=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'linear_correlation_hist.png'))
    plt.close()

    # histogram
    weak_color = '#6B6666'  # weak correlation ~ grey
    intermediate_color = '#EFA125'  # intermediate correlation ~ orange
    strong_color = '#ED2415'  # strong correlation ~ red
    plt.figure(figsize=(6, 6), dpi=300)
    fig, axe = plt.subplots()
    sorted_linear_correlation_array_abs = np.sort(r_array_abs)
    weak_correlation = r_array_abs[np.where(r_array_abs < intermediate_linear_threshold)[0]]
    intermediate_correlation = r_array_abs[np.where(
        (r_array_abs >= intermediate_linear_threshold) & (r_array_abs < high_linear_threshold))[0]
    ]
    strong_correlation = r_array_abs[np.where(r_array_abs >= high_linear_threshold)[0]]

    plt.hist([weak_correlation, intermediate_correlation, strong_correlation],
             color=[weak_color, intermediate_color, strong_color],
             label=['Weak', 'Moderate', 'Strong'], linewidth=0.5, bins=50)

    # tick range and format
    ax = plt.gca()
    ax.set_xlim([0.0, 1.0])
    plt.yscale('log')
    plt.yticks(ticks=[0.1, 1, 10], labels=['', 1, 10])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=32)
    plt.xlabel('Absolute linear correlation |$\\rho|$', fontsize=32)
    plt.ylabel('Counts', fontsize=32)
    # plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sorted_linear_correlation_hist.png'), dpi=1200)
    plt.savefig(os.path.join(save_dir, f'sorted_linear_correlation_hist.svg'), format='svg', dpi=1200)
    plt.close()




