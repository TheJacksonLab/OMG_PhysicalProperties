import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from matplotlib import rcParams


property_columns_Boltzmann = \
        ['molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi'] + \
        ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity'] + \
        ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
        ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy']
property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

matrix_columns = \
        ['$MW$ [g/mol]', '$LogP$ [unitless]', '$QED$ [unitless]', '$TPSA$ [$\mathring{A}^2$]', '$\Phi_{monomer}$ [unitless]', '$\Phi_{backbone}$ [unitless]'] + \
        ['$\Omega_\mathrm{A}$ [unitless]', '$\epsilon$ [unitless]', '$S_\mathrm{I}\;$[$\mathring{A}^{-2}g^{-1}$mol]', '$R_\mathrm{g}\;$[$\mathring{A}$]', '$\Omega_\mathrm{S}$ [unitless]'] + \
        ['$E_{\mathrm{HOMO}\minus1}}$ [eV]', '$E_{\mathrm{HOMO}}$ [eV]', '$E_{\mathrm{LUMO}}$ [eV]', '$E_{\mathrm{LUMO}\plus1}$ [eV]', '$\mu$ [$e\cdot$$a_0$]',
         '$q$ [$e\cdot$$a_0^2$]', '$\\alpha$ [$a_0^3$]'] + \
        ['$E_{\mathrm{S}_1}$ [eV]', "$E^{\prime}_{\mathrm{singlet}}$ [eV]", "$f^{\prime}_{\mathrm{osc}}$ [unitless]", '$E_{\mathrm{T}_1}$ [eV]'] + \
        ['$\chi_{\mathrm{water}}$ [unitless]', '$\chi_{\mathrm{ethanol}}$ [unitless]', '$\chi_{\mathrm{chloroform}}$ [unitless]']


if __name__ == '__main__':
    # plot columns
    plot_columns_list = [f'{column}_Boltzmann_average' for column in property_columns_Boltzmann] + [f'{column}_mean' for column in property_columns_mean]
    matrix_columns_list = [f'{column}' for column in matrix_columns]

    # load .csv
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/diversity_sampled_predictions_AL3.csv')  # AL3

    # number of properties
    number_of_properties = 25

    # append
    scale = True  # doesn't affect the linear correlation and rank correlation
    intermediate_linear_threshold = 0.57  # threshold to draw plots for a linear correlation
    high_linear_threshold = 0.80  # threshold to draw plots for a linear correlation
    r_list = list()

    # save dir
    save_dir = Path('./heatmap/AL3_diverse')
    save_dir.mkdir(parents=True, exist_ok=True)

    # get rank correlations np.array [num_properties, num_properties]. Only fill up the upper diagonal
    heatmap_arr = np.zeros(shape=(number_of_properties, number_of_properties))  # [i, j] > correlation between property i and j
    intermediate_correlation_idx_list, high_correlation_idx_list = list(), list()
    for row_idx in range(number_of_properties):
        for col_idx in range(number_of_properties):
            row_idx_flipped = number_of_properties - 1 - row_idx  # needed for flipping
            col_idx_flipped = number_of_properties - 1 - col_idx  # needed for flipping
            if row_idx >= col_idx_flipped:
                continue

            x_target = plot_columns_list[col_idx]  # col_idx -> x
            y_target = plot_columns_list[row_idx_flipped]  # row_idx -> y

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
            r_list.append(r)

            # append
            heatmap_arr[row_idx, col_idx] = np.abs(r)  # absolute value
            if np.abs(r) >= high_linear_threshold:
                high_correlation_idx_list.append([row_idx, col_idx])
                print(x_target, y_target)
            elif np.abs(r) >= intermediate_linear_threshold:
                intermediate_correlation_idx_list.append([row_idx, col_idx])
                print(x_target, y_target)

    # r
    r_arr = np.array(r_list)
    print(r_arr.shape)
    print(np.sum(np.abs(r_arr) >= intermediate_linear_threshold))

    # create heatmap df
    df_heatmap = pd.DataFrame(columns=matrix_columns_list[:number_of_properties], data=heatmap_arr,
                              index=list(reversed(matrix_columns_list))[:number_of_properties])  # index (row) reversed

    # annotation arr
    annotation_arr = np.zeros_like(heatmap_arr)
    for idx_pair in high_correlation_idx_list:
        row_idx, col_idx = idx_pair[0], idx_pair[1]
        annotation_arr[row_idx, col_idx] = heatmap_arr[row_idx, col_idx]
    for idx_pair in intermediate_correlation_idx_list:
        row_idx, col_idx = idx_pair[0], idx_pair[1]
        annotation_arr[row_idx, col_idx] = heatmap_arr[row_idx, col_idx]
    annotation_str = np.vectorize(lambda x: '' if x == 0 else f'{x:.2f}')(annotation_arr)

    # draw heatmap
    weak_color = '#6B6666'  # weak correlation ~ grey
    intermediate_color = '#EFA125'  # intermediate correlation ~ orange
    strong_color = '#ED2415'  # strong correlation ~ red
    cmap = LinearSegmentedColormap.from_list("", [weak_color, intermediate_color, strong_color])
    mask = 1 - np.flip(np.tril(np.ones_like(heatmap_arr) - np.eye(heatmap_arr.shape[0])), axis=0)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(data=df_heatmap, annot=annotation_str, fmt='', cmap=cmap, mask=mask, linewidths=1.0, ax=ax,
                annot_kws={'size': 10}, square=True)

    # save fig
    plt.savefig(os.path.join(save_dir, f'flip_heatmap.png'))
    plt.savefig(os.path.join(save_dir, f'flip_heatmap.svg'), format='svg', dpi=1200)
