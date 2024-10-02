import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import LinearSegmentedColormap


def kernel_density_estimation(arr: np.array, label, plot_min, plot_max, scatter_map,
                              reaction_id, low_z_reaction_id, high_z_reaction_id, z_ticks,
                              low_eps, high_eps, save_file_path: str):
    """
    This function plots & saves a kernel density estimation of a given arr.
    arr is "scaled" arr.
    :return: None
    """
    # get mean and std
    mean_value = z_target_mean
    std_value = z_target_std

    # get a low z
    low_z_lower_bound = (-1.5 - low_eps) * std_value + mean_value  # unscaled
    low_z_upper_bound = (-1.5 + low_eps) * std_value + mean_value  # unscaled
    low_z_value_idx = np.where((arr >= low_z_lower_bound) & (arr < low_z_upper_bound))[0]

    # find a molecule z value
    low_z_reaction_id_index = np.where(reaction_id == low_z_reaction_id)[0]
    possible_low_z_reaction_id = reaction_id[low_z_value_idx]
    # print(f'Possible low Z reaction id: {possible_low_z_reaction_id}')
    low_z_unscaled = arr[low_z_reaction_id_index]
    print(f"The number of possible low z values: {low_z_value_idx.shape[0]}")
    print(f"Low Z value reaction id: {low_z_reaction_id}")
    print(f"Low Z value (unscaled): {low_z_unscaled}")

    # get a high z
    high_z_lower_bound = (1.5 - high_eps) * std_value + mean_value  # unscaled
    high_z_upper_bound = (1.5 + high_eps) * std_value + mean_value  # unscaled
    high_z_value_idx = np.where((arr >= high_z_lower_bound) & (arr < high_z_upper_bound))[0]

    # find a molecule z value
    high_z_reaction_id_index = np.where(reaction_id == high_z_reaction_id)[0]
    possible_high_z_reaction_id = reaction_id[high_z_value_idx]
    # print(f'Possible high Z reaction id: {possible_high_z_reaction_id}')
    high_z_unscaled = arr[high_z_reaction_id_index]
    print(f"The number of possible high z values: {high_z_value_idx.shape[0]}")
    print(f"High Z value reaction id: {high_z_reaction_id}")
    print(f"High Z value (unscaled): {high_z_unscaled}")

    # KDE
    density = stats.gaussian_kde(arr)

    # plot x
    plot_x_list = np.arange(plot_min, plot_max, 0.001)

    # plot
    plt.figure(figsize=(6, 6))
    # kde_min, kde_max = 1.0, 0.0
    kde_min, kde_max = 0.0, 1.6
    for plot_x in plot_x_list:
        density_plot = density(plot_x)
        plt.scatter(plot_x, density_plot, color=scatter_map.to_rgba(plot_x, norm=True))  # plot_x will be normalized inside the function.

        # # assign min & max
        # if density_plot < kde_min:
        #     kde_min = density_plot
        # if density_plot > kde_max:
        #     kde_max = density_plot

    # vertical lines
    plt.vlines(low_z_lower_bound, ymin=kde_min, ymax=kde_max, color="#7C8E8D", linestyles='dashed', linewidth=1.0)
    plt.vlines(low_z_upper_bound, ymin=kde_min, ymax=kde_max, color="#7C8E8D", linestyles='dashed', linewidth=1.0)
    plt.vlines(high_z_lower_bound, ymin=kde_min, ymax=kde_max, color="#7C8E8D", linestyles='dashed', linewidth=1.0)
    plt.vlines(high_z_upper_bound, ymin=kde_min, ymax=kde_max, color="#7C8E8D", linestyles='dashed', linewidth=1.0)

    # plot low z (unscaled)
    low_z_unscaled = arr[low_z_reaction_id_index]
    plt.scatter(low_z_unscaled, density(low_z_unscaled), color='#231F20', alpha=1.0, marker='*', s=200.0, zorder=2)

    # plot high z (unscaled)
    high_z_unscaled = arr[high_z_reaction_id_index]
    plt.scatter(high_z_unscaled, density(high_z_unscaled), color='#231F20', alpha=1.0, marker='*', s=200.0, zorder=2)

    plt.xticks(z_ticks, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel('Probabilty density', fontsize=24)
    plt.yticks(ticks=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
    plt.tight_layout()
    plt.savefig(save_file_path, format='svg', dpi=1200)
    plt.savefig(save_file_path.replace('.svg', '.png'), dpi=1200)  # png
    plt.close()

    return


if __name__ == '__main__':
    """
    This script draws kernel density estimation from "unscaled" z values to get correct values of probability density.
    """
    # plot to show a freedom of design. -> "un"correlated property-property correlation.
    plot_columns_list = ['normalized_monomer_phi_Boltzmann_average', 'dominant_transition_energy_Boltzmann_average',
                         'chi_parameter_water_mean']
    plot_name = ['$\Phi_{\mathrm{monomer}}$ [unitless]', "$E^{\prime}_{\mathrm{singlet}}$ [eV]",
                 '$\chi_{\mathrm{water}}$ [unitless]']
    color_bar = ["#2F65ED", "#F5EF7E", "#F89A3F"]

    # load .csv
    df = pd.read_csv(
        '/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/diversity_sampled_predictions_AL3.csv')

    # reaction id arr
    reaction_id_arr = df['reaction_id'].to_numpy(dtype='int')

    # save dir
    save_dir = Path(os.path.join('./AL3_diverse/figure'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # arr
    x_target = plot_columns_list[0]
    y_target = plot_columns_list[1]
    z_target = plot_columns_list[2]

    x_target_arr = df[x_target].to_numpy().copy()
    y_target_arr = df[y_target].to_numpy().copy()
    z_target_arr = df[z_target].to_numpy().copy()

    # manual ticks
    manual_tick = True
    manual_x_ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    manual_y_ticks = [0, 2, 4, 6, 8, 10]
    manual_z_ticks = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

    # get mean and std
    x_target_std = np.std(x_target_arr, ddof=0)
    x_target_mean = np.mean(x_target_arr)
    y_target_std = np.std(y_target_arr, ddof=0)
    y_target_mean = np.mean(y_target_arr)
    z_target_std = np.std(z_target_arr, ddof=0)
    z_target_mean = np.mean(z_target_arr)

    # scale doesn't affect the linear correlation and rank correlation
    x_target_arr = (x_target_arr - x_target_mean) / x_target_std
    y_target_arr = (y_target_arr - y_target_mean) / y_target_std
    # z_target_arr = (z_target_arr - z_target_mean) / z_target_std  # no scale z_target_arr

    # print(f"Z mean: {z_target_mean}")
    # print(f"Z std: {z_target_std}")

    # high & low threshold - scaled. 1 means -> one standard deviation
    x_high_threshold = 1
    x_low_threshold = -1
    y_high_threshold = 1
    y_low_threshold = -1
    epsilon = 0.2

    # choose four different area -> plus minus epsilon range
    # (1) mean + 1 sigma (x) / mean + 1 sigma (y)
    x_1 = x_high_threshold
    x_1_lower, x_1_upper = x_1 - epsilon, x_1 + epsilon
    x_1_bool_arr = np.logical_and(x_target_arr >= x_1_lower, x_target_arr < x_1_upper)

    y_1 = y_high_threshold
    y_1_lower, y_1_upper = y_1 - epsilon, y_1 + epsilon
    y_1_bool_arr = np.logical_and(y_target_arr >= y_1_lower, y_target_arr < y_1_upper)

    bool_1_arr = np.logical_and(x_1_bool_arr, y_1_bool_arr)

    # (1) target arr (normalized)
    x_1_target_arr = x_target_arr[bool_1_arr]
    y_1_target_arr = y_target_arr[bool_1_arr]
    z_1_target_arr = z_target_arr[bool_1_arr]
    reaction_id_arr_1 = reaction_id_arr[bool_1_arr]

    # (2) mean - 1 sigma (x) / mean + 1 sigma (y)
    x_2 = x_low_threshold
    x_2_lower, x_2_upper = x_2 - epsilon, x_2 + epsilon
    x_2_bool_arr = np.logical_and(x_target_arr >= x_2_lower, x_target_arr < x_2_upper)

    y_2 = y_high_threshold
    y_2_lower, y_2_upper = y_2 - epsilon, y_2 + epsilon
    y_2_bool_arr = np.logical_and(y_target_arr >= y_2_lower, y_target_arr < y_2_upper)

    bool_2_arr = np.logical_and(x_2_bool_arr, y_2_bool_arr)

    # (2) target arr (normalized)
    x_2_target_arr = x_target_arr[bool_2_arr]
    y_2_target_arr = y_target_arr[bool_2_arr]
    z_2_target_arr = z_target_arr[bool_2_arr]
    reaction_id_arr_2 = reaction_id_arr[bool_2_arr]

    # (3) mean - 1 sigma (x) / mean - 1 sigma (y)
    x_3 = x_low_threshold
    x_3_lower, x_3_upper = x_3 - epsilon, x_3 + epsilon
    x_3_bool_arr = np.logical_and(x_target_arr >= x_3_lower, x_target_arr < x_3_upper)

    y_3 = y_low_threshold
    y_3_lower, y_3_upper = y_3 - epsilon, y_3 + epsilon
    y_3_bool_arr = np.logical_and(y_target_arr >= y_3_lower, y_target_arr < y_3_upper)

    bool_3_arr = np.logical_and(x_3_bool_arr, y_3_bool_arr)

    # (3) target arr (normalized)
    x_3_target_arr = x_target_arr[bool_3_arr]
    y_3_target_arr = y_target_arr[bool_3_arr]
    z_3_target_arr = z_target_arr[bool_3_arr]
    reaction_id_arr_3 = reaction_id_arr[bool_3_arr]

    # (4) mean + 1 sigma (x) / mean - 1 sigma (y)
    x_4 = x_high_threshold
    x_4_lower, x_4_upper = x_4 - epsilon, x_4 + epsilon
    x_4_bool_arr = np.logical_and(x_target_arr >= x_4_lower, x_target_arr < x_4_upper)

    y_4 = y_low_threshold
    y_4_lower, y_4_upper = y_4 - epsilon, y_4 + epsilon
    y_4_bool_arr = np.logical_and(y_target_arr >= y_4_lower, y_target_arr < y_4_upper)

    bool_4_arr = np.logical_and(x_4_bool_arr, y_4_bool_arr)

    # (4) target arr (normalized)
    x_4_target_arr = x_target_arr[bool_4_arr]
    y_4_target_arr = y_target_arr[bool_4_arr]
    z_4_target_arr = z_target_arr[bool_4_arr]
    reaction_id_arr_4 = reaction_id_arr[bool_4_arr]

    # plot
    g = sns.JointGrid()

    # plot df
    plot_df = pd.DataFrame()
    plot_df[x_target] = x_target_arr
    plot_df[y_target] = y_target_arr
    plot_df[z_target] = z_target_arr

    cmap = LinearSegmentedColormap.from_list("", color_bar)

    # scatter plot
    ax = sns.scatterplot(x=x_target, y=y_target, hue=z_target, data=plot_df, palette=cmap, ax=g.ax_joint,
                         edgecolor=None, alpha=0.75, legend=False, s=7.5)

    # plot square region 1
    g.ax_joint.vlines(x=x_1_lower, ymin=y_1_lower, ymax=y_1_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.vlines(x=x_1_upper, ymin=y_1_lower, ymax=y_1_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_1_lower, xmin=x_1_lower, xmax=x_1_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_1_upper, xmin=x_1_lower, xmax=x_1_upper, color='#F2364D', linewidth=2.0)

    # plot square region 2
    g.ax_joint.vlines(x=x_2_lower, ymin=y_2_lower, ymax=y_2_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.vlines(x=x_2_upper, ymin=y_2_lower, ymax=y_2_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_2_lower, xmin=x_2_lower, xmax=x_2_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_2_upper, xmin=x_2_lower, xmax=x_2_upper, color='#F2364D', linewidth=2.0)

    # plot square region 3
    g.ax_joint.vlines(x=x_3_lower, ymin=y_3_lower, ymax=y_3_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.vlines(x=x_3_upper, ymin=y_3_lower, ymax=y_3_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_3_lower, xmin=x_3_lower, xmax=x_3_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_3_upper, xmin=x_3_lower, xmax=x_3_upper, color='#F2364D', linewidth=2.0)

    # plot square region 4
    g.ax_joint.vlines(x=x_4_lower, ymin=y_4_lower, ymax=y_4_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.vlines(x=x_4_upper, ymin=y_4_lower, ymax=y_4_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_4_lower, xmin=x_4_lower, xmax=x_4_upper, color='#F2364D', linewidth=2.0)
    g.ax_joint.hlines(y=y_4_upper, xmin=x_4_lower, xmax=x_4_upper, color='#F2364D', linewidth=2.0)

    # kde plots on the marginal axes
    sns.kdeplot(x=x_target, data=plot_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
    sns.kdeplot(y=y_target, data=plot_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

    # tick ranges. Modify the tick values to the original scales.
    if not manual_tick:
        xticks_range_to_plot_scaled = np.linspace(x_target_arr.min(), x_target_arr.max(), num=5, endpoint=True)
        xticks_range_to_plot_unscaled = (xticks_range_to_plot_scaled * x_target_std) + x_target_mean
        ax.set_xticks(xticks_range_to_plot_scaled)  # ticks
        if abs(xticks_range_to_plot_unscaled.max()) >= 3:  # not 1
            xtick_label = xticks_range_to_plot_unscaled.astype('int')
        else:
            xtick_label = [f'{value:.2f}' for value in xticks_range_to_plot_unscaled]
        ax.set_xticklabels(xtick_label, fontsize=12)  # ticks
    else:
        # normalize
        xticks_normalized = (np.array(manual_x_ticks) - x_target_mean) / x_target_std
        ax.set_xticks(xticks_normalized)  # ticks
        ax.set_xticklabels(manual_x_ticks)  # tick labels

    if not manual_tick:
        yticks_range_to_plot_scaled = np.linspace(y_target_arr.min(), y_target_arr.max(), num=5, endpoint=True)
        yticks_range_to_plot_unscaled = (yticks_range_to_plot_scaled * y_target_std) + y_target_mean
        ax.set_yticks(yticks_range_to_plot_scaled)  # ticks
        if abs(yticks_range_to_plot_unscaled.max()) >= 3:  # not 1
            ytick_label = yticks_range_to_plot_unscaled.astype('int')
        else:
            ytick_label = [f'{value:.2f}' for value in yticks_range_to_plot_unscaled]
        ax.set_yticklabels(ytick_label, fontsize=12)  # ticks
    else:
        # normalize
        yticks_normalized = (np.array(manual_y_ticks) - y_target_mean) / y_target_std
        ax.set_yticks(yticks_normalized)  # ticks
        ax.set_yticklabels(manual_y_ticks)  # tick labels

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.set_axis_labels(plot_name[0], plot_name[1], fontsize=16)
    g.fig.tight_layout()

    # color bars
    norm = plt.Normalize(plot_df[z_target].min(), plot_df[z_target].max())
    scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Make space for the colorbar
    g.fig.subplots_adjust(bottom=0.2)
    cax = g.fig.add_axes([0.21, 0.06, 0.6, 0.02])  # l, b, w, h
    cbar = g.fig.colorbar(scatter_map, orientation='horizontal', cax=cax)
    cbar.set_ticks(manual_z_ticks)
    cbar.ax.tick_params(labelsize=16)

    g.fig.savefig(os.path.join(save_dir, "unscaled_freedom_square.png"), dpi=1200)
    plt.close()

    # histogram 1
    print(z_1_target_arr.shape)
    kernel_density_estimation(z_1_target_arr, label=plot_name[-1], plot_min=z_target_arr.min(),
                              plot_max=z_target_arr.max(), scatter_map=scatter_map,
                              z_ticks=manual_z_ticks,
                              save_file_path='./AL3_diverse/figure/unscaled_freedom_region_1.svg',
                              reaction_id=reaction_id_arr_1,
                              low_z_reaction_id=1772481, high_z_reaction_id=3655215, low_eps=0.2, high_eps=0.2)

    # histogram 2
    print(z_2_target_arr.shape)
    kernel_density_estimation(z_2_target_arr, label=plot_name[-1], plot_min=z_target_arr.min(),
                              plot_max=z_target_arr.max(), scatter_map=scatter_map,
                              z_ticks=manual_z_ticks,
                              save_file_path='./AL3_diverse/figure/unscaled_freedom_region_2.svg',
                              reaction_id=reaction_id_arr_2,
                              low_z_reaction_id=3999522, high_z_reaction_id=9258272, low_eps=0.2, high_eps=0.2)

    # histogram 3
    print(z_3_target_arr.shape)
    kernel_density_estimation(z_3_target_arr, label=plot_name[-1], plot_min=z_target_arr.min(),
                              plot_max=z_target_arr.max(), scatter_map=scatter_map,
                              z_ticks=manual_z_ticks,
                              save_file_path='./AL3_diverse/figure/unscaled_freedom_region_3.svg',
                              reaction_id=reaction_id_arr_3,
                              low_z_reaction_id=4044344, high_z_reaction_id=4002812, low_eps=0.2, high_eps=0.2)

    # histogram 4
    print(z_4_target_arr.shape)
    kernel_density_estimation(z_4_target_arr, label=plot_name[-1], plot_min=z_target_arr.min(),
                              plot_max=z_target_arr.max(), scatter_map=scatter_map,
                              z_ticks=manual_z_ticks,
                              save_file_path='./AL3_diverse/figure/unscaled_freedom_region_4.svg',
                              reaction_id=reaction_id_arr_4,
                              low_z_reaction_id=11888541, high_z_reaction_id=6215927, low_eps=0.2, high_eps=0.2)
