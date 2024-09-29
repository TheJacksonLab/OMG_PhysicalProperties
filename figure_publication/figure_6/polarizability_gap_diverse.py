import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './nds-py'))  # search pareto front

from nds import ndomsort  # https://github.com/KernelA/nds-py/tree/master

def func(x, a):
    return a / x


if __name__ == '__main__':
    # figure 6 -> Polarizability and HOMO-LUMO gap
    plot_name = ['$\\alpha$ [$a_0^3$]', 'HOMO-LUMO gap [eV]']  # y, x

    manual_tick = True
    manual_x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    manual_y_ticks = [70, 400, 700, 1000, 1300]

    # load .csv
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/diversity_sampled_predictions_AL3.csv')
    print(df.shape)
    print(df['chi_parameter_water_mean'].describe())
    print(df['chi_parameter_chloroform_mean'].describe())

    # exclude a negative homo-lumo gap (small portion). This can happen as properties are machine-learned properties.
    df['gap_Boltzmann_average'] = df['LUMO_Boltzmann_average'].to_numpy().copy() - df['HOMO_Boltzmann_average'].to_numpy().copy()
    df = df[df['gap_Boltzmann_average'] >= 0].reset_index(drop=True)
    # df = df[df['chi_parameter_water_mean'] < 0.961292].reset_index(drop=True)  # water solubility. mean - std
    df = df[df['chi_parameter_chloroform_mean'] < -0.245265].reset_index(drop=True)  # chloroform solubility. mean - std

    # To answer the question of Nick
    # methyl_terminated_product_list = df['methyl_terminated_product'].tolist()
    # for product in methyl_terminated_product_list:
    #     if 'cc' in product:
    #         print(product)
    # df = df.sort_values(by='polarizability_Boltzmann_average', ascending=False)
    # print(df['methyl_terminated_product'].tolist()[:30])
    # exit()

    # reaction id list
    reaction_id_arr = df['reaction_id'].to_numpy(dtype='int')
    methyl_terminated_product_arr = df['methyl_terminated_product'].to_numpy(dtype='str')
    polymerization_mechanism_idx_arr = df['polymerization_mechanism_idx'].to_numpy()

    # save dir
    save_dir = Path(os.path.join('./figure/AL3_diverse'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # get HOMO-LUMO gap
    gap_arr = df['gap_Boltzmann_average'].to_numpy().copy()
    polarizability_arr = df['polarizability_Boltzmann_average'].to_numpy().copy()

    # plot
    scale = True  # doesn't affect the linear correleation and rank correlation
    shift = False

    # arr
    x_target = 'gap'
    y_target = 'polarizability'
    x_target_arr = gap_arr
    y_target_arr = polarizability_arr

    if scale:
        # get mean and std
        x_target_std = np.std(x_target_arr, ddof=0)
        x_target_mean = np.mean(x_target_arr)
        y_target_std = np.std(y_target_arr, ddof=0)
        y_target_mean = np.mean(y_target_arr)

        # mean shift
        if shift:
            x_target_arr = (x_target_arr - x_target_mean) / x_target_std
            y_target_arr = (y_target_arr - y_target_mean) / y_target_std
        else:
            x_target_arr = (x_target_arr) / x_target_std
            y_target_arr = (y_target_arr) / y_target_std

    # rank correlation
    # rank_correlation_object = spearmanr(a=x_target_arr, b=y_target_arr)  # the same when a and b are swapped.
    # rank_correlation_coefficient = rank_correlation_object.statistic
    # print(rank_correlation_coefficient)

    # pearson r - symmetric
    pearson_r = pearsonr(x_target_arr, y_target_arr)
    r = pearson_r.statistic
    print(r)

    # pareto search
    seq_for_pareto = np.hstack((x_target_arr.reshape(-1, 1), y_target_arr.reshape(-1, 1)))  # (num_data, 2)
    seq_for_pareto *= -1  # multiply (-1) to minimize seq value (maximize both)
    front_idx = np.array(ndomsort.non_domin_sort(seq_for_pareto, only_front_indices=True))  # return a front index (0 -> optimal pareto front). (num_data,)

    # get pareto optimal points
    pareto_idx = np.where(front_idx == 0)[0]
    non_pareto_idx = np.where(front_idx != 0)[0]
    print(pareto_idx.shape)
    print(non_pareto_idx.shape)

    # fitting
    popt, pcov = curve_fit(func, x_target_arr, y_target_arr)  # fit
    x_target_arr_sorted = np.sort(x_target_arr)  # for plot
    x_target_arr_sorted = x_target_arr_sorted[x_target_arr_sorted >= 0.7]
    y_target_fit_arr_sorted = func(x_target_arr_sorted, *popt)

    # plot
    plot_df = pd.DataFrame()
    plot_df[x_target] = x_target_arr[non_pareto_idx]
    plot_df[y_target] = y_target_arr[non_pareto_idx]

    # get idx for polymerization mechanisms
    non_pareto_polymerization_mechanism_idx_arr = polymerization_mechanism_idx_arr[non_pareto_idx]
    non_pareto_step_growth_idx = np.where((non_pareto_polymerization_mechanism_idx_arr >= 1) & (non_pareto_polymerization_mechanism_idx_arr <= 7))[0]
    non_pareto_chain_growth_addition_idx = np.where((non_pareto_polymerization_mechanism_idx_arr >= 8) & (non_pareto_polymerization_mechanism_idx_arr <= 9))[0]
    non_pareto_ring_opening_idx = np.where((non_pareto_polymerization_mechanism_idx_arr >= 10) & (non_pareto_polymerization_mechanism_idx_arr <= 15))[0]
    non_pareto_metathesis_idx = np.where((non_pareto_polymerization_mechanism_idx_arr >= 16) & (non_pareto_polymerization_mechanism_idx_arr <= 17))[0]

    # scatter plot
    g = sns.JointGrid()

    # step growth
    step_growth_df = plot_df.loc[non_pareto_step_growth_idx]
    ax = sns.scatterplot(x=x_target, y=y_target, data=step_growth_df, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0, color='#EB2329', linewidth=0.0)  # label='Step growth'

    # chain growth addition
    chain_growth_df = plot_df.loc[non_pareto_chain_growth_addition_idx]
    sns.scatterplot(x=x_target, y=y_target, data=chain_growth_df, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0, color='#2AB565', linewidth=0.0)  # label='Chain growth'

    # ring opening
    ring_opening_df = plot_df.loc[non_pareto_ring_opening_idx]
    sns.scatterplot(x=x_target, y=y_target, data=ring_opening_df, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0, color='#3E76BB', linewidth=0.0)  # label='Ring Opening'

    # metathesis
    metathesis_df = plot_df.loc[non_pareto_metathesis_idx]
    sns.scatterplot(x=x_target, y=y_target, data=metathesis_df, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0, color='#7B54A2', linewidth=0.0)  # label='Metathesis'

    # plot pareto
    pareto_x = x_target_arr[pareto_idx]
    pareto_y = y_target_arr[pareto_idx]
    plot_pareto_idx = np.argsort(pareto_x)
    plot_pareto_x, plot_pareto_y = pareto_x[plot_pareto_idx], pareto_y[plot_pareto_idx]

    # print molecules
    pareto_reaction_id = reaction_id_arr[pareto_idx][plot_pareto_idx]  # sort
    pareto_methyl_terminated_monomers = methyl_terminated_product_arr[pareto_idx][plot_pareto_idx]  # sort
    pareto_polymerization_idx = polymerization_mechanism_idx_arr[pareto_idx][plot_pareto_idx]  # sort

    for point_idx, (reaction_id, methyl_terminated_monomer, polymerization_idx) in enumerate(
            zip(pareto_reaction_id, pareto_methyl_terminated_monomers, pareto_polymerization_idx)):
        print(f'===== POINT {point_idx + 1} =====')
        print(f'Reaction id: {reaction_id}')
        print(f'Methyl_terminated_monomer: {methyl_terminated_monomer}')
        print(f'Polymerization_idx: {polymerization_idx}')

    # plot pareto
    pareto_step_growth_idx = np.where((pareto_polymerization_idx >= 1) & (pareto_polymerization_idx <= 7))[0]
    pareto_chain_growth_addition_idx = np.where((pareto_polymerization_idx >= 8) & (pareto_polymerization_idx <= 9))[0]
    pareto_ring_opening_idx = np.where((pareto_polymerization_idx >= 10) & (pareto_polymerization_idx <= 15))[0]
    pareto_metathesis_idx = np.where((pareto_polymerization_idx >= 16) & (pareto_polymerization_idx <= 17))[0]

    # plot
    plot_pareto_df = pd.DataFrame()
    plot_pareto_df[x_target] = plot_pareto_x
    plot_pareto_df[y_target] = plot_pareto_y

    # step growth
    pareto_step_growth_df = plot_pareto_df.loc[pareto_step_growth_idx]
    sns.scatterplot(x=x_target, y=y_target, data=pareto_step_growth_df, ax=g.ax_joint,
                         edgecolor='#231F20', alpha=0.75, legend=False, size=3.0, color='#EB2329', linewidth=1.0)

    # chain growth addition
    pareto_chain_growth_df = plot_pareto_df.loc[pareto_chain_growth_addition_idx]
    sns.scatterplot(x=x_target, y=y_target, data=pareto_chain_growth_df, ax=g.ax_joint,
                    edgecolor='#231F20', alpha=0.75, legend=False, size=3.0, color='#2AB565', linewidth=1.0)

    # ring opening
    pareto_ring_opening_df = plot_pareto_df.loc[pareto_ring_opening_idx]
    sns.scatterplot(x=x_target, y=y_target, data=pareto_ring_opening_df, ax=g.ax_joint,
                    edgecolor='#231F20', alpha=0.75, legend=False, size=3.0, color='#3E76BB', linewidth=1.0)

    # metathesis
    pareto_metathesis_df = plot_pareto_df.loc[pareto_metathesis_idx]
    sns.scatterplot(x=x_target, y=y_target, data=pareto_metathesis_df, ax=g.ax_joint,
                    edgecolor='#231F20', alpha=0.75, legend=False, size=3.0, color='#7B54A2', linewidth=1.0)

    # line plot
    g.ax_joint.plot(plot_pareto_x, plot_pareto_y, c='#231F20', ls='--', linewidth=1.0)

    # kde plots on the marginal axes
    total_plot_df = pd.concat([plot_df, plot_pareto_df], axis=0)
    sns.kdeplot(x=x_target, data=total_plot_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
    sns.kdeplot(y=y_target, data=total_plot_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.set_axis_labels(plot_name[1], plot_name[0], fontsize=16)

    # fitting
    g.ax_joint.plot(x_target_arr_sorted, y_target_fit_arr_sorted, color='#665A5E')

    # tick ranges. Modify the tick values to the original scales.
    if not manual_tick:
        xticks_range_to_plot_scaled = np.linspace(x_target_arr.min(), x_target_arr.max(), num=5, endpoint=True)
        if shift:
            xticks_range_to_plot_unscaled = (xticks_range_to_plot_scaled * x_target_std) + x_target_mean
        else:
            xticks_range_to_plot_unscaled = (xticks_range_to_plot_scaled * x_target_std)
        g.ax_joint.set_xticks(xticks_range_to_plot_scaled)  # ticks
        if abs(xticks_range_to_plot_unscaled.max()) >= 1:
            xtick_label = xticks_range_to_plot_unscaled.astype('int')
        else:
            xtick_label = [f'{value:.3f}' for value in xticks_range_to_plot_unscaled]
        g.ax_joint.set_xticklabels(xtick_label)  # ticks
    else:
        if shift:
            xticks_normalized = (np.array(manual_x_ticks) - x_target_mean) / x_target_std
        else:
            xticks_normalized = np.array(manual_x_ticks) / x_target_std
        g.ax_joint.set_xticks(xticks_normalized)  # ticks
        g.ax_joint.set_xticklabels(manual_x_ticks)  # tick labels

    if not manual_tick:
        yticks_range_to_plot_scaled = np.linspace(y_target_arr.min(), y_target_arr.max(), num=5, endpoint=True)
        if shift:
            yticks_range_to_plot_unscaled = (yticks_range_to_plot_scaled * y_target_std) + y_target_mean
        else:
            yticks_range_to_plot_unscaled = yticks_range_to_plot_scaled * y_target_std
        g.ax_joint.set_yticks(yticks_range_to_plot_scaled)  # ticks
        if abs(yticks_range_to_plot_unscaled.max()) >= 1:
            ytick_label = yticks_range_to_plot_unscaled.astype('int')
        else:
            ytick_label = [f'{value:.3f}' for value in yticks_range_to_plot_unscaled]
        g.ax_joint.set_yticklabels(ytick_label)  # ticks
    else:
        if shift:
            yticks_normalized = (np.array(manual_y_ticks) - y_target_mean) / y_target_std
        else:
            yticks_normalized = np.array(manual_y_ticks) / y_target_std
        g.ax_joint.set_yticks(yticks_normalized)  # ticks
        g.ax_joint.set_yticklabels(manual_y_ticks)  # tick labels

    # g.fig.legend(fontsize=16)
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(save_dir, f'{x_target}_{y_target}_pareto.png'), dpi=1200)
