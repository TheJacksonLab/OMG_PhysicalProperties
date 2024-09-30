import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path

EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './chemprop_evidential/chemprop'))  # chemprop evidential -> to load pickle.


def load_object(filepath):
    """
    This function loads an object using pickle
    """
    with open(filepath, "rb") as input_file:
        object = pickle.load(input_file)
        input_file.close()
    return object


NUM_GNNS = 4
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
        'polarizability_Boltzmann_average', ],  # without partial charges

    ['s1_energy_Boltzmann_average',
     'dominant_transition_energy_Boltzmann_average',
     'dominant_transition_oscillator_strength_Boltzmann_average',
     't1_energy_Boltzmann_average', ],

    ['chi_parameter_water_mean',
     'chi_parameter_ethanol_mean',
     'chi_parameter_chloroform_mean', ]
]

DATA_DIR = '/home/sk77/PycharmProjects/omg_database_publication/data'
PARETO_GREEDY_DIR = os.path.join(DATA_DIR, 'active_learning/pareto_greedy')
UNCERTAINTY_PRED_DIR = os.path.join(PARETO_GREEDY_DIR, 'uncertainty_pred')
UNCERTAINTY_SAMPLING_DIR = os.path.join(PARETO_GREEDY_DIR, 'uncertainty_sampling')


if __name__ == '__main__':
    """
    This script samples molecules from the first Pareto and plot results.
    """
    ### MODIFY ###
    current_batch = 2  # initial train batch -> 0.
    pareto_partial_ratio_n_th = 5
    plot = True
    plot_x_property_idx = 0  # pareto front plot
    plot_y_property_idx = 1  # pareto front plot

    # sampling from the first pareto
    sample = True  # bool to perform random sampling or not
    sample_num = 5000
    NP_RANDOM_SEED = 42  # for a random choice
    ### MODIFY ###

    # load df
    uncertainty_csv_path = os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_first_pareto.csv')  # plus 1 from current batch
    df = pd.read_csv(uncertainty_csv_path)  # -> 12,870,976 polymers

    # load df partial cut
    partial_cut_path = os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_partial_cut.csv')  # plus 1 from current batch
    df_partial_cut = pd.read_csv(partial_cut_path)
    partial_cut_reaction_id = df_partial_cut[df_partial_cut['batch'] == current_batch + 1]['reaction_id'].to_numpy().copy()

    # index per batch - check the number of molecules
    for batch in range(0, current_batch + 1 + 1):
        # batch molecule idx
        print(f'=== BATCH {batch} ===', flush=True)
        df_batch = df[df['batch'] == batch]
        print(df_batch.shape, flush=True)

    # sample "sample_num" molecules among the pareto front
    np.random.seed(NP_RANDOM_SEED)  # seed
    first_pareto_molecule_reaction_id = df[df['batch'] == current_batch + 1]['reaction_id'].to_numpy().copy()
    if sample:
        sampled_first_pareto_molecule_reaction_id = np.random.choice(first_pareto_molecule_reaction_id, size=sample_num, replace=False)  # (sample_num,)
    else:  # no sample
        sampled_first_pareto_molecule_reaction_id = first_pareto_molecule_reaction_id.copy()

    # get idx to modify the idx
    df_sampled_pareto = df.copy()
    label_first_pareto = df_sampled_pareto['reaction_id'].isin(first_pareto_molecule_reaction_id).to_numpy(dtype=int).copy()  # (12870976,). "1" indicates the label of the first pareto. "0" doesn't
    label_sampled_first_pareto = df_sampled_pareto['reaction_id'].isin(sampled_first_pareto_molecule_reaction_id).to_numpy(dtype=int).copy()  # (12870976,). "1" indicates the label of the sampled molecules among first pareto. "0" doesn't. (sample_num,)
    label_exclude_from_first_pareto = label_first_pareto - label_sampled_first_pareto  # 1 -> needs to be excluded

    # modify the labeled .csv for the following calculations
    df_sampled_pareto['batch'] = df_sampled_pareto['batch'].to_numpy().copy() - label_exclude_from_first_pareto * (current_batch + 1 + 1)  # revert to -1. (12870976)
    print(df_sampled_pareto['batch'].value_counts())

    # newly added molecule
    df_newly_added = df_sampled_pareto[df_sampled_pareto['batch'] == current_batch + 1]  # (sample_num,)

    # save newly added molecules csv
    save_path = os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_sampled_from_first_pareto_only_newly_added.csv')
    df_newly_added.to_csv(save_path, index=False)

    # save a batch csv that will be used.
    save_path = os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_sampled_from_first_pareto.csv')
    df_sampled_pareto.to_csv(save_path, index=False)

    if plot:
        # load total std df
        print('Loading uncertainty dataframe ...', flush=True)
        df_std_path = os.path.join(UNCERTAINTY_PRED_DIR, f'current_batch_{current_batch}', 'total_std_results.csv')
        df_std = pd.read_csv(df_std_path)

        # first pareto df
        df_std_first_pareto = df_std[df_std['reaction_id'].isin(first_pareto_molecule_reaction_id)]

        # sampled first pareto df
        df_std_sampled_first_pareto = df_std[df_std['reaction_id'].isin(sampled_first_pareto_molecule_reaction_id)]  # (sample_num,)

        # pareto partial cut df
        df_std_partial_cut = df_std[df_std['reaction_id'].isin(partial_cut_reaction_id)]

        # get std arr
        std_cols_list = [f'std_{col}' for gnn_idx in range(NUM_GNNS) for col in target_cols_list[gnn_idx]]  # length 19
        total_std_arr = df_std[std_cols_list].to_numpy()  # total train
        first_pareto_std_arr = df_std_first_pareto[std_cols_list].to_numpy()  # first pareto
        sampled_first_pareto_std_arr = df_std_sampled_first_pareto[std_cols_list].to_numpy()  # sampled from first pareto
        partial_cut_std_arr = df_std_partial_cut[std_cols_list].to_numpy()  # partial cut

        # scaler
        print('Scaling ...', flush=True)
        scaler_std = list()
        SCALER_SAVE_DIR = os.path.join(DATA_DIR, 'active_learning', 'pareto_greedy', 'uncertainty_pred', f'current_batch_{current_batch}')
        for gnn_idx in range(NUM_GNNS):
            scaler = load_object(os.path.join(SCALER_SAVE_DIR, f'gnn_{gnn_idx}', 'scaler', f'smi_batch_idx_0_scaler.pkl'))  # the same for "smi_batch_idx_0", "smi_batch_idx_1", ...
            scaler_std.extend(scaler.stds.tolist())  # convert np.array to list
        scaler_std = np.array(scaler_std)  # convert list to np.array. (19,)

        # scale total arr
        scaler_std_tile = np.tile(scaler_std, reps=(total_std_arr.shape[0], 1))
        scaled_total_std_arr = total_std_arr / scaler_std_tile  # scaled uncertainty.

        # scale a first pareto arr
        scaler_std_tile = np.tile(scaler_std, reps=(first_pareto_std_arr.shape[0], 1))
        scaled_first_pareto_std_arr = first_pareto_std_arr / scaler_std_tile  # scaled uncertainty.

        # scale a sampled pareto arr
        scaler_std_tile = np.tile(scaler_std, reps=(sampled_first_pareto_std_arr.shape[0], 1))
        scaled_sampled_first_pareto_std_arr = sampled_first_pareto_std_arr / scaler_std_tile  # scaled uncertainty.

        # scale a partial cut
        scaler_std_tile = np.tile(scaler_std, reps=(partial_cut_std_arr.shape[0], 1))
        scaled_partial_cut_std_arr = partial_cut_std_arr / scaler_std_tile  # scaled uncertainty.

        # plot
        print('Plot ...', flush=True)
        total_plot_x_std = scaled_total_std_arr[:, plot_x_property_idx]
        total_plot_y_std = scaled_total_std_arr[:, plot_y_property_idx]
        first_pareto_plot_x_std = scaled_first_pareto_std_arr[:, plot_x_property_idx]
        first_pareto_plot_y_std = scaled_first_pareto_std_arr[:, plot_y_property_idx]
        sampled_first_pareto_plot_x_std = scaled_sampled_first_pareto_std_arr[:, plot_x_property_idx]
        sampled_first_pareto_plot_y_std = scaled_sampled_first_pareto_std_arr[:, plot_y_property_idx]
        partial_cut_plot_x_std = scaled_partial_cut_std_arr[:, plot_x_property_idx]
        partial_cut_plot_y_std = scaled_partial_cut_std_arr[:, plot_y_property_idx]

        # 1) total points vs partial cut
        plt.figure(figsize=(6, 6))
        plt.plot(total_plot_x_std, total_plot_y_std, 'ko', label='Total', alpha=0.5)
        plt.plot(partial_cut_plot_x_std, partial_cut_plot_y_std, 'co', label='Partial', alpha=0.5)
        plt.xlabel(f'Property idx {plot_x_property_idx} uncertainty (scaled)', fontsize=14)
        plt.ylabel(f'Property idx {plot_y_property_idx} uncertainty (scaled)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # save fig
        fig_dir = Path(os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', 'figure'))
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'pareto_front_partial_{pareto_partial_ratio_n_th}th_property_{plot_x_property_idx}_{plot_y_property_idx}_total_partial.png'))
        plt.close()

        # 2) partial and pareto cut
        plt.figure(figsize=(6, 6))
        plt.plot(partial_cut_plot_x_std, partial_cut_plot_y_std, 'co', label='Partial', alpha=0.5)
        plt.plot(first_pareto_plot_x_std, first_pareto_plot_y_std, 'go', label='First Pareto among partial')
        plt.plot(sampled_first_pareto_plot_x_std, sampled_first_pareto_plot_y_std, 'ro', label='Sampled from First Pareto')
        plt.xlabel(f'Property idx {plot_x_property_idx} uncertainty (scaled)', fontsize=14)
        plt.ylabel(f'Property idx {plot_y_property_idx} uncertainty (scaled)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # save fig
        fig_dir = Path(os.path.join(UNCERTAINTY_SAMPLING_DIR, f'current_batch_{current_batch}', 'figure'))
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.legend(fontsize=12, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'pareto_front_partial_{pareto_partial_ratio_n_th}th_property_{plot_x_property_idx}_{plot_y_property_idx}_partial_pareto.png'))
        plt.close()
