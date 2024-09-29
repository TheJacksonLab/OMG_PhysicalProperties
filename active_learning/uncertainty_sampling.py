import os
import sys
EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './nds-py'))  # search pareto front
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './chemprop_evidential/chemprop'))  # chemprop evidential

import argparse
import pandas as pd
import pickle

from nds import ndomsort  # https://github.com/KernelA/nds-py/tree/master

"""Train a model using active learning on a dataset."""
import numpy as np

from pathlib import Path


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='argument parser for uncertainty sampling')

    # Add positional arguments
    # parser.add_argument('--data_path', type=str, help='Data (.csv) file path to run', default=None)
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning. ["random", "pareto_greedy"]', default=None)
    parser.add_argument('--current_batch', type=int, help='Current batch for active learning', default=None)
    parser.add_argument('--pareto_partial_ratio_n_th', type=int, help='Pareto partial filter ratio. "n" means 1/n', default=None)

    return parser.parse_args()  # Parse the arguments


def load_object(filepath):
    """
    This function loads an object using pickle
    """
    with open(filepath, "rb") as input_file:
        object = pickle.load(input_file)
        input_file.close()
    return object

############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################

if __name__ == '__main__':
    """
    This script estimate uncertainties.
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred.py
    4) fold_plot_learning_curve.py
    5) uncertainty_pred.py
    6) uncertainty_sampling.py
    -> repeat to 1)
    """
    ############################## ARGUMENTS ##############################
    # argument parser
    args = get_args()
    strategy = args.active_learning_strategy  # ['pareto_greedy']
    current_batch = args.current_batch  # 0 for initial training data
    pareto_partial_ratio_n_th = args.pareto_partial_ratio_n_th

    # debug
    # strategy = 'pareto_greedy'  # ['pareto_greedy']
    # current_batch = 0  # 0 for initial training data
    # pareto_partial_ratio_n_th = 16
    ############################## ARGUMENTS ##############################

    # except for (1) molecular_weight, (2) logP, (3) qed, (4) TPSA, (5) normalized_monomer_phi, (6) normalized_backbone_phi
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
         'polarizability_Boltzmann_average', ],

        ['s1_energy_Boltzmann_average',
         'dominant_transition_energy_Boltzmann_average',
         'dominant_transition_oscillator_strength_Boltzmann_average',
         't1_energy_Boltzmann_average', ],

        ['chi_parameter_water_mean',
         'chi_parameter_ethanol_mean',
         'chi_parameter_chloroform_mean', ]
    ]

    ############################## GET UNCERTAINTY DATAFRAME ##############################
    NUM_GNNS = 4
    TOTAL_TRAIN_DATA_WITH_BATCH = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_labeled.csv')  # 12,870,976 polymers (total train polymers) with the batch column
    UNCERTAINTY_DIR = os.path.join(DATA_DIR, 'active_learning', strategy, 'uncertainty_pred')
    TOTAL_UNCERTAINTY_CSV_PATH_TO_USE = os.path.join(UNCERTAINTY_DIR, f'current_batch_{current_batch}', 'total_std_results.csv')  # 12,870,976 polymers (total train polymers) with 21 columns (19 property stds + methyl terminate polymer + reaction id)

    # get data
    df_total_train_data_with_batch = pd.read_csv(TOTAL_TRAIN_DATA_WITH_BATCH)  # 12,870,976 polymers (total train polymers) with the batch column
    df_total_std = pd.read_csv(TOTAL_UNCERTAINTY_CSV_PATH_TO_USE)  # 12,870,976 polymers (total train polymers) with 21 columns (19 property stds + methyl terminate polymer + reaction id)
    print('Is there any nan value in a std .csv file?')
    print(df_total_std.isnull().values.any())  # debug
    print(f'(1/n) -> n = {pareto_partial_ratio_n_th} pareto filtering scheme')

    # get std arr
    std_cols_list = [f'std_{col}' for gnn_idx in range(NUM_GNNS) for col in target_cols_list[gnn_idx]]  # length 19
    std_arr = df_total_std[std_cols_list].to_numpy()  # std arr. unscaled uncertainties. (12870976, 19)
    print('Are uncertainties positive?')
    check_arr = std_arr > 0.0
    print(check_arr.all())
    df_total_std['batch'] = df_total_train_data_with_batch['batch'].to_numpy().copy()  # add batch information. (12870976, 21 + 1). NEEDS TO ADD to_numpy(). nd_array copy -> default "deep copy". Changes to the copied one don't affect the original one.
    ############################## GET UNCERTAINTY DATAFRAME ##############################

    ############################## ACTIVE SAMPLING ##############################
    NP_RANDOM_SEED = 42  # for a random choice (random sampling)
    number_of_samples_to_add = 10000  # linear space

    # save dir
    SAMPLING_SAVE_DIR = Path(os.path.join(DATA_DIR, 'active_learning', strategy, 'uncertainty_sampling', f'current_batch_{current_batch}'))
    SAMPLING_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # scaler
    scaler_std = list()
    SCALER_SAVE_DIR = os.path.join(DATA_DIR, 'active_learning', strategy, 'uncertainty_pred', f'current_batch_{current_batch}')
    for gnn_idx in range(NUM_GNNS):
        scaler = load_object(os.path.join(SCALER_SAVE_DIR, f'gnn_{gnn_idx}', 'scaler', f'smi_batch_idx_0_scaler.pkl'))  # the same for "smi_batch_idx_0", "smi_batch_idx_1", ...
        scaler_std.extend(scaler.stds.tolist())  # convert np.array to list
    scaler_std = np.array(scaler_std)  # convert list to np.array. (19,)

    # normalize
    scaler_std_tile = np.tile(scaler_std, reps=(std_arr.shape[0], 1))  # (12870976, 19)
    scaled_std_arr = std_arr / scaler_std_tile  # scaled uncertainty.  # (12870976, 19)
    mean_std = np.mean(scaled_std_arr, axis=1)  # mean std of scaled uncertainty. (12870976,)

    # Coley's approach -> mean of "unscaled" uncertainties
    # mean_std = np.mean(std_arr, axis=1)

    # get already chosen index
    total_number_of_train_data = df_total_std.shape[0]  # 12870976
    train_subset_inds = df_total_std['batch'].to_numpy().copy()  # copy is needed. np -> deepcopy.
    train_subset_inds *= -1  # data with the value of "1" have not been chosen yet. Data with the value less than or equal to 0 have been chosen.
    train_subset_inds = np.clip(train_subset_inds, a_min=0, a_max=None)  # value is now either 1 (not chosen yet) or 0 (already chosen)

    # active learning based on a model predictive uncertainty
    if strategy == 'random':
        per_sample_prob = np.ones_like(mean_std)  # random
    elif strategy == 'pareto_greedy':  # find a pareto front -> maximize each property uncertainty
        # ===== Faster version ===== -> pareto greedy partial (after filtering with scaled mean uncertainty)
        # exclude already chosen molecules
        seq = mean_std * train_subset_inds  # make zero mean uncertainty for already chosen molecules. (12870976,)

        # filter unchosen molecules -> large uncertainty
        seq_idx_sorted = np.argsort(seq)  # smallest to largest of scaled uncertainty
        number_of_unchosen_molecules = np.count_nonzero(seq)  # count unchosen molecules -> non zero mean_std
        seq_large_uncertainty_idx_sorted = seq_idx_sorted[-int(number_of_unchosen_molecules / pareto_partial_ratio_n_th):]  # choose unchosen molecules with large uncertainty (1/n)

        # save partial space cutting results
        df_partial_cut = df_total_train_data_with_batch.copy()  # default deep copy for pandas.
        partial_cut_to_append = np.zeros(shape=(total_number_of_train_data,))  # default 0. (12870976,)
        for idx in seq_large_uncertainty_idx_sorted:  # (number_of_samples_to_add (10,000),)
            partial_cut_to_append[idx] = current_batch + 1 + 1  # next batch = current batch + 1. Additional 1 comes from a default -1 for not being chosen data. Not overlapping with already chosen one.
        df_partial_cut['batch'] = df_total_train_data_with_batch['batch'].to_numpy().copy() + partial_cut_to_append
        save_path = os.path.join(SAMPLING_SAVE_DIR, f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_partial_cut.csv')
        df_partial_cut.to_csv(save_path, index=False)

        # construct a mask
        uncertainty_filter_mask = np.zeros_like(seq)  # (12870976,)
        uncertainty_filter_mask[seq_large_uncertainty_idx_sorted] = 1.0  # make value 1.0 for molecule candidates

        # seq to filter
        uncertainty_filter_mask = uncertainty_filter_mask.reshape(-1, 1)  # (12870976, 1)
        uncertainty_filter_mask = np.tile(uncertainty_filter_mask, (1, scaled_std_arr.shape[1]))  # (12870976, 19)
        seq_for_pareto = scaled_std_arr * uncertainty_filter_mask
        seq_for_pareto *= -1  # multiply (-1) to minimize seq value (maximize uncertainty)
        front_idx = np.array(ndomsort.non_domin_sort(seq_for_pareto, only_front_indices=True))  # return a front index (0 -> optimal pareto front)

        # print the front components
        unique, counts = np.unique(front_idx, return_counts=True)  # count unique values
        value_count_arr = np.asarray((unique, counts), dtype=int).T
        print(value_count_arr)

        # get per sample probability
        front_idx_from_1 = front_idx + 1  # return a front index (1 -> optimal pareto front)
        per_sample_prob = 1 / front_idx_from_1  # (all positive values) the larger, the more optimal pareto front

    else:
        raise ValueError('Check your active learning strategy!')

    # per sample normalization
    per_sample_prob = per_sample_prob * train_subset_inds
    per_sample_prob = per_sample_prob / per_sample_prob.sum()  # probability

    # greedy choice, just pick the highest probability indices
    if strategy == 'random':
        np.random.seed(NP_RANDOM_SEED)
        train_inds_to_add = np.random.choice(total_number_of_train_data, size=number_of_samples_to_add, p=per_sample_prob, replace=False)

    elif strategy == 'pareto_greedy':
        # stable sort to make Pareto greedy partial (n=1) converging to Pareto greedy without filter
        inds_sorted = np.argsort(per_sample_prob, kind='stable')  # smallest to largest. (choose non-dominated pareto optimal sets for pareto_greedy)

        # count the number of molecules at the first Pareto optimal front
        max_sample_prob = per_sample_prob.max()
        comparison = per_sample_prob == max_sample_prob
        number_of_molecules_at_the_first_pareto = np.count_nonzero(comparison)
        if number_of_molecules_at_the_first_pareto > number_of_samples_to_add:  # if more molecules in the first pareto front
            train_inds_to_add = inds_sorted[-number_of_molecules_at_the_first_pareto:]  # returns all of the first pareto front molecules.
        else:
            train_inds_to_add = inds_sorted[-number_of_samples_to_add:]  # grab the last k inds (greedy choice)  -> if less than the number of molecules to sample.

    else:
        raise ValueError('Check your active learning strategy!')

    # create encoding vector with the information of the current batch
    active_mask_arr_to_append = np.zeros(shape=(total_number_of_train_data,))  # default 0. (12870976,)
    for idx in train_inds_to_add:  # (number_of_samples_to_add (10,000),)
        active_mask_arr_to_append[idx] = current_batch + 1 + 1  # next batch = current batch + 1. Additional 1 comes from a default -1 for not being chosen data. Not overlapping with already chosen one.

    # save
    df_result = df_total_train_data_with_batch.copy()  # default deep copy for pandas.
    df_result['batch'] = df_total_train_data_with_batch['batch'].to_numpy().copy() + active_mask_arr_to_append
    save_path = os.path.join(SAMPLING_SAVE_DIR, f'OMG_train_batch_{current_batch + 1}_labeled_pareto_partial_{pareto_partial_ratio_n_th}th_first_pareto.csv')
    df_result.to_csv(save_path, index=False)
