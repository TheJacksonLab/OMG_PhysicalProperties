import os
import sys

EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, 'chemprop_evidential/chemprop'))  # chemprop evidential
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, 'nds-py'))  # pareto front

import pandas as pd

from chemprop.parsing import parse_train_args_script, parse_predict_args_script
from chemprop.train import cross_validate
from chemprop.utils import create_logger

from nds import ndomsort  # https://github.com/KernelA/nds-py/tree/master

"""Train a model using active learning on a dataset."""
from copy import deepcopy
import numpy as np

from chemprop.train import make_predictions

from scipy.stats.mstats import gmean

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity

# get taniomoto distance between batch training set and new data
radius, nBits = 2, 1024


def bulk_tanimoto_similarity(new_smi, existing_fps_list, radius=2, nBits=1024):
    """
    This function calculates a tanimoto similarity between a new smi and existing smis
    :param new_smi: a new molecule to calculate a tanimoto similarity
    :param existing_fps_list: fps lists for a tanimoto similarity calculation
    The following two lines will be performed outside of this function to avoid a duplicated process

            mol_obj_list = [Chem.MolFromSmiles(smi) for smi in existing_smi_list]
            fps_list = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits) for mol in mol_obj_list]

    :param radius: for morgan finger print calculation
    :param nBits: for morgan finger prints calculation

    :return: np.array [len(existing_smi_list)]
    """
    # fps of a new smi
    new_mol = Chem.MolFromSmiles(new_smi)
    new_fps = AllChem.GetMorganFingerprintAsBitVect(new_mol, radius=radius, nBits=nBits)

    return np.array(BulkTanimotoSimilarity(new_fps, existing_fps_list))


if __name__ == '__main__':
    ########### MODIFY ###########
    strategy = sys.argv[1]  # ['random', 'pareto_greedy_stable_sort', 'pareto_greedy_partial_stable_sort']
    gpu_index = sys.argv[2]  # [0, 1, 2, 3]
    torch_seed = 42  # torch seed to initialize weight & bias
    random_split_seed = sys.argv[3]  # random seed used to split data
    data_seed = 42  # fixed
    ########### MODIFY ###########

    # log space sampling
    active_learning_cycles = 10  # batch 0, 1, 2, ..., active_learning_cycles

    # active learning
    batch_list = range(active_learning_cycles)
    target_cols = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    for batch in batch_list:
        # set path
        DATA_DIR = './active_learning_data'
        CURRENT_BATCH = batch  # 0 is a initial random sampling
        TRAIN_DATA_FILE_PATH = os.path.join(DATA_DIR, f'{strategy}_train_batch_{CURRENT_BATCH}_random_seed_{random_split_seed}.csv')
        TEST_DATA_FILE_PATH = os.path.join(DATA_DIR, f'test_split_random_seed_{random_split_seed}.csv')  # -> held-out data
        SAVE_DIR = f'./{strategy}_check_point/chemprop_current_batch_{CURRENT_BATCH}'

        # features path
        TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, f'{strategy}_train_batch_{CURRENT_BATCH}.npy')  # batch training
        TOTAL_TRAIN_FEATURES_PATH = f'/home/sk77/PycharmProjects/omg_database_publication/qm9_active_learning/qm9_rdkit_features/train_features_random_seed_{random_split_seed}.npy'
        TEST_FEATURES_PATH = f'/home/sk77/PycharmProjects/omg_database_publication/qm9_active_learning/qm9_rdkit_features/test_features_random_seed_{random_split_seed}.npy'

        # load data
        df_train_target = pd.read_csv(TRAIN_DATA_FILE_PATH)
        df_train_target_mask = df_train_target['batch'] >= 0.0  # only include chosen samples
        df_train_target = df_train_target[df_train_target_mask]
        df_train_target = df_train_target.drop(columns=['batch'], axis=1)

        # load features
        total_train_features = np.load(TOTAL_TRAIN_FEATURES_PATH)
        df_train_features = total_train_features[df_train_target_mask.to_list()]

        # drop cols
        if batch != 0:  # contains preds
            df_train_target = df_train_target.drop(columns=[target for target in target_cols], axis=1)  # pred values
            df_train_target = df_train_target.drop(columns=['uncertainty_' + target for target in target_cols], axis=1)  # uncertainty values
            df_train_target = df_train_target.drop(columns=['std_' + target for target in target_cols], axis=1)  # std values
            true_col_to_be_replaced = ['true_' + target for target in target_cols]
            df_train_target = df_train_target.rename(columns={key: value for key, value in zip(true_col_to_be_replaced, target_cols)})

        # save
        TRAIN_DATA_FILE_CHEMPROP_PATH = os.path.join(DATA_DIR, f'{strategy}_train_batch_{CURRENT_BATCH}_chemprop.csv')
        df_train_target.to_csv(TRAIN_DATA_FILE_CHEMPROP_PATH, index=False)
        np.save(TRAIN_FEATURES_PATH, df_train_features)

        # General arguments
        train_arguments = [
            '--gpu', f'{gpu_index}',
            '--data_path', f'{TRAIN_DATA_FILE_CHEMPROP_PATH}',
            '--save_dir', f'{SAVE_DIR}',
            '--dataset_type', 'regression',
            '--separate_val_path', f'{TEST_DATA_FILE_PATH}',  # the same as test
            '--separate_test_path', f'{TEST_DATA_FILE_PATH}',
            '--num_folds', '1',  # cross validation
            '--torch_seed', f'{torch_seed}',  # torch weight initialization
            '--seed', f'{data_seed}',  # used for data splitting + np.random seed and random.seed + pytorch seed
            '--metric', 'rmse',
            '--log_frequency', '10',
            '--show_individual_scores',

            # features
            '--features_path', TRAIN_FEATURES_PATH,  # batch train features path
            '--separate_val_features_path', TEST_FEATURES_PATH,  # the same as test
            '--separate_test_features_path', TEST_FEATURES_PATH,  # Path to file with features for separate test set
            '--no_features_scaling',  # Turn off scaling of features -> already scaled
        ]

        # Training arguments
        train_arguments.extend([
            '--epoch', '100',  # Number of epochs to run
            '--batch_size', '50',
            '--final_lr', '1e-4',  # Final learning rate
            '--init_lr', '1e-4',  # Initial learning rate
            '--max_lr', '1e-3',  # Maximum learning rate
            '--warmup_epochs', '2',
            # Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`. Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
            '--stokes_balance', '1',  # ?
        ])

        # Model arguments
        train_arguments.extend([
            '--ensemble_size', '1',  # Number of models in ensemble
            '--threads', '1',  # Number of parallel threads to spawn ensembles in. 1 thread trains in serial.
            '--hidden_size', '300',  # Dimensionality of hidden layers in MPN
            '--bias',  # Whether to add bias to linear layers
            '--depth', '3',  # Number of message passing steps
            '--activation', 'ReLU',  # ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
            '--ffn_num_layers', '2',  # Number of layers in FFN after MPN encoding.
        ])

        # Confidence arguments
        train_arguments.extend([
            '--confidence', 'evidence',  # Measure confidence values for the prediction.
            '--regularizer_coeff', '0.2',  # Coefficient to scale the loss function regularizer
            '--new_loss',  # Use the new evidence loss with the model,
            '--no_dropout_inference',  # If true, don't use dropout for mean inference
            '--use_entropy',  # If true, also output the entropy for each prediction with the model
            '--save_confidence', 'confidence.txt',  # Measure confidence values for the prediction.
            '--confidence_evaluation_methods', 'cutoff',
        ])

        # train - Common arguments - sed in both :class:`TrainArgs` and :class:`PredictArgs`
        train_arguments.extend([
            '--batch_size', '50',  # Batch size
        ])

        args = parse_train_args_script(train_arguments)
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
        mean_score, std_score = cross_validate(args, logger)

        # active sampling
        ####### MODIFY #######
        CURRENT_ITERATION_TRAIN_DATA_PATH = os.path.join(DATA_DIR, f'{strategy}_train_batch_{CURRENT_BATCH}_random_seed_{random_split_seed}.csv')
        TRAIN_DATA_FILE_PATH = os.path.join(DATA_DIR, f'train_split_random_seed_{random_split_seed}.csv')
        CHECKPOINT_DIR = f'./{strategy}_check_point/chemprop_current_batch_{CURRENT_BATCH}'
        NP_RANDOM_SEED = 42  # for a random choice
        number_of_samples_to_add = 100
        ####### MODIFY #######

        # evaluate models on a training data
        pred_arguments = [
            '--gpu', f'{gpu_index}',
            '--test_path', f'{TRAIN_DATA_FILE_PATH}',  # Path to CSV file containing testing data for which predictions will be made
            '--preds_path', f'NULL',  # No effects
            '--checkpoint_dir', f'{CHECKPOINT_DIR}',
            '--batch_size', '50',

            # features
            '--features_path', TOTAL_TRAIN_FEATURES_PATH,  # total train features
            '--no_features_scaling',  # Turn off scaling of features -> already scaled
        ]
        args = parse_predict_args_script(pred_arguments)
        df_result, df_std, scaler = make_predictions(args)

        # active learning based on a model predictive uncertainty
        std_arr = df_std.to_numpy()  # unscaled uncertainty

        # normalize
        scaler_std = scaler.stds
        scaler_std_tile = np.tile(scaler_std, reps=(std_arr.shape[0], 1))
        scaled_std_arr = std_arr / scaler_std_tile  # scaled uncertainty
        mean_std = np.mean(scaled_std_arr, axis=1)  # mean std of scaled uncertainty

        # Coley's approach -> mean of unscaled uncertainties
        # mean_std = np.mean(std_arr, axis=1)

        # get already chosen index
        current_iteration_train_data = pd.read_csv(CURRENT_ITERATION_TRAIN_DATA_PATH)
        total_number_of_train_data = current_iteration_train_data.shape[0]
        train_subset_inds = current_iteration_train_data['batch'].to_numpy().copy()  # copy is needed
        train_subset_inds *= -1  # data with the value of "1" have not been chosen yet. Data with the value less than or equal to 0 have been chosen.
        train_subset_inds = np.clip(train_subset_inds, a_min=0, a_max=None)  # value is now either 1 (not chosen yet) or 0 (already chosen)

        # active learning based on a model predictive uncertainty
        if strategy in ['explorative_greedy', 'explorative_diverse']:
            per_sample_prob = deepcopy(mean_std)  # explorative_greedy or explorative_diverse
        elif strategy == 'random':
            per_sample_prob = np.ones_like(mean_std)  # random
        elif strategy in ['pareto_greedy_stable_sort', 'pareto_diverse']:  # find a pareto front -> maximize each property uncertainty
            # remove train examples first
            train_subset_inds_mask = train_subset_inds.copy().reshape(-1, 1)  # reshape
            train_subset_inds_mask = np.tile(train_subset_inds_mask, (1, scaled_std_arr.shape[1]))
            seq_for_pareto = scaled_std_arr * train_subset_inds_mask
            seq_for_pareto *= -1  # multiply (-1) to minimize seq value (maximize uncertainty)

            front_idx = np.array(ndomsort.non_domin_sort(seq_for_pareto, only_front_indices=True))  # return a front index (0 -> optimal pareto front)

            # print Pareto components
            unique, counts = np.unique(front_idx, return_counts=True)  # count unique values
            value_count_arr = np.asarray((unique, counts), dtype=int).T
            print(value_count_arr)

            front_idx_from_1 = front_idx + 1  # return a front index (1 -> optimal pareto front)
            per_sample_prob = 1 / front_idx_from_1  # (all positive values) the larger, the more optimal pareto front
        elif strategy == 'rank_mean':
            rank_arr = np.argsort(scaled_std_arr, axis=0)  # sort smallest to largest per each target property (axis=1). rank arr. The same shape as scaled_std_arr
            per_sample_prob = rank_arr.mean(axis=1)  # [number of molecules,]. The larger, the more uncertain. Arithmetic mean to highlight the large value.
        elif strategy == 'pareto_greedy_partial_stable_sort':
            # exclude already chosen molecules
            seq = mean_std * train_subset_inds  # make zero mean uncertainty for already chosen molecules

            # filter unchosen molecules -> large uncertainty
            seq_idx_sorted = np.argsort(seq)  # smallest to largest of scaled uncertainty
            number_of_unchosen_molecules = np.count_nonzero(seq)  # count unchosen molecules -> non zero mean_std
            # seq_large_uncertainty_idx_sorted = seq_idx_sorted[-int(number_of_unchosen_molecules / 2):]  # choose unchosen molecules with larget uncertainty than median
            seq_large_uncertainty_idx_sorted = seq_idx_sorted[-int(number_of_unchosen_molecules / 8):]  # choose unchosen molecules with larget uncertainty than median

            # construct a mask
            uncertainty_filter_mask = np.zeros_like(seq)
            uncertainty_filter_mask[seq_large_uncertainty_idx_sorted] = 1.0  # make value 1.0 for molecule candidates
            uncertainty_filter_mask = uncertainty_filter_mask.reshape(-1, 1)  # reshape
            uncertainty_filter_mask = np.tile(uncertainty_filter_mask, (1, scaled_std_arr.shape[1]))
            seq_for_pareto = scaled_std_arr * uncertainty_filter_mask

            # seq to filter
            seq_for_pareto *= -1  # multiply (-1) to minimize seq value (maximize uncertainty)
            front_idx = np.array(ndomsort.non_domin_sort(seq_for_pareto, only_front_indices=True))  # return a front index (0 -> optimal pareto front)

            # print Pareto components
            unique, counts = np.unique(front_idx, return_counts=True)  # count unique values
            value_count_arr = np.asarray((unique, counts), dtype=int).T
            print(value_count_arr)

            front_idx_from_1 = front_idx + 1  # return a front index (1 -> optimal pareto front)
            per_sample_prob = 1 / front_idx_from_1  # (all positive values) the larger, the more optimal pareto front

        else:
            raise ValueError('Check your active learning strategy!')

        # per sample prob.
        per_sample_prob = per_sample_prob * train_subset_inds
        per_sample_prob = per_sample_prob / per_sample_prob.sum()  # probability

        # greedy choice, just pick the highest probability indices
        if strategy in ['explorative_greedy', 'pareto_greedy_stable_sort', 'rank_mean', 'pareto_greedy_partial_stable_sort']:
            inds_sorted = np.argsort(per_sample_prob, kind='stable')  # smallest to largest. (choose non-dominated pareto optimal sets for pareto_greedy)
            train_inds_to_add = inds_sorted[-number_of_samples_to_add:]  # grab the last k inds (greedy choice)

        elif strategy == 'random':
            np.random.seed(NP_RANDOM_SEED)
            train_inds_to_add = np.random.choice(total_number_of_train_data, size=number_of_samples_to_add, p=per_sample_prob, replace=False)

        elif strategy in ['explorative_diverse', 'pareto_diverse']:
            inds_sorted = np.argsort(per_sample_prob)  # smallest to largest
            multiply_const_for_diverse_sampling = 3
            candidate_train_inds_to_add = inds_sorted[-multiply_const_for_diverse_sampling * number_of_samples_to_add:]  # grab the last k inds * constant
            existing_smi_list = [current_iteration_train_data['smi'][idx] for idx in range(total_number_of_train_data) if train_subset_inds[idx] == 0]  # chosen
            mol_obj_list = [Chem.MolFromSmiles(smi) for smi in existing_smi_list]
            existing_fps_list = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits) for mol in mol_obj_list]
            geomean_arr = np.zeros(shape=candidate_train_inds_to_add.shape, dtype=float)  # changed from np.zeros(shape=multiply_const_for_diverse_sampling * number_of_samples_to_add) -> when candidate_train_inds_to_add doesn't have enough samples.
            for arr_idx, idx in enumerate(candidate_train_inds_to_add):
                new_smi = current_iteration_train_data['smi'][idx]
                tanimoto_arr = bulk_tanimoto_similarity(new_smi=new_smi, existing_fps_list=existing_fps_list, radius=radius, nBits=nBits)
                geomean = gmean(tanimoto_arr)  # to reduce an effect from a large similarity
                geomean_arr[arr_idx] = geomean
            candidate_inds_sorted = np.argsort(geomean_arr)  # smallest to largest
            train_inds_to_add = candidate_train_inds_to_add[candidate_inds_sorted[: number_of_samples_to_add]]  # samples with a small similarity

        else:
            raise ValueError('Check your active learning strategy!')

        # create encoding vector with the information of the current batch
        active_mask_arr_to_append = np.zeros(shape=(total_number_of_train_data,))  # default 0
        for idx in train_inds_to_add:
            active_mask_arr_to_append[idx] = CURRENT_BATCH + 1 + 1  # next batch = current batch + 1. Additional 1 comes from a default -1 for not being chosen data. Not overlapping with already chosen one.

        # save
        df_result['batch'] = current_iteration_train_data['batch'].to_numpy() + active_mask_arr_to_append
        save_path = os.path.join(DATA_DIR, f'{strategy}_train_batch_{CURRENT_BATCH + 1}_random_seed_{random_split_seed}.csv')
        df_result.to_csv(save_path, index=False)
