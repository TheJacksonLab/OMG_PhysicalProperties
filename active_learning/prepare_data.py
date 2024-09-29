import os
import pandas as pd
import numpy as np

from pathlib import Path
from argparse import ArgumentParser


def get_args():
    # Create an argument parser
    parser = ArgumentParser(description='Prepare data before training chemprop evidential')

    # Add positional arguments
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning. ["pareto_greedy"]', default=None)
    parser.add_argument('--current_batch', type=int, help='Current batch for active learning', default=None)

    return parser.parse_args()  # Parse the arguments

############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
PROPERTY_DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation/calculation_results_csv')
FEATURES_NPZ_PATH = '/home/sk77/PycharmProjects/omg_database_publication/data/rdkit_features/rdkit_features.npz'
############################## TARGET DIR & PATH ##############################

############################## TARGET COLUMNS ##############################
property_columns_Boltzmann = \
    ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity',
     'molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi'] + \
    ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
    ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy']
property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']
############################## TARGET COLUMNS ##############################


if __name__ == '__main__':
    """
    This script prepares dataframe and .npy features before training chemprop.
    Run order:
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
    ############################## ARGUMENTS ##############################

    ############################## TARGET COLUMNS ##############################
    # except for (1) molecular_weight, (2) logP, (3) qed, (4) TPSA, (5) normalized_monomer_phi, (6) normalized_backbone_phi
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
        'polarizability_Boltzmann_average',],  # without partial charges

        ['s1_energy_Boltzmann_average',
        'dominant_transition_energy_Boltzmann_average',
        'dominant_transition_oscillator_strength_Boltzmann_average',
        't1_energy_Boltzmann_average',],

        ['chi_parameter_water_mean',
        'chi_parameter_ethanol_mean',
        'chi_parameter_chloroform_mean',]
    ]
    ############################## TARGET COLUMNS ##############################

    ############################## SET PATH ##############################
    # train and test data to only list SMILES
    TRAIN_SMILES_BATCH_DATA_FILE_PATH = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_labeled.csv')  # with batch column -> 12,870,976 polymers
    TEST_SMILES_BATCH_DATA_FILE_PATH = os.path.join(DATA_DIR, 'active_learning', 'OMG_test.csv')  # -> held-out data -> need to update with properties.

    # OMG property calculations data
    TRAIN_PROPERTY_DATA_FILE_PATH = os.path.join(PROPERTY_DATA_DIR, strategy, f'{current_batch}', 'total_results', 'total_results.csv')
    TEST_PROPERTY_DATA_FILE_PATH = os.path.join(PROPERTY_DATA_DIR, 'test', '0', 'total_results', 'total_results.csv')

    # feature dir
    TRAIN_TARGET_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', strategy, f'train_batch_{current_batch}.npy')
    TEST_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', 'test', f'test_rdkit_features.npy')
    ############################## SET PATH ##############################

    ############################## TRAIN DATA ##############################
    # load train SMILES data to train
    df_train_smiles_target = pd.read_csv(TRAIN_SMILES_BATCH_DATA_FILE_PATH)
    df_train_smiles_target_mask = df_train_smiles_target['batch'] >= 0.0  # only include chosen samples
    df_train_smiles_target = df_train_smiles_target[df_train_smiles_target_mask]

    # make a dataframe to train ML
    df_train_property_target = pd.DataFrame()
    for current_batch_idx in range(current_batch):  # add data from previous batches
        df_previous_train_batch_path = os.path.join(PROPERTY_DATA_DIR, strategy, f'{current_batch_idx}', 'total_results', 'total_results.csv')
        df_previous_train_batch = pd.read_csv(df_previous_train_batch_path)
        df_train_property_target = pd.concat([df_train_property_target, df_previous_train_batch], axis=0)

    # current batch
    df_train_current_batch = pd.read_csv(TRAIN_PROPERTY_DATA_FILE_PATH)
    df_train_property_target = pd.concat([df_train_property_target, df_train_current_batch], axis=0)

    # create df
    property_target_columns = [f'{column}_Boltzmann_average' for column in property_columns_Boltzmann] + [f'{column}_mean' for column in property_columns_mean]
    df_train_target = pd.DataFrame(columns=['methyl_terminated_product'] + property_target_columns + ['reaction_id'])  # to append
    for reaction_id, smi in zip(df_train_smiles_target['reaction_id'], df_train_smiles_target['methyl_terminated_product']):
        target_sub_df = df_train_property_target[df_train_property_target['reaction_id'] == reaction_id]
        if target_sub_df.shape[0] == 1 and not target_sub_df.isnull().any().any():  # find a df with no NaN values!
            df_sub_to_append = target_sub_df[property_target_columns].copy()
            df_sub_to_append['methyl_terminated_product'] = [smi]  # append smi (methyl terminated)
            df_sub_to_append['reaction_id'] = [reaction_id]  # append reaction id
            df_train_target = pd.concat([df_train_target, df_sub_to_append], axis=0)

    # dtype
    df_train_target['methyl_terminated_product'] = df_train_target['methyl_terminated_product'].astype('str')
    df_train_target['reaction_id'] = df_train_target['reaction_id'].astype('int')

    # save df to train
    path_to_save_train_with_reaction_id = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_chemprop_with_reaction_id.csv')
    df_train_target.to_csv(path_to_save_train_with_reaction_id, index=False)
    ############################## TRAIN DATA ##############################

    ############################## TEST DATA ##############################
    # NOT NEEDED AFTER THE INITIAL TRAINING
    # make a dataframe for test
    # df_test_smiles_target = pd.read_csv(TEST_SMILES_BATCH_DATA_FILE_PATH)
    # df_test_property_target = pd.read_csv(TEST_PROPERTY_DATA_FILE_PATH)
    # df_test_target = pd.DataFrame(columns=['methyl_terminated_product'] + property_target_columns + ['reaction_id'])  # to append
    # for reaction_id, smi in zip(df_test_smiles_target['reaction_id'], df_test_smiles_target['methyl_terminated_product']):
    #     target_sub_df = df_test_property_target[df_test_property_target['reaction_id'] == reaction_id]
    #     if target_sub_df.shape[0] == 1 and not target_sub_df.isnull().any().any():  # find a df with no NaN values!
    #         df_sub_to_append = target_sub_df[property_target_columns].copy()
    #         df_sub_to_append['methyl_terminated_product'] = [smi]  # append smi (methyl terminated)
    #         df_sub_to_append['reaction_id'] = [reaction_id]  # append reaction id
    #         df_test_target = pd.concat([df_test_target, df_sub_to_append], axis=0)
    #
    # # dtype
    # df_test_target['methyl_terminated_product'] = df_test_target['methyl_terminated_product'].astype('str')
    # df_test_target['reaction_id'] = df_test_target['reaction_id'].astype('int')
    #
    # # save df to test
    # path_to_save_test_with_reaction_id = os.path.join(DATA_DIR, 'active_learning', 'test', 'test_chemprop_with_reaction_id.csv')
    # df_test_target.to_csv(path_to_save_test_with_reaction_id, index=False)  # with reaction id
    ############################## TEST DATA ##############################

    ############################## FEATURES ##############################
    # load train features
    features_npz = np.load(FEATURES_NPZ_PATH)
    target_idx = df_train_target['reaction_id'].to_numpy()
    lst = features_npz.files
    for item in lst:
        normalized_arr = features_npz[item]
        target_train_features = normalized_arr[target_idx]
        np.save(TRAIN_TARGET_FEATURES_PATH, target_train_features)  # save

    # NOT NEEDED AFTER THE INITIAL TRAINING
    # load test features
    # target_test_idx = df_test_target['reaction_id'].to_numpy()
    # lst = features_npz.files
    # for item in lst:
    #     normalized_arr = features_npz[item]
    #     test_features = normalized_arr[target_test_idx]
    #     np.save(TEST_FEATURES_PATH, test_features)  # save
    ############################## FEATURES ##############################

    ############################## SAVE FOR CHEMPROP ##############################
    # drop train reaction id columns
    df_train_target = df_train_target.drop(columns=['reaction_id'], axis=1)
    for gnn_idx, target_cols in enumerate(target_cols_list):
        df_train_target_to_save = df_train_target[['methyl_terminated_product'] + target_cols]  # include target cols
        PATH_CHEMPROP_TRAIN = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_chemprop_train_gnn_{gnn_idx}.csv')
        df_train_target_to_save.to_csv(PATH_CHEMPROP_TRAIN, index=False)

    # NOT NEEDED AFTER THE INITIAL TRAINING
    # drop test reaction id columns
    # df_test_target = df_test_target.drop(columns=['reaction_id'], axis=1)
    # for gnn_idx, target_cols in enumerate(target_cols_list):
    #     df_test_target_to_save = df_test_target[['methyl_terminated_product'] + target_cols]  # include target cols
    #     PATH_CHEMPROP_TEST = os.path.join(DATA_DIR, 'active_learning', 'test', f'test_chemprop_gnn_{gnn_idx}.csv')
    #     df_test_target_to_save.to_csv(PATH_CHEMPROP_TEST, index=False)
    ############################## SAVE FOR CHEMPROP ##############################

    # NOT USED
    # drop cols - need to be changed later
    # if current_batch != 0:  # contains preds
    #     df_train_target = df_train_target.drop(columns=[target for target in target_cols], axis=1)  # pred values
    #     df_train_target = df_train_target.drop(columns=['uncertainty_' + target for target in target_cols], axis=1)  # uncertainty values
    #     df_train_target = df_train_target.drop(columns=['std_' + target for target in target_cols], axis=1)  # std values
    #     true_col_to_be_replaced = ['true_' + target for target in target_cols]
    #     df_train_target = df_train_target.rename(columns={key: value for key, value in zip(true_col_to_be_replaced, target_cols)})
