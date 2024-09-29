import os
import sys
EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './chemprop_evidential/chemprop'))  # chemprop evidential

import argparse
import pandas as pd
import pickle

from pathlib import Path
from chemprop.parsing import parse_train_args_script, parse_predict_args_script

"""Train a model using active learning on a dataset."""
from copy import deepcopy
import numpy as np

from chemprop.train import make_uncertainty_predictions_without_true_values

############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='argument parser for uncertainty sampling')

    # Add positional arguments
    parser.add_argument('--data_path', type=str, help='Polymer .csv to use (with only one column)', default=None)
    parser.add_argument('--rdkit_features_path', type=str, help='rdkit features path to use', default=None)
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning. ["random", "pareto_greedy"]', default=None)
    parser.add_argument('--current_batch', type=int, help='Current batch for active learning', default=None)
    parser.add_argument('--gpu_idx', type=int, help='gpu idx to use', default=None)
    parser.add_argument('--gnn_idx', type=int, help='gnn idx', default=None)
    parser.add_argument('--smi_batch_idx', type=int, help='smi batch index', default=None)
    parser.add_argument('--model_save_dir_name_gnn', type=str, help='Model save dir name for gnn', default=None)

    return parser.parse_args()  # Parse the arguments


def save_object(obj, filename):
    """ This function saves an object
    e.g.) save_object(company1, 'company1.pkl')
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    This scripts predicts uncertainties of OMG polymers and save a dataframe with uncertainties predicted
    from each D-MPNN evidential learning.
    
    In active learning champaign,
    1) prepare_data.py 
    2) train.py
    3) fold_plot_learning_curve.py
    4) uncertainty_pred.py
    5) uncertainty_sampling.py
    
    Go back to 1)
    """
    ############################## SHELL ARGUMENTS ##############################
    args = get_args()
    data_path = args.data_path
    rdkit_features_path = args.rdkit_features_path
    active_learning_strategy = args.active_learning_strategy
    current_batch = args.current_batch
    gpu_idx = args.gpu_idx
    gnn_idx = args.gnn_idx
    smi_batch_idx = args.smi_batch_idx
    model_save_dir_name_gnn = args.model_save_dir_name_gnn
    ############################## SHELL ARGUMENTS ##############################

    ############################## SET PATH ##############################
    DF_SAVE_DIR = Path(os.path.join(DATA_DIR, 'active_learning', active_learning_strategy, 'uncertainty_pred', f'current_batch_{current_batch}'))
    DF_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_LEARNING_CHECK_POINT_DIR = os.path.join(ACTIVE_LEARNING_DIR, f'./{active_learning_strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}')  # torch seed 42
    MODEL_SAVE_DIR = os.path.join(ACTIVE_LEARNING_CHECK_POINT_DIR, model_save_dir_name_gnn, 'fold_0')
    ############################## SET PATH ##############################

    ############################## PREDICT ##############################
    # evaluate models on a training data
    pred_arguments = [
        '--gpu', f'{gpu_idx}',  # for GPU running
        '--test_path', f'{data_path}', # Path to CSV file containing testing data for which predictions will be made. First column (methyl_terminated_product will be used to predict properties)
        '--preds_path', f'NULL',  # No effects
        '--checkpoint_dir', f'{MODEL_SAVE_DIR}',
        '--batch_size', '50',

        # features
        '--features_path', rdkit_features_path,  # rdkit featuers path to use
        '--no_features_scaling',  # Turn off scaling of features -> already scaled
    ]
    args = parse_predict_args_script(pred_arguments)
    df_result, df_std, scaler = make_uncertainty_predictions_without_true_values(args)

    # active learning based on a model predictive uncertainty
    # std_arr = df_std.to_numpy()  # unscaled uncertainty
    ############################## PREDICT ##############################

    ############################## SAVE ##############################
    # dir path
    PREDICTION_DIR = Path(os.path.join(DF_SAVE_DIR, f'gnn_{gnn_idx}', 'prediction'))
    UNCERTAINTY_DIR = Path(os.path.join(DF_SAVE_DIR, f'gnn_{gnn_idx}', 'uncertainty'))
    SCALER_DIR = Path(os.path.join(DF_SAVE_DIR, f'gnn_{gnn_idx}', 'scaler'))

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    # save df
    df_result.to_csv(os.path.join(PREDICTION_DIR, f'smi_batch_idx_{smi_batch_idx}_pred.csv'), index=False)
    df_std.to_csv(os.path.join(UNCERTAINTY_DIR, f'smi_batch_idx_{smi_batch_idx}_std.csv'), index=False)
    save_object(scaler, os.path.join(SCALER_DIR, f'smi_batch_idx_{smi_batch_idx}_scaler.pkl'))  # should be all the same.
    ############################## SAVE ##############################
