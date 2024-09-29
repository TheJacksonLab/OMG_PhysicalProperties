import os
import sys
EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './chemprop_evidential/chemprop'))  # chemprop evidential

import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from chemprop.parsing import parse_predict_args_script
from chemprop.train import make_predictions


def get_args():
    # Create an argument parser
    parser = ArgumentParser(description='Prd chemprop')

    # Add positional arguments
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning. ["pareto_greedy"]', default=None)
    parser.add_argument('--current_batch', type=int, help='Current batch for active learning', default=None)
    parser.add_argument('--gnn_idx', type=int,
                        help='GNN idx for training. 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter',
                        default=None)
    parser.add_argument('--dir_name', type=str, help='Dir name under gnn_idx. Ex) 240125-...', default=None)
    parser.add_argument('--gpu_idx', type=int, help='GPU to use [0, 1, 2, 3]', default=None)

    return parser.parse_args()  # Parse the arguments


############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################


if __name__ == '__main__':
    """
    This script predicts results.
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred.py
    4) uncertainty_sampling.py
    -> repeat to 1)
    """
    # argument parser
    args = get_args()
    strategy = args.active_learning_strategy  # ['random', 'pareto_greedy']
    current_batch = args.current_batch  # 0 for initial training data
    gnn_idx = args.gnn_idx  # 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter
    dir_name = args.dir_name  # ex) 240125-...
    gpu_idx = args.gpu_idx
    fold_num = 5  # five-fold training for error bars

    ################################# MODIFY #################################
    # strategy = 'pareto_greedy'  # try
    # current_batch = 0  # 0 for initial train
    # gnn_idx = 1  # GNN idx for training. 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter
    # gpu_idx = 0  # GPU to use [0, 1, 2, 3]
    ################################# MODIFY #################################

    ############################## SET PATH ##############################
    # feature dir
    TRAIN_TARGET_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', strategy, f'train_batch_{current_batch}.npy')
    TEST_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', 'test', f'test_rdkit_features.npy')

    # data to train
    PATH_CHEMPROP_TRAIN = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_chemprop_train_gnn_{gnn_idx}.csv')
    PATH_CHEMPROP_TEST = os.path.join(DATA_DIR, 'active_learning', 'test', f'test_chemprop_gnn_{gnn_idx}.csv')
    ############################## SET PATH ##############################

    # data to train
    for fold_idx in range(fold_num):
        ACTIVE_LEARNING_CHECK_POINT_DIR = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/{dir_name}/fold_{fold_idx}')

        # train and test
        target_csv_features_path = [
            ['train', PATH_CHEMPROP_TRAIN, TRAIN_TARGET_FEATURES_PATH],
            ['test', PATH_CHEMPROP_TEST, TEST_FEATURES_PATH]
        ]
        for target_list in target_csv_features_path:
            pred_arguments = [
                '--gpu', f'{gpu_idx}',
                '--test_path', target_list[1],  # Path to CSV file containing testing data for which predictions will be made
                '--preds_path', f'NULL',  # No effects
                '--checkpoint_dir', f'{ACTIVE_LEARNING_CHECK_POINT_DIR}',
                '--batch_size', '50',

                # features
                '--features_path', target_list[2],  # total train features
                '--no_features_scaling',  # Turn off scaling of features -> already scaled
            ]
            args = parse_predict_args_script(pred_arguments)
            df_result, df_std, scaler = make_predictions(args)  # unscaled df
            df_result.to_csv(os.path.join(ACTIVE_LEARNING_CHECK_POINT_DIR, f'{target_list[0]}_pred.csv'), index=False)
