import os
import sys
EXTERNAL_PACKAGE_DIR = '/home/sk77/PycharmProjects/omg_database_publication/external_packages'
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './chemprop_evidential/chemprop'))  # chemprop evidential
sys.path.append(os.path.join(EXTERNAL_PACKAGE_DIR, './nds-py'))  # search pareto front

import pandas as pd
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from chemprop.parsing import parse_train_args_script
from chemprop.train import cross_validate
from chemprop.utils import create_logger


def get_args():
    # Create an argument parser
    parser = ArgumentParser(description='train chemprop')

    # Add positional arguments
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning. ["pareto_greedy"]', default=None)
    parser.add_argument('--current_batch', type=int, help='Current batch for active learning', default=None)
    parser.add_argument('--gnn_idx', type=int,
                        help='GNN idx for training. 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter',
                        default=None)
    parser.add_argument('--gpu_idx', type=int, help='GPU to use [0, 1, 2, 3]', default=None)

    return parser.parse_args()  # Parse the arguments

############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################


if __name__ == '__main__':
    """
    This script trains a chemprop.
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
    gnn_idx = args.gnn_idx  # 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter
    gpu_idx = args.gpu_idx

    # strategy = 'pareto_greedy'  # try
    # current_batch = 0  # 0 for initial train
    # gnn_idx = 1  # GNN idx for training. 0 -> 3D geometry / 1 -> DFT / 2 -> TD-DFT / 3 -> chi parameter
    # gpu_idx = 0  # GPU to use [0, 1, 2, 3]
    ############################## ARGUMENTS ##############################

    ############################## SET PATH ##############################
    # active learning dir
    ACTIVE_LEARNING_CHECK_POINT_DIR = os.path.join(ACTIVE_LEARNING_DIR, f'./{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}')

    # feature dir
    TRAIN_TARGET_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', strategy, f'train_batch_{current_batch}.npy')
    TEST_FEATURES_PATH = os.path.join(DATA_DIR, 'rdkit_features', 'test', f'test_rdkit_features.npy')

    # data to train
    PATH_CHEMPROP_TRAIN = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch}_chemprop_train_gnn_{gnn_idx}.csv')
    PATH_CHEMPROP_TEST = os.path.join(DATA_DIR, 'active_learning', 'test', f'test_chemprop_gnn_{gnn_idx}.csv')
    ############################## SET PATH ##############################

    ############################## RUN CHEMPROP ##############################
    # General arguments
    train_arguments = [
        '--gpu', f'{gpu_idx}',
        '--data_path', f'{PATH_CHEMPROP_TRAIN}',
        '--save_dir', f'{ACTIVE_LEARNING_CHECK_POINT_DIR}',
        '--dataset_type', 'regression',
        '--separate_test_path', f'{PATH_CHEMPROP_TEST}',
        '--separate_val_path', f'{PATH_CHEMPROP_TEST}',  # same as the test path
        '--num_folds', '5',  # to generate error bars
        '--torch_seed', '42',  # pytorch seed (weight initialization)
        '--seed', '42',  # used for data splitting + np.random seed and random.seed + CUDA seed
        '--metric', 'rmse',
        '--log_frequency', '10',
        '--show_individual_scores',

        # features
        '--features_path', TRAIN_TARGET_FEATURES_PATH,  # batch train features path
        '--separate_test_features_path', TEST_FEATURES_PATH,  # Path to file with features for separate test set
        '--separate_val_features_path', TEST_FEATURES_PATH,  # Path to file with features for separate val set. Same as the test features.
        '--no_features_scaling',  # Turn off scaling of features -> already scaled
    ]

    # Training arguments
    train_arguments.extend([
        '--epoch', '100',  # Number of epochs to run
        '--batch_size', '50',
        '--final_lr', '1e-4',  # Final learning rate
        '--init_lr', '1e-4',  # Initial learning rate
        '--max_lr', '1e-3',  # Maximum learning rate
        '--warmup_epochs', '2',  # Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`. Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
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

    args = parse_train_args_script(train_arguments)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    mean_score, std_score = cross_validate(args, logger)
    ############################## RUN CHEMPROP ##############################
