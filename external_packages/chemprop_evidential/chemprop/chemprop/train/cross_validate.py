from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np
import random

from .confidence_evaluator import ConfidenceEvaluator
from .run_training import run_training, get_dataset_splits, get_atomistic_splits, evaluate_models
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    torch_seed = args.torch_seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.torch_seed = torch_seed + fold_num
        info(f'Torch weight initialization seed {args.torch_seed}')
        # args.seed = init_seed + fold_num
        info(f'Random seed {args.seed} for data split and random number generations')
        random.seed(args.seed)  # random seed
        np.random.seed(args.seed)  # random seed
        torch.cuda.manual_seed_all(args.seed)  # random seed for generating random numbers for the current GPU.
        torch.manual_seed(args.torch_seed)  # torch seed to initialize models
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)

        # Load the data
        (train_data, val_data, test_data), features_scaler, scaler = \
            get_dataset_splits(args.data_path, args, logger)

        # Train with the data, return the best models
        models = run_training(
            train_data, val_data, scaler, features_scaler, args, logger)

        # Evaluate the models on both val and test data
        val_scores, val_preds, val_conf, val_std, val_entropy = evaluate_models(
            models, train_data, val_data, scaler, args, logger,
            export_std=True)
        test_scores, test_preds, test_conf, test_std, test_entropy = evaluate_models(
            models, train_data, test_data, scaler, args, logger, 
            export_std=True)

        # Log the confidence plots if desired
        if args.confidence:
            ConfidenceEvaluator.save(
                val_preds, val_data.targets(), val_conf, val_std, val_data.smiles(),
                test_preds, test_data.targets(), test_conf, test_std, test_data.smiles(),
                val_entropy, test_entropy, args)

            ConfidenceEvaluator.visualize(
                args.save_confidence, args.confidence_evaluation_methods,
                info, args.save_dir, draw=False)

        all_scores.append(test_scores)
        # all_scores.append(val_scores)  # for hyperparameter optimization
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        # info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')
        info(f'Seed {torch_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                # info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')
                info(f'Seed {torch_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


def cross_validate_atomistic(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = "U0" #get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)

        # Load the data
        (train_data, val_data, test_data), features_scaler, scaler = \
            get_atomistic_splits(args.data_path, args, logger)

        # Train with the data, return the best models
        models = run_training(
            train_data, val_data, scaler, features_scaler, args, logger)

        # Evaluate the models on both val and test data
        val_scores, val_preds, val_conf, val_std, val_entropy = evaluate_models(
            models, train_data, val_data, scaler, args, logger,
            export_std=True)
        test_scores, test_preds, test_conf, test_std, test_entropy = evaluate_models(
            models, train_data, test_data, scaler, args, logger, 
            export_std=True)

        # Log the confidence plots if desired
        if args.confidence:
            ConfidenceEvaluator.save(
                val_preds, val_data.targets(), val_conf, val_std, val_data.smiles(),
                test_preds, test_data.targets(), test_conf, test_std, test_data.smiles(),
                val_entropy, test_entropy, args)

            ConfidenceEvaluator.visualize(
                args.save_confidence, args.confidence_evaluation_methods,
                info, args.save_dir, draw=False)


        all_scores.append(test_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
