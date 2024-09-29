import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################

############################## TARGET COLUMNS ##############################
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
    'polarizability_Boltzmann_average',],

    ['s1_energy_Boltzmann_average',
    'dominant_transition_energy_Boltzmann_average',
    'dominant_transition_oscillator_strength_Boltzmann_average',
    't1_energy_Boltzmann_average',],

    ['chi_parameter_water_mean',
    'chi_parameter_ethanol_mean',
    'chi_parameter_chloroform_mean',]
]
############################## TARGET COLUMNS ##############################

if __name__ == '__main__':
    """
    This script plots a learning curve for fold learning (different random seeds)
    Run order:
    1) prepare_data.py
    2) train.py
    3) pred.py & plot_pred.py
    4) fold_plot_learning_curve.py
    5) uncertainty_pred.py
    6) uncertainty_sampling.py
    -> repeat to 1)
    """
    ################### MODIFY ###################
    gnn_idx = 0
    strategy = 'pareto_greedy'  # ['random', 'pareto_greedy']
    current_batch = 0  # 0 for initial training data
    num_folds = 1
    num_epochs = 100

    # get path
    CHECKPOINT_DIR = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}')
    MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence')
    ################### MODIFY ###################

    # target cols
    target_cols = target_cols_list[gnn_idx]

    # file path
    verbose_log_file_path = os.path.join(MODEL_SAVE_DIR, 'fold_0', 'model_0', 'verbose.log')  # this contains all verbose information for num_folds training

    # list to contain values
    fold_validation_rmse_list = list()  # all folds
    validation_rmse_list = list()  # the validation data is the same as the test data
    fold_training_loss_list = list()  # all folds
    training_loss_list = list()
    fold_target_col_rmse_list = list()  # all folds
    target_col_rmse_list = list()

    # gather
    with open(verbose_log_file_path, 'r') as file:
        lines = file.readlines()
        training_loss_per_epoch_list = list()  # gather train evidential loss for each epoch -> take a mean value to plot
        for line in lines:
            tmp_line = line.split()
            if line.startswith('Loss'):  # training loss
                training_loss_per_epoch_list.append(float(tmp_line[2][:-1]))  # exclude comma
            elif line.startswith('Validation rmse'):  # validation rmse
                validation_rmse_list.append(float(tmp_line[-1]))  # "arithmetic" mean of scaled-recovered properties. Not validation loss. Not geometric mean.
                training_loss_list.append(np.mean(training_loss_per_epoch_list))  # append training losses. Not always the same number of training loss is printed per epoch.
                training_loss_per_epoch_list = list()  # reset
            elif line.startswith('Validation') and not line.startswith('Validation rmse'):
                target_col_rmse_list.append(float(tmp_line[-1]))
            elif line.startswith('Epoch 0') and len(validation_rmse_list) != 0:  # save (excluding the first epoch 0)
                fold_validation_rmse_list.append(validation_rmse_list)  # save
                fold_training_loss_list.append(training_loss_list)  # save
                fold_target_col_rmse_list.append(target_col_rmse_list)  # save

                # reset
                validation_rmse_list = list()
                training_loss_list = list()
                target_col_rmse_list = list()

        # add the last epoch
        fold_validation_rmse_list.append(validation_rmse_list)  # save
        fold_training_loss_list.append(training_loss_list)  # save
        fold_target_col_rmse_list.append(target_col_rmse_list)  # save
        # reset
        validation_rmse_list = list()
        training_loss_list = list()
        target_col_rmse_list = list()
        file.close()

    # arr
    fold_validation_rmse_arr = np.array(fold_validation_rmse_list)  # [num_folds, num_epochs (100)]
    fold_training_loss_arr = np.array(fold_training_loss_list)  # [num_folds, num_epochs (100)]
    fold_target_col_rmse_arr = np.array(fold_target_col_rmse_list)  # [num_folds, num_targets * num_epochs (100)]
    fold_target_col_rmse_arr = fold_target_col_rmse_arr.reshape(num_folds, num_epochs, -1)  # [num_folds, num_epochs, num_targets]

    # draw a learning curve
    main_validation_rmse_arr = fold_validation_rmse_arr[0]  # torch seed 42
    main_training_loss_arr = fold_training_loss_arr[0]  # torch seed 42
    main_target_col_rmse_arr = fold_target_col_rmse_arr[0]  # torch seed 42

    # mean arr
    mean_validation_rmse_arr = fold_validation_rmse_arr.mean(axis=0)
    mean_training_loss_arr = fold_training_loss_arr.mean(axis=0)
    mean_target_col_rmse_arr = fold_target_col_rmse_arr.mean(axis=0)

    # get error bars
    std_validation_rmse_arr = fold_validation_rmse_arr.std(axis=0, ddof=0)  # [num_epochs (100),]
    std_training_loss_arr = fold_training_loss_arr.std(axis=0, ddof=0)  # [num_epochs (100),]
    std_target_col_rmse_arr = fold_target_col_rmse_arr.std(axis=0, ddof=0)  # [num_epochs (100), num_targets]

    # plot - averaged validation error
    plt.figure(figsize=(6, 6), dpi=300)
    num_epoch = len(std_validation_rmse_arr)
    fig, ax1 = plt.subplots()
    color = 'c'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Train Evidential Loss', fontsize=14, color=color)
    ax1.plot(range(num_epoch), main_training_loss_arr, color=color)
    ax1.fill_between(range(num_epoch), mean_training_loss_arr - std_training_loss_arr, mean_training_loss_arr + std_training_loss_arr,
                     color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'm'
    ax2.set_ylabel('RMSE (mean RMSE of all tasks)', fontsize=14, color=color)  # unscaled
    ax2.plot(range(num_epoch), main_validation_rmse_arr, color=color, label='Model used')
    ax2.legend(fontsize=14)
    ax2.fill_between(range(num_epoch), mean_validation_rmse_arr - std_validation_rmse_arr, mean_validation_rmse_arr + std_validation_rmse_arr,
                     color=color, alpha=0.2, label='Mean ($\pm$std)')
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'learning_curve_with_test_rmse.png'))
    plt.close()

    # plot - averaged validation error per target
    for target_col_idx, target_col in enumerate(target_cols):
        plt.figure(figsize=(6, 6), dpi=300)

        # plot - averaged validation error
        num_epoch = len(std_validation_rmse_arr)
        main_target_arr_to_plot = main_target_col_rmse_arr[:, target_col_idx]
        mean_target_arr_to_plot = mean_target_col_rmse_arr[:, target_col_idx]
        std_target_arr_to_plot = std_target_col_rmse_arr[:, target_col_idx]

        # plot
        color = 'm'
        plt.plot(range(num_epoch), main_target_arr_to_plot, color=color, label='Model used')
        plt.fill_between(range(num_epoch), mean_target_arr_to_plot - std_target_arr_to_plot, mean_target_arr_to_plot + std_target_arr_to_plot,
                     color=color, alpha=0.2, label='Mean ($\pm$std)')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(f'RMSE (unscaled) {target_col}', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, f'learning_curve_with_test_rmse_{target_col}.png'))
        plt.close()
