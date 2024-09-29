import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################


if __name__ == '__main__':
    """
    This script plots a learning curve.
    """
    ################### MODIFY ###################
    # arguments
    gnn_idx = 3
    strategy = 'pareto_greedy'
    current_batch = 0  # 0 for initial training data

    # get path
    CHECKPOINT_DIR = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '231214-141052601222_random_train_batch_0_chemprop_train_gnn_3_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence')
    # MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence')
    MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, '240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence')
    ################### MODIFY ###################

    # file path
    verbose_log_file_path = os.path.join(MODEL_SAVE_DIR, 'fold_0', 'model_0', 'verbose.log')

    # gather values
    validation_rmse_list = list()
    training_loss_list = list()
    fold_cnt = 0
    with open(verbose_log_file_path, 'r') as file:
        lines = file.readlines()
        training_loss_per_epoch_list = list()  # gather train evidential loss for each epoch -> take a mean value to plot
        for line in lines:
            if 'Building model' in line:
                fold_cnt += 1
            if fold_cnt == 2:
                break
            if line.startswith('Loss'):  # training loss
                tmp_line = line.split()
                training_loss_per_epoch_list.append(float(tmp_line[2][:-1]))  # exclude comma
            elif line.startswith('Validation rmse'):  # validation rmse
                tmp_line = line.split()
                validation_rmse_list.append(float(tmp_line[-1]))  # "arithmetic" mean of scaled-recovered properties. Not validation loss. Not geometric mean.
                training_loss_list.append(np.mean(training_loss_per_epoch_list))  # append training losses. Not always the same number of training loss is printed per epoch.
                training_loss_per_epoch_list = list()  # reset
        file.close()

    # draw a learning curve
    num_epoch = len(validation_rmse_list)
    plt.figure(figsize=(6, 6), dpi=300)
    fig, ax1 = plt.subplots()

    color = 'c'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Train Evidential Loss', fontsize=14, color=color)
    ax1.plot(range(num_epoch), training_loss_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'm'
    ax2.set_ylabel('RMSE (mean RMSE of all tasks)', fontsize=14, color=color)  # scaled-recovered
    ax2.plot(range(num_epoch), validation_rmse_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'learning_curve_with_validation_rmse.png'))
    plt.close()
