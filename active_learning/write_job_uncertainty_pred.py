import os
import numpy as np
import pandas as pd

from pathlib import Path
from subprocess import call

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')  # run dir contains a python script to run


if __name__ == '__main__':
    ################## MODIFY #####################
    data_dir = '/home/sk77/PycharmProjects/omg_database_publication/data'
    active_learning_strategy = 'pareto_greedy'  # ['pareto_greedy']
    current_batch = 3  # 0 for initial training

    # save results
    script_type = 'uncertainty_pred.py'

    # gnn idx
    gnn_idx = 3

    # initial batch
    # model_save_dir_name_gnn_list = [
    #     '240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence',
    #     '240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence',
    #     '240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence',
    #     '240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence',
    # ]

    # AL1
    # model_save_dir_name_gnn_list = [
    #     '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence',
    #     '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence',
    #     '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence',
    #     '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence',
    # ]

    # AL2
    # model_save_dir_name_gnn_list = [
    #     '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence',
    #     '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence',
    #     '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence',
    #     '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence',
    # ]

    # AL3
    model_save_dir_name_gnn_list = [
        '240531-100412913979_OMG_train_batch_3_chemprop_train_gnn_0_evidence',
        '240531-100828584294_OMG_train_batch_3_chemprop_train_gnn_1_evidence',
        '240531-100906010104_OMG_train_batch_3_chemprop_train_gnn_2_evidence',
        '240531-101615032820_OMG_train_batch_3_chemprop_train_gnn_3_evidence',
    ]

    # node info
    node_number_1 = 0
    node_number_2 = 7
    gpu_idx = 3  # gpu card index
    number_of_processor_per_batch = 1

    # batch prediction
    total_batch_num = 48
    ################## MODIFY #####################

    # data path
    batch_smi_dir = os.path.join(data_dir, 'active_learning', active_learning_strategy, 'uncertainty_pred', 'split_polymers')
    batch_features_dir = os.path.join(data_dir, 'active_learning', active_learning_strategy, 'uncertainty_pred', 'split_features')

    # save path
    SH_DIR = Path(os.path.join(RUN_DIR, 'sh_dir'))
    SAVE_SH_FILE_DIRECTORY = Path(os.path.join(SH_DIR, 'uncertainty_pred', f'{current_batch}'))
    SAVE_SH_FILE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUT_SH_FILE_DIRECTORY = Path(os.path.join(SH_DIR, 'out_dir', 'uncertainty_pred', f'{current_batch}'))
    OUT_SH_FILE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # dir to save .sh files
    python_script = os.path.join(RUN_DIR, script_type)

    # .sh script
    sh_script = [
        f'#!/bin/csh\n',
        f'#$ -N GNN_{gnn_idx}\n',
        f'#$ -wd {OUT_SH_FILE_DIRECTORY}\n'
        f'#$ -j y\n',
        f'#$ -o gnn_{gnn_idx}.out\n',
        f'#$ -pe orte {number_of_processor_per_batch}\n',
        f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
        f'#$ -q gpu\n',
    ]
    for smi_batch_idx in range(total_batch_num):
        data_path = os.path.join(batch_smi_dir, f'OMG_train_smi_batch_{smi_batch_idx}.csv')
        rdkit_features_path = os.path.join(batch_features_dir, f'OMG_train_smi_batch_features_{smi_batch_idx}.npy')
        cmd_line = f'--data_path {data_path} --rdkit_features_path {rdkit_features_path} --active_learning_strategy {active_learning_strategy} ' \
                   f'--gpu_idx {gpu_idx} --gnn_idx {gnn_idx} --current_batch {current_batch} --smi_batch_idx {smi_batch_idx}  ' \
                   f'--model_save_dir_name_gnn {model_save_dir_name_gnn_list[gnn_idx]}'

        # write a .sh file
        sh_script.extend([
            f'/home/sk77/.conda/envs/chemprop_evidential/bin/python {python_script} ' + cmd_line + '\n',
        ])

    # save .sh file
    sh_file_path = os.path.join(SAVE_SH_FILE_DIRECTORY, f'gnn_{gnn_idx}.sh')

    with open(sh_file_path, 'w') as sh_file:
        sh_file.writelines(sh_script)
        sh_file.close()

    # submit
    # qsub_call = f'qsub {sh_file_path}'
    # call(qsub_call, shell=True)
