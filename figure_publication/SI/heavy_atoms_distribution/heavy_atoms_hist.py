import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem


if __name__ == '__main__':
    #### MODIFY ####
    current_batch = 3
    #### MODIFY ####

    # df path
    uncertainty_sampling_dir = './'
    df_path_list = [
        os.path.join(f'./OMG_train_batch_0.csv'),
        os.path.join(f'./OMG_train_batch_1.csv'),
        os.path.join(f'./OMG_train_batch_2.csv'),
        os.path.join(f'./OMG_train_batch_3.csv'),
    ]

    # df
    df_list = [pd.read_csv(df_path) for df_path in df_path_list]

    # print shape & calculate heavy atoms
    for df in df_list:
        print(df.shape)
        print(df['polymerization_mechanism_idx'].value_counts())
        df['heavy_atoms'] = df['methyl_terminated_product'].apply(lambda x: Chem.MolFromSmiles(x).GetNumHeavyAtoms())
        print(df['heavy_atoms'].describe())

    # plot
    color_list = ['#6DC9C9', '#BA5BBA', '#EF2525', '#345BBF']
    name_list = ['Initial train', 'For round 1', 'For round 2', 'For round 3']
    for df_idx, df in enumerate(df_list):
        plt.hist(df['heavy_atoms'].to_numpy(), color=color_list[df_idx], label=name_list[df_idx], alpha=0.6, bins=50)

    plt.xlabel('Number of heavy atoms', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f'heavy_atoms_current_batch_{current_batch}.png')  # datasets include until current_batch + 1
