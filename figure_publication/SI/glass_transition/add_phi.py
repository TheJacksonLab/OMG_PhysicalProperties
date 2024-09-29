import os
import sys
import numpy as np
sys.path.append('/home/sk77/PycharmProjects/omg_database_publication')

import matplotlib.pyplot as plt
import pandas as pd

from math import ceil
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcPhi

from sklearn.linear_model import LinearRegression

from utils import combine_polymer_recursively_terminating_CH3, extract_backbone_from_molecule_depth_one


if __name__ == '__main__':
    # argument values
    df_path = './processed_Tg.csv'
    df = pd.read_csv(df_path)

    # cnt number of heavy atoms
    # df['num_heavy_atoms'] = df['canon_p_smi'].apply(lambda x: Chem.MolFromSmiles(x).GetNumHeavyAtoms())  # without asterisk

    molecule_idx = range(0, df.shape[0])
    canon_p_smi_list = df['canon_p_smi'].to_list()
    normalized_monomer_phi_idx_list = list()
    normalized_backbone_phi_idx_list = list()

    for idx in molecule_idx:
        print(f'Reaction {idx}')
        # monomer smi
        repeating_unit_smi = canon_p_smi_list[idx]
        repeating_unit_mol = Chem.MolFromSmiles(repeating_unit_smi)
        total_number_of_atoms_in_monomer = repeating_unit_mol.GetNumHeavyAtoms()  # without asterisks

        # get backbone smi
        backbone_repeating_unit_smi = extract_backbone_from_molecule_depth_one(repeating_unit_smi)

        # get phi idx
        repeating_number_list = [0, 1]
        phi_mon_list = list()
        phi_bb_list = list()
        for repeating_number in repeating_number_list:
            repeating_mol = combine_polymer_recursively_terminating_CH3(repeating_unit_smi, repeating_number=repeating_number)
            backbone_mol = combine_polymer_recursively_terminating_CH3(backbone_repeating_unit_smi, repeating_number=repeating_number)

            # repeating unit
            phi_mon = CalcPhi(repeating_mol)
            phi_mon_list.append(phi_mon)

            # backbone
            phi_bb = CalcPhi(backbone_mol)
            phi_bb_list.append(phi_bb)

        # change to arr
        phi_mon_arr = np.array(phi_mon_list)
        phi_bb_arr = np.array(phi_bb_list)

        # phi idx
        phi_mon_increase = phi_mon_arr[1] - phi_mon_arr[0]
        phi_bb_increase = phi_bb_arr[1] - phi_bb_arr[0]

        # normalized
        normalized_monomer_phi_idx = phi_mon_increase / total_number_of_atoms_in_monomer
        normalized_backbone_phi_idx = phi_bb_increase / total_number_of_atoms_in_monomer

        # append
        normalized_monomer_phi_idx_list.append(normalized_monomer_phi_idx)
        normalized_backbone_phi_idx_list.append(normalized_backbone_phi_idx)

    # save
    df_save = pd.DataFrame({
        'normalized_monomer_phi': np.array(normalized_monomer_phi_idx_list),
        'normalized_backbone_phi': np.array(normalized_backbone_phi_idx_list),
    })
    df_save = pd.concat([df, df_save], axis=1)
    df_save.to_csv('./normalized_phi_estimation.csv', index=False)
