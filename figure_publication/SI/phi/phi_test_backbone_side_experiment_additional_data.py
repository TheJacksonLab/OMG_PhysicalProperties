import os
import sys
import numpy as np
sys.path.append('/home/sk77/PycharmProjects/omg_database_publication')

import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcPhi

from sklearn.linear_model import LinearRegression

from utils import combine_polymer_recursively_terminating_CH3, extract_backbone_from_molecule_depth_one


if __name__ == '__main__':
    # argument values
    df_path = './csv_files/413K_clear_experiment_flexibility_with_copolymers.csv'
    df = pd.read_csv(df_path)
    df['repeating_unit_smi_canon'] = df['repeating_unit_smi_non_canon'].apply(lambda x: Chem.CanonSmiles(x))

    # divide data
    reference_idx = 0  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = df.shape[0]
    molecule_idx = range(reference_idx, reference_idx + number_of_reactions)

    repeating_number_list = [0, 1]

    monomer_phi_per_total_atom_list = list()
    backbone_phi_per_total_atom_list = list()

    for idx in molecule_idx:
        print(f'Reaction {idx}')
        repeating_unit_smi = df['repeating_unit_smi_canon'][idx]
        repeating_unit_mol = Chem.MolFromSmiles(repeating_unit_smi)

        # remove backbone
        backbone_repeating_unit_smi = extract_backbone_from_molecule_depth_one(repeating_unit_smi)
        backbone_repeating_unit_mol = Chem.MolFromSmiles(backbone_repeating_unit_smi)

        # get the number of atoms in a monomer
        total_number_of_atoms_in_monomer = repeating_unit_mol.GetNumHeavyAtoms()  # without asterisks

        repeating_phi_list = list()
        backbone_phi_list = list()
        for repeating_number in repeating_number_list:
            repeating_mol = combine_polymer_recursively_terminating_CH3(repeating_unit_smi, repeating_number=repeating_number)
            backbone_mol = combine_polymer_recursively_terminating_CH3(backbone_repeating_unit_smi, repeating_number=repeating_number)

            # repeating unit
            repeating_phi = CalcPhi(repeating_mol)
            repeating_phi_list.append(repeating_phi)

            # backbone
            backbone_phi = CalcPhi(backbone_mol)
            backbone_phi_list.append(backbone_phi)

        # linear fitting: y = a * (n - 1) + b ~ a * n
        monomer_phi_arr = np.array(repeating_phi_list)
        backbone_phi_arr = np.array(backbone_phi_list)

        # phi idx
        monomer_phi_idx = monomer_phi_arr[1] - monomer_phi_arr[0]
        backbone_phi_idx = backbone_phi_arr[1] - backbone_phi_arr[0]

        monomer_phi_per_total_atom_list.append(monomer_phi_idx / total_number_of_atoms_in_monomer)
        backbone_phi_per_total_atom_list.append(backbone_phi_idx / total_number_of_atoms_in_monomer)

    # save
    df_save = pd.DataFrame({
        'monomer_phi_per_total_atom': np.array(monomer_phi_per_total_atom_list),
        'backbone_phi_per_total_atom': np.array(backbone_phi_per_total_atom_list)
    })
    df_save.to_csv('./csv_files/phi_estimation.csv', index=False)
