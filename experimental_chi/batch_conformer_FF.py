import os
import sys
import numpy as np
import pandas as pd

import argparse

from rdkit import Chem
from pathlib import Path
from subprocess import call

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/experimental_chi')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils
from utils import make_directory, draw_molecule, draw_molecule_from_mol, combine_polymer_recursively_terminating_CH3


if __name__ == '__main__':
    # load df
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/experimental_chi/data/reference_chi_canon.csv')
    df = df.drop_duplicates(subset='Polymer')  # drop duplicates. The same shape with drop duplicate with canon smi
    df = df.reset_index(drop=True)

    # divide data
    reference_idx = 0  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = df.shape[0]

    # parameters
    chain_length_list = [1]
    number_of_conformers = 30
    force_field = 'UFF'

    # computation setting
    computing = 'cpu'
    node_number_1 = 0
    node_number_2 = 0
    number_of_processor_per_job = 1

    # save path
    SAVE_DIRECTORY = Path(os.path.join(RUN_DIR, 'calculation_results'))
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    BATCH_DIRECTORY = Path(os.path.join(RUN_DIR, 'batch_script'))
    BATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # batch
    number_of_workers = number_of_reactions
    reaction_idx_arr = np.arange(reference_idx, reference_idx + number_of_reactions)
    split_reaction_idx_arr = np.array_split(reaction_idx_arr, number_of_workers)
    for worker_idx, submit_idx in enumerate(split_reaction_idx_arr):
        # append reaction
        product_list = list()
        for row_idx in submit_idx:
            # append product .smi (canonical smiles)
            product_list.append(df['repeating_unit_smi_canon'][row_idx])

        # .sh script
        sh_script = [
            f'#!/bin/csh\n',
            f'#$ -N Batch_{worker_idx}\n',
            f'#$ -wd {BATCH_DIRECTORY}\n'
            f'#$ -j y\n',
            f'#$ -o {reference_idx}_Batch_conformer_{worker_idx}.out\n',
            f'#$ -pe orte {number_of_processor_per_job}\n',
            f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
            f'#$ -q all.q\n',
            f'SECONDS=0\n',
        ]

        # geometry optimization with GAFF
        for reaction_idx, repeating_unit_smi in zip(submit_idx, product_list):
            # save dir
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_idx}')
            make_directory(save_dir)

            # draw molecules
            # draw_molecule(repeating_unit_smi, os.path.join(save_dir, 'repeating_units.png'))

            # for interpolation
            for chain_length in chain_length_list:
                # create directory
                save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')  # save dir for product
                make_directory(save_opt_dir)  # make dir

                # combine units repeatedly
                combined_p_mol = combine_polymer_recursively_terminating_CH3(
                    repeating_unit_smi,
                    repeating_number=(chain_length - 1)
                )  # implicit hydrogen

                # draw a molecule
                # draw_molecule_from_mol(combined_p_mol, os.path.join(save_opt_dir, 'molecule.png'))

                # explicit hydrogen
                explicit_H_combined_p_mol = Chem.AddHs(combined_p_mol)

                # .smi with explicit hydrogen
                explicit_H_combined_p_smi = Chem.MolToSmiles(explicit_H_combined_p_mol, allHsExplicit=True)

                # save .smi file
                smi_path = os.path.join(save_opt_dir, 'polymer.smi')  # with terminating hydrogen
                with open(smi_path, 'w') as file:
                    file.write(explicit_H_combined_p_smi)
                    file.close()
                asterisk_smi_path = os.path.join(save_opt_dir, 'repeating_unit_asterisk.smi')  # with asterisks
                with open(asterisk_smi_path, 'w') as file:
                    file.write(repeating_unit_smi)
                    file.close()

                # .mol path to save with 3D
                p_mol_path = os.path.join(save_opt_dir, 'polymer.mol')

                # .sdf path to save conformers (weighted rotor)
                conformer_path = os.path.join(save_opt_dir, 'conformers.sdf')
                conformer_xyz_path = os.path.join(save_opt_dir, 'conformers.xyz')

                # write a .sh file for each polymer
                sh_script.extend([
                    f'/home/sk77/.conda/envs/chemprop_evidential/bin/obabel {smi_path} -O {p_mol_path} --gen3d\n',
                    f'/home/sk77/.conda/envs/chemprop_evidential/bin/obabel {p_mol_path} -O {conformer_path} --conformer --nconf {number_of_conformers} --children 5 --mutability 5 --convergence 5 --score rmsd --writeconformers --log\n',
                    f'/home/sk77/.conda/envs/chemprop_evidential/bin/obabel {conformer_path} -O {conformer_xyz_path} --minimize  --cg --steps 5000 --ff {force_field} -c 1e-8 --append "Energy"\n'
                ])

        # append
        sh_script.extend([
            f'duration=$SECONDS\n',
            f'echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."\n',
        ])

        # save .sh file
        sh_file_path = os.path.join(BATCH_DIRECTORY, f'{reference_idx}_batch_conformer_{worker_idx}.sh')
        with open(sh_file_path, 'w') as sh_file:
            sh_file.writelines(sh_script)
            sh_file.close()

        # submit
        qsub_call = f'qsub {sh_file_path}'
        call(qsub_call, shell=True)
