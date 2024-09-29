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
from utils import extract_energy_and_xyz_coordinates_from_multimolecule_xyz, make_directory


if __name__ == '__main__':
    # load df
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/experimental_chi/data/reference_chi_canon.csv')
    df = df.drop_duplicates(subset='Polymer')  # drop duplicates. The same shape with drop duplicate with canon smi
    df = df.reset_index(drop=True)

    # divide data
    reference_idx = 0  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = df.shape[0]  # batch

    # parameters
    chain_length_list = [1]
    topn = 15

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
            f'#$ -N XTB_{worker_idx}\n',
            f'#$ -wd {BATCH_DIRECTORY}\n'
            f'#$ -j y\n',
            f'#$ -o {reference_idx}_Batch_XTB_{worker_idx}.out\n',
            f'#$ -pe orte {number_of_processor_per_job}\n',
            f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
            f'#$ -q all.q\n',
            f'SECONDS=0\n',
            f'xtb_path=/home/sk77/PycharmProjects/xtb/xtb_6.6.1\n',
        ]

        # geometry optimization with GAFF
        for reaction_idx, repeating_unit_smi in zip(submit_idx, product_list):
            # save dir
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_idx}')

            # for interpolation
            for chain_length in chain_length_list:
                # create directory
                save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')  # save dir for product

                # pick topn conformers with lowest energy
                conformer_xyz_energy_path = os.path.join(save_opt_dir, 'conformers.xyz')
                polymer_mol_path = os.path.join(save_opt_dir, 'polymer.mol')
                polymer_mol = Chem.MolFromMolFile(polymer_mol_path, removeHs=False)
                number_of_atoms = polymer_mol.GetNumAtoms()

                # extract energy and corresponding .xyz coordinates
                sorted_energy_xyz_list = extract_energy_and_xyz_coordinates_from_multimolecule_xyz(
                    file_path=conformer_xyz_energy_path,
                    number_of_atoms=number_of_atoms,
                    topn=topn,
                )

                # geometry optimization with xtb
                for idx, energy_xyz in enumerate(sorted_energy_xyz_list):
                    # make directory
                    ff_opt_dir = os.path.join(save_opt_dir, f'conformer_{idx}')  # The lower idx, the lower energy
                    make_directory(ff_opt_dir)  # make dir

                    # write .xyz file
                    ff_optimized_geometry = os.path.join(ff_opt_dir, 'ff_optimized.xyz')
                    with open(ff_optimized_geometry, 'w') as file:
                        file.writelines(energy_xyz)
                        file.close()

                    # geometry optimization with XTB2 -- directly use the XTB2 (COSMO solvent)
                    sh_script.extend([
                        f'$xtb_path {ff_optimized_geometry} --opt tight -P {number_of_processor_per_job} --namespace {ff_opt_dir}/geometry --cosmo toluene --cycles 300 > {ff_opt_dir}/geometry.out\n',
                        f'cd {ff_opt_dir}\n'
                        f'rm geometry.charges geometry.wbo geometry.xtb.cosmo geometry.xtbopt.log geometry.xtbrestart geometry.xtbtopo.mol\n',
                    ])


        # append
        sh_script.extend([
            f'duration=$SECONDS\n',
            f'echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."\n',
        ])

        # save .sh file
        sh_file_path = os.path.join(BATCH_DIRECTORY, f'{reference_idx}_batch_xtb_{worker_idx}.sh')
        with open(sh_file_path, 'w') as sh_file:
            sh_file.writelines(sh_script)
            sh_file.close()

        # submit
        qsub_call = f'qsub {sh_file_path}'
        call(qsub_call, shell=True)
