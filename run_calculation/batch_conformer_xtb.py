import os
import sys
import numpy as np
import pandas as pd

import argparse

from pathlib import Path
from rdkit import Chem

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils
from utils import extract_energy_and_xyz_coordinates_from_multimolecule_xyz, make_directory


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='argument parser for run calculations')

    # Add positional arguments
    parser.add_argument('--data_path', type=str, help='Data (.csv) file path to run', default=None)
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning', default=None)
    parser.add_argument('--active_learning_iter', type=str, help='Iteration for active learning', default=None)
    parser.add_argument('--start_idx', type=int, help='Reaction start idx', default=None)
    parser.add_argument('--number_of_reactions', type=int, help='Reaction end idx.', default=None)
    parser.add_argument('--node_number_1', type=int, help='Node number 1 compute-{node_number_1}-{node_number_2}', default=None)
    parser.add_argument('--node_number_2', type=int, help='Node number 2 compute-{node_number_1}-{node_number_2}', default=None)
    parser.add_argument('--number_of_cpus', type=int, help='Number of CPUs to use', default=None)

    return parser.parse_args()  # Parse the arguments


if __name__ == '__main__':
    # argument parser
    args = get_args()
    data_path = args.data_path
    active_learning_strategy = args.active_learning_strategy
    active_learning_iter = args.active_learning_iter
    reference_idx = args.start_idx  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = args.number_of_reactions
    node_number_1 = args.node_number_1
    node_number_2 = args.node_number_2
    number_of_workers = args.number_of_cpus

    # save path
    SAVE_DIRECTORY = Path(os.path.join(RUN_DIR, 'calculation_results', active_learning_strategy, active_learning_iter))
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    BATCH_DIRECTORY = Path(os.path.join(RUN_DIR, './batch_script/conformer_xtb', active_learning_strategy, active_learning_iter))
    BATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # dir to save .sh files
    computing = 'cpu'
    number_of_processor_per_batch = 1
    SAVE_SH_FILE_DIRECTORY = Path(os.path.join(BATCH_DIRECTORY), f'start_idx_{reference_idx}_end_idx_{reference_idx + number_of_reactions}')
    SAVE_SH_FILE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # argument values
    save_directory = SAVE_DIRECTORY

    # load df
    df = pd.read_csv(data_path)

    # parameters
    chain_length_list = [1]
    topn = 15

    # batch
    reaction_idx_arr = np.arange(reference_idx, reference_idx + number_of_reactions)
    split_reaction_idx_arr = np.array_split(reaction_idx_arr, number_of_workers)
    for worker_idx, submit_idx in enumerate(split_reaction_idx_arr):
        # append reaction
        reaction_list = list()
        product_list = list()
        reaction_id_list = list()
        for row_idx in submit_idx:
            # append reactant .smi
            sub_list = [df['reactant_1'][row_idx], df['reactant_2'][row_idx]]
            reaction_list.append(sub_list)
            # append product .smi (canonical smiles)
            product_list.append(df['product'][row_idx])
            # reaction id list
            reaction_id_list.append(df['reaction_id'][row_idx])

        # .sh script
        sh_script = [
            f'#!/bin/csh\n',
            f'#$ -N XTB_{worker_idx}\n',
            f'#$ -wd {BATCH_DIRECTORY}\n'
            f'#$ -j y\n',
            f'#$ -o {reference_idx}_Batch_XTB_{worker_idx}.out\n',
            f'#$ -pe orte {number_of_processor_per_batch}\n',
            f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
            f'#$ -q all.q\n',
            f'SECONDS=0\n',
            f'xtb_path=/home/sk77/PycharmProjects/xtb/xtb_6.6.1\n',
        ]

        # geometry optimization with GAFF
        for reaction_id, reaction, repeating_unit_smi in zip(reaction_id_list, reaction_list, product_list):
            # save dir
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_id}')

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
                        f'$xtb_path {ff_optimized_geometry} --opt tight -P {number_of_processor_per_batch} --namespace {ff_opt_dir}/geometry --cosmo toluene --cycles 300 > {ff_opt_dir}/geometry.out\n',
                        f'cd {ff_opt_dir}\n'
                        f'rm geometry.charges geometry.wbo geometry.xtb.cosmo geometry.xtbopt.log geometry.xtbrestart geometry.xtbtopo.mol\n',
                    ])

        # append
        sh_script.extend([
            f'duration=$SECONDS\n',
            f'echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."\n',
        ])

        # save .sh file
        sh_file_path = os.path.join(SAVE_SH_FILE_DIRECTORY, f'{reference_idx}_batch_xtb_{worker_idx}.sh')
        with open(sh_file_path, 'w') as sh_file:
            sh_file.writelines(sh_script)
            sh_file.close()
