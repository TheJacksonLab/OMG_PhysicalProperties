import os
import sys
import numpy as np
import pandas as pd

import argparse

from rdkit import Chem
from pathlib import Path

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils
from utils import make_directory, draw_molecule, draw_molecule_from_mol, combine_polymer_recursively_terminating_CH3


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
    BATCH_DIRECTORY = Path(os.path.join(RUN_DIR, './batch_script/conformer_FF', active_learning_strategy, active_learning_iter))
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
    number_of_conformers = 30
    force_field = 'UFF'

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
            f'#$ -N Batch_{worker_idx}\n',
            f'#$ -wd {BATCH_DIRECTORY}\n'
            f'#$ -j y\n',
            f'#$ -o {reference_idx}_Batch_conformer_{worker_idx}.out\n',
            f'#$ -pe orte {number_of_processor_per_batch}\n',
            f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
            f'#$ -q all.q\n',
            f'SECONDS=0\n',
        ]

        # geometry optimization with GAFF
        for reaction_id, reaction, repeating_unit_smi in zip(reaction_id_list, reaction_list, product_list):
            # save dir
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_id}')
            make_directory(save_dir)

            # assign reactants
            reactant_1 = reaction[0]
            reactant_2 = reaction[1]

            # draw molecules
            # draw_molecule(reactant_1, os.path.join(save_dir, 'reactant_1.png'))
            # draw_molecule(reactant_2, os.path.join(save_dir, 'reactant_2.png'))
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

                # .sdf path to save conformers
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
        sh_file_path = os.path.join(SAVE_SH_FILE_DIRECTORY, f'{reference_idx}_batch_conformer_{worker_idx}.sh')
        with open(sh_file_path, 'w') as sh_file:
            sh_file.writelines(sh_script)
            sh_file.close()
