import os
import sys
import numpy as np
import pandas as pd

import argparse

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.Descriptors3D import Asphericity, Eccentricity, InertialShapeFactor, RadiusOfGyration, SpherocityIndex
from rdkit.Chem.Descriptors import ExactMolWt, TPSA
from rdkit.Chem.rdMolDescriptors import CalcPhi
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

from pathlib import Path
RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils
from utils import check_xtb_normally_finished
from utils import extract_backbone_from_molecule_depth_one, extract_coordinates_from_xyz
from utils import select_conformers, get_adjacency_matrix_from_combined_mol_path, get_adjacency_matrix_from_mol_with_modified_xyz
from utils import combine_polymer_recursively_terminating_CH3, OrcaPropertyCalculator


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
    BATCH_DIRECTORY = Path(os.path.join(RUN_DIR, './batch_script/dft', active_learning_strategy, active_learning_iter))
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
    dft_chain_length_list = [1]  # for chain length 1
    topn = 15

    # computation setting
    solvent = 'Toluene'  # solvent
    solvent_model = 'CPCMC'  # solvent model -> CPCMC: COSMO solvation (with COSMO epsilon function applied to CPCM)

    # result analysis
    geometry_fail_cnt = 0
    bond_matrix_fail_cnt = 0

    # rdkit columns
    rdkit_columns = ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity',
                     'molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi']

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
            f'#$ -N DFT_{worker_idx}\n',
            f'#$ -wd {BATCH_DIRECTORY}\n'
            f'#$ -j y\n',
            f'#$ -o {reference_idx}_Batch_DFT_{worker_idx}.out\n',
            f'#$ -pe orte {number_of_processor_per_batch}\n',
            f'#$ -l hostname=compute-{node_number_1}-{node_number_2}.local\n',
            f'#$ -q all.q\n',
            f'module load openmpi/4.1.1\n',
            f'SECONDS=0\n',
        ]

        # geometry optimization with GAFF
        for reaction_id, reaction, repeating_unit_smi in zip(reaction_id_list, reaction_list, product_list):
            # save dir
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_id}')

            # for interpolation -- DFT calculation
            for chain_length in dft_chain_length_list:
                # create directory
                save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')  # save dir for product
                polymer_mol_path = os.path.join(save_opt_dir, 'polymer.mol')  # with terminating explicit hydrogen

                # select conformers among geometries that (1) converged and (2) the same covalent topology
                xtb_geometry_xyz_path_list = list()  # contain only converged geometries
                xtb_geometry_output_path_list = list()
                for conformer_idx in range(topn):
                    conformer_dir = os.path.join(save_opt_dir, f'conformer_{conformer_idx}')

                    # check geometry optimization result
                    geometry_out = os.path.join(conformer_dir, 'geometry.out')
                    if not os.path.exists(geometry_out):
                        # In case there is no conformer (in case of a simple molecule)
                        continue

                    # check xtb output
                    geometry_check = check_xtb_normally_finished(geometry_out)
                    if geometry_check != 1:
                        # print(f'Please check the geometry output!: {geometry_out}')
                        geometry_fail_cnt += 1
                        continue
                    geometry_to_import = os.path.join(conformer_dir, 'geometry.xtbopt.xyz')  # optimized geometry

                    # compare based on a bond configuration matrix
                    desirable_adjacency_matrix = get_adjacency_matrix_from_combined_mol_path(polymer_mol_path)
                    adjacency_matrix_of_geometry_optimized_with_xtb = get_adjacency_matrix_from_mol_with_modified_xyz(
                        mol_file_path=polymer_mol_path, target_xyz_file_path=geometry_to_import
                    )

                    # compare
                    diff_matrix = np.abs(
                        np.triu(adjacency_matrix_of_geometry_optimized_with_xtb - desirable_adjacency_matrix)
                    )  # absolute values
                    if diff_matrix.sum() != 0.0:
                        bond_matrix_fail_cnt += 1
                        continue

                    # append
                    xtb_geometry_xyz_path_list.append(geometry_to_import)
                    xtb_geometry_output_path_list.append(geometry_out)

                # select conformers
                if len(xtb_geometry_xyz_path_list) == 0:  # no conformers
                    continue
                selected_idx_list = select_conformers(
                    list_of_geometry_xyz_path=xtb_geometry_xyz_path_list, list_of_geometry_output_path=xtb_geometry_output_path_list,
                    mol_file_path=polymer_mol_path, max_num_conformers=5
                )
                for selected_idx in selected_idx_list:
                    conformer_dir = os.path.dirname(xtb_geometry_xyz_path_list[selected_idx])
                    geometry_to_import = xtb_geometry_xyz_path_list[selected_idx]

                    # property calculation with DFT
                    polymer_property_calculator_orca = OrcaPropertyCalculator(conformer_dir)
                    polymer_property_calculation_input_file_path = polymer_property_calculator_orca.write(
                        dir_name=conformer_dir,
                        functional='revPBE',
                        basis_set=['def2-SVP', 'def2/J'],
                        geometry_to_import=geometry_to_import,
                        solvent_model=solvent_model,
                        solvent=solvent,
                        number_of_processor_per_job=number_of_processor_per_batch,
                        TD_DFT=True,
                        chain_cosmo=True,  # chain cosmo calculations for sigma profiles with an ideal solvent
                    )
                    polymer_property_output_file_path = os.path.join(conformer_dir, 'property.out')
                    sh_script.extend([
                        f'/share/apps/orca/orca_5_0_3/orca {polymer_property_calculation_input_file_path} > {polymer_property_output_file_path}\n'
                        f'cd {conformer_dir}\n',
                        f'rm property.gbw property_property.txt property.cis\n',
                    ])

                    # calculate rdkit properties - 1) 3D properties + 2) Chemical information from RDKit
                    mol_to_calculate = Chem.MolFromMolFile(polymer_mol_path, removeHs=False)  # create a mol object with explicit hydrogen
                    conformer = mol_to_calculate.GetConformer(id=0)  # only one conformer
                    xyz_to_use = extract_coordinates_from_xyz(geometry_to_import)  # load geometry to use
                    for atom_idx in range(mol_to_calculate.GetNumAtoms()):  # update coordinates
                        x, y, z = xyz_to_use[atom_idx]
                        conformer.SetAtomPosition(atom_idx, Point3D(x, y, z))

                    # 1) 3D geometry properties depending on a molecular geometry
                    asphericity = Asphericity(mol_to_calculate, confId=0, useAtomicMasses=True)  # Asphericity
                    eccentricity = Eccentricity(mol_to_calculate, confId=0, useAtomicMasses=True)  # Eccentricity
                    inertial_shape_factor = InertialShapeFactor(mol_to_calculate, confId=0, useAtomicMasses=True)  # InertialShapeFactor
                    radius_of_gyration = RadiusOfGyration(mol_to_calculate, confId=0, useAtomicMasses=True)  # RadiusOfGyration
                    spherocity = SpherocityIndex(mol_to_calculate, confId=0)  # SpherocityIndex without weights

                    # 2) chemical information "not" depending on a molecular geometry
                    molecular_weight = ExactMolWt(mol_to_calculate)  # Exact molecular weight with hydrogen
                    logP = MolLogP(mol_to_calculate, includeHs=True)  # LogP
                    qed_score = qed(mol_to_calculate)  # QED
                    TPSA_score = TPSA(mol_to_calculate, includeSandP=False)  # TPSA - ignore contributions from S and P as the original paper

                    # 3) calculate Phi idx to estimate a chain flexibility
                    repeating_unit_mol = Chem.MolFromSmiles(repeating_unit_smi)  # .smi with non-explicit hydrogen
                    total_number_of_atoms_in_monomer = repeating_unit_mol.GetNumHeavyAtoms()  # without asterisks
                    backbone_repeating_unit_smi = extract_backbone_from_molecule_depth_one(repeating_unit_smi)  # .smi with non-explicit hydrogen
                    repeating_number_list = [0, 1]
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

                    # phi idx
                    monomer_phi_idx = repeating_phi_list[1] - repeating_phi_list[0]
                    backbone_phi_idx = backbone_phi_list[1] - backbone_phi_list[0]

                    # normalized
                    normalized_monomer_phi_idx = monomer_phi_idx / total_number_of_atoms_in_monomer
                    normalized_backbone_phi_idx = backbone_phi_idx / total_number_of_atoms_in_monomer

                    ### save rdkit properties ###
                    data_array = np.array([asphericity, eccentricity, inertial_shape_factor, radius_of_gyration, spherocity, molecular_weight, logP, qed_score, TPSA_score, normalized_monomer_phi_idx, normalized_backbone_phi_idx])
                    df_rdkit = pd.DataFrame(columns=rdkit_columns, data=[data_array])
                    df_rdkit.to_csv(os.path.join(conformer_dir, 'rdkit.csv'), index=False)

        # append
        sh_script.extend([
            f'duration=$SECONDS\n',
            f'echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."\n',
        ])

        # save .sh file
        sh_file_path = os.path.join(SAVE_SH_FILE_DIRECTORY, f'{reference_idx}_batch_dft_{worker_idx}.sh')
        with open(sh_file_path, 'w') as sh_file:
            sh_file.writelines(sh_script)
            sh_file.close()

    print(f'Geometry fail {geometry_fail_cnt}')
    print(f'Bond matrix fail {bond_matrix_fail_cnt}')
