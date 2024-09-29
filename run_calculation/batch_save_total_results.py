import os
import sys
import math
import numpy as np
import pandas as pd

import argparse

from pathlib import Path

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils


# Directory
SAVE_DIRECTORY = Path(os.path.join(RUN_DIR, './calculation_results'))
CSV_SAVE_DIRECTORY = Path(os.path.join(RUN_DIR, './calculation_results_csv'))
CSV_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Unit Conversion
TEMPERATURE = 298.15  # K
Boltzmann_constant = 1.38 * 10 ** (-23)  # (J/K)
joule_to_kcal_mol = 6.02 / 4184 * 10 ** 23
beta = 1 / (Boltzmann_constant * TEMPERATURE * joule_to_kcal_mol)  # 1 / (kcal/mol)


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='argument parser for run calculations')

    # Add positional arguments
    parser.add_argument('--data_path', type=str, help='Data (.csv) file path to run', default=None)
    parser.add_argument('--active_learning_strategy', type=str, help='Strategy for active learning', default=None)
    parser.add_argument('--active_learning_iter', type=str, help='Iteration for active learning', default=None)
    parser.add_argument('--start_idx', type=int, help='Reaction start idx', default=None)
    parser.add_argument('--number_of_reactions', type=int, help='Reaction end idx.', default=None)

    return parser.parse_args()  # Parse the arguments


if __name__ == '__main__':
    # argument parser
    args = get_args()
    data_path = args.data_path
    active_learning_strategy = args.active_learning_strategy
    active_learning_iter = args.active_learning_iter
    reference_idx = args.start_idx  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = args.number_of_reactions

    # argument values
    save_directory = os.path.join(SAVE_DIRECTORY, active_learning_strategy, active_learning_iter)

    # load df
    df = pd.read_csv(data_path)

    # parameter
    dft_chain_length_list = [1]  # for chain length 1
    topn = 15

    # batch
    reaction_idx_arr = np.arange(reference_idx, reference_idx + number_of_reactions)
    polymerization_mechanism_idx_list = list()
    reaction_id_list = list()
    for row_idx in reaction_idx_arr:
        # polymerization mechanism idx
        polymerization_mechanism_idx_list.append(df['polymerization_mechanism_idx'][row_idx])
        # reaction id list
        reaction_id_list.append(df['reaction_id'][row_idx])

    # save final results (Boltzmann average, mean, and std)
    property_columns = ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity'] + \
                       ['molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi'] + \
                       ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
                       ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy'] + \
                       ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

    default_columns = ['reaction_id', 'polymerization_mechanism_idx', 'chain_length']
    save_columns = default_columns + ['number_of_conformers'] + [f'{column}_Boltzmann_average' for column in property_columns] + \
                   [f'{column}_Boltzmann_std' for column in property_columns] + [f'{column}_mean' for column in property_columns] + \
                   [f'{column}_std' for column in property_columns]
    df_save_property = pd.DataFrame(columns=save_columns)

    # used to gather data for each conformer
    columns_for_conformer = default_columns + ['conformer_idx', 'final_single_point_energy_(kcal_mol)'] + property_columns
    df_property = pd.DataFrame(columns=columns_for_conformer)

    # gather property data
    for reaction_id, polymerization_mechanism_idx in zip(reaction_id_list, polymerization_mechanism_idx_list):
        print(reaction_id)
        # sub property df
        df_property_per_reaction_id = pd.DataFrame(columns=columns_for_conformer)
        number_of_points_per_reaction = list()

        for chain_length in dft_chain_length_list:  # for extrapolation
            # set the input & save file path
            save_dir = os.path.join(save_directory, f'reaction_{reaction_id}')
            save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')
            monomer_mol_path = os.path.join(save_opt_dir, 'polymer.mol')
            repeating_unit_smi_asterisk_path = os.path.join(save_opt_dir, 'repeating_unit_asterisk.smi')

            # count data points
            point_count = 0

            # gather property data
            for conformer_idx in range(topn):
                conformer_dir = os.path.join(save_opt_dir, f'conformer_{conformer_idx}')

                # check dft csv
                df_dft_results_path = os.path.join(conformer_dir, 'dft_results.csv')
                if not os.path.exists(df_dft_results_path):
                    continue

                # check chi csv
                df_chi_results_path = os.path.join(conformer_dir, 'chi_parameter.csv')
                if not os.path.exists(df_chi_results_path):
                    continue

                # info list
                df_dft_results = pd.read_csv(df_dft_results_path)
                sub_result = list()
                sub_result.append(reaction_id)  # 0) default column
                sub_result.append(polymerization_mechanism_idx)  # 0) default column
                sub_result.append(chain_length)  # 0) default column
                sub_result.append(conformer_idx)  # 0) conformer_idx for data collection
                single_point_energy_with_solvation = df_dft_results['final_single_point_energy_(kcal_mol)'][0]
                sub_result.append(single_point_energy_with_solvation)  # 0) final single point energy (kcal/mol) with the solvation effect for data collection

                # load the rdkit.csv
                rdkit_csv = pd.read_csv(os.path.join(conformer_dir, 'rdkit.csv'))
                sub_result.append(rdkit_csv['asphericity'][0])  # 1) asphericity
                sub_result.append(rdkit_csv['eccentricity'][0])  # 2) eccentricity
                sub_result.append(rdkit_csv['inertial_shape_factor'][0])  # 3) inertial_shape_factor
                sub_result.append(rdkit_csv['radius_of_gyration'][0])  # 4) radius_of_gyration
                sub_result.append(rdkit_csv['spherocity'][0])  # 5) spherocity
                sub_result.append(rdkit_csv['molecular_weight'][0])  # 6) molecular_weight
                sub_result.append(rdkit_csv['logP'][0])  # 7) logP
                sub_result.append(rdkit_csv['qed'][0])  # 8) qed
                sub_result.append(rdkit_csv['TPSA'][0])  # 9) TPSA
                sub_result.append(rdkit_csv['normalized_monomer_phi'][0])  # 10) normalized monomer phi idx
                sub_result.append(rdkit_csv['normalized_backbone_phi'][0])  # 11) normalized backbone phi idx

                # DFT
                sub_result.append(df_dft_results['HOMO_minus_1'][0])  # 12) HOMO_minus_1
                sub_result.append(df_dft_results['HOMO'][0])  # 13) HOMO
                sub_result.append(df_dft_results['LUMO'][0])  # 14) LUMO
                sub_result.append(df_dft_results['LUMO_plus_1'][0])  # 15) LUMO_plus_1
                sub_result.append(df_dft_results['dipole_moment'][0])  # 16) dipole_moment
                sub_result.append(df_dft_results['quadrupole_moment'][0])  # 17) quadrupole_moment
                sub_result.append(df_dft_results['polarizability'][0])  # 18) polarizability

                # TD-DFT
                sub_result.append(df_dft_results['s1_energy'][0])  # 19) s1 energy
                sub_result.append(df_dft_results['dominant_transition_energy'][0])  # 20) dominant singlet transition energy
                sub_result.append(df_dft_results['dominant_transition_oscillator_strength'][0])  # 21) dominant singlet oscillator strength
                sub_result.append(df_dft_results['t1_energy'][0])  # 22) t1 energy

                # chi parameters
                df_chi_results = pd.read_csv(df_chi_results_path)
                sub_result.append(df_chi_results['chi_parameter_water'][0])  # 23) chi_parameter for water
                sub_result.append(df_chi_results['chi_parameter_ethanol'][0])  # 24) chi_parameter for ethanol
                sub_result.append(df_chi_results['chi_parameter_chloroform'][0])  # 25) chi_parameter for chloroform

                # append
                point_count += 1  # passes all criteria.
                df_sub = pd.DataFrame(data=[sub_result], columns=columns_for_conformer)
                df_property_per_reaction_id = pd.concat([df_property_per_reaction_id, df_sub], axis=0, ignore_index=True)

            # number_of_points per chain length update
            number_of_points_per_reaction.append(point_count)

        # get statistics
        df_group_by_chain_length = df_property_per_reaction_id.groupby('chain_length')
        df_group_by_chain_length = df_group_by_chain_length.agg(['mean', 'std'])

        # add number of points per reaction
        if len(df_group_by_chain_length.index) != len(number_of_points_per_reaction):  # no data
            continue

        else:
            # get Boltzmann weight
            energy_arr = df_property_per_reaction_id['final_single_point_energy_(kcal_mol)'].to_numpy().copy()
            energy_arr -= energy_arr.mean()  # mean shift for the calculation
            boltzmann_weight_arr = np.exp((-1) * beta * energy_arr)
            boltzmann_weight_arr /= boltzmann_weight_arr.sum()

            # append
            df_property_append = pd.DataFrame(columns=save_columns)
            df_property_append['reaction_id'] = [int(df_group_by_chain_length[('reaction_id', 'mean')].item())]  # reaction_id
            df_property_append['polymerization_mechanism_idx'] = [int(df_group_by_chain_length[('polymerization_mechanism_idx', 'mean')].item())]  # reaction_mechanism_idx
            df_property_append['chain_length'] = [1]  # chain_length fixed to 1
            df_property_append['number_of_conformers'] = [number_of_points_per_reaction]

            # property columns
            for add_col in property_columns:
                boltzmann_average = np.average(df_property_per_reaction_id[f'{add_col}'].to_numpy(),
                                               weights=boltzmann_weight_arr)  # Boltzmann average
                boltzmann_variance = np.average(
                    (df_property_per_reaction_id[f'{add_col}'].to_numpy() - boltzmann_average) ** 2,
                    weights=boltzmann_weight_arr)  # Boltzmann variance
                mean = np.average(df_property_per_reaction_id[f'{add_col}'].to_numpy())
                variance = np.average((df_property_per_reaction_id[f'{add_col}'].to_numpy() - mean) ** 2)

                # append
                df_property_append[f'{add_col}_Boltzmann_average'] = boltzmann_average
                df_property_append[f'{add_col}_Boltzmann_std'] = math.sqrt(boltzmann_variance)
                df_property_append[f'{add_col}_mean'] = mean  # mean
                df_property_append[f'{add_col}_std'] = math.sqrt(variance)  # std

            # concat
            df_save_property = pd.concat([df_save_property, df_property_append], axis=0, ignore_index=True)

    # save
    result_save_dir = Path(os.path.join(CSV_SAVE_DIRECTORY, active_learning_strategy, active_learning_iter))
    result_save_dir.mkdir(exist_ok=True, parents=True)
    df_save_property.to_csv(
        os.path.join(result_save_dir, f'start_idx_{reference_idx}_end_idx_{reference_idx + number_of_reactions}.csv'),
        index=False)
