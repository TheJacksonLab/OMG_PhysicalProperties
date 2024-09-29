import os
import sys
import numpy as np
import pandas as pd

import argparse

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/experimental_chi')
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils
from utils import OrcaPropertyJobSubmit
from utils import read_final_single_point_energy_with_solvation, read_HOMO_and_LUMO_adjacent_energy
from utils import read_min_max_charge, read_electric_properties, read_singlet_triplet_transition
from utils import read_absorption_spectrum_via_transition_electric_dipole_moments
from utils import get_sigma_profiles
from utils import calculate_activity_coefficients_of_polymer_solute_and_solvent
from utils import calculate_flory_huggins_chi_parameter, convert_volume_fraction_to_mol_fraction

SOLVENT_LIST = ['water', 'ethanol', 'chloroform']
NUMBER_OF_BACKBONE_ATOMS = 1000  # for polymers / excluding terminating hydrogen

# Unit Conversion
TEMPERATURE = 298.15  # K
Boltzmann_constant = 1.38 * 10 ** (-23)  # (J/K)
joule_to_kcal_mol = 6.02 / 4184 * 10**23
beta = 1 / (Boltzmann_constant * TEMPERATURE * joule_to_kcal_mol)  # 1 / (kcal/mol)


if __name__ == '__main__':
    # load df
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/experimental_chi/data/reference_chi_canon.csv')
    df_drop_duplicate = df.drop_duplicates(subset='Polymer')  # drop duplicates. The same shape with drop duplicate with canon smi
    df_drop_duplicate = df_drop_duplicate.reset_index(drop=True)

    # make dict
    polymer_dict = dict(zip(df_drop_duplicate['Polymer'], df_drop_duplicate.index))  # key: polymer, value: reaction idx

    # divide data
    reference_idx = 0  # for indexing. start_idx: reference_idx, end_idx: reference_idx + number_of_reactions
    number_of_reactions = df.shape[0]  # number of reactions to calculate chi

    # parameter
    chain_length_list = [1]
    topn = 15

    # idx
    reaction_start_idx = reference_idx
    reaction_end_idx = number_of_reactions + reference_idx
    submit_idx = list(range(reaction_start_idx, reaction_end_idx))  # submit idx

    # Orca property calculation class
    computing = 'cpu'
    property_job_manager = OrcaPropertyJobSubmit(computing=computing)

    # save final results (Boltzmann average, mean, and std) - HOMO (eV), LUMO (eV), DipoleMoment (au), QuadrupoleMoment (au), Polarizability (au)
    property_columns = ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity'] + \
                       ['molecular_weight', 'logP', 'qed', 'TPSA'] + \
                       ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
                       ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy'] + \
                       ['chi_parameter']
    # property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

    # save path
    SAVE_DIRECTORY = '/home/sk77/PycharmProjects/omg_database_publication/experimental_chi/calculation_results'
    SOLVENT_DIRECTORY = '/home/sk77/PycharmProjects/omg_database_publication/run_calculation/solvent_sigma_profile_cosmo'  # geometry under ideal solvent

    default_columns = ['reaction_idx', 'chain_length']
    save_columns = default_columns + ['number_of_conformers'] + [f'{column}_Boltzmann_average' for column in property_columns] + [f'{column}_mean' for column in property_columns] + [f'{column}_std' for column in property_columns]
    df_save_property = pd.DataFrame(columns=save_columns)

    # used to gather data for each conformer
    columns_for_conformer = default_columns + ['conformer_idx', 'final_single_point_energy_(kcal_mol)'] + property_columns
    df_property = pd.DataFrame(columns=columns_for_conformer)

    # gather property data
    for reaction_raw_idx in submit_idx:
        print(reaction_raw_idx)
        # convert to reaction_idx (without duplicates)
        polymer_name = df.iloc[reaction_raw_idx]['Polymer']
        reaction_idx = polymer_dict[polymer_name]

        # sub property df
        df_property_per_reaction_idx = pd.DataFrame(columns=columns_for_conformer)
        number_of_points_per_reaction = list()

        for chain_length in chain_length_list:  # for extrapolation
            # set the input & save file path
            save_dir = os.path.join(SAVE_DIRECTORY, f'reaction_{reaction_idx}')
            save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')
            monomer_mol_path = os.path.join(save_opt_dir, 'polymer.mol')
            repeating_unit_smi_asterisk_path = os.path.join(save_opt_dir, 'repeating_unit_asterisk.smi')

            # count data points
            point_count = 0

            # gather property data
            for conformer_idx in range(topn):
                conformer_dir = os.path.join(save_opt_dir, f'conformer_{conformer_idx}')

                # check property calculation results
                property_out = os.path.join(conformer_dir, 'property.out')
                if not os.path.exists(property_out):  # doesnt' exist (previous procedure may have failed)
                    continue

                property_check = property_job_manager.check_if_orca_normally_finished(property_out)
                if property_check != 1:
                    # print(f'Please check the property output!: {property_out}')
                    continue

                # check cosmo (.cpcm) output file
                cpcm_path = os.path.join(conformer_dir, 'property_cosmo.cpcm')
                if not os.path.exists(cpcm_path):  # no output file
                    continue
                sigma_output_path = os.path.join(conformer_dir, 'sigma_cosmo.out')  # set a sigma profile path
                monomer_geometry_path = os.path.join(conformer_dir, 'geometry.xtbopt.xyz')  # set a geometry path

                # info list
                point_count += 1
                sub_result = list()  # initialize
                sub_result.append(reaction_idx)  # 0) default column
                sub_result.append(chain_length)  # 0) default column
                sub_result.append(conformer_idx)  # 0) conformer_idx for data collection
                single_point_energy_with_solvation = read_final_single_point_energy_with_solvation(property_out)
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
                # sub_result.append(rdkit_csv['normalized_monomer_phi'][0])  # 10) normalized monomer phi idx
                # sub_result.append(rdkit_csv['normalized_backbone_phi'][0])  # 11) normalized backbone phi idx

                # DFT
                HOMO_minus_1, HOMO, LUMO, LUMO_plus_1 = read_HOMO_and_LUMO_adjacent_energy(property_out)
                dipole_moment, quadrupole_moment, polarizability = read_electric_properties(property_out)
                sub_result.append(HOMO_minus_1)  # 12) HOMO_minus_1
                sub_result.append(HOMO)  # 13) HOMO
                sub_result.append(LUMO)  # 14) LUMO
                sub_result.append(LUMO_plus_1)  # 15) LUMO_plus_1
                sub_result.append(dipole_moment)  # 16) dipole_moment
                sub_result.append(quadrupole_moment)  # 17) quadrupole_moment
                sub_result.append(polarizability)  # 18) polarizability

                # TD-DFT
                singlet_transition_energy_arr, triplet_transition_energy_arr = read_singlet_triplet_transition(property_out)
                s1_oscillator_strength, dominant_transition_energy, dominant_transition_oscillator_strength, singlet_oscillator_strength_arr = read_absorption_spectrum_via_transition_electric_dipole_moments(
                    property_out, cutoff_wavelength=0)  # no cutoff. This function DOESN'T sort the spectrum based on their values.

                # Caution -> "s1_oscillator_strength" from the above function may not be always positive!
                singlet_transition_positive_energy_idx = np.where(singlet_transition_energy_arr >= 0)[0]
                triplet_transition_positive_energy_idx = np.where(triplet_transition_energy_arr >= 0)[0]

                # check negative transition
                if (singlet_transition_energy_arr.shape[0] > singlet_transition_positive_energy_idx.shape[0]) or (
                        triplet_transition_energy_arr.shape[0] > triplet_transition_positive_energy_idx.shape[0]):  # nagative idx
                    print(f"Following has negative transitions {property_out}")
                    continue

                # s1 energy
                sorted_singlet_transition_energy_arr = np.sort(singlet_transition_energy_arr)
                sub_result.append(sorted_singlet_transition_energy_arr[0])  # 19) s1 energy

                # check dominant transitions
                if (dominant_transition_energy < 0) or (dominant_transition_oscillator_strength < 0):
                    print(f"Following has negative dominant transitions {property_out}")
                    continue

                # append
                sub_result.append(dominant_transition_energy)  # 20) dominant singlet transition energy
                sub_result.append(dominant_transition_oscillator_strength)  # 21) dominant singlet oscillator strength

                # t1 energy
                sorted_triplet_transition_energy_arr = np.sort(triplet_transition_energy_arr)
                sub_result.append(sorted_triplet_transition_energy_arr[0])  # 22) t1 energy

                # sigma profiles
                try:
                    get_sigma_profiles(inpath=cpcm_path, outpath=sigma_output_path, geometry_path=monomer_geometry_path,
                                       num_profiles=1, averaging='Mullins')
                except ValueError:  # error happened during sigma profile calculations
                    print(f"Following has numerical errors in getting sigma profile {property_out}")
                    continue

                # activity coefficient calculations
                # polymer_volume_fraction = 0.5  # outside the critical regime
                # for solvent in SOLVENT_LIST:  # calculate activity coefficients
                #     solvent_path = os.path.join(SOLVENT_DIRECTORY, solvent)
                #     # solvent_sigma_profile_path = os.path.join(solvent_path, 'sigma_toluene.out')  # get solvent sigma profile
                #     solvent_sigma_profile_path = os.path.join(solvent_path, 'sigma_cosmo.out')  # get solvent sigma profile
                #     solvent_mol_path = os.path.join(solvent_path, 'molecule.mol')
                #     solvent_geometry_path = os.path.join(solvent_path, 'geometry.xtbopt.xyz')
                #     mol_fraction_polymer_solute = convert_volume_fraction_to_mol_fraction(
                #         volume_fraction_solute=polymer_volume_fraction, monomer_solute_sigma_path=sigma_path,
                #         solvent_sigma_path=solvent_sigma_profile_path,
                #         repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path,
                #         number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS)
                #
                #     try:
                #         log_activity_polymer_solute, log_activity_solvent, volume_polymer_solute, volume_solvent = calculate_activity_coefficients_of_polymer_solute_and_solvent(
                #             # activity coefficients
                #             monomer_solute_sigma_path=sigma_path, polymer_solute_mol_fraction=mol_fraction_polymer_solute,
                #             solvent_sigma_path=solvent_sigma_profile_path,
                #             temperature=TEMPERATURE, number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS,
                #             monomer_solute_mol_path=monomer_mol_path,
                #             monomer_solute_geometry_path_to_import=monomer_geometry_path, solvent_mol_path=solvent_mol_path,
                #             solvent_geometry_path_to_import=solvent_geometry_path,
                #             repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path)
                #     except ValueError:
                #         print(f"Following has numerical errors in getting activity coefficients {property_out}")
                #         continue
                #
                #     chi_parameter = calculate_flory_huggins_chi_parameter(
                #         solute_mol_fraction=mol_fraction_polymer_solute, solute_cosmo_volume=volume_polymer_solute,
                #         log_solute_activity_coefficient=log_activity_polymer_solute,
                #         solvent_cosmo_volume=volume_solvent, log_solvent_activity_coefficient=log_activity_solvent)


                # get information
                solvent, temperature_C, polymer_volume_fraction = df.iloc[reaction_raw_idx]['solvent'], df.iloc[reaction_raw_idx]['temperature_C'], df.iloc[reaction_raw_idx]['polymer_volume_fraction']

                # solvent
                solvent_path = os.path.join(SOLVENT_DIRECTORY, solvent)
                solvent_sigma_profile_path = os.path.join(solvent_path, 'sigma_cosmo.out')  # get solvent sigma profile
                solvent_mol_path = os.path.join(solvent_path, 'molecule.mol')
                solvent_geometry_path = os.path.join(solvent_path, 'geometry.xtbopt.xyz')

                # convert volume fraction to a mol fraction
                if polymer_volume_fraction == 0.0:
                    polymer_volume_fraction = 1e-3  # avoid dividing by zero
                if polymer_volume_fraction == 1.0:
                    polymer_volume_fraction -= 1e-3  # avoid dividing by zero for a solvent

                mol_fraction_polymer_solute = convert_volume_fraction_to_mol_fraction(
                    volume_fraction_solute=polymer_volume_fraction, monomer_solute_sigma_path=sigma_output_path, solvent_sigma_path=solvent_sigma_profile_path,
                    repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path, number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS)
                temperature_K = temperature_C + 273.15

                # activity coefficients
                try:
                    log_activity_polymer_solute, log_activity_solvent, volume_polymer_solute, volume_solvent = calculate_activity_coefficients_of_polymer_solute_and_solvent( # activity coefficients
                        monomer_solute_sigma_path=sigma_output_path, polymer_solute_mol_fraction=mol_fraction_polymer_solute, solvent_sigma_path=solvent_sigma_profile_path,
                        temperature=temperature_K, number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS, monomer_solute_mol_path=monomer_mol_path,
                        monomer_solute_geometry_path_to_import=monomer_geometry_path, solvent_mol_path=solvent_mol_path,
                        solvent_geometry_path_to_import=solvent_geometry_path, repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path)
                except ValueError:
                    print(f"Following has numerical errors in getting activity coefficients {property_out}")
                    continue

                # chi parameter
                chi_parameter = calculate_flory_huggins_chi_parameter(
                    solute_mol_fraction=mol_fraction_polymer_solute, solute_cosmo_volume=volume_polymer_solute,
                    log_solute_activity_coefficient=log_activity_polymer_solute, solvent_cosmo_volume=volume_solvent, log_solvent_activity_coefficient=log_activity_solvent)
                sub_result.append(chi_parameter)  # 21) chi_parameter

                # append
                df_sub = pd.DataFrame(data=[sub_result], columns=columns_for_conformer)
                df_property_per_reaction_idx = pd.concat([df_property_per_reaction_idx, df_sub], axis=0, ignore_index=True)

            # number_of_points per chain length update
            number_of_points_per_reaction.append(point_count)

        # get statistics
        df_group_by_chain_length = df_property_per_reaction_idx.groupby('chain_length')
        df_group_by_chain_length = df_group_by_chain_length.agg(['mean', 'std'])

        # add number of points per reaction
        if len(df_group_by_chain_length.index) != len(number_of_points_per_reaction):  # no data
            continue

        else:
            df_group_by_chain_length['number_of_points'] = number_of_points_per_reaction

            # get Boltzmann weight
            energy_arr = df_property_per_reaction_idx['final_single_point_energy_(kcal_mol)'].to_numpy()
            energy_arr -= energy_arr.mean()  # mean shift for the calculation
            boltzmann_weight_arr = np.exp((-1) * beta * energy_arr)
            boltzmann_weight_arr /= boltzmann_weight_arr.sum()

            # append
            df_property_append = pd.DataFrame(columns=save_columns)
            df_property_append['reaction_idx'] = [int(df_group_by_chain_length[('reaction_idx', 'mean')].item())]  # reaction_idx
            df_property_append['chain_length'] = [1]  # chain_length fixed to 1
            df_property_append['number_of_conformers'] = [df_group_by_chain_length['number_of_points'].item()]

            # property columns
            for add_col in property_columns:
                df_property_append[f'{add_col}_Boltzmann_average'] = np.sum(df_property_per_reaction_idx[f'{add_col}'].to_numpy() * boltzmann_weight_arr)  # Boltzmann average at the room temperature
                df_property_append[f'{add_col}_mean'] = [df_group_by_chain_length[(f'{add_col}', 'mean')].item()]  # mean
                df_property_append[f'{add_col}_std'] = [df_group_by_chain_length[(f'{add_col}', 'std')].item()]  # std

            # concat
            df_save_property = pd.concat([df_save_property, df_property_append], axis=0, ignore_index=True)

    print(df_save_property)
    df_save_property.to_csv('./chi_results/chi_prediction_0_025_solute_toluene_cosmo_solvent_cosmo_cosmo.csv')
