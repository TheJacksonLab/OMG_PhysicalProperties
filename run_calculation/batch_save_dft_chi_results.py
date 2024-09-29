import os
import sys
import math
import numpy as np
import pandas as pd

import argparse

from pathlib import Path

RUN_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation')  # run dir
sys.path.append(RUN_DIR.parents[0].as_posix())  # to call utils

from utils import OrcaPropertyJobSubmit
from utils import read_final_single_point_energy_with_solvation, read_HOMO_and_LUMO_adjacent_energy
from utils import read_min_max_charge, read_electric_properties, read_singlet_triplet_transition
from utils import read_absorption_spectrum_via_transition_electric_dipole_moments
from utils import get_sigma_profiles
from utils import calculate_activity_coefficients_of_polymer_solute_and_solvent
from utils import calculate_flory_huggins_chi_parameter, convert_volume_fraction_to_mol_fraction

# Directory
SAVE_DIRECTORY = Path(os.path.join(RUN_DIR, './calculation_results'))
SOLVENT_DIRECTORY = os.path.join(RUN_DIR, 'solvent_sigma_profile_cosmo')  # geometry optimization under ideal solvent.
SOLVENT_LIST = ['water', 'ethanol', 'chloroform']
NUMBER_OF_BACKBONE_ATOMS = 1000  # for polymers / excluding terminating hydrogen

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
    save_df_dft_columns = ['final_single_point_energy_(kcal_mol)'] + \
                          ['min_partial_charge', 'max_partial_charge', 'HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1',
                           'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
                          ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength',
                           't1_energy']
    save_df_chi_columns = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

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

    # Orca property calculation class
    computing = 'cpu'
    property_job_manager = OrcaPropertyJobSubmit(computing=computing)

    # gather property data
    for reaction_id, polymerization_mechanism_idx in zip(reaction_id_list, polymerization_mechanism_idx_list):
        print(reaction_id)
        for chain_length in dft_chain_length_list:  # for extrapolation
            # set the input & save file path
            save_dir = os.path.join(save_directory, f'reaction_{reaction_id}')
            save_opt_dir = os.path.join(save_dir, f'opt_n_{chain_length}')
            monomer_mol_path = os.path.join(save_opt_dir, 'polymer.mol')
            repeating_unit_smi_asterisk_path = os.path.join(save_opt_dir, 'repeating_unit_asterisk.smi')

            # gather property data
            for conformer_idx in range(topn):
                conformer_dir = os.path.join(save_opt_dir, f'conformer_{conformer_idx}')

                # check property calculation results
                property_out = os.path.join(conformer_dir, 'property.out')
                if not os.path.exists(property_out):  # doesn't exist (previous procedure may have failed)
                    continue

                property_check = property_job_manager.check_if_orca_normally_finished(property_out)
                if property_check != 1:
                    # print(f'Please check the property output!: {property_out}')
                    continue

                # check cosmo (.cpcm) output file
                cpcm_path = os.path.join(conformer_dir, 'property_cosmo.cpcm')
                if not os.path.exists(cpcm_path):  # no output file
                    continue
                sigma_output_path = os.path.join(conformer_dir,
                                                 'sigma_cosmo.out')  # sigma profile with an ideal solvent.
                monomer_geometry_path = os.path.join(conformer_dir, 'geometry.xtbopt.xyz')  # set a geometry path

                # dft info list
                save_dft_csv_path = os.path.join(conformer_dir, 'dft_results.csv')
                if not os.path.exists(save_dft_csv_path):  # not gathered yet
                    sub_dft_result = list()
                    single_point_energy_with_solvation = read_final_single_point_energy_with_solvation(property_out)
                    sub_dft_result.append(
                        single_point_energy_with_solvation)  # final single point energy (kcal/mol) with the solvation effect for data collection

                    # DFT
                    min_partial_charge, max_partial_charge = read_min_max_charge(property_out, partitioning='Hirshfeld')
                    HOMO_minus_1, HOMO, LUMO, LUMO_plus_1 = read_HOMO_and_LUMO_adjacent_energy(property_out)
                    dipole_moment, quadrupole_moment, polarizability = read_electric_properties(property_out)
                    sub_dft_result.append(min_partial_charge)  # minimum partial charge
                    sub_dft_result.append(max_partial_charge)  # maximum partial charge
                    sub_dft_result.append(HOMO_minus_1)  # HOMO_minus_1
                    sub_dft_result.append(HOMO)  # HOMO
                    sub_dft_result.append(LUMO)  # LUMO
                    sub_dft_result.append(LUMO_plus_1)  # LUMO_plus_1
                    sub_dft_result.append(dipole_moment)  # dipole_moment
                    sub_dft_result.append(quadrupole_moment)  # quadrupole_moment
                    sub_dft_result.append(polarizability)  # polarizability

                    # TD-DFT
                    singlet_transition_energy_arr, triplet_transition_energy_arr = read_singlet_triplet_transition(
                        property_out)
                    s1_oscillator_strength, dominant_transition_energy, dominant_transition_oscillator_strength, singlet_oscillator_strength_arr = read_absorption_spectrum_via_transition_electric_dipole_moments(
                        property_out,
                        cutoff_wavelength=0)  # no cutoff. This function DOESN'T sort the spectrum based on their values.

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
                    sub_dft_result.append(sorted_singlet_transition_energy_arr[0])  # s1 energy

                    # check dominant transitions
                    if (dominant_transition_energy < 0) or (dominant_transition_oscillator_strength < 0):
                        print(f"Following has negative dominant transitions {property_out}")
                        continue

                    # append
                    sub_dft_result.append(dominant_transition_energy)  # dominant singlet transition energy
                    sub_dft_result.append(
                        dominant_transition_oscillator_strength)  # dominant singlet oscillator strength

                    # t1 energy
                    sorted_triplet_transition_energy_arr = np.sort(triplet_transition_energy_arr)
                    sub_dft_result.append(sorted_triplet_transition_energy_arr[0])  # t1 energy

                    # save dft results
                    if len(sub_dft_result) == len(save_df_dft_columns):  # all succeeds
                        df_dft_results = pd.DataFrame(columns=save_df_dft_columns, data=[sub_dft_result])
                        df_dft_results.to_csv(save_dft_csv_path, index=False)
                    else:
                        print(f"Some DFT results are missing: {conformer_dir}")

                # sigma profiles
                max_sigma_list = [0.025, 0.050]
                for max_sigma in max_sigma_list:
                    if not os.path.exists(sigma_output_path):  # if there is no sigma output path.
                        try:
                            get_sigma_profiles(inpath=cpcm_path, outpath=sigma_output_path,
                                               geometry_path=monomer_geometry_path,
                                               num_profiles=1, averaging='Mullins', max_sigma=max_sigma)
                        except ValueError:  # error happened during sigma profile calculations
                            pass

                # check
                if not os.path.exists(sigma_output_path):  # check .out from get_sigma_profiles
                    print(f"Failure to obtain sigma profile for {property_out}")
                    continue

                # activity coefficient calculations
                save_chi_csv_path = os.path.join(conformer_dir, 'chi_parameter.csv')
                if not os.path.exists(save_chi_csv_path):  # not gathered yet
                    sub_result = list()
                    polymer_volume_fraction = 0.2  # outside the critical regime.
                    for solvent in SOLVENT_LIST:  # calculate activity coefficients
                        solvent_path = os.path.join(SOLVENT_DIRECTORY, solvent)
                        # determine solvent sigma profile based on a solute sigma profile (shape)
                        sigma_lines = open(sigma_output_path, 'r').readlines()
                        max_sigma_from_profile = float(sigma_lines[-1].split()[0])
                        if max_sigma_from_profile == 0.025:
                            solvent_sigma_profile_path = os.path.join(solvent_path, 'sigma_cosmo.out')  # get solvent sigma profile
                        elif max_sigma_from_profile == 0.050:
                            solvent_sigma_profile_path = os.path.join(solvent_path, 'sigma_cosmo_0.050.out')  # get solvent sigma profile
                        else:
                            raise ValueError  # should be 0.025 or 0.050
                        solvent_mol_path = os.path.join(solvent_path, 'molecule.mol')
                        solvent_geometry_path = os.path.join(solvent_path, 'geometry.xtbopt.xyz')
                        mol_fraction_polymer_solute = convert_volume_fraction_to_mol_fraction(
                            volume_fraction_solute=polymer_volume_fraction, monomer_solute_sigma_path=sigma_output_path,
                            solvent_sigma_path=solvent_sigma_profile_path,
                            repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path,
                            number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS)

                        try:
                            log_activity_polymer_solute, log_activity_solvent, volume_polymer_solute, volume_solvent = calculate_activity_coefficients_of_polymer_solute_and_solvent(
                                # activity coefficients
                                monomer_solute_sigma_path=sigma_output_path,
                                polymer_solute_mol_fraction=mol_fraction_polymer_solute,
                                solvent_sigma_path=solvent_sigma_profile_path,
                                temperature=TEMPERATURE, number_of_backbone_atoms=NUMBER_OF_BACKBONE_ATOMS,
                                monomer_solute_mol_path=monomer_mol_path,
                                monomer_solute_geometry_path_to_import=monomer_geometry_path,
                                solvent_mol_path=solvent_mol_path,
                                solvent_geometry_path_to_import=solvent_geometry_path,
                                repeating_unit_smi_asterisk_path=repeating_unit_smi_asterisk_path)
                        except ValueError:
                            print(f"Following has numerical errors in getting activity coefficients for {solvent}: {property_out}")
                            continue

                        chi_parameter = calculate_flory_huggins_chi_parameter(
                            solute_mol_fraction=mol_fraction_polymer_solute, solute_cosmo_volume=volume_polymer_solute,
                            log_solute_activity_coefficient=log_activity_polymer_solute,
                            solvent_cosmo_volume=volume_solvent, log_solvent_activity_coefficient=log_activity_solvent)

                        sub_result.append(chi_parameter)  # chi_parameter for three different solvents

                    # save chi parameters
                    if len(sub_result) == len(save_df_chi_columns):  # all succeeds
                        df_chi_parameters = pd.DataFrame(columns=save_df_chi_columns, data=[sub_result])
                        df_chi_parameters.to_csv(save_chi_csv_path, index=False)
