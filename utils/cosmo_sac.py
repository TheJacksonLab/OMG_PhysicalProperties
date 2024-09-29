import re
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit.Geometry import Point3D

from io import StringIO

from math import ceil

# constants from COSMO-SAC 2002 by Mullins (2005)
# Table 1 from the paper:https://pubs.acs.org/doi/epdf/10.1021/ie901947m
q0 = 79.53  # [A^2]
r0 = 66.69  # [A^3]
z_coordination = 10
# r_av
AEFFPRIME = 7.5  # [A^2] -> effective area
c_hb = 85580.0  # kcal A^4 / mol/e^2
sigma_hb = 0.0084
EPS = 3.667  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
EO = 2.395e-4
FPOL = (EPS-1.0)/(EPS+0.5)
ALPHA = (0.3*AEFFPRIME**(1.5))/(EO)
alpha_prime = FPOL*ALPHA
R = 8.3144598/4184  # 0.001987 # but really: 8.3144598/4184


def read_sigma_profile(sigma_path):
    """
    This function reads the sigma profile from .sigma file. Note that this function only works for one sigma profile.
    Source: https://github.com/usnistgov/COSMOSAC/blob/master/COSMO-PurePython.ipynb
    """
    # regular expression
    sigma_contents = open(sigma_path).read()
    re_search = re.search('[\s\S]+#[\s\S]+#[\s\S]+T\n+([\s\S]+)', sigma_contents).group(1).rstrip()

    # get area & volume
    re_search_area = float(re.search('"area \[A\^2\]": +([0-9.]+)', sigma_contents).group(1).rstrip())
    re_search_volume = float(re.search('"volume \[A\^3\]": +([0-9.]+)', sigma_contents).group(1).rstrip())

    # dataframe
    columns = ['sigma [e/A^2]', 'p(sigma)*A [A^2]']
    df = pd.read_csv(StringIO(re_search), names=columns, sep=r'\s+', engine='python')
    df['p(sigma)'] = df['p(sigma)*A [A^2]'] / re_search_area

    return df, re_search_volume, re_search_area


def get_lngamma_comb(x, i, V_A3_list, A_A2_list):
    """
    This function calculates the combinatorial part of ln(γ_i).
    x: mol_fraction_list [solute, solvent]
    i: index to calculate the ln(γ_i), e.g. idx: 0 -> solute
    V_A3_list: volume list [solute, solvent]
    A_A2_list: area list [solute, solvent]
    """
    # get normalized area and volume
    q_arr = np.array(A_A2_list) / q0
    r_arr = np.array(V_A3_list) / r0

    # in case of infinite dilution
    theta_i_over_x_i = q_arr[i] / np.dot(x, q_arr)
    phi_i_over_x_i = r_arr[i] / np.dot(x, r_arr)
    theta_i_over_phi_i = theta_i_over_x_i / phi_i_over_x_i

    # get l
    l_arr = z_coordination/2 * (r_arr - q_arr) - (r_arr - 1)

    return np.log(phi_i_over_x_i) + z_coordination/2 * q_arr[i] * np.log(theta_i_over_phi_i) + l_arr[i] - phi_i_over_x_i * np.dot(x, l_arr)


def get_lngamma_free_volume_comb(x, i, free_volume_A3_list):
    """
    This function calculates the modified combinatorial part of ln(γ_i) with a "free volume argument".
    x: mol_fraction_list [polymer_solute, solvent]
    i: index to calculate the ln(γ_i), e.g. idx: 0 -> solute
    free_volume_A3_list: free volume list [polymer_solute, solvent]
    """
    # free volume fraction
    free_volume_fraction_i = x[i] * free_volume_A3_list[i] / np.dot(x, free_volume_A3_list)

    # calculate ln(γ_i)
    ln_gamma = np.log(free_volume_fraction_i / x[i]) + 1 - (free_volume_fraction_i / x[i])

    return ln_gamma


def get_lngamma_total_with_polymer_combinatorial_part(T, x, i, V_A3_list, free_volume_A3_list, A_A2_list, profile_list, lnGamma_mix, free_volume: bool):
    """
    Sum of the contributions (residual + "polymer" combinatorial) to ln(γ_i)
    T: temperature to calculate (K)
    x: mol_fraction_list [solute, solvent]
    i: index to calculate the ln(γ_i), e.g. idx: 0 -> solute
    V_A3_list: volume list [solute, solvent]
    free_volume_A3_list: free volume list [solute, solvent]
    A_A2_list: area list [solute, solvent]
    profile_list: unnormalized sigma profile list
    lnGamma_mix: activity coefficient array of a mixture [51,]

    return ln(γ_total)
    """
    # combinatorial term
    if free_volume:
        lnGamma_comb = get_lngamma_free_volume_comb(x=x, i=i, free_volume_A3_list=free_volume_A3_list)  # free volume
    else:
        lnGamma_comb = get_lngamma_comb(x=x, i=i, V_A3_list=V_A3_list, A_A2_list=A_A2_list)  # small molecule

    # residual term
    lnGamma_comb_resid = get_lngamma_resid(
        T=T, i=i, profile_list=profile_list, A_A2_list=A_A2_list, lnGamma_mix=lnGamma_mix
    )

    return lnGamma_comb + lnGamma_comb_resid


def get_Gamma(T, p_sigma, max_iteration=500):
    """
    Get the value of Γ (capital gamma) for the given "normalized" sigma profile.
    The sigma profile can be either from a mixture or a pure solute
    """
    # get shape of the p_sigma
    number_of_bins = p_sigma.shape[0]
    if number_of_bins == 51:
        max_sigma = 0.025
    elif number_of_bins == 101:
        max_sigma = 0.050
    else:
        raise ValueError("Number of bins must be 51 or 101")

    # get an exchange term
    sigma_tabulated = np.linspace(-max_sigma, max_sigma, number_of_bins)
    sigma_m = np.tile(sigma_tabulated, (len(sigma_tabulated), 1))  # shape -> [51, 51] -> charge density array
    sigma_n = np.tile(np.array(sigma_tabulated, ndmin=2).T, (1, len(sigma_tabulated)))  # transpose of sigma_m
    sigma_acc = np.tril(sigma_n) + np.triu(sigma_m, 1)  # each component: max(sigma_m, sigma_n)
    sigma_don = np.tril(sigma_m) + np.triu(sigma_n, 1)  # each component: min(sigma_m, sigma_n)

    # exchange term
    DELTAW = (alpha_prime / 2) * (sigma_m + sigma_n) ** 2 + c_hb * np.maximum(0, sigma_acc - sigma_hb) * np.minimum(0, sigma_don + sigma_hb)

    Gamma = np.ones_like(p_sigma)  # array of the shape: [51,] -> assumes an ideal behavior -> 1
    AA = np.exp(-DELTAW/(R*T)) * p_sigma  # constant and can be pre-calculated outside of the loop

    # solve the self-consistent equation
    difference = np.zeros_like(p_sigma)
    for i in range(max_iteration):
        Gammanew = 1 / np.sum(AA*Gamma, axis=1)
        difference = np.abs((Gamma-Gammanew)/Gamma)  # relative difference
        Gamma = (Gammanew + Gamma)/2  # update
        if np.max(difference) < 1e-8:
            break
        else:
            pass
    if np.max(difference) >= 1e-8:
        print(f"[WARNING] Iterative cycles for activity coefficient calculations yielded a relative difference: {np.max(difference)}")
        raise ValueError
    return Gamma, np.max(difference)


def get_lngamma_resid(T, i, profile_list, A_A2_list, lnGamma_mix):
    """
    This function calculates the residual contribution to ln(γ_i).
    T: temperature to calculate (K)
    i: index to calculate the ln(γ_i), e.g. idx: 0 -> solute
    profile_list: unnormalized sigma profile list
    A_A2_list: area list [solute, solvent]  [A^2]
    lnGamma_mix: activity coefficient array of a mixture [51,]
    """
    # For this component
    p_sigma_arr = profile_list[i]['p(sigma)'].to_numpy()
    area = A_A2_list[i]

    # Get the lnGamma_i
    lnGamma_i = np.log(get_Gamma(T, p_sigma_arr)[0])

    # calculate the residual term
    lngamma_resid_i = area / AEFFPRIME * np.sum(p_sigma_arr * (lnGamma_mix - lnGamma_i))

    return lngamma_resid_i


def get_lngamma_total(T, x, i, V_A3_list, A_A2_list, profile_list, lnGamma_mix):
    """
    Sum of the contributions (residual + combinatorial) to ln(γ_i)
    T: temperature to calculate (K)
    x: mol_fraction_list [solute, solvent]
    i: index to calculate the ln(γ_i), e.g. idx: 0 -> solute
    V_A3_list: volume list [solute, solvent]
    A_A2_list: area list [solute, solvent]
    profile_list: unnormalized sigma profile list
    lnGamma_mix: activity coefficient array of a mixture [51,]

    get_lngamma_comb(x, i, V_A3_list, A_A2_list)
    get_lngamma_resid(T, i, profile_list, A_A2_list, lnGamma_mix)
    """
    # combinatorial term
    lnGamma_comb = get_lngamma_comb(x=x, i=i, V_A3_list=V_A3_list, A_A2_list=A_A2_list)

    # residual term
    lnGamma_comb_resid = get_lngamma_resid(
        T=T, i=i, profile_list=profile_list, A_A2_list=A_A2_list, lnGamma_mix=lnGamma_mix
    )
    return lnGamma_comb + lnGamma_comb_resid


def convert_volume_fraction_to_mol_fraction(volume_fraction_solute, monomer_solute_sigma_path, solvent_sigma_path, repeating_unit_smi_asterisk_path, number_of_backbone_atoms):
    """
    This function converts a volume fraction to a mol fraction (polymer solute mol fraction)
    :return: mol fraction
    """
    # get information for monomer "solute" sigma profiles
    df_monomer_solute, volume_monomer_solute, area_monomer_solute = read_sigma_profile(monomer_solute_sigma_path)  # volume -> radius is 20% larger than the Van der Waals radius

    # scale up to a polymer sigma profile
    number_of_backbone_atoms_in_monomer = get_number_of_backbone_atoms(
        target_smi_path=repeating_unit_smi_asterisk_path)  # get the number of backbone atoms
    scale_constant = ceil(number_of_backbone_atoms / number_of_backbone_atoms_in_monomer)  # get the multiply constant
    df_polymer_solute = df_monomer_solute.copy()
    df_polymer_solute['p(sigma)*A [A^2]'] *= scale_constant  # scale up to get a polymer sigma profile
    volume_polymer_solute = volume_monomer_solute * scale_constant

    # get information for "solvent" sigma profiles
    df_solvent, volume_solvent, area_solvent = read_sigma_profile(solvent_sigma_path)  # volume -> radius is 20% larger than the Van der Waals radius

    # volume fraction
    volume_fraction_solvent = 1 - volume_fraction_solute

    # convert to a mol fraction
    mol_fraction_solute_over_solvent = (volume_fraction_solute / volume_fraction_solvent) * (volume_solvent / volume_polymer_solute)
    mol_fraction_solute = 1 / (1 + 1 / mol_fraction_solute_over_solvent)

    return mol_fraction_solute


def calculate_activity_coefficients_of_polymer_solute_and_solvent(
        monomer_solute_sigma_path, polymer_solute_mol_fraction, solvent_sigma_path, temperature, number_of_backbone_atoms,
        monomer_solute_mol_path, monomer_solute_geometry_path_to_import, solvent_mol_path, solvent_geometry_path_to_import,
        repeating_unit_smi_asterisk_path
):
    """
    This function calculates the activity coefficient of "polymer" solute and solvent following COSMO-SAC (2002 or Mullins).
    The sigma profile of a monomer is scaled up to have the given number of atoms (excluding terminating hydrogen) in the backbone.
    This function uses a modified combinatorial term for activity calculations.
    This function also returns cosmo volume of a solute and a solvent for Flory-Huggins Chi parameter calculation.
    :return: np.array([log activity coefficient of "polymer solute", log activity coefficient of solvent, "polymer solute" cosmo volume, solvent cosmo volume])
    """
    # get information for monomer "solute" sigma profiles
    df_monomer_solute, volume_monomer_solute, area_monomer_solute = read_sigma_profile(monomer_solute_sigma_path)  # volume -> radius is 20% larger than the Van der Waals radius

    # scale up to a polymer sigma profile
    number_of_backbone_atoms_in_monomer = get_number_of_backbone_atoms(target_smi_path=repeating_unit_smi_asterisk_path)  # get the number of backbone atoms
    scale_constant = ceil(number_of_backbone_atoms / number_of_backbone_atoms_in_monomer)  # get the multiply constant

    df_polymer_solute = df_monomer_solute.copy()
    df_polymer_solute['p(sigma)*A [A^2]'] *= scale_constant  # scale up to get a polymer sigma profile
    volume_polymer_solute = volume_monomer_solute * scale_constant
    area_polymer_solute = area_monomer_solute * scale_constant

    # get information for "solvent" sigma profiles
    df_solvent, volume_solvent, area_solvent = read_sigma_profile(solvent_sigma_path)  # volume -> radius is 20% larger than the Van der Waals radius
    solvent_mol_fraction = 1 - polymer_solute_mol_fraction

    # calculate a free volume - polymer
    hard_core_volume_monomer_solute = calculate_molecular_hard_core_volume(mol_path=monomer_solute_mol_path, geometry_path_to_import=monomer_solute_geometry_path_to_import)
    hard_core_volume_polymer_solute = hard_core_volume_monomer_solute * scale_constant  # scale up to a polymer
    free_volume_polymer = volume_polymer_solute - hard_core_volume_polymer_solute  # A^3

    # calculate a free volume - solvent
    hard_core_volume_solvent = calculate_molecular_hard_core_volume(mol_path=solvent_mol_path, geometry_path_to_import=solvent_geometry_path_to_import)
    free_volume_solvent = volume_solvent - hard_core_volume_solvent  # A^3

    # get the sigma profile of a mixture -- independent assumption
    profile_list = [df_polymer_solute, df_solvent]  # unnormalized sigma profile
    area_list = [area_polymer_solute, area_solvent]  # A^2
    volume_list = [volume_polymer_solute, volume_solvent]  # A^3
    free_volume_A3_list = [free_volume_polymer, free_volume_solvent]
    mol_fraction_list = [polymer_solute_mol_fraction, solvent_mol_fraction]  # solute & solvent
    profile_mixture = np.array([mol_fraction * profile['p(sigma)*A [A^2]'].to_numpy() for mol_fraction, profile in zip(mol_fraction_list, profile_list)]).sum(axis=0)
    averaged_area = sum([mol_fraction * area for mol_fraction, area in zip(mol_fraction_list, area_list)])  # averaged area
    p_sigma_mixture = profile_mixture / averaged_area  # note this is a normalized sigma profile -> sum ~ 1.0

    # get gamma of the mixture (activity coefficient of charge segments)
    Gamma, max_difference = get_Gamma(T=temperature, p_sigma=p_sigma_mixture, max_iteration=500)
    lnGamma_mix = np.log(Gamma)

    # get activity coefficients of solute and solvent molecules
    lngamma_solute = get_lngamma_total_with_polymer_combinatorial_part(
        T=temperature, x=mol_fraction_list, i=0, V_A3_list=volume_list, free_volume_A3_list=free_volume_A3_list, A_A2_list=area_list, profile_list=profile_list,
        lnGamma_mix=lnGamma_mix, free_volume=True
    )
    lngamma_solvent = get_lngamma_total_with_polymer_combinatorial_part(
        T=temperature, x=mol_fraction_list, i=1, V_A3_list=volume_list, free_volume_A3_list=free_volume_A3_list, A_A2_list=area_list, profile_list=profile_list,
        lnGamma_mix=lnGamma_mix, free_volume=True
    )

    return np.array([lngamma_solute, lngamma_solvent, volume_polymer_solute, volume_solvent])


def calculate_molecular_hard_core_volume(mol_path, geometry_path_to_import):
    """
    This function calculates a molecular hard core volume (van der Waals volume) via RDKit.
    :return: molecular hard core volume (A^3)
    """
    mol_to_calculate = import_geometry_to_mol_file(mol_path=mol_path, geometry_path_to_import=geometry_path_to_import)
    van_der_Waals_volume = AllChem.ComputeMolVolume(mol_to_calculate, confId=0, gridSpacing=0.2, boxMargin=2.0)  # default option & A^3

    return van_der_Waals_volume


def extract_coordinates_from_xyz(xyz_file_path):
    """
    This function extracts coordinate vectors from .xyz file
    :return: np.array[N, 3] where N is the number of atoms
    """
    coordinate_ls = list()
    with open(xyz_file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            if line_idx >= 2:  # coordinate information
                tmp_line = line.split()
                coordinate_ls.append([float(tmp_line[1]), float(tmp_line[2]), float(tmp_line[3])])

    return np.array(coordinate_ls)


def import_geometry_to_mol_file(mol_path, geometry_path_to_import):
    """
    This function overwrites the geometry (confId=0) to a .mol object and returns the mol .object
    :return: .mol object
    """
    mol_to_calculate = Chem.MolFromMolFile(mol_path, removeHs=False)  # create a mol object with explicit hydrogen
    conformer = mol_to_calculate.GetConformer(id=0)  # only one conformer
    xyz_to_use = extract_coordinates_from_xyz(geometry_path_to_import)  # load geometry to use
    for atom_idx in range(mol_to_calculate.GetNumAtoms()):  # update coordinates
        x, y, z = xyz_to_use[atom_idx]
        conformer.SetAtomPosition(atom_idx, Point3D(x, y, z))

    return mol_to_calculate


def get_number_of_backbone_atoms(target_smi_path):
    """
    This function calculates the number of backbone atoms excluding terminating hydrogen
    :return: number of backbone atoms
    """
    target_smi = open(target_smi_path, 'r').read()
    target_mol = Chem.MolFromSmiles(target_smi)
    asterisk_idx_list = list()
    for idx in range(target_mol.GetNumAtoms()):
        atom = target_mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(idx)
    dist = GetShortestPath(target_mol, asterisk_idx_list[0], asterisk_idx_list[1])
    backbone_length = len(dist) - 2

    return backbone_length


def calculate_flory_huggins_chi_parameter(solute_mol_fraction, solute_cosmo_volume, log_solute_activity_coefficient,
                                          solvent_cosmo_volume, log_solvent_activity_coefficient):
    """
    This function calculates Flory-Huggins chi parameter.
    return: Flory-Huggins chi parameter
    """
    # volume fraction
    solvent_mol_fraction = 1 - solute_mol_fraction
    volume_fraction_solute_over_solvent = (solute_cosmo_volume / solvent_cosmo_volume) * (solute_mol_fraction / solvent_mol_fraction)
    volume_fraction_solute = 1 / (1 + 1 / volume_fraction_solute_over_solvent)
    volume_fraction_solvent = volume_fraction_solute / volume_fraction_solute_over_solvent  # volume_fraction_solute + volume_fraction_solvent = 1

    # Flory-Huggins chi parameter -- chi dependent with solute
    term_1 = np.log(solute_mol_fraction) + log_solute_activity_coefficient - np.log(volume_fraction_solute)
    term_2 = (1 - (solute_cosmo_volume / solvent_cosmo_volume)) * volume_fraction_solvent
    term_3 = solute_cosmo_volume / solvent_cosmo_volume * volume_fraction_solvent**2
    chi_solute_depended_composition = (term_1 - term_2) / term_3

    # Flory-Huggins chi parameter -- chi dependent with solvent
    term_1 = np.log(solvent_mol_fraction) + log_solvent_activity_coefficient - np.log(volume_fraction_solvent)
    term_2 = (1 - (solvent_cosmo_volume / solute_cosmo_volume)) * volume_fraction_solute
    term_3 = volume_fraction_solute**2
    chi_solvent_depended_composition = (term_1 - term_2) / term_3

    # Flory-Huggins chi parameter -- total
    chi_parameter = chi_solute_depended_composition * volume_fraction_solvent + chi_solvent_depended_composition * volume_fraction_solute
    
    return chi_parameter
