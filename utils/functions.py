import os
import re
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolfiles import MolToXYZBlock
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, GetShortestPath, GetMolFrags
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdchem import RWMol, BondType, Atom
from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolAlign import AlignMol

from scipy import stats
from scipy.spatial import distance_matrix
NUMBER_OF_PRINTED_MO_PER_LINE = 6
ATOMIC_SYMBOL_AFTER_CARTESIAN_COORDINATES_LINE = 2

# checked
def read_final_single_point_energy_with_solvation(output_file_path):
    """
    This function reads the final single point energy from a property calculation of one molecule. If this function is
    used for a chain calculation, this function reads the energy from the first job.
    This function considers a solvation effect (gas phase reaction at 1 atm dissolved at a solvent)
    Note that the molecule is still in a gas phase (additional 1.89 kcal/mol should be added for a liquid phase)
    :return: Final single point energy (kcal/mol) (ORCA - 5.0.3) -> everything is included.
    """
    # reactant
    flag = 0
    Eh_to_kcal_mol = 627.5
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'FINAL SINGLE POINT ENERGY' in line:
                tmp_line = line.split()
                single_point_energy = float(tmp_line[-1])  # Eh
                single_point_energy *= Eh_to_kcal_mol  # kcal/mol
                flag += 1

            if flag == 1:
                break
        f.close()

    return single_point_energy

# checked
def read_min_max_charge(file_path, partitioning):
    """
    This function reads out atomic partial min & max charges according to the inserted paritioning from an orca property calculation file.
    This function only supports for CHELPG and Hirshfeld
    :return: [min_charge, max_charge]
    """
    # read charge
    flag = 0
    charge_lines = list()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if partitioning.lower() == 'chelpg':
            for line in lines:
                if 'CHELPG Charges  ' in line:
                    flag = 1
                if 'Total charge:' in line:
                    flag = 0
                if flag == 1:
                    charge_lines.append(line)
                if 'JOB NUMBER  2' in line:  # stop if moving over to the second job.
                    break
            charge_lines = charge_lines[2:-1]

        elif partitioning.lower() == 'hirshfeld':
            for line in lines:
                if 'HIRSHFELD ANALYSIS' in line:
                    flag = 1
                if '  TOTAL' in line:
                    flag = 0
                if flag == 1:
                    charge_lines.append(line)
                if 'JOB NUMBER  2' in line:  # stop if moving over to the second job.
                    break
            charge_lines = charge_lines[7:-1]

        else:
            raise ValueError('Please check the input partitioning method!')

    # get min & max charge
    charge_list = list()
    if partitioning.lower() == 'chelpg':
        extract_idx = -1
    elif partitioning.lower() == 'hirshfeld':
        extract_idx = -2
    else:
        raise ValueError('Please check the input partitioning method!')

    for line in charge_lines:
        line_split = line.split()
        charge_list.append(float(line_split[extract_idx]))
    charge_arr = np.array(charge_list)

    return [np.min(charge_arr), np.max(charge_arr)]

# checked
def read_singlet_triplet_transition(file_path):
    """
    This function reads out singlet & triplet absorption (vertical transitions) from an orca output file with a solvation model.
    This function should be used with single TD-DFT job.
    :return: np.arr (singlet transition energy in eV), np.arr (triplet transition energy in eV)
    e.g.) [s1 energy, s2 energy, s3 energy, ...], [t1 energy, t2 energy, t3 energy, ...]
    """
    # extract singlet & triplet absorption transition energies
    td_dft_contents = open(file_path).read()
    re_search = re.search('(TD-DFT/TDA EXCITED STATES \(SINGLETS\)[\s\S]+)(TD-DFT/TDA EXCITED STATES \(TRIPLETS\)[\s\S]+-----------------------------\nTD-DFT/TDA-EXCITATION SPECTRA)', td_dft_contents, re.DOTALL)

    singlet_contents, triplet_contents = re_search.group(1), re_search.group(2)  # singlet & triplet contents

    # get singlet transition energies
    singlet_transition_energy_list = list()
    singlet_lines = singlet_contents.splitlines()
    for line in singlet_lines:
        if line.startswith('STATE'):
            tmp_line = line.split()
            singlet_transition_energy_list.append(float(tmp_line[-7]))  # <S**2> =   2.000000 are added

    # get triplet transition energies
    triplet_transition_energy_list = list()
    triplet_lines = triplet_contents.splitlines()
    for line in triplet_lines:
        if line.startswith('STATE'):
            tmp_line = line.split()
            triplet_transition_energy_list.append(float(tmp_line[-7]))  # <S**2> =   2.000000 are added

    return np.array(singlet_transition_energy_list), np.array(triplet_transition_energy_list)

# checked
def read_absorption_spectrum_via_transition_electric_dipole_moments(file_path, cutoff_wavelength=200):
    """
    This function reads out absorption spectrum via transition electric dipole moments from an orca output file with a solvation model.
    The cutoff wavelength is set to pick the most dominant transition.
    This function should be used with single TD-DFT job.
    absorption_spectrum_arr -> np.array([number of states, 3])  (3: Absorption energy (eV), wavelength (nm), oscillator strength)
    :return: s1_oscillator_strength, dominant_transition_energy (eV), dominant_transition_oscillator_strength, singlet_oscillator_strength_arr
    """
    # extract absorption spectrum via transition electric dipole moments
    td_dft_contents = open(file_path).read()
    re_search = re.search('(ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS[\s\S]+)(ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS[\s\S]+CD SPECTRUM)', td_dft_contents, re.DOTALL)
    absorption_spectrum_electric_dipole_moments, absorption_spectrum_velocity_dipole_moments = re_search.group(1), re_search.group(2)

    # get spectrum
    spectrum_line_list = list()
    spectrum_lines = absorption_spectrum_electric_dipole_moments.splitlines()
    for line in spectrum_lines:
        tmp_line = line.split()
        try:  # convert to float
            check = float(tmp_line[-1]) if len(tmp_line) > 0 else False
            if len(tmp_line) == 8:
                float_line = [float(comp) for comp in tmp_line]
                energy = float_line[1]  # cm-1
                absorption_energy = energy * 0.00012398  # cm-1 -> eV
                spectrum_line_list.append([absorption_energy, float_line[2], float_line[3]])  # [absorption energy (eV), wavelength (nm), fosc]
        except ValueError:  # error to convert to float
            pass

    # pick the s1 oscillator strength (probability of absorption)
    spectrum_arr = np.array(spectrum_line_list)
    s1_oscillator_strength = spectrum_arr[0][2]
    singlet_oscillator_strength_arr = spectrum_arr[:, 2]
    try:  # if the dominant electric dipole single transition energy is less than cutoff (longer wavelength than cutoff)
        # pick the dominant oscillator strength and the corresponding dominant transition energy
        idx_to_pick = np.where(spectrum_arr[:, 1] >= cutoff_wavelength)[0]
        filtered_spectrum_arr = spectrum_arr[idx_to_pick]
        dominant_transition_idx = np.argmax(filtered_spectrum_arr[:, 2])
        dominant_transition = filtered_spectrum_arr[dominant_transition_idx]
        dominant_transition_energy, dominant_transition_oscillator_strength = dominant_transition[0], dominant_transition[2]

    except ValueError:  # there is no dominant electric dipole singlet transition whose transition energy is less than cutoff
        dominant_transition_energy, dominant_transition_oscillator_strength = 0.0, 0.0

    return s1_oscillator_strength, dominant_transition_energy, dominant_transition_oscillator_strength, singlet_oscillator_strength_arr

# checked
def read_HOMO_and_LUMO_adjacent_energy(file_path) -> list:
    """
    This function reads HOMO-1, HOMO, LUMO, and LUMO + 1 energy (eV)
    :return: [HOMO-1, HOMO, LUMO, LUMO+1]
    """
    # get line idx to start the orbital energies
    orbital_line_start_idx = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            tmp_line = line.split()
            if len(tmp_line) == 0:  # no information
                continue
            if tmp_line[0] == 'NO' and tmp_line[1] == 'OCC' and tmp_line[2] == 'E(Eh)' and tmp_line[3] == 'E(eV)':
                orbital_line_start_idx = line_idx
            if 'JOB NUMBER  2' in line:  # stop if moving over to the second job.
                break
        file.close()

    # gather HOMO and LUMO line idx
    LUMO_line_idx = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines[orbital_line_start_idx + 1:]):
            tmp_line = line.split()
            occ = float(tmp_line[1])
            if occ == 0.0:  # LUMO idx
                LUMO_line_idx = orbital_line_start_idx + line_idx + 1  # start from adding 1
                break
        file.close()
    HOMO_line_idx = LUMO_line_idx - 1
    LUMO_plus_1_line_idx = LUMO_line_idx + 1
    HOMO_minus_1_line_idx = HOMO_line_idx - 1

    # gather HOMO and LUMO adjacent energy
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # HOMO minus 1 energy (eV)
        HOMO_minus_1_line = lines[HOMO_minus_1_line_idx].split()
        HOMO_minus_1_energy = float(HOMO_minus_1_line[-1])

        # HOMO energy (eV)
        HOMO_line = lines[HOMO_line_idx].split()
        HOMO_energy = float(HOMO_line[-1])

        # LUMO energy (eV)
        LUMO_line = lines[LUMO_line_idx].split()
        LUMO_energy = float(LUMO_line[-1])

        # LUMO plus 1 energy (eV)
        LUMO_plus_1_line = lines[LUMO_plus_1_line_idx].split()
        LUMO_plus_1_energy = float(LUMO_plus_1_line[-1])
        file.close()

    return [HOMO_minus_1_energy, HOMO_energy, LUMO_energy, LUMO_plus_1_energy]

# checked
def read_electric_properties(file_path) -> list:
    """
    This function reads dipole moment, quadrupole moment, and polarizability (all of them are a.u.)
    :return: [dipole moment, quadrupole moment, polarizability]
    * quadrupole moment -> isotropic quadrupole moment
    * polarizability -> isotropic polarizability
    """
    # get electric properties
    dipole_moment = None
    quadrupole_moment = None
    polarizability = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            tmp_line = line.split()
            if 'Magnitude (a.u.)' in line:
                dipole_moment = float(tmp_line[-1])
            if 'Isotropic quadrupole' in line:
                quadrupole_moment = float(tmp_line[-1])
            if 'Isotropic polarizability' in line:
                polarizability = float(tmp_line[-1])
            if 'JOB NUMBER  2' in line:  # stop if moving over to the second job.
                break
        file.close()

    return [dipole_moment, quadrupole_moment, polarizability]


def draw_molecule(smi, save_path, width=600, height=600):
    """
    This function draws molecules with explicit hydrogen and atom idx
    :param mol: Rdkit.Mol object
    :param save_path: path to save an image
    "https://stackoverflow.com/questions/67189346/how-to-save-ipython-core-display-svg-as-png-file"

    return None
    """
    mol = Chem.MolFromSmiles(smi)  # get mol
    # mol = Chem.AddHs(mol)  # add explicit hydrogen
    number_of_atoms = mol.GetNumAtoms()
    # for atom_idx in range(number_of_atoms):
    #     mol.GetAtomWithIdx(atom_idx).SetProp('atomNote', str(mol.GetAtomWithIdx(atom_idx).GetIdx()))

    # create png
    d2d = Draw.MolDraw2DCairo(width, height)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()

    # save png to file
    with open(save_path, 'wb') as png_file:
        png_file.write(png_data)

    return


def draw_molecule_from_mol(inp_mol, save_file_path, width=600, height=600):
    """
    This function draws molecules with explicit hydrogen and atom idx
    :param mol: Rdkit.Mol object
    :param save_directory: path to save an image
    "https://stackoverflow.com/questions/67189346/how-to-save-ipython-core-display-svg-as-png-file"

    return None
    """
    mol = copy.deepcopy(inp_mol)
    # number_of_atoms = mol.GetNumAtoms()
    # for atom_idx in range(number_of_atoms):
    #     mol.GetAtomWithIdx(atom_idx).SetProp('atomNote', str(mol.GetAtomWithIdx(atom_idx).GetIdx()))

    # create png
    d2d = Draw.MolDraw2DCairo(width, height)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()

    # save png to file
    with open(save_file_path, 'wb') as png_file:
        png_file.write(png_data)

    return


def get_number_of_atoms(smi):
    """
    This function calculates the number of atoms in a molecule including hydrogen
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    number_of_atoms = 0
    for _ in mol.GetAtoms():
        number_of_atoms += 1

    return number_of_atoms

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


def make_directory(dir_path):
    """
    This function makes a directory if not exist
    :return: None
    """
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True)

    return


def read_standard_state_enthalpy(output_file_path):
    """
    This function reads enthalpy from a frequency calculation of one molecule.
    This function doesn't consider a solvation effect (gas phase reaction at 1 atm)
    :return: enthalpy of the molecule
    """
    # reactant
    flag = 0
    enthalpy = None
    Eh_to_kcal_mol = 627.5
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Total Enthalpy' in line:
                tmp_line = line.split()
                enthalpy = float(tmp_line[-2])  # Eh
                enthalpy *= Eh_to_kcal_mol  # kcal/mol
                flag += 1
            if flag == 1:
                break
        f.close()

    return enthalpy


def read_standard_state_gibbs_free_energy_with_solvation(output_file_path):
    """
    This function reads Gibbs free energy from a frequency calculation of one molecule.
    This function considers a solvation effect (gas phase reaction at 1 atm dissolved at a solvent)
    Note that the molecule is still in a gas phase (additional 1.89 kcal/mol should be added for a liquid phase)
    :return: Gibbs free energy of the molecule
    """
    # reactant
    flag = 0
    Eh_to_kcal_mol = 627.5
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Charge-correction' in line:
                tmp_line = line.split()
                electrostatic_free_energy = float(tmp_line[-4])  # Eh
                electrostatic_free_energy *= Eh_to_kcal_mol  # kcal/mol

            if 'Free-energy (cav+disp)' in line:
                tmp_line = line.split()
                cavity_dispersion_free_energy = float(tmp_line[-4])  # Eh
                cavity_dispersion_free_energy *= Eh_to_kcal_mol  # kcal/mol

            if 'Final Gibbs free energy' in line:
                tmp_line = line.split()
                gibbs_free_energy = float(tmp_line[-2])  # Eh
                gibbs_free_energy *= Eh_to_kcal_mol  # kcal/mol
                flag += 1

            if flag == 1:
                break
        f.close()

    return electrostatic_free_energy + cavity_dispersion_free_energy + gibbs_free_energy


def read_standard_state_gibbs_free_energy(output_file_path):
    """
    This function reads Gibbs free energy from a frequency calculation of one molecule.
    This function doesn't consider a solvation effect (gas phase reaction at 1 atm)
    :return: Gibbs free energy of the molecule
    """
    # reactant
    flag = 0
    gibbs_free_energy = None
    Eh_to_kcal_mol = 627.5
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Final Gibbs free energy' in line:
                tmp_line = line.split()
                gibbs_free_energy = float(tmp_line[-2])  # Eh
                gibbs_free_energy *= Eh_to_kcal_mol  # kcal/mol
                flag += 1
            if flag == 1:
                break
        f.close()

    return gibbs_free_energy


def read_magnitude_of_dipole_moment(output_file_path):
    """
    This functions reads a dipole moment of the output file (one molecule)
    :return: dipole_moment (a.u.)
    """
    with open(output_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Magnitude (a.u.)' in line:
                tmp_line = line.split()
                dipole_moment = float(tmp_line[-1])  # a.u.
                break

    return dipole_moment


def combine_polymer(p_smile_1, p_smile_2):
    """
    This function combines the two polymer smiles containing asterisks to specify repeating units
    :return: combined polymer smi (str)
    """
    mol_1 = Chem.MolFromSmiles(p_smile_1)
    mol_2 = Chem.MolFromSmiles(p_smile_2)
    mw_1 = RWMol(mol_1)
    mw_2 = RWMol(mol_2)

    # store
    asterisk_idx = list()
    mol_1_del_list = list()
    mol_2_del_list = list()

    # find asterisk idx
    for idx, atom in enumerate(mol_1.GetAtoms()):
        if atom.GetSymbol() == '*':
            asterisk_idx.append(idx)
    mol_1_del_list.append(asterisk_idx[1])
    mol_2_del_list.append(asterisk_idx[0])

    # modify index of monomer 2
    modified_mol_2_del_list = [idx + mol_1.GetNumAtoms() for idx in mol_2_del_list]

    # combine
    new_polymer = RWMol(Chem.CombineMols(mw_1, mw_2))
    new_polymer.AddBond(mol_1_del_list[0], modified_mol_2_del_list[0], BondType.SINGLE)

    # rearrange atom idx
    new_polymer_smi = Chem.MolToSmiles(new_polymer)

    asterisk_idx_smi = list()
    target_asterisk_idx = new_polymer_smi.find('**')  # consecutive asterisks
    if target_asterisk_idx == -1:  # couldn't find
        raise ValueError('Please check the polymer smile output!')

    asterisk_idx_smi = [target_asterisk_idx, target_asterisk_idx + 1]
    new_polymer_smi = new_polymer_smi[:asterisk_idx_smi[0]] + new_polymer_smi[asterisk_idx_smi[1] + 1:]

    # terminal diene
    # if '==' in new_polymer_smi:
    #     target_idx = new_polymer_smi.find('==')
    #     new_polymer_smi = new_polymer_smi[:target_idx] + new_polymer_smi[target_idx + 1:]

    return Chem.CanonSmiles(new_polymer_smi)


def remove_atoms_and_relabel(mol, del_list, bnd_list):
    """
    This function removes atom_idx in del_list from the mol object and modifies indices in bnd_list according
    to the removed atom indices
    :return: modified_mol, modified_bnd_list
    """
    # reverse sort del_list, so we can delete without affecting other elements of del_list
    arr = del_list.copy()
    arr.sort(reverse=True)
    for del_id in arr:
        mol.RemoveAtom(del_id)
        # modify bnd_list idx
        for j in range(len(bnd_list)):
            if bnd_list[j] > del_id:
                bnd_list[j] -= 1

    return mol, bnd_list


def combine_polymer_recursively_terminating_CH3(p_smile, repeating_number: int):
    """
    This function combines the given polymer smile repeating_number times.
    if repeating number is 1 -> dimer is formed. 
    End groups are terminated with CH3.
    Note that polymerization function (mechanism 7 ~ 17) only works for the same reactant

    :return: polymer mol object (.mol)
    """
    # connect a polymer repeating unit multiple times
    p_smile_output = copy.copy(p_smile)
    for num in range(repeating_number):
        p_smile_output = combine_polymer(p_smile_output, p_smile)

    # p_mol & p_mw object
    p_mol = Chem.MolFromSmiles(p_smile_output)
    p_mw = RWMol(p_mol)

    # store asterisk indices & nearest atom information
    asterisk_idx_list = list()
    nearest_atom_idx_list = list()

    # find asterisk idx -> ordered from low to high
    for atom_idx, atom in enumerate(p_mol.GetAtoms()):
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(atom_idx)

    # find a nearest atom idx
    for asterisk_idx in asterisk_idx_list:
        for atom_idx, atom in enumerate(p_mol.GetAtoms()):
            bond = p_mol.GetBondBetweenAtoms(asterisk_idx, atom_idx)  # bond object
            if bond is not None:
                # add a nearest atom idx
                nearest_atom_idx_list.append(atom_idx)

    # remove asterisks from the mw object and update indices
    bnd_list = nearest_atom_idx_list.copy()
    modified_mol, modified_nearest_list = remove_atoms_and_relabel(
        mol=p_mw, del_list=asterisk_idx_list, bnd_list=bnd_list
    )  # modified_nearest_list is ordered as asterisk_idx_list

    # attach hydrogen
    for nearest_atom_idx in modified_nearest_list:
        new_atom_id = modified_mol.AddAtom(Atom('C'))
        modified_mol.AddBond(nearest_atom_idx, new_atom_id, BondType.SINGLE)

    # canonical mol
    canon_smi = Chem.MolToSmiles(modified_mol)
    canon_mol = Chem.MolFromSmiles(canon_smi)

    return canon_mol


def get_condensates_thermodynamics(reaction_mechanism_idx: int):
    """
    This function returns thermodynamic information of condensates to compensate for
    a polymerization propagation free energy change
    :param reaction_mechanism_idx:
    :return: [free_energy (kcal/mol), enthalpy (kcal/mol), entropy (kcal/mol kelvin)] of condensates

    * Polymerization mechanism were recognized by the input, reaction_mechanism_idx, as follows:
        '[step_growth]_[di_amine]_[di_carboxylic_acid]': 1      -> condensate: H2O (two)
        '[step_growth]_[di_acid_chloride]_[di_amine]': 2        -> condensate: HCl (two)
        '[step_growth]_[di_carboxylic_acid]_[di_ol]': 3         -> condensate: H2O (two)
        '[step_growth]_[di_acid_chloride]_[di_ol]': 4           -> condensate: HCl (two)
        '[step_growth]_[di_amine]_[di_isocyanate]': 5           -> No condensate
        '[step_growth]_[di_isocyanate]_[di_ol]': 6              -> No condensate
        '[step_growth]_[hydroxy_carboxylic_acid]': 7            -> condensate: H2O (one)
        '[chain_growth]_[vinyl]': 8                             -> No condensate
        '[chain_growth]_[acetylene]': 9                         -> No condensate
        '[chain_growth_ring_opening]_[lactone]': 10             -> No condensate
        '[chain_growth_ring_opening]_[lactam]': 11              -> No condensate
        '[chain_growth_ring_opening]_[cyclic_ether]': 12        -> No condensate
        '[chain_growth_ring_opening]_[cyclic_olefin]': 13       -> No condensate
        '[chain_growth_ring_opening]_[cyclic_carbonate]': 14    -> No condensate
        '[chain_growth_ring_opening]_[cyclic_sulfide]': 15      -> No condensate
        '[metathesis]_[terminal_diene]': 16                     -> condensate: Ethylene (one)
        '[metathesis]_[conjugated_di_bromide]': 17              -> Br2 (one)
    """
    # df
    temperature = 298.15  # kelvin
    df_condensate = pd.DataFrame(columns=['molecule', 'xtb_G_room_temperature', 'xtb_H', 'xtb_S'])
    df_condensate['molecule'] = ['H2O', 'HCl', 'Ethylene', 'Br2']

    # use data from ./data/condensate_thermodynamics
    Eh_to_kcal_mol = 627.5
    free_energy_arr = np.array([-5.077858546025, -5.071159739234, -6.246041409599, -8.197008940447])  # Eh
    free_energy_arr *= Eh_to_kcal_mol  # kcal/mol
    df_condensate['xtb_G_room_temperature'] = free_energy_arr

    enthalpy_arr = np.array([-5.056441684498, -5.049964653903, -6.221190610921, -8.169022235905])  # Eh
    enthalpy_arr *= Eh_to_kcal_mol  # kcal/mol
    df_condensate['xtb_H'] = enthalpy_arr

    entropy_arr = np.array([0.214169E-01 / temperature, 0.211951E-01 / temperature, 0.248508E-01 / temperature,
                            0.279867E-01 / temperature])   # Eh / kelvin
    entropy_arr *= Eh_to_kcal_mol  # kcal/ (mol kelvin)
    df_condensate['xtb_S'] = entropy_arr

    # return thermodynamic information of condensates
    if reaction_mechanism_idx in [1, 3]:  # two H2O condensates
        df_return = df_condensate[df_condensate['molecule'] == 'H2O']
        df_return = df_return.drop(columns='molecule')
        arr_thermo = df_return.to_numpy().flatten().copy()
        return 2 * arr_thermo

    elif reaction_mechanism_idx in [7]:  # one H2O condensates
        df_return = df_condensate[df_condensate['molecule'] == 'H2O']
        df_return = df_return.drop(columns='molecule')
        arr_thermo = df_return.to_numpy().flatten().copy()
        return arr_thermo

    elif reaction_mechanism_idx in [2, 4]:  # two HCl condensates
        df_return = df_condensate[df_condensate['molecule'] == 'HCl']
        df_return = df_return.drop(columns='molecule')
        arr_thermo = df_return.to_numpy().flatten().copy()
        return 2 * arr_thermo

    elif reaction_mechanism_idx in [16]:  # one ethylene condensates
        df_return = df_condensate[df_condensate['molecule'] == 'Ethylene']
        df_return = df_return.drop(columns='molecule')
        arr_thermo = df_return.to_numpy().flatten().copy()
        return arr_thermo

    elif reaction_mechanism_idx in [17]:  # one bromine
        df_return = df_condensate[df_condensate['molecule'] == 'Br2']
        df_return = df_return.drop(columns='molecule')
        arr_thermo = df_return.to_numpy().flatten().copy()
        return arr_thermo

    else:
        return None


def get_adjacency_matrix_from_combined_mol_path(combined_mol_path):
    """
    This function generates an adjacency matrix of a combined mol path.
    :return: adjacency_matrix [N, N]  (N is the total number of a reactant and product atoms. Components are either 0 or 1)
    """
    # load a mol file
    combined_mol = Chem.MolFromMolFile(combined_mol_path, removeHs=False)
    adjacency_matrix = GetAdjacencyMatrix(combined_mol)

    return adjacency_matrix


def get_adjacency_matrix_from_mol_with_modified_xyz(mol_file_path, target_xyz_file_path=None, scale_factor=1.2):
    """
    This function get the adjacency matrix from a mol file with modified coordinates from "target_xyz_file".
    The equilibrium bond length (single bond, units: angstroms) is based on UFF.
    These bond length parameters are used to determine whether the bond exists or not between two atoms.
    Question: Can there exist a different bond order of a molecule with the same adjacency matrix?
    :return: adjacency_matrix (not weighted)
    """
    # from https://github.com/zhaoqy1996/YARP/blob/main/version1.0/utilities/taffi_functions.py
    UFF_Radii = {  'H':0.354, 'He':0.849,
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,
              'default' : 0.7}

    # load a mol file
    mol = Chem.MolFromMolFile(mol_file_path, removeHs=False)
    conformer = mol.GetConformer()

    # modify the .xyz coordinates
    if target_xyz_file_path is None:
        position_vector = conformer.GetPositions()

    else:  # there exist a target xyz file
        position_vector = extract_coordinates_from_xyz(target_xyz_file_path)

    # get adjacency matrix
    dist_matrix = np.triu(distance_matrix(position_vector, position_vector))  # upper triangle
    adj_matrix = np.zeros_like(dist_matrix)

    # find the connection
    for row_idx in range(mol.GetNumAtoms()):
        for col_idx in range(row_idx + 1, mol.GetNumAtoms()):
            # get row atoms
            row_atom = mol.GetAtoms()[row_idx]
            row_atom_symbol = row_atom.GetSymbol()

            # get col atoms
            col_atom = mol.GetAtoms()[col_idx]
            col_atom_symbol = col_atom.GetSymbol()

            # get UFF distance threshold for bonding
            threshold = (UFF_Radii[row_atom_symbol] + UFF_Radii[col_atom_symbol]) * scale_factor

            # decide
            if dist_matrix[row_idx, col_idx] <= threshold:
                adj_matrix[row_idx, col_idx] = 1.0  # linked. Note that the edge is not bond order.
                adj_matrix[col_idx, row_idx] = 1.0  # linked

    return adj_matrix


def extract_energy_and_xyz_coordinates_from_multimolecule_xyz(file_path, number_of_atoms, topn=10):
    """
    This function extracts minimzed energy and xyz coordinates from multimolecule xyz.
    This functions returns "topn" molecules with lowest energy
    :return: [Energy list (size of top n), .xyz list (size of top n)]
    """
    # read file and extract energy & .xyz
    split_number = number_of_atoms + 2  # two lines more
    with open(file_path, 'r') as file:
        lines = file.readlines()
        energy_xyz_list = [lines[idx:idx + split_number] for idx in range(0, len(lines), split_number)]
        file.close()

    # extract energy
    energy_list = [float(energy_xyz[1].rstrip()) for energy_xyz in energy_xyz_list]

    # sort - ascending order
    sorted_idx = sorted(range(len(energy_list)), key=lambda i: energy_list[i])
    sorted_energy_xyz_list = [energy_xyz_list[idx] for idx in sorted_idx]

    # return
    return sorted_energy_xyz_list[:topn]


def check_xtb_normally_finished(xtb_geometry_output_path):
    """
    This function checks if the XTB geometry optimization normally finished.
    :return: bool
    """
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0
    output_lines = open(xtb_geometry_output_path).readlines()
    for line in output_lines:
        if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
            flag_1 = 1

        if 'Geometry Summary' in line:
            flag_2 = 1

        if 'finished run on' in line:
            flag_3 = 1

    return flag_1 * flag_2 * flag_3


def check_xtb_hessian_normally_finished(xtb_hessian_output_path):
    """
    This function checks if the XTB hessian normally finished.
    :return: bool
    """
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0
    output_lines = open(xtb_hessian_output_path).readlines()
    for line in output_lines:
        if 'convergence criteria satisfied' in line:
            flag_1 = 1

        if 'SUMMARY' in line:
            flag_2 = 1

        if 'finished run on' in line:
            flag_3 = 1

    return flag_1 * flag_2 * flag_3


def check_CREST_normally_finished(crest_output_path):
    """
    This function checks if the CREST conformer search normally finished.
    :return: bool
    """
    flag = 0
    output_lines = open(crest_output_path).readlines()
    for line in output_lines:
        if 'CREST terminated normally' in line:
            flag = 1

    return bool(flag)


def gather_calculation_time_orca(orca_output_path):
    """
    This function gathers the calculation time from the Orca output file (especially for a single point calculation)
    :param orca_output_path:
    :return: Calculation time (mins)
    """
    orca_contents = open(orca_output_path).read()
    re_search = re.search('\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*\nTOTAL RUN TIME: ([0-9]+) days ([0-9]+) hours ([0-9]+) minutes ([0-9]+) seconds', orca_contents, re.DOTALL)
    days, hours, minutes, seconds = float(re_search.group(1)), float(re_search.group(2)), float(re_search.group(3)), float(re_search.group(4))
    computation_time = days * 24. * 60. + hours * 60. + minutes + seconds / 60.  # minutes

    return computation_time


def gather_calculation_time_xtb(xtb_output_path):
    """
    This function gathers the calculation time from the xtb output file (geometry optimization)
    :param xtb_output_path:
    :return: Calculation time (mins)
    """
    xtb_contents = open(xtb_output_path).read()
    re_search = re.search('total:[\s]+\* wall-time:[\s]+([0-9.]+) d,[\s]+([0-9.]+) h,[\s]+([0-9.]+) min,[\s]+([0-9.]+) sec\n', xtb_contents, re.DOTALL)
    days, hours, minutes, seconds = float(re_search.group(1)), float(re_search.group(2)), float(re_search.group(3)), float(re_search.group(4))
    computation_time = days * 24. * 60. + hours * 60. + minutes + seconds / 60.  # minutes

    return computation_time


def gather_calculation_time_crest(crest_output_path):
    """
    This function gathers the calculation time from the crest output file (conformer search)
    :param crest_output_path:
    :return: Calculation time (mins)
    """
    crest_contents = open(crest_output_path).read()
    re_search = re.search('Overall wall time[\s]+:([\s0-9.]+)h[\s]+:([\s0-9.]+)m[\s]+:([\s0-9.]+)s', crest_contents, re.DOTALL)
    hours, minutes, seconds = float(re_search.group(1)), float(re_search.group(2)), float(re_search.group(3))
    computation_time = hours * 60. + minutes + seconds / 60.0  # minutes

    return computation_time


def prepare_mol_with_xyz(geometry_xyz_path, mol_file_path):  # function to obtain a mol object
    """
    This function obtains a mol object with a modified xyz with geometry_xyz_path
    :return: mol object
    """
    mol_to_calculate = Chem.MolFromMolFile(mol_file_path, removeHs=False)  # create a mol object with explicit hydrogen
    conformer = mol_to_calculate.GetConformer(id=0)  # only one conformer / # print(conformer.GetPositions())
    xyz_to_use = extract_coordinates_from_xyz(geometry_xyz_path)  # load geometry to use
    for atom_idx in range(mol_to_calculate.GetNumAtoms()):  # update coordinates
        x, y, z = xyz_to_use[atom_idx]
        conformer.SetAtomPosition(atom_idx, Point3D(x, y, z))

    return mol_to_calculate


def calculate_rmsd(list_of_geometry_xyz_path, mol_file_path):
    """
    :return: mean_pairwise_rmsd, std_pairwise_rmsd (angstrom)
    """
    # obtain mol objects -- ConfId 0 has been modified
    mol_list = [prepare_mol_with_xyz(geometry_xyz_path=geometry_xyz_path, mol_file_path=mol_file_path) for geometry_xyz_path in list_of_geometry_xyz_path]

    # calculate rmsd
    number_of_mols = len(mol_list)
    rmsd_list = list()
    for row_idx in range(number_of_mols):
        for col_idx in range(row_idx + 1, number_of_mols):
            # conformer = mol_list[col_idx].GetConformer(id=0) / # print(conformer.GetPositions())
            # rmsd = GetBestRMS(prbMol=mol_list[row_idx], refMol=mol_list[col_idx], prbId=0, refId=0)  # GetBestRMS -> Changes the atom order. The geometry of a probmoleule is changed. The geometry of a "refMol" is not changed. -> more rigorous
            rmsd = AlignMol(prbMol=mol_list[row_idx], refMol=mol_list[col_idx], prbCid=0, refCid=0)  # AlignMol -> doesn't change the atom order (not considering permutation). The geometry of a probmoleule is changed. The geometry of a "refMol" is not changed.
            # conformer = mol_list[col_idx].GetConformer(id=0) / # print(conformer.GetPositions())
            rmsd_list.append(rmsd)

    # arr
    rmsd_arr = np.array(rmsd_list)

    return rmsd_arr.mean(), rmsd_arr.std()


def extract_rotational_constants_xtb(xtb_geometry_output_path):
    """
    Ths function extracts the rotational constants from the XTB geometry optimization output
    :return: np.array with rotational constants (MHz) (length: 3)
    """
    xtb_contents = open(xtb_geometry_output_path).read()
    re_search = re.search('rotational constants/cm⁻¹[\s]+:[\s]+([-0-9.E]+)[\s]+([-0-9.E]+)[\s]+([-0-9.E]+)\n', xtb_contents, re.DOTALL)
    rot_1, rot_2, rot_3 = float(re_search.group(1)), float(re_search.group(2)), float(re_search.group(3))  # rot_1 > rot_2 > rot_3. Unit: cm-1
    rot_1, rot_2, rot_3 = rot_1 * 29979.2458, rot_2 * 29979.2458, rot_3 * 29979.2458  # MHz

    return np.array([rot_1, rot_2, rot_3])


def extract_energy_xtb(xtb_geometry_output_path):
    """
    This function extracts the energy from the XTB geometry optimization output file
    :return: Energy (kcal/mol)
    """
    xtb_contents = open(xtb_geometry_output_path).read()
    re_search = re.search('TOTAL ENERGY[\s]+([-0-9.]+) Eh', xtb_contents, re.DOTALL)
    energy = float(re_search.group(1))  # Eh. Eh_to_kcal_mol = 627.5
    energy *= 627.5  # kcal/mol

    return energy


def extract_free_energy_xtb(output_path):
    """
    This function extracts a calculated free energy, enthalpy, and entropy from the XTB.hess.out
    :return: np.array([free energy (kcal/mol), enthalpy (kcal/mol), entropy (kcal/(mol K))])
    """
    free_energy = None
    enthalpy = None
    temperature = 298.15  # Kelvin
    Eh_to_kcal_mol = 627.5
    with open(output_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'TOTAL FREE ENERGY' in line:
                tmp_line = line.split()
                free_energy = float(tmp_line[-3])
            if 'TOTAL ENTHALPY' in line:
                tmp_line = line.split()
                enthalpy = float(tmp_line[-3])
        file.close()
    entropy = (enthalpy - free_energy) / temperature  # Eh / K

    # unit conversion to kcal/mol
    free_energy *= Eh_to_kcal_mol
    enthalpy *= Eh_to_kcal_mol
    entropy *= Eh_to_kcal_mol  # kcal/(mol K)

    return [free_energy, enthalpy, entropy]


def calculate_rmsd_between_two_molecules(geometry_xyz_path_1, geometry_xyz_path_2, mol_file_path):
    """
    This function calculates the RMSD between two molecules considering symmetry
    :return: RMSD (angstorms)
    """
    # obtain mol objects -- ConfId 0 has been modified
    mol_1 = prepare_mol_with_xyz(geometry_xyz_path=geometry_xyz_path_1, mol_file_path=mol_file_path)
    mol_2 = prepare_mol_with_xyz(geometry_xyz_path=geometry_xyz_path_2, mol_file_path=mol_file_path)

    # calculate rmsd
    rmsd = AlignMol(prbMol=mol_1, refMol=mol_2, prbCid=0, refCid=0)  # AlignMol -> doesn't change the atom order (not considering permutation). The geometry of a probmoleule is changed. The geometry of a "refMol" is not changed.

    return rmsd


def select_conformers(list_of_geometry_xyz_path, list_of_geometry_output_path, mol_file_path, max_num_conformers=5, energy_window=6.0, energy_threshold=0.1, rmsd_threshold=0.125, rotatation_constant_threshold=15.0):
    """
    This functions selects conformers from the given list of geometries with the criteria of (1) energy window, (2) energy threshold,
    (3) rmsd_threshold, and (4) rotational constant threshold. This threshold values are from the CREST paper: https://pubs.rsc.org/en/content/articlelanding/2020/CP/C9CP06869D (Pracht, P.; Bohle, F.; Grimme, S. Automated Exploration of the Low-Energy Chemical Space with Fast Quantum Chemical Methods. Phys. Chem. Chem. Phys. 2020)

    :param list_of_geometry_xyz_path: geometry list of conformers to be selected
    :param list_of_geometry_output_path: geometry output list of conformers to be selected. The order should be matched with "list_of_geometry_xyz_path"
    :param mol_file_path: the reference mol file to be used for RMSD calculations by replacing xyz coordinates
    :param max_num_conformers: the maximum number of conformers to be selected
    :param energy_window: the relative energy (kcal/mol) from the lowest conformer energy. Default: 6.0 kcal/mol
    :param energy_threshold: the energy threshold (kcal/mol) to be used to remove duplicates (Fig. 3 from the paper). Default: 0.1 kcal/mol
    :param rmsd_threshold: the rmsd threshold to be used to remove duplicates (Fig. 3 from the paper). Default: 0.125 angstroms
    :param rotatation_constant_threshold: the rotation constant threshold (MHz) to be used to remove duplicates (Fig. 3 from the paper). Default: 15.0 MHz
    B = (h_bar)**2 / 2I
    B_bar (rotational constant) = B / hc
    The output of XTB2 sorts the rotational constant from large to small
    :return: the list of selected conformer idx (maximum length: num_conformers). (energy has the ascending order)
    Note: This is index of list_of_geometry_xyz_path and list_of_geometry_output_path, not "conformer idx"
    """
    # choose the lowest energy conformer
    energy_arr = np.array([extract_energy_xtb(geometry_output) for geometry_output in list_of_geometry_output_path])
    sorted_idx = np.argsort(energy_arr)  # ascending order
    
    # filter geometry based on the energy window
    max_energy_allowed = energy_arr[sorted_idx[0]] + energy_window
    filtered_sorted_idx_list = sorted_idx[np.where(energy_arr[sorted_idx] <= max_energy_allowed)[0]]

    # pairwise comparison to drop duplicates
    selected_idx_list = [filtered_sorted_idx_list[0]]  # start with the lowest energy conformer
    for candidate_idx in filtered_sorted_idx_list[1:]:
        flag = 1  # assume to be added
        for selected_idx in selected_idx_list:  # pairwise comparison
            # apply energy threshold
            cond_1 = energy_arr[candidate_idx] - energy_arr[selected_idx] <= energy_threshold

            # apply rmsd threshold
            rmsd = calculate_rmsd_between_two_molecules(geometry_xyz_path_1=list_of_geometry_xyz_path[candidate_idx], geometry_xyz_path_2=list_of_geometry_xyz_path[selected_idx], mol_file_path=mol_file_path)
            cond_2 = rmsd <= rmsd_threshold

            # apply rotational constants threshold -- to differentiate between conformer and rotamer
            # candidate_rotational_constants = extract_rotational_constants_xtb(list_of_geometry_output_path[candidate_idx])
            # selected_rotational_constants = extract_rotational_constants_xtb(list_of_geometry_output_path[selected_idx])
            # cond_3 = len(np.where(np.abs(candidate_rotational_constants - selected_rotational_constants) <= rotatation_constant_threshold)[0]) == 3  # all values are less than the threshold

            # decide
            # if cond_1 & cond_2 & cond_3:  # energy <= threshold & rmsd <= threshold & rotational constant <= threshold
            if cond_1 & cond_2:  # energy <= threshold & rmsd <= threshold
                flag = 0
                break  # no need for next iterations

        # update
        if flag == 1:
            selected_idx_list.append(candidate_idx)

        # check num_conformers
        if len(selected_idx_list) == max_num_conformers:
            break

    return selected_idx_list


def remove_atoms(mol, del_list, keep_list):
    """
    This function removes atom_idx in del_list from the mol object (with explicit hydrogen -- to avoid a kekulization error) and modifies the mol object.
    When removing atoms attached to atoms of keep list, attach hydrogen to main the valency.
    If idx is hydrogen, hydrogen is not removed.
    :return: modified_mol
    """
    # reverse sort del_list, so we can delete without affecting other elements of del_list
    arr = del_list.copy()
    arr.sort(reverse=True)
    for del_id in arr:
        if mol.GetAtomWithIdx(del_id).GetSymbol() == 'H':
            continue
        mol.RemoveAtom(del_id)

    return mol


def extract_backbone_from_molecule_depth_one(target_smi):
    """
    This function extracts a backbone molecule of .smi with asterisks.
    1) Find the shortest path connecting two asterisks
    2) Include additional atom indices if they are in the same ring as the atoms in the shortest path.
    3) Including length one side chain.
    :return: modified smi (without side chains)
    """
    # target_smi = open(path_to_smi_with_asterisks, 'r').read()
    target_mol = Chem.MolFromSmiles(target_smi)
    # draw_molecule_from_mol(target_mol, f'./original.png')
    target_mol = RWMol(Chem.AddHs(target_mol))
    Chem.Kekulize(target_mol)  # change aromatic to single or double keeping an aromatic flag

    # find the shortest path
    asterisk_idx_list = list()
    for idx in range(target_mol.GetNumAtoms()):
        atom = target_mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(idx)
    backbone_atom_indices_set = set(GetShortestPath(target_mol, asterisk_idx_list[0], asterisk_idx_list[1]))

    # obtain additional atom indices if they are in the same ring
    additional_atom_indices_set = set()

    # First update -- search rings
    Chem.FastFindRings(target_mol)
    ring_info = target_mol.GetRingInfo().AtomRings()
    for ring_indices_tuple in ring_info:
        intersection = backbone_atom_indices_set.intersection(set(ring_indices_tuple))
        if len(intersection) >= 2:  # at least two backbone atoms are included in the ring to decide this is not a side ring.
            indices_to_add = set(ring_indices_tuple) - intersection  # indices to add
            additional_atom_indices_set = additional_atom_indices_set.union(indices_to_add)
    backbone_atom_indices_set = backbone_atom_indices_set.union(additional_atom_indices_set)

    # Second update -- include aromatic rings (conjugated) attached to the main backbone recursively
    additional_ring_to_main_backbone = True
    while additional_ring_to_main_backbone:
        decision_for_next_iteration = False  # initially assume False
        indices_set_to_add = set()
        for ring_indices_tuple in ring_info:
            if set(ring_indices_tuple).issubset(backbone_atom_indices_set):  # already included to th main backbone
                continue
            linkage_flag, aromatic_flag = False, False  # initially assume False
            if len(set(ring_indices_tuple).intersection(backbone_atom_indices_set)) != 0:  # linked to the main backbone
                linkage_flag = True
            aromatic_sum = np.array([target_mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_indices_tuple]).sum()
            if aromatic_sum == len(ring_indices_tuple):  # all aromatic
                aromatic_flag = True

            # A ring linked to the main backbone and all atoms in a ring are aromatic (conjugated)
            if linkage_flag and aromatic_flag:
                decision_for_next_iteration = True
                indices_set_to_add = indices_set_to_add.union(set(ring_indices_tuple))

        # add to the main backbone
        backbone_atom_indices_set = backbone_atom_indices_set.union(indices_set_to_add)
        if not decision_for_next_iteration:  # stop the next iteration in while
            additional_ring_to_main_backbone = False

    # include a side chain (length 1) if connected
    indices_set_to_add = set()
    for idx in range(target_mol.GetNumAtoms()):
        for backbone_atom_index in backbone_atom_indices_set:
            bond = target_mol.GetBondBetweenAtoms(idx, backbone_atom_index)  # bond object
            if bond is None:  # not connected
                continue
            else:
                indices_set_to_add.add(idx)
    backbone_atom_indices_set = backbone_atom_indices_set.union(indices_set_to_add)

    # indices to remove
    total_atom_indices_set = set(idx for idx in range(target_mol.GetNumAtoms()))
    atom_indices_to_remove_set = total_atom_indices_set - backbone_atom_indices_set  # indices to remove

    # remove unnecessary atom indices
    for atom_index_to_remove in atom_indices_to_remove_set:
        for backbone_atom_index in backbone_atom_indices_set:
            bond = target_mol.GetBondBetweenAtoms(atom_index_to_remove, backbone_atom_index)  # bond object
            if bond is None:  # not connected
                continue
            bond_type = bond.GetBondTypeAsDouble()  # SINGLE: 1.0, DOUBLE: 2.0, and TRIPLE: 3.0. (There is no AROMATIC: 1.5 due to kukulize)
            target_mol.RemoveBond(atom_index_to_remove, backbone_atom_index)  # remove bond
            for _ in range(int(bond_type)):  # add hydrogen to conserve a valency. e.g.) int(1.5) = 1
                # for a backbone molecule to conserve a valency
                new_hydrogen_idx = target_mol.AddAtom(Atom('H'))
                target_mol.AddBond(new_hydrogen_idx, backbone_atom_index, BondType.SINGLE)

                # for a fragment to conserve a valency
                new_hydrogen_idx = target_mol.AddAtom(Atom('H'))
                target_mol.AddBond(new_hydrogen_idx, atom_index_to_remove, BondType.SINGLE)

    # remedy aromatic flag
    for atom in target_mol.GetAtoms():  # aromatic flag for an atom
        if (not atom.IsInRing()) and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    for bond in target_mol.GetBonds():  # aromatic flag for a bond
        if (not bond.IsInRing()) and bond.GetIsAromatic():
            bond.SetIsAromatic(False)

    # choose the backbone fragment
    # draw_molecule_from_mol(target_mol, f'./bond_break.png')
    mol_list, idx_list = list(GetMolFrags(target_mol, asMols=True)), list(GetMolFrags(target_mol))
    assert mol_list is not None
    subset_bool_list = list(map(lambda indices: backbone_atom_indices_set.issubset(set(indices)), idx_list))
    backbone_mol_idx = [idx for idx, boolean_value in enumerate(subset_bool_list) if boolean_value][0]
    backbone_mol = mol_list[backbone_mol_idx]
    backbone_mol = Chem.RemoveHs(backbone_mol)  # implicit hydrogen
    # draw_molecule_from_mol(backbone_mol, f'./bond_break.png')
    # exit()

    return Chem.MolToSmiles(backbone_mol)


def get_number_of_backbone_atoms(target_smi):
    """
    This function calculates the number of backbone atoms excluding terminating hydrogen
    :return: number of backbone atoms
    """
    target_mol = Chem.MolFromSmiles(target_smi)
    asterisk_idx_list = list()
    for idx in range(target_mol.GetNumAtoms()):
        atom = target_mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(idx)
    dist = GetShortestPath(target_mol, asterisk_idx_list[0], asterisk_idx_list[1])
    backbone_length = len(dist) - 2

    return backbone_length


def construct_density_of_states(transition_energy_arr, gaussian_smearing=0.2, resolution=0.1,
                                min_energy=0, max_energy=8):
    """
    This function constructs a continuous density of states a given transition energy array (unit: eV) using a
    gaussian smearing (eV) to make a discrete transition to continuous DOS. (plot resolution (eV))
    """
    energy_bins = np.arange(min_energy, max_energy, resolution)
    dos_arr = np.zeros(shape=(len(transition_energy_arr), len(energy_bins)))
    for transition_idx, transition_energy in enumerate(transition_energy_arr):
        gaussian_dist = stats.norm(loc=transition_energy, scale=gaussian_smearing)
        density_list = list()
        for energy in energy_bins:
            density_list.append(gaussian_dist.pdf(energy))
        dos_arr[transition_idx] = np.array(density_list)
    dos_arr = dos_arr.sum(axis=0)  # sum over different transition idx

    return dos_arr


def construct_weighted_density_of_states(transition_energy_arr, oscillator_strength_arr, gaussian_smearing=0.2,
                                         resolution=0.1, min_energy=0, max_energy=8):
    """
    This function constructs a continuous density of states multiplied by oscillator strength with a given transition energy array (unit: eV)
    using a gaussian smearing (eV) to make a discrete transition to continuous DOS. (plot resolution (eV))
    """
    energy_bins = np.arange(min_energy, max_energy, resolution)
    dos_arr = np.zeros(shape=(len(transition_energy_arr), len(energy_bins)))
    for transition_idx, transition_energy in enumerate(transition_energy_arr):
        gaussian_dist = stats.norm(loc=transition_energy, scale=gaussian_smearing)
        scale_const = oscillator_strength_arr[transition_idx]
        density_list = list()
        for energy in energy_bins:
            density_list.append(gaussian_dist.pdf(energy) * scale_const)
        dos_arr[transition_idx] = np.array(density_list)
    dos_arr = dos_arr.sum(axis=0)  # sum over different transition idx

    return dos_arr