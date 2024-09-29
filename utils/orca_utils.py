import os
import sys

from subprocess import call
from pathlib import Path

import warnings
warnings.simplefilter("ignore", UserWarning)

Eh_to_kcal_mol = 627.5
gas_to_liquid_phase = 1.89  # kcal/mol. But.. standard Gibbs free energy -> gas phase (?) -> for now gas dissolved in liquids


class OrcaGeometryOptimizer(object):
    """
    This class manages Orca geometry optimization jobs
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            Path(self.save_dir).mkdir(parents=True)

    def get_orca_script_geometry_optimization(
            self,
            initial_coordinate_file_path: str,
            xc_functional: str,
            basis_set: list,
            solvent_model: str,
            solvent: str,
            number_of_processor_per_job: int,
            thermo: bool,
    ):
        """
        This function generates a script for geometry optimization with Orca
        :return orca_script_lines [str]
        """
        # set spin multiplicity
        charge = 0
        spin_multiplicity = 1 if charge == 0 else 2

        # read initial coordinates that will be used in geometry optimization
        xyz_block = ''
        with open(initial_coordinate_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tmp_line = line.split()
                if len(tmp_line) == 4:
                    xyz_block += line

        # calculation type
        if thermo:
            # calculation_type = 'OPT FREQ'
            calculation_type = 'RIJCOSX Grid4 GridX4 OPT NumFreq'
        else:
            # calculation_type = 'OPT'
            calculation_type = 'TightOPT'  # for OMG property calculations

        # solvent model
        if 'cpcm' == solvent_model.lower():
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} {solvent_model}({solvent}) {calculation_type}\n',  # approach basis aux_basis
                f'%geom\n',
                f'MaxIter 300\n',
                f'end\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'* xyz {charge} {spin_multiplicity}\n',
                f'{xyz_block}',
                f'*\n'
            ]

        elif 'cpcmc' == solvent_model.lower():  # COSMO
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} {solvent_model}({solvent}) {calculation_type}\n',  # approach basis aux_basis
                f'%geom\n',
                f'MaxIter 300\n',
                f'end\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'* xyz {charge} {spin_multiplicity}\n',
                f'{xyz_block}',
                f'*\n'
            ]

        elif 'smd' == solvent_model.lower():
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} CPCM {calculation_type}\n',  # approach basis aux_basis
                f'%cpcm\n'
                f'SMD true\n',
                f'SMDsolvent "{solvent}"\n',
                f'end\n'
                f'%geom\n',
                f'MaxIter 300\n',
                f'end\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'* xyz {charge} {spin_multiplicity}\n',
                f'{xyz_block}',
                f'*\n'
            ]

        else:  # no solvents
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} {calculation_type}\n',  # approach basis aux_basis
                f'%geom\n',
                f'MaxIter 300\n',
                f'end\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'* xyz {charge} {spin_multiplicity}\n',
                f'{xyz_block}',
                f'*\n'
            ]

        return orca_script

    def write(
            self,
            dir_name: str,
            geometry_opt_functional: str,
            geometry_opt_basis_set: list,
            geometry_to_import: str,
            solvent_model: str,
            solvent: str,
            number_of_processor_per_job: int,
            thermo: bool = False,
    ):
        """
        This function writes a inp file for Orca
        :return: filepath of geometry.inp
        """
        # create save directory
        result_save_dir = os.path.join(self.save_dir, dir_name)
        if not os.path.exists(result_save_dir):
            Path(result_save_dir).mkdir(parents=True)

        # get geometry optimization script
        geometry_file_path = os.path.join(result_save_dir, 'geometry.inp')
        geometry_script = self.get_orca_script_geometry_optimization(
            initial_coordinate_file_path=geometry_to_import, xc_functional=geometry_opt_functional,
            basis_set=geometry_opt_basis_set, solvent_model=solvent_model, solvent=solvent,
            number_of_processor_per_job=number_of_processor_per_job, thermo=thermo
        )

        # save script
        with open(geometry_file_path, 'w') as bash_file:
            bash_file.writelines(geometry_script)
            bash_file.close()

        return geometry_file_path


class OrcaGeometryJobSubmit(object):
    """
    This class manages a geometry optimization job
    """
    def __init__(self, computing: str):
        self.computing = computing

    def write_job(self, geometry_inp, sh_file_name, job_name, number_of_processors, node_number, save_dir):
        """
        This function writes a .sh file at the save_dir
        :return: None
        """
        # script path
        file_path = os.path.join(save_dir, f'{sh_file_name}.sh')

        # write a script
        if self.computing.lower() == 'cpu':
            bash_lines = [
                f'#!/bin/csh\n',
                f'#$ -N {job_name}\n',
                f'#$ -wd {save_dir}\n'
                f'#$ -j y\n',
                f'#$ -o $JOB_ID.out\n',
                f'#$ -pe orte {number_of_processors}\n',
                f'#$ -l hostname=compute-0-{node_number}.local\n',
                f'#$ -q all.q\n',
                # f'module load openmpi/4.1.1\n',
                f'module load openbabel/3.1.1\n',
                f'/home/sk77/PycharmProjects/orca/orca_4_2_1_linux_x86-64_openmpi216/orca {geometry_inp} > geometry.out\n'
                f'rm *gbw *prop *txt *tmp *engrad *opt *smd.grd *smd.out\n'
            ]

        elif self.computing.lower() == 'gpu':
            bash_lines = [
                f'#!/bin/csh\n',
                f'#$ -N {job_name}\n',
                f'#$ -wd {save_dir}\n'
                f'#$ -j y\n',
                f'#$ -o $JOB_ID.out\n',
                f'#$ -pe orte {number_of_processors}\n',
                f'#$ -l hostname=compute-0-{node_number}.local\n',
                f'module load openmpi/2.1.6\n',
                f'module load openbabel/3.1.1\n',
                f'/home/sk77/PycharmProjects/orca/orca_4_2_1_linux_x86-64_openmpi216/orca {geometry_inp} > geometry.out\n'
                f'rm *gbw *prop *txt *tmp *engrad *opt\n'
            ]

        else:
            bash_lines = ''

        # save the script
        with open(file_path, 'w') as bash_file:
            bash_file.writelines(bash_lines)
            bash_file.close()

        return

    def submit(self, file_path):
        """
        This function submits a job of a file_path (.sh)
        :return: None
        """
        qsub_call = f'qsub {file_path}'
        call(qsub_call, shell=True)

    @ staticmethod
    def check_if_orca_geometry_optimization_finished(output_file_path):
        flag_1 = 0
        flag_2 = 0
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'HURRAY' in line:
                    flag_1 = 1
                if 'ORCA TERMINATED NORMALLY' in line:
                    flag_2 = 1

        return flag_1 * flag_2

    @ staticmethod
    def check_if_orca_normally_finished(output_file_path):
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'ORCA TERMINATED NORMALLY' in line:
                    return True
            return False

# checked for orca 5.0 -> OMG
class OrcaPropertyCalculator(object):
    """
    This class manages Orca property calculation jobs (single point)
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            Path(self.save_dir).mkdir(parents=True)

    def get_orca_script_property_calculation(
            self,
            initial_coordinate_file_path: str,
            xc_functional: str,
            basis_set: list,
            solvent_model: str,
            solvent: str,
            number_of_processor_per_job: int,
            TD_DFT: bool,
            chain_cosmo: bool,
    ):
        """
        This function generates a script for geometry optimization with Orca.
        :return orca_script_lines [str]
        """
        # set spin multiplicity
        charge = 0
        spin_multiplicity = 1  # neutral molecule

        # calculation type
        if xc_functional == 'wB97X-D3':
            calculation_type = 'RIJCOSX Energy'
        elif xc_functional == 'B97-D3':
            calculation_type = 'RIJONX Energy'  # RI-J -> default for GGA
        else:
            calculation_type = 'RIJONX D3 Energy'  # RI-J -> default for GGA (e.g., revPBE)

        # TD-DFT for UV-Vis spectra and triplet density of states
        num_roots = 15
        TD_DFT_block = f'%TDDFT\nNRoots {num_roots}\nTriplets True\nend' if TD_DFT else ''

        # solvent model
        if 'cpcm' == solvent_model.lower():
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} {solvent_model}({solvent}) Hirshfeld CHELPG {calculation_type}\n',  # approach basis aux_basis
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'%maxcore 3000\n',
                f'%elprop\n',
                f'Polar 1\n',
                f'Quadrupole True\n',
                f'end\n',
                f'{TD_DFT_block}\n',
                f'* xyzfile {charge} {spin_multiplicity} {initial_coordinate_file_path}\n',
            ]

        elif 'cpcmc' == solvent_model.lower():  # cosmo solvation
            if solvent.lower() == 'cosmo':
                solvation_option = f'{solvent_model}'
            else:  # e.g. DMSO, THF
                solvation_option = f'{solvent_model}({solvent})'
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} {solvation_option} Hirshfeld CHELPG {calculation_type}\n',  # approach basis aux_basis
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'%maxcore 3000\n',
                f'%elprop\n',
                f'Polar 1\n',
                f'Quadrupole True\n',
                f'end\n',
                f'{TD_DFT_block}\n',
                f'* xyzfile {charge} {spin_multiplicity} {initial_coordinate_file_path}\n',
            ]

        elif 'smd' == solvent_model.lower():
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} CPCM Hirshfeld CHELPG {calculation_type}\n',  # approach basis aux_basis
                f'%cpcm\n'
                f'SMD true\n',
                f'SMDsolvent "{solvent}"\n',
                f'end\n'
                f'%maxcore 3000\n',
                f'%elprop\n'
                f'Polar 1\n',
                f'Quadrupole True\n',
                f'end\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'{TD_DFT_block}\n',
                f'* xyzfile {charge} {spin_multiplicity} {initial_coordinate_file_path}\n',
            ]

        else:  # no solvents
            # geometry optimization
            orca_script = [
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} Hirshfeld CHELPG {calculation_type}\n',  # approach basis aux_basis
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'%maxcore 3000\n',
                f'%elprop\n',
                f'Polar 1\n',
                f'Quadrupole True\n',
                f'end\n',
                f'{TD_DFT_block}\n',
                f'* xyzfile {charge} {spin_multiplicity} {initial_coordinate_file_path}\n',
            ]

        # chain reactions
        if chain_cosmo:  # connect a following job under COSMO solvation (only DFT single point energy calculations)
            base_name = os.path.join(Path(initial_coordinate_file_path).parents[0], 'property_cosmo')  # CAUTION: the same path with the geometry
            orca_script.extend([
                '\n',
                '# for cosmo solvent -> Flory Huggins\n',
                '$new_job\n',
                f'! {xc_functional} {basis_set[0]} {basis_set[1]} CPCMC Hirshfeld CHELPG {calculation_type}\n',  # approach basis aux_basis,
                f'%base "{base_name}"\n',
                f'%pal\n',
                f'nprocs {number_of_processor_per_job}\n'
                f'end\n',
                f'%maxcore 3000\n',
                f'* xyzfile {charge} {spin_multiplicity} {initial_coordinate_file_path}\n',
            ])

        return orca_script

    def write(
            self,
            dir_name: str,
            functional: str,
            basis_set: list,
            geometry_to_import: str,
            solvent_model: str,
            solvent: str,
            number_of_processor_per_job: int,
            TD_DFT: bool,
            chain_cosmo: bool,
    ):
        """
        This function writes a inp file for Orca
        :return: filepath of geometry.inp
        """
        # create save directory
        result_save_dir = os.path.join(self.save_dir, dir_name)
        if not os.path.exists(result_save_dir):
            Path(result_save_dir).mkdir(parents=True)

        # get property calculation script
        property_file_path = os.path.join(result_save_dir, 'property.inp')
        property_script = self.get_orca_script_property_calculation(
            initial_coordinate_file_path=geometry_to_import, xc_functional=functional,
            basis_set=basis_set, solvent_model=solvent_model, solvent=solvent,
            number_of_processor_per_job=number_of_processor_per_job, TD_DFT=TD_DFT, chain_cosmo=chain_cosmo
        )

        # save script
        with open(property_file_path, 'w') as bash_file:
            bash_file.writelines(property_script)
            bash_file.close()

        return property_file_path


class OrcaPropertyJobSubmit(object):
    """
    This class manages a property job submission
    """
    def __init__(self, computing: str):
        self.computing = computing

    def write_job(self, property_inp, sh_file_name, job_name, number_of_processors, node_number, save_dir):
        """
        This function writes a .sh file at the save_dir
        :return: None
        """
        # script path
        file_path = os.path.join(save_dir, f'{sh_file_name}.sh')

        # write a script
        if self.computing.lower() == 'cpu':
            bash_lines = [
                f'#!/bin/csh\n',
                f'#$ -N {job_name}\n',
                f'#$ -wd {save_dir}\n'
                f'#$ -j y\n',
                f'#$ -o $JOB_ID.out\n',
                f'#$ -pe orte {number_of_processors}\n',
                f'#$ -l hostname=compute-0-{node_number}.local\n',
                f'#$ -q all.q\n',
                f'module load openmpi/2.1.6\n',
                f'/home/sk77/PycharmProjects/orca/orca_4_2_1_linux_x86-64_openmpi216/orca {property_inp} > property.out\n'
                f'rm *gbw *prop *txt *tmp *engrad *opt *smd.grd *smd.out\n'
            ]

        elif self.computing.lower() == 'gpu':
            bash_lines = [
                f'#!/bin/csh\n',
                f'#$ -N {job_name}\n',
                f'#$ -wd {save_dir}\n'
                f'#$ -j y\n',
                f'#$ -o $JOB_ID.out\n',
                f'#$ -pe orte {number_of_processors}\n',
                f'#$ -l hostname=compute-0-{node_number}.local\n',
                f'module load openmpi/2.1.6\n',
                f'module load openbabel/3.1.1\n',
                f'/home/sk77/PycharmProjects/orca/orca_4_2_1_linux_x86-64_openmpi216/orca {property_inp} > property.out\n'
                f'rm *gbw *prop *txt *tmp *engrad *opt *smd.grd *smd.out\n'
            ]

        else:
            bash_lines = ''

        # save the script
        with open(file_path, 'w') as bash_file:
            bash_file.writelines(bash_lines)
            bash_file.close()

        return

    def submit(self, file_path):
        """
        This function submits a job of a file_path (.sh)
        :return: None
        """
        qsub_call = f'qsub {file_path}'
        call(qsub_call, shell=True)

    @ staticmethod
    def check_if_orca_property_calculation_finished(output_file_path):
        # TODO
        flag_1 = 0
        flag_2 = 0
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'HURRAY' in line:
                    flag_1 = 1
                if 'ORCA TERMINATED NORMALLY' in line:
                    flag_2 = 1

        return flag_1 * flag_2

    @ staticmethod
    # checked for orca 5.0
    def check_if_orca_normally_finished(output_file_path):
        # TODO
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Check your MOs and check whether a frozen core calculation is appropriate' in line:
                    return False  # Frozen core approximation -> sometime weird results
                if 'ORCA TERMINATED NORMALLY' in line:
                    return True
            return False


