import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path


target_cols_list = [
        ['asphericity_Boltzmann_average',
        'eccentricity_Boltzmann_average',
        'inertial_shape_factor_Boltzmann_average',
        'radius_of_gyration_Boltzmann_average',
        'spherocity_Boltzmann_average'],

        [
        'HOMO_minus_1_Boltzmann_average',
        'HOMO_Boltzmann_average',
        'LUMO_Boltzmann_average',
        'LUMO_plus_1_Boltzmann_average',
        'dipole_moment_Boltzmann_average',
        'quadrupole_moment_Boltzmann_average',
        'polarizability_Boltzmann_average',],  # without partial charges

        ['s1_energy_Boltzmann_average',
        'dominant_transition_energy_Boltzmann_average',
        'dominant_transition_oscillator_strength_Boltzmann_average',
        't1_energy_Boltzmann_average',],

        ['chi_parameter_water_mean',
        'chi_parameter_ethanol_mean',
        'chi_parameter_chloroform_mean',]
    ]

############################## TARGET DIR & PATH ##############################
DATA_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/data')
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
CALCULATION_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/run_calculation/calculation_results_csv/pareto_greedy')
############################## TARGET DIR & PATH ##############################


if __name__ == '__main__':
    """
    This script saves a prediction and true values for actively sampled points to draw plots to decide when to stop the active learning cycle.
    The saved dataframe is used to draw a plot of error of active sampled points to set active learning stopping criteria
    based on the training data.
    
    * run order
    1) save_data_for_stopping_criteria.py
    2) plot_for_stopping_criteria.py
    
    * To draw a test RMSE please run the following scripts:
    1) save_fold_pred_error.py
    2) plot_fold_pred_error.py
    """
    ### MODIFY ###
    current_batch = 2  # 0 for initial data. If "0", this function draws prediction errors of active sampled points for the AL "1".
    strategy = 'pareto_greedy'
    ### MODIFY ###

    # data to evaluate
    data_path_sampled = os.path.join(DATA_DIR, 'active_learning', strategy, f'OMG_train_batch_{current_batch + 1}.csv')
    df_sampled = pd.read_csv(data_path_sampled)
    reaction_id_sampled = df_sampled['reaction_id'].to_numpy()

    # property prediction of total train polymers
    data_path_total_prediction = os.path.join(DATA_DIR, 'active_learning', strategy, 'uncertainty_pred', f'current_batch_{current_batch}', 'total_prediction_results.csv')
    df_total_prediction = pd.read_csv(data_path_total_prediction)

    # prediction of sampled polymers
    df_prediction_sampled = df_total_prediction[df_total_prediction['reaction_id'].isin(reaction_id_sampled)]
    df_prediction_sampled_reaction_id = df_prediction_sampled['reaction_id'].to_numpy()

    # load calculation values
    target_cols = ['reaction_id'] + [col for target_cols in target_cols_list for col in target_cols]
    data_path_calculation = os.path.join(CALCULATION_DIR, f'{current_batch + 1}', 'total_results', 'total_results.csv')   # current batch + 1
    df_calculation = pd.read_csv(data_path_calculation)[target_cols]  # probably smaller than sampled due to the calculation failures.

    # prepare df
    df_save = pd.DataFrame()
    converged_reaction_id = df_calculation['reaction_id'].to_numpy()
    df_prediction_sampled_converged_in_calculation = df_prediction_sampled[df_prediction_sampled['reaction_id'].isin(converged_reaction_id)]

    # sort
    df_prediction_sampled_converged_in_calculation = df_prediction_sampled_converged_in_calculation.sort_values(by=['reaction_id'], ascending=True).reset_index(drop=True)
    df_calculation = df_calculation.sort_values(by=['reaction_id'], ascending=True).reset_index(drop=True)

    # fill in the data
    df_save['reaction_id'] = df_calculation['reaction_id'].to_numpy().copy()
    for target_cols in target_cols_list:
        for col in target_cols:
            df_save[f'true_{col}'] = df_calculation[col].to_numpy()
            df_save[f'pred_{col}'] = df_prediction_sampled_converged_in_calculation[col].to_numpy()

    # save
    save_path = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point', f'current_batch_{current_batch}_train', 'stopping_criteria_train_unscaled.csv')
    df_save.to_csv(save_path, index=False)

