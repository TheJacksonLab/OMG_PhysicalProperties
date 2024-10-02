import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.isotonic import IsotonicRegression

from scipy import stats
from scipy.integrate import simps
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize, Bounds



############################## TARGET COLUMNS ##############################
target_cols_list = [
    ['asphericity_Boltzmann_average',
    'eccentricity_Boltzmann_average',
    'inertial_shape_factor_Boltzmann_average',
    'radius_of_gyration_Boltzmann_average',
    'spherocity_Boltzmann_average'],

    ['HOMO_minus_1_Boltzmann_average',
    'HOMO_Boltzmann_average',
    'LUMO_Boltzmann_average',
    'LUMO_plus_1_Boltzmann_average',
    'dipole_moment_Boltzmann_average',
    'quadrupole_moment_Boltzmann_average',
    'polarizability_Boltzmann_average',],

    ['s1_energy_Boltzmann_average',
    'dominant_transition_energy_Boltzmann_average',
    'dominant_transition_oscillator_strength_Boltzmann_average',
    't1_energy_Boltzmann_average',],

    ['chi_parameter_water_mean',
    'chi_parameter_ethanol_mean',
    'chi_parameter_chloroform_mean',]
]

# label_name_list_all = [
#     ['$\Omega_A$ [unitless]', '$\epsilon$ [unitless]', '$S_I\;$[$\mathring{A}^{-2}g^{-1}$mol]',
#      '$R_g\;$[$\mathring{A}$]', '$\Omega_s$ [unitless]'],
#     ['$E_{HOMO\minus1}$ [eV]', '$E_{HOMO}$ [eV]', '$E_{LUMO}$ [eV]', '$E_{LUMO\plus1}$ [eV]', '$\mu$ [a.u.]',
#      '$q$ [a.u.]', '$\\alpha$ [a.u.]'],
#     ['$E_{S_1}$ [eV]', "$E^{\prime}_{singlet}$ [eV]", "$f^{\prime}_{osc}$ [unitless]", '$E_{T_1}$ [eV]'],
#     ['$\chi_{water}$ [unitless]', '$\chi_{ethanol}$ [unitless]', '$\chi_{chloroform}$ [unitless]']
# ]
label_name_list_all = [
    ['$\Omega_\mathrm{A}$ [unitless]', '$\epsilon$ [unitless]', '$S_\mathrm{I}\;$[$\mathring{A}^{-2}g^{-1}$mol]',
     '$R_\mathrm{g}\;$[$\mathring{A}$]', '$\Omega_\mathrm{S}$ [unitless]'],
    ['$E_{\mathrm{HOMO}\minus1}}$ [eV]', '$E_{\mathrm{HOMO}}$ [eV]', '$E_{\mathrm{LUMO}}$ [eV]', '$E_{\mathrm{LUMO}\plus1}$ [eV]', '$\mu$ [a.u.]',
     '$q$ [a.u.]', '$\\alpha$ [a.u.]'],
    ['$E_{\mathrm{S}_1}$ [eV]', "$E^{\prime}_{\mathrm{singlet}}$ [eV]", "$f^{\prime}_{\mathrm{osc}}$ [unitless]", '$E_{\mathrm{T}_1}$ [eV]'],
    ['$\chi_{\mathrm{water}}$ [unitless]', '$\chi_{\mathrm{ethanol}}$ [unitless]', '$\chi_{\mathrm{chloroform}}$ [unitless]']
]
############################## TARGET COLUMNS ##############################


############################## TARGET DIR & PATH ##############################
ACTIVE_LEARNING_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/active_learning')
############################## TARGET DIR & PATH ##############################


# refer to: https://github.com/uncertainty-toolbox/uncertainty-toolbox/tree/89c42138d3028c8573a1a007ea8bef80ad2ed8e6
def calibration_curve(std_arr_to_calibrate: np.array, train_target_arr: np.array, train_pred_arr: np.array,
                      property_name, save_name, num_partitions=40, plot=True):
    """
    This function plots & saves a calibration curve.
    :param std_arr_to_calibrate: std_arr to calibrate
    :return: expected_cumulative, observed_cumulative, miscalibration_area
    """
    expected_p = np.arange(num_partitions + 1) / num_partitions
    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - expected_p / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + expected_p / 2.0)

    # normalize residuals with pred & trues
    residuals = train_pred_arr - train_target_arr
    normalized_residuals = (residuals / std_arr_to_calibrate).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound
    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    # function to integrate ideal calibration
    x_integrate = expected_p
    y_integrate = np.abs(obs_proportions - expected_p)
    miscalibration_area = simps(y=y_integrate, x=x_integrate)

    # plot
    if plot:
        plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(expected_p, obs_proportions, color='#2B84A0', label=f'Miscalibration area {miscalibration_area:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ideal')
        plt.title(property_name, fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Expected cumulative distribution', fontsize=14)
        plt.ylabel('Observed cumulative distribution', fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'calibration_{save_name}.png'))
        plt.close()

    return expected_p, obs_proportions, miscalibration_area


def decide_linear_scaling(train_std_arr, train_target_arr, train_pred_arr):
    """
    This function decides a proper linear scaling to uncertainty estimates.
    :param: train_std_arr: std_arr to use in calibration
    :param: train_target_arr: train_target_arr
    :param: train_pred_arr: train_pred_arr
    :return: linear transformation function.
    """
    # calibration to minimize the miscalibration area as https://pubs.acs.org/doi/10.1021/acscentsci.1c00546 did.
    scaler_num = 200
    scalar_arr = np.linspace(start=0.01, stop=2.0, num=scaler_num)
    area_arr = np.zeros(shape=(scaler_num,))
    for scaler_idx, scaler in enumerate(scalar_arr):
        _, _, area = calibration_curve(std_arr_to_calibrate=train_std_arr * scaler,
                                       train_target_arr=train_target_arr,
                                       train_pred_arr=train_pred_arr,
                                       property_name=label_name, plot=False,
                                       save_name=target_columns)
        area_arr[scaler_idx] = area  # append

    # result
    optimal_scaler_idx = area_arr.argmin()  # min scaler
    optimal_scaler = scalar_arr[optimal_scaler_idx]
    optimal_area = area_arr[optimal_scaler_idx]

    def uncertainty_linear_transformation(std_arr_to_transform: np.array):
        return std_arr_to_transform * optimal_scaler

    # return
    return uncertainty_linear_transformation


def decide_isotonic_scaling(train_std_arr, train_target_arr, train_pred_arr):
    """
    This function decides isotonic regression for uncertainty transformation.
    :param: train_std_arr: std_arr to use in calibration
    :param: train_target_arr
    :param: train_pred_arr
    :return: isotonic uncertainty transformation function
    """
    # train abs error
    train_abs_error = np.abs(train_pred_arr - train_target_arr)  # absolute error

    # isotonic regression
    iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")

    # fit
    try:
        iso_model = iso_model.fit(train_std_arr, train_abs_error)
    except Exception:
        raise RuntimeError("Failed to fit isotonic regression model")

    # predict
    calibrated_std_arr = iso_model.predict(train_std_arr)

    # plot isotonic regression
    fig = plt.figure(figsize=(6, 6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # Draw the scatter plot and marginals.
    ax_histx.tick_params(axis="x", labelbottom=False, labelsize=12)
    ax_histy.tick_params(axis="y", labelleft=False, labelsize=12)

    # the scatter plot
    ax.scatter(train_std_arr, train_abs_error, color='m', s=5.0, label='Original distribution')
    sorted_for_plot = np.argsort(train_std_arr)
    ax.plot(train_std_arr[sorted_for_plot], calibrated_std_arr[sorted_for_plot], label=f'Isotonic regression', color='c', linewidth=2.0)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlabel('Uncertainty', fontsize=14)
    ax.set_ylabel('Absolute error', fontsize=14)

    ax_histx.hist(train_std_arr, density=True, bins=100, color='m', alpha=0.5)
    ax_histy.hist(train_abs_error, orientation='horizontal', density=True, bins=100, color='m', alpha=0.5)
    ax.legend(fontsize=12)
    plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'train_isotonic.png'))
    plt.close()

    # plt.figure(figsize=(6, 6), dpi=300)
    # plt.plot(train_std_arr, train_abs_error, 'mo', label='Original distribution', markersize=5.0)
    # sorted_for_plot = np.argsort(train_std_arr)
    # plt.plot(train_std_arr[sorted_for_plot], calibrated_std_arr[sorted_for_plot], 'c', label='Isotonic regression', linewidth=2.0)
    # plt.title(label_name, fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Estimated prediction uncertainty', fontsize=14)
    # plt.ylabel('Train prediction error', fontsize=14)
    # plt.legend(fontsize=14)
    # plt.tight_layout()
    # plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'train_isotonic.png'))
    # plt.close()

    # decide the optimal linear scaling to minimize the miscalibration area
    scaler_num = 200
    scalar_arr = np.linspace(start=0.01, stop=2.0, num=scaler_num)
    area_arr = np.zeros(shape=(scaler_num,))
    for scaler_idx, scaler in enumerate(scalar_arr):
        _, _, area = calibration_curve(std_arr_to_calibrate=calibrated_std_arr * scaler,
                                       train_target_arr=train_target_arr,
                                       train_pred_arr=train_pred_arr,
                                       property_name=label_name, plot=False,
                                       save_name=target_columns)
        area_arr[scaler_idx] = area  # append

    # result
    optimal_scaler_idx = area_arr.argmin()  # min scaler
    optimal_scaler = scalar_arr[optimal_scaler_idx]
    optimal_area = area_arr[optimal_scaler_idx]

    def isotonic_transformation(std_arr_to_transform: np.array):
        return iso_model.predict(std_arr_to_transform) * optimal_scaler

    # return
    return isotonic_transformation


def evaluate_calibration(linear_transformed_std_arr, isotonic_transformed_std_arr, train_pred_arr, train_target_arr):
    """
    This function evaluates the uncertainty calibration in terms of the following metrics:
    1) miscalibration area
    2) Sharpness: root-mean-squared of uncertainty estimates.
    3) coefficient of variation -> secondary condition (not considered).
    4) negative log likelihood

    return 1 (if isotonic), 0 (if linear)
    """
    # compare
    linear_cnt, isotonic_cnt = 0, 0

    # 1) miscalibration area
    _, _, linear_miscalibration_area = calibration_curve(std_arr_to_calibrate=linear_transformed_std_arr,
                                                         train_target_arr=train_target_arr, train_pred_arr=train_pred_arr,
                                                         property_name=label_name, plot=False,
                                                         save_name=target_columns)
    _, _, isotonic_miscalibration_area = calibration_curve(std_arr_to_calibrate=isotonic_transformed_std_arr,
                                                           train_target_arr=train_target_arr,
                                                           train_pred_arr=train_pred_arr,
                                                           property_name=label_name, plot=False,
                                                           save_name=target_columns)
    # compare
    if linear_miscalibration_area <= isotonic_miscalibration_area:
        linear_cnt += 1
    else:
        isotonic_cnt += 1

    # 2) sharpenss
    linear_sharp_metric = np.sqrt(np.mean(linear_transformed_std_arr ** 2))
    isotonic_sharp_metric = np.sqrt(np.mean(isotonic_transformed_std_arr ** 2))

    # compare
    if linear_sharp_metric <= isotonic_sharp_metric:
        linear_cnt += 1
    else:
        isotonic_cnt += 1

    # 3) coefficient of variation
    # linear_coefficient_of_variation = np.std(linear_transformed_std_arr, ddof=1) / np.mean(linear_transformed_std_arr)
    # isotonic_coefficient_of_variation = np.std(isotonic_transformed_std_arr, ddof=1) / np.mean(isotonic_transformed_std_arr)
    #
    # # compare
    # if linear_coefficient_of_variation >= isotonic_coefficient_of_variation:
    #     linear_cnt += 1
    # else:
    #     isotonic_cnt += 1

    # 4) NLL
    residuals = train_pred_arr - train_target_arr  # residuals

    # linear NLL sum
    linear_nll_arr = stats.norm.logpdf(residuals, scale=linear_transformed_std_arr) * (-1)
    linear_nll_mean = np.mean(linear_nll_arr)

    # isotonic NLL sum
    isotonic_nll_arr = stats.norm.logpdf(residuals, scale=isotonic_transformed_std_arr) * (-1)
    isotonic_nll_mean = np.mean(isotonic_nll_arr)

    if linear_nll_mean <= isotonic_nll_mean:
        linear_cnt += 1
    else:
        isotonic_cnt += 1

    # compare final results (in terms of three criteria)
    if linear_cnt > isotonic_cnt:
        return 0
    else:
        return 1


if __name__ == '__main__':
    """
    This script plots prediction results with uncertainty estimated with evidential deep learning. 
    """
    ################### MODIFY ###################
    dir_name_list = [
        ['240228-134523889858_OMG_train_batch_0_chemprop_train_gnn_0_evidence',
         '240414-143341632187_OMG_train_batch_1_chemprop_train_gnn_0_evidence',
         '240510-115829385918_OMG_train_batch_2_chemprop_train_gnn_0_evidence',
         '240531-100412913979_OMG_train_batch_3_chemprop_train_gnn_0_evidence'],

        ['240228-140859795607_OMG_train_batch_0_chemprop_train_gnn_1_evidence',
         '240414-143927227661_OMG_train_batch_1_chemprop_train_gnn_1_evidence',
         '240510-115831803916_OMG_train_batch_2_chemprop_train_gnn_1_evidence',
         '240531-100828584294_OMG_train_batch_3_chemprop_train_gnn_1_evidence'],

        ['240228-140947749059_OMG_train_batch_0_chemprop_train_gnn_2_evidence',
         '240414-143957455116_OMG_train_batch_1_chemprop_train_gnn_2_evidence',
         '240510-115856198900_OMG_train_batch_2_chemprop_train_gnn_2_evidence',
         '240531-100906010104_OMG_train_batch_3_chemprop_train_gnn_2_evidence'],

        ['240228-141034359440_OMG_train_batch_0_chemprop_train_gnn_3_evidence',
         '240414-144012634335_OMG_train_batch_1_chemprop_train_gnn_3_evidence',
         '240510-115915034483_OMG_train_batch_2_chemprop_train_gnn_3_evidence',
         '240531-101615032820_OMG_train_batch_3_chemprop_train_gnn_3_evidence']
    ]
    strategy = 'pareto_greedy'
    current_batch = 3  # most recent result

    SAVE_DIR = Path('/home/sk77/PycharmProjects/omg_database_publication/figure_publication/model_performance/pred')
    ################### MODIFY ###################

    # plot
    num_gnn = 4
    for gnn_idx in range(num_gnn):
        target_columns_list = target_cols_list[gnn_idx]
        label_name_list = label_name_list_all[gnn_idx]
        dir_name = dir_name_list[gnn_idx][current_batch]
        for target_columns, label_name in zip(target_columns_list, label_name_list):
            print(target_columns)
            # fig save dir
            FIGURE_SAVE_DIR = Path(os.path.join(SAVE_DIR, target_columns))
            FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            # load .csv
            csv_file_dir = os.path.join(ACTIVE_LEARNING_DIR, f'{strategy}_check_point/current_batch_{current_batch}_train/gnn_{gnn_idx}/{dir_name}/fold_0')  # used for active learning.
            train_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'train_pred.csv'))
            test_pred_results = pd.read_csv(os.path.join(csv_file_dir, 'test_pred.csv'))

            # random shuffle for plot
            train_pred_results = train_pred_results.sample(frac=1).reset_index(drop=True)
            test_pred_results = test_pred_results.sample(frac=1).reset_index(drop=True)

            # train
            train_target_arr = train_pred_results[f'true_{target_columns}'].to_numpy()  # same for different fold_idx
            train_pred_arr = train_pred_results[f'{target_columns}'].to_numpy()
            train_std_arr = train_pred_results[f'std_{target_columns}'].to_numpy()

            # test
            test_target_arr = test_pred_results[f'true_{target_columns}'].to_numpy()  # same for different fold_idx
            test_pred_arr = test_pred_results[f'{target_columns}'].to_numpy()
            test_std_arr = test_pred_results[f'std_{target_columns}'].to_numpy()

            # abs error
            train_abs_error = np.abs(train_pred_arr - train_target_arr)  # absolute error
            test_abs_error = np.abs(test_pred_arr - test_target_arr)  # absolute error

            # before calibration
            calibration_curve(std_arr_to_calibrate=train_std_arr, property_name=label_name, plot=True,
                              train_target_arr=train_target_arr, train_pred_arr=train_pred_arr,
                              save_name=target_columns + '_before')  # before calibration
            # uncertainty
            rank_correlation_object = spearmanr(a=train_abs_error, b=train_std_arr)  # the same when a and b are swapped.
            rank_correlation_coefficient = rank_correlation_object.statistic

            # plot
            fig = plt.figure(figsize=(6, 6), dpi=300)
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            # Create the Axes.
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

            # Draw the scatter plot and marginals.
            ax_histx.tick_params(axis="x", labelbottom=False, labelsize=12)
            ax_histy.tick_params(axis="y", labelleft=False, labelsize=12)

            # the scatter plot
            ax.scatter(train_std_arr, train_abs_error, label=f'Rank correlation: {rank_correlation_coefficient:.3f}', s=5, color='m')
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.set_xlabel('Uncertainty', fontsize=14)
            ax.set_ylabel('Absolute error', fontsize=14)

            # now determine nice limits by hand:
            # binwidth = 0.25
            # xymax = max(np.max(np.abs(train_std_arr)), np.max(np.abs(train_abs_error)))
            # lim = (int(xymax / binwidth) + 1) * binwidth

            # bins = np.arange(-lim, lim + binwidth, binwidth)
            ax_histx.hist(train_std_arr, density=True, bins=100, color='m', alpha=0.5)
            ax_histy.hist(train_abs_error, orientation='horizontal', density=True, bins=100, color='m', alpha=0.5)
            ax.legend(fontsize=12)
            plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'uncertainty_{target_columns}_before.png'))
            plt.close()

            # plt.figure(figsize=(6, 6), dpi=300)
            # # plt.plot(train_std_arr, train_abs_error, 'mo', label=f'Rank correlation: {rank_correlation_coefficient:.3f}', markersize=5)
            # # plt.scatter(train_std_arr, train_abs_error, c=z, label=f'Rank correlation: {rank_correlation_coefficient:.3f}', s=5)
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)
            # plt.xlabel('Uncertainty', fontsize=14)
            # plt.ylabel('Absolute error', fontsize=14)
            # plt.legend(fontsize=14)
            # plt.tight_layout()
            # plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'uncertainty_{target_columns}_before.png'))
            # plt.close()

            # 1) linear uncertainty scaling
            optimal_uncertainty_linear_transformation = decide_linear_scaling(train_std_arr=train_std_arr,
                                                                              train_target_arr=train_target_arr,
                                                                              train_pred_arr=train_pred_arr)
            # 2) isotonic uncertainty scaling
            optimal_isotonic_transformation = decide_isotonic_scaling(train_std_arr=train_std_arr,
                                                                      train_target_arr=train_target_arr,
                                                                      train_pred_arr=train_pred_arr)

            # compare linear vs isotonic uncertainty calibration. (1 -> isotonic / 0 -> linear)
            decision = evaluate_calibration(linear_transformed_std_arr=optimal_uncertainty_linear_transformation(train_std_arr),
                                            isotonic_transformed_std_arr=optimal_isotonic_transformation(train_std_arr),
                                            train_target_arr=train_target_arr, train_pred_arr=train_pred_arr)
            if decision == 1:  # isotonic
                uncertainty_transformation_function = optimal_isotonic_transformation
            else:  # linear
                print("Linear is better than isotonic!")
                raise ValueError
                uncertainty_transformation_function = optimal_uncertainty_linear_transformation

            # plot
            transformed_train_std_arr = uncertainty_transformation_function(train_std_arr)
            calibration_curve(std_arr_to_calibrate=transformed_train_std_arr, property_name=label_name, plot=True,
                              train_target_arr=train_target_arr, train_pred_arr=train_pred_arr,
                              save_name=target_columns)

            # check the standardized distribution - seems working except for the tails.
            train_diff_arr = train_pred_arr - train_target_arr
            normalized_train_diff_arr = train_diff_arr / transformed_train_std_arr
            sorted_normalized_train_diff_arr = np.sort(normalized_train_diff_arr)
            plt.figure(figsize=(6, 6), dpi=300)
            plt.hist(sorted_normalized_train_diff_arr[10:-10], density=True, bins=50)
            plt.savefig(os.path.join(FIGURE_SAVE_DIR, 'train_normalize_check.png'))
            plt.close()

            # after calibration
            # uncertainty
            rank_correlation_object = spearmanr(a=train_abs_error, b=transformed_train_std_arr)  # the same when a and b are swapped.
            rank_correlation_coefficient = rank_correlation_object.statistic

            # plot
            fig = plt.figure(figsize=(6, 6), dpi=300)
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            # Create the Axes.
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

            # Draw the scatter plot and marginals.
            ax_histx.tick_params(axis="x", labelbottom=False, labelsize=12)
            ax_histy.tick_params(axis="y", labelleft=False, labelsize=12)

            # the scatter plot
            ax.scatter(transformed_train_std_arr, train_abs_error, label=f'Rank correlation: {rank_correlation_coefficient:.3f}', s=5, color='m')
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.set_xlabel('Uncertainty', fontsize=14)
            ax.set_ylabel('Absolute error', fontsize=14)

            ax_histx.hist(transformed_train_std_arr, density=True, bins=100, color='m', alpha=0.5)
            ax_histy.hist(train_abs_error, orientation='horizontal', density=True, bins=100, color='m', alpha=0.5)
            ax.legend(fontsize=12)
            plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'uncertainty_{target_columns}.png'))
            plt.close()

            # plt.figure(figsize=(6, 6), dpi=300)
            # plt.plot(transformed_train_std_arr, train_abs_error, 'mo', label=f'Rank correlation: {rank_correlation_coefficient:.3f}', markersize=5.0)
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)
            # plt.xlabel('Uncertainty', fontsize=14)
            # plt.ylabel('Absolute error', fontsize=14)
            # plt.legend(fontsize=14)
            # plt.tight_layout()
            # plt.savefig(os.path.join(FIGURE_SAVE_DIR, f'uncertainty_{target_columns}.png'))
            # plt.close()

            # 5) train seaborn plot
            # linear correlation & R2
            linear_correlation_object = pearsonr(x=train_target_arr, y=train_pred_arr)  # the same when x and y are swapped.
            linear_correlation_coefficient = linear_correlation_object.statistic
            r2 = r2_score(y_true=train_target_arr, y_pred=train_pred_arr)

            # get 95% confidence interval (optional)
            norm = stats.norm(loc=0, scale=1)
            z_score = norm.ppf(0.975)  # two-tailed 95%.
            confidence_interval = transformed_train_std_arr * z_score

            # check confidence interval
            confidence_check = train_abs_error <= confidence_interval
            confidence_percent = confidence_check.sum() / train_abs_error.shape[0] * 100
            print(f"The percent of train abs error within 95% confidence interval: {confidence_percent:.2f}%")

            # seaborn
            plot_df = pd.DataFrame()
            plot_df['target'] = train_target_arr
            plot_df['pred'] = train_pred_arr
            plot_df['std'] = transformed_train_std_arr
            plot_min = train_target_arr.min()
            plot_max = train_target_arr.max()

            # scatter plot
            g = sns.JointGrid()
            color_bar = ["#2F65ED", "#F5EF7E", "#EB2329"]
            cmap = LinearSegmentedColormap.from_list("", color_bar)

            # scatter plot
            ax = sns.scatterplot(x='target', y='pred', hue='std', data=plot_df, palette=cmap, ax=g.ax_joint,
                                 edgecolor=None, alpha=0.75, legend=False, size=3.0)

            # kde plots on the marginal axes
            sns.kdeplot(x='target', data=plot_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
            sns.kdeplot(y='pred', data=plot_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

            # tick parameters
            g.ax_joint.tick_params(labelsize=16)
            g.set_axis_labels(f'True {label_name}', f'Prediction {label_name}', fontsize=16)

            # y=x
            g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)
            g.ax_joint.plot([], [], label=f'$R^2$: {r2:.2f}')
            # g.ax_joint.plot([], [], label=f'$\\rho$  : {linear_correlation_coefficient:.2f}')
            g.ax_joint.legend(handlelength=0, fontsize=14, loc='upper left')

            # tight layout
            plt.tight_layout()

            # color bars
            norm = plt.Normalize(plot_df['std'].min(), plot_df['std'].max())
            scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

            # Make space for the colorbar
            g.fig.subplots_adjust(bottom=0.2)
            cax = g.fig.add_axes([0.20, 0.06, 0.6, 0.02])  # l, b, w, h
            cbar = g.fig.colorbar(scatter_map, orientation='horizontal', cax=cax)
            cbar.ax.tick_params(labelsize=14)
            g.fig.savefig(os.path.join(FIGURE_SAVE_DIR, f'pred_train_{target_columns}.png'))
            plt.close()

            # #6) test seaborn plot
            # linear correlation & R2
            linear_correlation_object = pearsonr(x=test_target_arr, y=test_pred_arr)  # the same when x and y are swapped.
            linear_correlation_coefficient = linear_correlation_object.statistic
            r2 = r2_score(y_true=test_target_arr, y_pred=test_pred_arr)

            # get calibrated uncertainties
            transformed_test_std_arr = uncertainty_transformation_function(test_std_arr)

            # get 95% confidence interval (optional)
            norm = stats.norm(loc=0, scale=1)
            z_score = norm.ppf(0.975)  # two-tailed 95%.
            confidence_interval = transformed_test_std_arr * z_score

            # check confidence interval
            confidence_check = test_abs_error <= confidence_interval
            confidence_percent = confidence_check.sum() / test_abs_error.shape[0] * 100
            print(f"The percent of test abs error within 95% confidence interval: {confidence_percent:.2f}%")

            # seaborn
            plot_df = pd.DataFrame()
            plot_df['target'] = test_target_arr
            plot_df['pred'] = test_pred_arr
            plot_df['std'] = transformed_test_std_arr
            plot_min = test_target_arr.min()
            plot_max = test_target_arr.max()

            # scatter plot
            g = sns.JointGrid()
            color_bar = ["#2F65ED", "#F5EF7E", "#EB2329"]
            cmap = LinearSegmentedColormap.from_list("", color_bar)

            # scatter plot
            ax = sns.scatterplot(x='target', y='pred', hue='std', data=plot_df, palette=cmap, ax=g.ax_joint,
                                 edgecolor=None, alpha=0.75, legend=False, size=3.0)

            # kde plots on the marginal axes
            sns.kdeplot(x='target', data=plot_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
            sns.kdeplot(y='pred', data=plot_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

            # tick parameters
            if target_columns == 'radius_of_gyration_Boltzmann_average':
                g.ax_joint.set_xticks(ticks=[0, 5, 10, 15, 20, 25, 30, 35])
                g.ax_joint.set_yticks(ticks=[0, 5, 10, 15, 20, 25, 30, 35])
            g.ax_joint.tick_params(labelsize=16)
            g.set_axis_labels(f'True {label_name}', f'Prediction {label_name}', fontsize=16)

            # y=x
            g.ax_joint.plot([plot_min, plot_max], [plot_min, plot_max], color='#665A5E', linestyle='--', linewidth=1.5)
            g.ax_joint.plot([], [], label=f'$R^2$: {r2:.2f}')
            # g.ax_joint.plot([], [], label=f'$\\rho$  : {linear_correlation_coefficient:.2f}')
            g.ax_joint.legend(handlelength=0, fontsize=14, loc='upper left', frameon=False)

            # tight layout
            plt.tight_layout()

            # color bars
            norm = plt.Normalize(plot_df['std'].min(), plot_df['std'].max())
            scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

            # Make space for the colorbar
            g.fig.subplots_adjust(bottom=0.2)
            cax = g.fig.add_axes([0.20, 0.06, 0.6, 0.02])  # l, b, w, h
            cbar = g.fig.colorbar(scatter_map, orientation='horizontal', cax=cax)
            cbar.ax.tick_params(labelsize=14)
            g.fig.savefig(os.path.join(FIGURE_SAVE_DIR, f'pred_test_{target_columns}.png'), dpi=1200)
            plt.close()

