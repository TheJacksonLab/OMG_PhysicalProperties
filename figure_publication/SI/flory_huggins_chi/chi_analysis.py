import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from rdkit import Chem


if __name__ == '__main__':
    # TODO
    # different colors for water, ethanol, and chloroform.
    
    # experimental chi
    df_experimental = pd.read_csv('./data/reference_chi_canon.csv')

    # add
    df_pred = pd.read_csv('./chi_results/chi_prediction_0_025_solute_toluene_cosmo_solvent_cosmo_cosmo.csv')

    df_experimental['pred_chi'] = df_pred['chi_parameter_mean'].to_numpy()
    df_experimental['std_chi'] = df_pred['chi_parameter_std'].to_numpy()

    # solvent
    df_experimental = df_experimental[df_experimental['polymer_volume_fraction'] >= 0.2]

    plt.figure(figsize=(6, 6), dpi=300)
    experimental_chi_arr = df_experimental['experimental_chi'].to_numpy()
    pred_chi_arr = df_experimental['pred_chi'].to_numpy()
    std_chi_arr = df_experimental['std_chi'].to_numpy()

    # linear fitting -- just plot
    reg = LinearRegression().fit(pred_chi_arr.reshape(-1, 1), experimental_chi_arr)  # (x, y)

    # pred
    chi_exp_predicted = pred_chi_arr * reg.coef_ + reg.intercept_
    r2_linear_fit = r2_score(y_true=experimental_chi_arr, y_pred=chi_exp_predicted)
    plt.plot(pred_chi_arr, chi_exp_predicted, 'r', label=f'Linear fitting ($R^2$={r2_linear_fit:.3f})', alpha=0.5)
    plt.plot(pred_chi_arr, chi_exp_predicted, 'r', alpha=0.4)

    prediction_r2_score = r2_score(y_true=experimental_chi_arr, y_pred=pred_chi_arr)
    print(prediction_r2_score)

    # y=x plot
    plot_min = min(pred_chi_arr.min(), experimental_chi_arr.min())
    plot_max = max(pred_chi_arr.max(), experimental_chi_arr.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'grey', linestyle='--')

    # depending on solvent
    solvent_arr = df_experimental['solvent'].to_numpy(dtype='str')
    # print(solvent_arr.shape)

    # water
    pred_chi_arr_water = pred_chi_arr[solvent_arr == 'water']
    experimental_chi_arr_water = experimental_chi_arr[solvent_arr == 'water']
    std_chi_arr_water = std_chi_arr[solvent_arr == 'water']
    plt.errorbar(x=pred_chi_arr_water, y=experimental_chi_arr_water, xerr=std_chi_arr_water, color='#1F50C9', ecolor='#1F50C9', linewidth=2.5,
                 elinewidth=1.5, capsize=4.0, fmt="o", markersize=4.0)
    # print(pred_chi_arr_water.shape, experimental_chi_arr_water.shape, std_chi_arr_water.shape)

    # ethanol
    pred_chi_arr_ethanol = pred_chi_arr[solvent_arr == 'ethanol']
    experimental_chi_arr_ethanol = experimental_chi_arr[solvent_arr == 'ethanol']
    std_chi_arr_ethanol = std_chi_arr[solvent_arr == 'ethanol']
    plt.errorbar(x=pred_chi_arr_ethanol, y=experimental_chi_arr_ethanol, xerr=std_chi_arr_ethanol, color='#1F7F28', ecolor='#1F7F28',
                 linewidth=2.5, elinewidth=1.5, capsize=4.0, fmt="o", markersize=4.0)
    # print(pred_chi_arr_ethanol.shape, experimental_chi_arr_ethanol.shape, std_chi_arr_ethanol.shape)

    # chloroform
    pred_chi_arr_chloroform = pred_chi_arr[solvent_arr == 'chloroform']
    experimental_chi_arr_chloroform = experimental_chi_arr[solvent_arr == 'chloroform']
    std_chi_arr_chloroform = std_chi_arr[solvent_arr == 'chloroform']
    plt.errorbar(x=pred_chi_arr_chloroform, y=experimental_chi_arr_chloroform, xerr=std_chi_arr_chloroform, color='#A3A838', ecolor='#A3A838',
                 linewidth=2.5, elinewidth=1.5, capsize=4.0, fmt="o", markersize=4.0)
    # print(pred_chi_arr_chloroform.shape, experimental_chi_arr_chloroform.shape, std_chi_arr_chloroform.shape)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('COSMO-SAC $\chi$  [unitless]', fontsize=16)
    plt.ylabel('Experimental $\chi$  [unitless]', fontsize=16)
    plt.legend(fontsize=15, loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig('./chi_results/chi_pred.png')
