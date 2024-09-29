import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    # load df
    df_phi_path = './csv_files/phi_estimation.csv'
    df_phi = pd.read_csv(df_phi_path)
    df_path = './csv_files/413K_clear_experiment_flexibility_with_copolymers.csv'
    df = pd.read_csv(df_path)

    # concat
    df_pred = pd.concat([df, df_phi], axis=1)

    # LOOCV
    number_of_molecules = df_pred.shape[0]
    target_square_end_to_end_distance_per_mass_list, pred_square_end_to_end_distance_per_mass_list = list(), list()
    for molecule_idx in range(number_of_molecules):
        # target value
        target_square_end_to_end_distance_per_mass = df_pred['square_end_to_end_distance_per_mass'][molecule_idx]
        target_variable = np.array([df_pred['monomer_phi_per_total_atom'][molecule_idx], df_pred['backbone_phi_per_total_atom'][molecule_idx]]).reshape(1, -1)

        # linear regression for LOOCV
        df_variable = df_pred.drop(labels=molecule_idx)
        x1 = df_variable['monomer_phi_per_total_atom'].to_numpy().reshape(-1, 1)
        x2 = df_variable['backbone_phi_per_total_atom'].to_numpy().reshape(-1, 1)
        x = np.concatenate([x1, x2], axis=-1)
        y_square_end_to_end_distance_per_mass = df_variable['square_end_to_end_distance_per_mass'].to_numpy()

        # fit
        reg_square_end_to_end_distance_per_mass = LinearRegression().fit(x, y_square_end_to_end_distance_per_mass)

        # pred for LOOCV
        pred_square_end_to_end_distance_per_mass = reg_square_end_to_end_distance_per_mass.predict(target_variable).flatten()[0]

        # append
        target_square_end_to_end_distance_per_mass_list.append(target_square_end_to_end_distance_per_mass)
        pred_square_end_to_end_distance_per_mass_list.append(pred_square_end_to_end_distance_per_mass)

    # plot
    plt.figure(figsize=(6, 6), dpi=300)

    # r2 score
    x, y = np.array(pred_square_end_to_end_distance_per_mass_list), np.array(target_square_end_to_end_distance_per_mass_list)
    r2 = r2_score(y_true=y, y_pred=x)
    plt.scatter(pred_square_end_to_end_distance_per_mass_list, target_square_end_to_end_distance_per_mass_list, color='#D31DD8',
                marker='o', s=20.0, label=f'Prediction $R^2$ = {r2:0.3f}')

    # y=x
    plot_min = min(x.min(), y.min())
    plot_max = max(x.max(), y.max())
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'grey', linestyle='--')

    plt.xlabel('Prediction $<h^2>/M$   [$\mathring{A}^{2}\;$mol$\;g^{-1}$]', fontsize=14)
    plt.ylabel('Experimental $<h^2>/M$   [$\mathring{A}^{2}\;$mol$\;g^{-1}$]', fontsize=14)
    plt.title('LOOCV prediction', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('./figure/square_end_to_end_distance_per_mass.png')
    plt.close()



