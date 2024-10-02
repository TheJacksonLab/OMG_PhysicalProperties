import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from matplotlib import rcParams

from sklearn.decomposition import PCA


property_columns_Boltzmann = \
        ['asphericity', 'eccentricity', 'inertial_shape_factor', 'radius_of_gyration', 'spherocity',
         'molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi'] + \
        ['HOMO_minus_1', 'HOMO', 'LUMO', 'LUMO_plus_1', 'dipole_moment', 'quadrupole_moment', 'polarizability'] + \
        ['s1_energy', 'dominant_transition_energy', 'dominant_transition_oscillator_strength', 't1_energy']
property_columns_mean = ['chi_parameter_water', 'chi_parameter_ethanol', 'chi_parameter_chloroform']

matrix_columns = \
        ['$\Omega_\mathrm{A}$', '$\epsilon$', '$S_\mathrm{I}$', '$R_\mathrm{g}$', '$\Omega_\mathrm{S}$',
         'MW', 'LogP', 'QED', 'TPSA', '$\Phi_\mathrm{monomer}$', '$\Phi_\mathrm{backbone}$'] + \
        ['$E_{\mathrm{HOMO}\minus1}}$', '$E_{\mathrm{HOMO}}$', '$E_{\mathrm{LUMO}}$', '$E_{\mathrm{LUMO}\plus1}$', '$\mu$',
         '$q$', '$\\alpha$'] + \
        ['$E_{\mathrm{S}_1}$', "$E^{\prime}_{\mathrm{singlet}}$", "$f^{\prime}_{\mathrm{osc}}$", '$E_{\mathrm{T}_1}$'] + \
        ['$\chi_{\mathrm{water}}$', '$\chi_{\mathrm{ethanol}}$', '$\chi_{\mathrm{chloroform}}$']


def search_center(pc_vectors):
    """
    This function searches the center molecule of the PC space (distance-based search)
    :param pc_vectors: PC vectors to search the center for
    :return: center_molecule_idx (not reaction idx. idx on the dataframe), pc vector of the center molecule
    """
    # get mean
    pc_mean = pc_vectors.mean(axis=0)  # (num_targets,)
    mean_tile = np.tile(pc_mean, reps=(pc_vectors.shape[0], 1))  # (num_data, num_targets)

    # get distance from the mean point
    distance_vector = pc_vectors - mean_tile
    l2_norm_vector = np.linalg.norm(distance_vector, axis=1)

    # get idx
    center_molecule_idx = l2_norm_vector.argmin()

    return center_molecule_idx, pc_vectors[center_molecule_idx]


def search_molecule(pc1_move, pc2_move, center_molecule_idx, pc_vectors):
    """
    This function search for the molecule with pc1_move, pc2_move from the center of the molecule (distance-based)
    :param pc1_move: amount of the PC1 move from the center of the molecule
    :param pc2_move:amount of the PC1 move from the center of the molecule
    :param center_molecule_idx: center molecule idx(not reaction idx. idx on the dataframe)
    :param pc_vectors: PC vectors
    :return: target_molecule_idx (not reaction idx. idx on the dataframe)
    """
    # get PC1 move vector
    pc1_move_vector = np.zeros(shape=(pc_vectors.shape[1],))
    pc1_move_vector[0] = pc1_move

    # get PC2 move vector
    pc2_move_vector = np.zeros(shape=(pc_vectors.shape[1],))
    pc2_move_vector[1] = pc2_move

    # center vector
    center_molecule_pc_vector = pc_vectors[center_molecule_idx].copy()

    # get target PC vector
    target_pc_vector = center_molecule_pc_vector + pc1_move_vector + pc2_move_vector

    # search
    target_pc_tile = np.tile(target_pc_vector, reps=(pc_vectors.shape[0], 1))  # (num_data, num_targets)
    distance_vector = pc_vectors - target_pc_tile
    distance_vector_target = distance_vector[:, [0, 1]]  # only care the distance of (PC1, PC2 points)
    l2_norm_vector = np.linalg.norm(distance_vector_target, axis=1)

    # get idx
    target_molecule_idx = l2_norm_vector.argmin()

    return target_molecule_idx


def find_reactants(reaction_id):
    """
    This function identifies OMG reactants to OMG polymers
    """
    df_OMG = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/OMG_methyl_terminated_polymers.csv')
    df_target = df_OMG[df_OMG['reaction_id'] == reaction_id]

    reactant_1 = df_target['reactant_1'].tolist()[0]
    reactant_2 = df_target['reactant_2'].tolist()[0]
    polymerization_mechanism_idx = df_target['polymerization_mechanism_idx'].tolist()[0]

    return reactant_1, reactant_2, polymerization_mechanism_idx


if __name__ == '__main__':
    # plot columns
    plot_columns_list = [f'{column}_Boltzmann_average' for column in property_columns_Boltzmann] + [f'{column}_mean' for column in property_columns_mean]
    matrix_columns_list = [f'{column}' for column in matrix_columns]

    # load .csv
    df = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/sampled_predictions_AL3_100K.csv')
    methyl_terminated_smi_list = df['methyl_terminated_product'].tolist()
    reaction_id_list = df['reaction_id'].tolist()

    # property list
    polarizability = df['polarizability_Boltzmann_average'].to_numpy()
    dominant_singlet = df['dominant_transition_energy_Boltzmann_average'].to_numpy()
    molecular_weight = df['molecular_weight_Boltzmann_average'].to_numpy()
    quadrupole_moment = df['quadrupole_moment_Boltzmann_average'].to_numpy()
    radius_of_gyration = df['radius_of_gyration_Boltzmann_average'].to_numpy()

    # property to use in a color bar
    property_name = 'radius_of_gyration_Boltzmann_average'
    property_label = '$R_g$'
    property_color_bar = df[property_name].to_numpy().copy()

    # get mean & std
    property_arr = df[plot_columns_list].to_numpy().copy()  # unnormalized. (num_data, num_targets = 25)
    mean_property_arr = property_arr.mean(axis=0)  # (num_targets,)
    std_property_arr = property_arr.std(axis=0, ddof=0)  # (num_targets,)

    # tile to unify the shape
    scaler_mean_tile = np.tile(mean_property_arr, reps=(property_arr.shape[0], 1))  # (num_data, num_targets)
    scaler_std_tile = np.tile(std_property_arr, reps=(property_arr.shape[0], 1))  # (num_data, num_targets)

    # normalize
    normalized_property_arr = (property_arr - scaler_mean_tile) / scaler_std_tile  # (num_data, num_targets)

    # PCA
    n_components = property_arr.shape[1]  # 25
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(normalized_property_arr)

    principal_df = pd.DataFrame(
        data=principal_components,
        columns=['PC%d' % (num + 1) for num in range(n_components)]
    )

    # append property
    principal_df['property'] = property_color_bar

    # search center of the PC space (over "all" PC axes)
    center_idx, center_pc_vector = search_center(principal_components)
    print('==================================================')
    print(f"The center molecule (methyl-terminated) is {methyl_terminated_smi_list[center_idx]}")
    print(f"The reaction_id is {reaction_id_list[center_idx]}")
    reactant_1, reactant_2, polymerization_mechanism_idx = find_reactants(reaction_id_list[center_idx])
    print(f"The reactants are {reactant_1, reactant_2}")
    print(f"The polymerization mechanism idx is {polymerization_mechanism_idx}")
    print(f"The polarizability is {polarizability[center_idx]}")
    print(f"The dominant_singlet is {dominant_singlet[center_idx]}")
    print(f"The molecular_weight is {molecular_weight[center_idx]}")
    print(f"The quadrupole_moment is {quadrupole_moment[center_idx]}")
    print(f"The radius_of_gyration is {radius_of_gyration[center_idx]}")

    # calculate std value
    pc_std = principal_components.std(axis=0, ddof=0)
    pc1_std_value, pc2_std_value = pc_std[0], pc_std[1]

    target_molecule_idx_pc1_minus_std = search_molecule(pc1_move=-pc1_std_value, pc2_move=0, center_molecule_idx=center_idx, pc_vectors=principal_components)
    print('==================================================')
    print(f"The PC1 - 1 std moved (methyl-terminated) is {methyl_terminated_smi_list[target_molecule_idx_pc1_minus_std]}")
    print(f"The reaction_id is {reaction_id_list[target_molecule_idx_pc1_minus_std]}")
    reactant_1, reactant_2, polymerization_mechanism_idx = find_reactants(reaction_id_list[target_molecule_idx_pc1_minus_std])
    print(f"The reactants are {reactant_1, reactant_2}")
    print(f"The polymerization mechanism idx is {polymerization_mechanism_idx}")
    print(f"The polarizability is {polarizability[target_molecule_idx_pc1_minus_std]}")
    print(f"The dominant_singlet is {dominant_singlet[target_molecule_idx_pc1_minus_std]}")
    print(f"The molecular_weight is {molecular_weight[target_molecule_idx_pc1_minus_std]}")
    print(f"The quadrupole_moment is {quadrupole_moment[target_molecule_idx_pc1_minus_std]}")
    print(f"The radius_of_gyration is {radius_of_gyration[target_molecule_idx_pc1_minus_std]}")

    # investigate a multi-dimensional property space (only PC1 move)
    target_molecule_idx_pc1_plus_std = search_molecule(pc1_move=pc1_std_value, pc2_move=0,
                                                       center_molecule_idx=center_idx, pc_vectors=principal_components)
    print('==================================================')
    print(f"The PC1 + 1 std moved (methyl-terminated) is {methyl_terminated_smi_list[target_molecule_idx_pc1_plus_std]}")
    print(f"The reaction_id is {reaction_id_list[target_molecule_idx_pc1_plus_std]}")
    reactant_1, reactant_2, polymerization_mechanism_idx = find_reactants(reaction_id_list[target_molecule_idx_pc1_plus_std])
    print(f"The reactants are {reactant_1, reactant_2}")
    print(f"The polymerization mechanism idx is {polymerization_mechanism_idx}")
    print(f"The polarizability is {polarizability[target_molecule_idx_pc1_plus_std]}")
    print(f"The dominant_singlet is {dominant_singlet[target_molecule_idx_pc1_plus_std]}")
    print(f"The molecular_weight is {molecular_weight[target_molecule_idx_pc1_plus_std]}")
    print(f"The quadrupole_moment is {quadrupole_moment[target_molecule_idx_pc1_plus_std]}")
    print(f"The radius_of_gyration is {radius_of_gyration[target_molecule_idx_pc1_plus_std]}")

    # plot explained variance by principal component analysis
    fig_save_dir = Path(os.path.join('./pca/AL3_figure_3'))
    fig_save_dir.mkdir(exist_ok=True, parents=True)
    exp_var_pca = pca.explained_variance_ratio_
    plt.bar(range(1, len(exp_var_pca) + 1), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'Explained_ratio.png'), dpi=300)
    plt.close()

    # eigenvector component 1
    top_n = 7  # 25 -> all
    pc1_vector = pca.components_[0]
    linear_contribution_pc1 = np.abs(pc1_vector)
    plot_idx = np.flip(np.argsort(linear_contribution_pc1))[:top_n]
    plt.figure(figsize=(7, 6))
    plt.bar([matrix_columns[idx] for idx in plot_idx], pc1_vector[plot_idx], alpha=0.5, align='center',
           color='#7C8E8D', width=0.25)
    plt.axhline(y=0.0, linewidth=1, color='k')  # zero line
    plt.ylabel('PC1 contribution', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'pc1_eigen_vector.png'), dpi=300)
    plt.savefig(os.path.join(fig_save_dir, 'pc1_eigen_vector.svg'), dpi=1200, format='svg')
    plt.close()

    # eigenvector component 2
    pc2_vector = pca.components_[1]
    linear_contribution_pc2 = np.abs(pc2_vector)
    plot_idx = np.flip(np.argsort(linear_contribution_pc2))[:top_n]
    plt.figure(figsize=(7, 6))
    plt.bar([matrix_columns[idx] for idx in plot_idx], pc2_vector[plot_idx], alpha=0.5, align='center',
           color='#7C8E8D', width=0.25)
    plt.axhline(y=0.0, linewidth=1, color='k')  # zero line
    plt.ylabel('PC2 contribution', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'pc2_eigen_vector.png'), dpi=300)
    plt.savefig(os.path.join(fig_save_dir, 'pc2_eigen_vector.svg'), dpi=1200, format='svg')
    plt.close()

    # plot
    g = sns.JointGrid(x='PC1', y='PC2', data=principal_df)

    # scatter plot
    cmap = LinearSegmentedColormap.from_list("", ["#3E76BB", "#90A8BF", "#DD999E", "#EC2027"])
    sns.scatterplot(x='PC1', y='PC2', data=principal_df, hue='property', ax=g.ax_joint, palette=cmap,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0)
    sns.kdeplot(x='PC1', data=principal_df, ax=g.ax_marg_x, fill=True, color='#4DC3C8')
    sns.kdeplot(y='PC2', data=principal_df, ax=g.ax_marg_y, fill=True, color='#4DC3C8')

    # investigation points
    g.ax_joint.scatter(principal_components[center_idx][0], principal_components[center_idx][1], marker='*', color="#231F20", s=100.0)
    g.ax_joint.scatter(principal_components[target_molecule_idx_pc1_plus_std][0], principal_components[target_molecule_idx_pc1_plus_std][1], marker='*', color="#231F20", s=100.0)
    g.ax_joint.scatter(principal_components[target_molecule_idx_pc1_minus_std][0], principal_components[target_molecule_idx_pc1_minus_std][1], marker='*', color="#231F20", s=100.0)

    # tick & label
    g.ax_joint.tick_params(labelsize=18)
    g.set_axis_labels('Principal component 1', 'Principal component 2', fontsize=18)
    g.fig.tight_layout()

    # color bars
    norm = plt.Normalize(principal_df[f'property'].min(), principal_df[f'property'].max())
    scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    g.fig.subplots_adjust(bottom=0.225)
    cax = g.fig.add_axes([0.22, 0.08, 0.6, 0.02])  # l, b, w, h
    cbar = g.fig.colorbar(scatter_map, cax=cax, orientation='horizontal', ticks=[5, 10, 15, 20, 25, 30, 35])
    cbar.set_label(label=property_label, size=12)
    cbar.ax.tick_params(labelsize=16)

    # save
    g.fig.savefig(os.path.join(fig_save_dir, f'latent_space_1_2_radius_of_gyration.png'), dpi=1200)
