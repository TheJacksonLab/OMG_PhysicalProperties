import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

if __name__ == '__main__':
    df = pd.read_csv('./normalized_phi_estimation.csv')  # (315, 4) -> ['canon_p_smi', 'experimental_Tg', 'polymer_phi', 'polymer_backbone_phi']

    # correlation
    experimental_Tg_arr = df['experimental_Tg'].to_numpy()
    normalized_monomer_phi_arr = df['normalized_monomer_phi'].to_numpy()
    normalized_backbone_phi_arr = df['normalized_backbone_phi'].to_numpy()

    # plot - Tg versus polymer phi
    color = 'm'
    pearson_r = pearsonr(x=normalized_monomer_phi_arr, y=experimental_Tg_arr)
    linear_correlation = pearson_r.statistic
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(normalized_monomer_phi_arr, experimental_Tg_arr, color=color, alpha=0.5,
                label=f'Linear correlation $\\rho$ = {linear_correlation:.3f}')
    plt.xlabel('$\Phi_\mathrm{mon}$ [unitless]', fontsize=16)
    plt.ylabel('Experimental $T_g$ [K]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig('./plot/correlation_normalized_monomer_phi.png')
    plt.close()

    # plot - Tg versus polymer backbone phi
    color = 'g'
    pearson_r = pearsonr(x=normalized_backbone_phi_arr, y=experimental_Tg_arr)
    linear_correlation = pearson_r.statistic
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(normalized_backbone_phi_arr, experimental_Tg_arr, color=color, alpha=0.5,
                label=f'Linear correlation $\\rho$ = {linear_correlation:.3f}')
    plt.xlabel('$\Phi_\mathrm{bb}$ [unitless]', fontsize=16)
    plt.ylabel('Experimental $T_g$ [K]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig('./plot/correlation_normalized_backbone_phi.png')
    plt.close()



