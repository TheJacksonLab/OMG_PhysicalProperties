import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    """
    This script plots scaled mean of active learning results. 
    """
    # random seed list
    random_split_seed_list = [42, 43, 44, 45, 46]

    # gather rmse
    # strategy_list = ['random', 'explorative_greedy', 'explorative_diverse', 'pareto_greedy', 'pareto_diverse', 'pareto_greedy_partial', 'rank_mean']
    strategy_list = ['random', 'explorative_greedy', 'pareto_greedy_partial_stable_sort']
    target_col_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    number_of_tasks = len(target_col_list)
    # color_list = ['c', 'm', 'r', 'g', 'b', 'k', 'y']
    color_list = ['c', 'm', 'b']
    plot_strategy = ['Random', 'Mean uncertainty', 'Pareto sampling (12.5% subspace)']
    if len(strategy_list) != len(color_list):
        raise ValueError
    if len(strategy_list) != len(plot_strategy):
        raise ValueError
    save_fig_file_name_png = 'figure_s5a.png'
    save_fig_file_name_svg = 'figure_s5a.svg'

    # plot type
    plot_type = 'test'
    if plot_type == 'test':
        fig_save_path_dir = Path(f'./{plot_type}')
        fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError('Check your plot type!')

    # plot batch
    num_train_batch = 10

    # scale std arr
    scaler_std_list = list()
    df_total_train_data = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/qm9_active_learning/qm9_data/num_tasks_12.csv')  # total QM9 data to compare different random seeds
    df_total_train_data_arr = df_total_train_data.to_numpy()[:, 1:]  # exclude smi columns
    total_train_scaler = StandardScaler()
    total_train_scaler.fit(df_total_train_data_arr)
    scaler_std_list.extend(total_train_scaler.scale_.tolist())
    scaler_std_list = scaler_std_list * num_train_batch  # (number_of_task * num_train_batch,)
    scaler_std_arr = np.array(scaler_std_list)  # (number_of_task * num_train_batch,)

    # gather RMSE
    plot_x_ratio_train_data = list()
    plot_y_mean_scaled_test_rmse_arr = list()
    for random_split_seed in random_split_seed_list:
        active_learning_dir = f'../active_learning/active_learning_random_seed_{random_split_seed}'
        for color, strategy in zip(color_list, strategy_list):
            check_point_dir = os.path.join(active_learning_dir, f'./{strategy}_check_point')
            plot_batch_list = range(num_train_batch)
            batch_train_dir = os.path.join(check_point_dir, f'chemprop_current_batch_0')  # all log files were written as the writer was not closed.
            verbose_path = os.path.join(batch_train_dir, os.listdir(batch_train_dir)[0], 'verbose.log')
            rmse_list = list()
            train_size_list = list()
            with open(verbose_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    tmp_line = line.split()
                    if line.startswith('Model 0 test'):
                        rmse = float(tmp_line[-1])
                        rmse_list.append(rmse)
                    if 'train size' in line:
                        number_of_train_data = tmp_line[8].replace(',', '')
                        train_size_list.append(int(number_of_train_data))
                file.close()

            # divide rmse loss test rmse (for now, duplicated errors)
            test_rmse_list = list()
            for idx, rmse in enumerate(rmse_list):
                if idx % (number_of_tasks * 2) >= number_of_tasks:  # test rmse
                    test_rmse_list.append(rmse)
            test_rmse_arr = np.array(test_rmse_list)  # [number_of_tasks * batch,]
            test_rmse_arr = test_rmse_arr[:number_of_tasks * num_train_batch]

            # convert train size to percentage
            total_train_data_size = 120496
            ratio_train_data = np.array(train_size_list) / total_train_data_size

            # scaled rmse by dividing a standard deviation
            scaled_test_rmse_arr = test_rmse_arr / scaler_std_arr
            # scaled_test_rmse_arr = test_rmse_arr  # unscaled

            # mean of RMSE
            scaled_test_rmse_arr = scaled_test_rmse_arr.reshape(num_train_batch, number_of_tasks)
            mean_scaled_test_rmse_arr = np.mean(scaled_test_rmse_arr, axis=1)  # (num_train_batch,)

            # append
            plot_x_ratio_train_data.append(ratio_train_data[:num_train_batch].tolist())  # append list
            plot_y_mean_scaled_test_rmse_arr.append(mean_scaled_test_rmse_arr.tolist())  # append list

    # reshape arr
    plot_x_ratio_train_data = np.array(plot_x_ratio_train_data)
    plot_x_ratio_train_data = plot_x_ratio_train_data.reshape(-1, len(strategy_list), num_train_batch)  # (number of random seeds, strategy, train_batch)
    plot_y_mean_scaled_test_rmse_arr = np.array(plot_y_mean_scaled_test_rmse_arr)
    plot_y_mean_scaled_test_rmse_arr = plot_y_mean_scaled_test_rmse_arr.reshape(-1, len(strategy_list), num_train_batch)  # (number of random seeds, strategy, train_batch)

    # plot
    plt.figure(figsize=(7, 6), dpi=300)
    plot_x_mean = plot_x_ratio_train_data.mean(axis=0)  # (strategy, train_batch)
    plot_x_std = plot_x_ratio_train_data.std(axis=0, ddof=0)  # (strategy, train_batch) -> all zero
    plot_y_mean = plot_y_mean_scaled_test_rmse_arr.mean(axis=0)  # (strategy, train_batch)
    plot_y_std = plot_y_mean_scaled_test_rmse_arr.std(axis=0, ddof=0)  # (strategy, train_batch)
    for idx, (color, strategy) in enumerate(zip(color_list, strategy_list)):
        plt.plot(plot_x_mean[idx] * 100, plot_y_mean[idx], color=color, label=plot_strategy[idx])
        plt.fill_between(plot_x_mean[idx] * 100, plot_y_mean[idx] - plot_y_std[idx], plot_y_mean[idx] + plot_y_std[idx],
                         color=color, alpha=0.2)
    plt.legend(fontsize=14, frameon=False)
    plt.xlabel('Train data ratio (%)', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(f'Active learning', fontsize=16)
    if plot_type == 'test':
        plt.ylabel(f'Test mean of RMSE (scaled)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_path_dir, save_fig_file_name_png))
        plt.savefig(os.path.join(fig_save_path_dir, save_fig_file_name_svg), format='svg', dpi=1200)
        plt.close()

