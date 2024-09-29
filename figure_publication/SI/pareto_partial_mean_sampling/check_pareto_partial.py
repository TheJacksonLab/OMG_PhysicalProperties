import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearFit(object):
    """
    Linear function class with specified gradient and one point to pass.
    """
    def __init__(self, gradient, point_x, point_y):
        """
        gradient, point_x, point_y -> float variables.
        """
        self.gradient = gradient
        self.point_x, self.point_y = point_x, point_y

    def extrapolate(self, x):
        """
        return y value at x
        """
        return self.gradient * (x - self.point_x) + self.point_y

    def get_x_intercept(self):
        """
        return x intercept
        """
        return -self.point_y / self.gradient + self.point_x

    def integrate(self, x1, x2):
        """
        integrate between [x1, x2]
        """
        y1, y2 = self.extrapolate(x1), self.extrapolate(x2)
        integrate_y1 = 1 / 2 * self.gradient * x1 ** 2 + (-self.gradient * x1 + y1) * x1
        integrate_y2 = 1 / 2 * self.gradient * x2 ** 2 + (-self.gradient * x2 + y2) * x2
        return integrate_y2 - integrate_y1


def plot_ratio_of_number_of_molecules_at_the_pareto_front(save_dir,
                                                          number_of_molecules_at_pareto_front_arr: np.array,
                                                          sub_space_size):
    """
    This function plots the number ratio of OMG polymers located to the Pareto front of the multi-dimensional (19)
    uncertainty space to ensure the partial space is a good approximation to the original whole space.

    :param sub_space_size -> subspace size used for partial Pareto
    """
    # number of round
    num_rounds = number_of_molecules_at_pareto_front_arr.shape[0]

    # plot - the number of molecules on the Pareto front
    color_list = ['#BA5BBA', '#EF2525', '#345BBF']
    label_list = ['For round 1', 'For round 2', 'For round 3']
    plt.figure(figsize=(6, 6), dpi=1200)
    for round_idx in range(num_rounds):
        number_of_molecules_at_pareto_front = number_of_molecules_at_pareto_front_arr[round_idx]
        color = color_list[round_idx]
        label = label_list[round_idx]
        plt.xlabel('Subspace size (%)', fontsize=16)
        plt.ylabel('Number of molecules at the Pareto front', fontsize=16)
        plt.plot(sub_space_size * 100, number_of_molecules_at_pareto_front, color=color)
        plt.scatter(sub_space_size * 100, number_of_molecules_at_pareto_front, color=color, marker='o', s=30.0, label=label)
    plt.xticks(fontsize=14)
    plt.yticks(ticks=[100000, 200000, 300000, 400000], labels=['100K', '200K', '300K', '400K'], fontsize=14)
    # plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'number_of_molecules_Pareto_front.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'number_of_molecules_Pareto_front.svg'), format='svg', dpi=1200)
    plt.close()

    # plot - gradient of the number of molecules at the Pareto front
    plt.figure(figsize=(6, 6), dpi=1200)
    for round_idx in range(num_rounds):
        number_of_molecules_at_pareto_front = number_of_molecules_at_pareto_front_arr[round_idx]
        color = color_list[round_idx]
        label = label_list[round_idx]

        # calculate the gradient of the first pareto molecules
        diff_number_of_molecules_at_pareto_front = np.array([
            number_of_molecules_at_pareto_front[idx + 1] - number_of_molecules_at_pareto_front[idx]
            for idx in range(len(number_of_molecules_at_pareto_front) - 1)
        ])
        diff_sub_space_size_percent = np.array([
            (sub_space_size[idx + 1] - sub_space_size[idx]) * 100
            for idx in range(len(sub_space_size) - 1)
        ])
        sub_space_middle_point_percent = np.array([
            (sub_space_size[idx + 1] + sub_space_size[idx]) / 2 * 100 for idx in range(len(sub_space_size) - 1)
        ])
        pareto_increase_gradient = diff_number_of_molecules_at_pareto_front / diff_sub_space_size_percent  # unit inverse percent.

        # plot
        plt.xlabel('Subspace size (%)', fontsize=16)  # va bottom not to overlap between axis and labels
        plt.ylabel('Gradient of the number of Pareto molecules ($\%^{-1}$)', fontsize=14)  # va bottom not to overlap between axis and labels
        plt.plot(sub_space_middle_point_percent, pareto_increase_gradient, color=color)
        plt.scatter(sub_space_middle_point_percent, pareto_increase_gradient, color=color, marker='o', s=30.0, label=label)

        # estimate the quality of mean cut approximation
        diff_sub_space_size_percent_of_middle_point = np.array([
            sub_space_middle_point_percent[idx + 1] - sub_space_middle_point_percent[idx]
            for idx in range(len(sub_space_middle_point_percent) - 1)
        ])
        diff_pareto_increase_gradient = np.array([
            pareto_increase_gradient[idx + 1] - pareto_increase_gradient[idx]
            for idx in range(len(pareto_increase_gradient) - 1)
        ])
        gradient_of_pareto_increase_gradient = diff_pareto_increase_gradient / diff_sub_space_size_percent_of_middle_point  # unit: inverse percent^2

        # get an approximated "linear" form of pareto_increase_gradient.
        gradient = gradient_of_pareto_increase_gradient[-1]  # approximate the linear form
        point_x = sub_space_middle_point_percent[-1]  # unit: percent.
        point_y = pareto_increase_gradient[-1]
        linear_fit = LinearFit(gradient=gradient, point_x=point_x, point_y=point_y)

        x_intercept = linear_fit.get_x_intercept()
        print(f"The pareto front is expected to converge at the subspace of {x_intercept:.2f}% filtering assuming the linear gradient.", flush=True)
        print(f"This value is set to the upper value of the integration\n", flush=True)

        # extrapolate for the lower bound of the unexplored molecules -> linear decay of gradient.
        lower_bound_number_of_unexplored_pareto_molecules = linear_fit.integrate(x1=sub_space_size[-1], x2=x_intercept)

        # the upper bound of the unexplored molecules -> constant gradient
        delta_percent = 100 - sub_space_size[-1] * 100
        upper_bound_number_of_unexplored_pareto_molecules = delta_percent * pareto_increase_gradient[-3]

        # calculate the percentage (lower bound of unexplored Pareto molecules)
        number_of_explored_pareto_molecules = number_of_molecules_at_pareto_front[-3]  # 1/8 cutting
        lower_bound_estimated_number_of_total_pareto_molecules = number_of_explored_pareto_molecules + lower_bound_number_of_unexplored_pareto_molecules
        upper_bound_explored_percentage = number_of_explored_pareto_molecules / lower_bound_estimated_number_of_total_pareto_molecules * 100
        print("=== Upper bound of exploration ===")
        print(f'The estimated LOWER BOUND of the first Pareto molecules is {lower_bound_estimated_number_of_total_pareto_molecules:.2f}', flush=True)
        print(f'The {upper_bound_explored_percentage:.2f}% of the Pareto molecules are explored (UPPER BOUND)\n', flush=True)

        # calculate the percentage (upper bound of unexplored Pareto molecules)
        upper_bound_estimated_number_of_total_pareto_molecules = number_of_explored_pareto_molecules + upper_bound_number_of_unexplored_pareto_molecules
        lower_bound_explored_percentage = number_of_explored_pareto_molecules / upper_bound_estimated_number_of_total_pareto_molecules * 100
        print("=== Lower bound of exploration ===")
        print(f'The estimated number UPPER BOUND of the first Pareto molecules is {upper_bound_estimated_number_of_total_pareto_molecules:.2f}', flush=True)
        print(f'The {lower_bound_explored_percentage:.2f}% of the Pareto molecules are explored (LOWER BOUND)', flush=True)

    plt.xticks(fontsize=14)
    plt.yticks(ticks=[10000, 20000, 30000, 40000, 50000, 60000], labels=['10K', '20K', '30K', '40K', '50K', '60K'], fontsize=14)
    # plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pareto_increase_gradient.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'pareto_increase_gradient.svg'), format='svg', dpi=1200)
    plt.close()


if __name__ == '__main__':
    """
    This function plots the number ratio of OMG polymers located to the Pareto front of the multi-dimensional (19) 
    uncertainty space to ensure the partial space is a good approximation to the original whole space.
    """
    ### MODIFY ###
    pareto_partial_ratio_n_th_list = [256, 128, 64, 32, 16, 14, 12, 10, 8, 6, 5]

    # sampling after initial train
    number_of_molecules_at_pareto_front_list_batch_0 = [12081, 31321, 78715, 148533, 238231, 257195, 280004, 308503, 344883, 394506, 426084]  # current batch 0. if known

    # AL1
    number_of_molecules_at_pareto_front_list_batch_1 = [23089, 44501, 84536, 142551, 214768, 231566, 253039, 281376, 319983, 376809, 416624]

    # AL2
    number_of_molecules_at_pareto_front_list_batch_2 = [21861, 39145, 66595, 109711, 165581, 177901, 192637, 211550, 235804, 269090, 291058]
    ### MODIFY ###

    # np array
    number_of_molecules_at_pareto_front_arr = np.array([
        number_of_molecules_at_pareto_front_list_batch_0,
        number_of_molecules_at_pareto_front_list_batch_1,
        number_of_molecules_at_pareto_front_list_batch_2,
    ])
    sub_space_size = np.array([1/pareto_partial_ratio_n_th for pareto_partial_ratio_n_th in pareto_partial_ratio_n_th_list])

    # plot
    fig_save_dir = f'./figure'
    plot_ratio_of_number_of_molecules_at_the_pareto_front(
        save_dir=fig_save_dir,
        number_of_molecules_at_pareto_front_arr=number_of_molecules_at_pareto_front_arr,
        sub_space_size=sub_space_size,
    )
