import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ratio_of_number_of_molecules_at_the_pareto_front(save_dir, uncertainty_sampling_cycle,
                                                          number_of_molecules_at_pareto_front: list or tuple,
                                                          sub_space_size):
    """
    This function plots the number ratio of OMG polymers located to the Pareto front of the multi-dimensional (21)
    uncertainty space to ensure the partial space is a good approximation to the original whole space.

    :param uncertainty_sampling_cycle -> uncertainty sampling with the trained model of the active learning train batch "n".
    0 means initial training set.
    :param sub_space_size -> subspace size used for partial Pareto
    """
    diff_number_of_molecules_at_pareto_front = np.array([
        number_of_molecules_at_pareto_front[idx + 1] - number_of_molecules_at_pareto_front[idx]
        for idx in range(len(number_of_molecules_at_pareto_front) - 1)
    ])
    diff_sub_space_size_percent = np.array([
        (sub_space_size[idx + 1] - sub_space_size[idx]) * 100
        for idx in range(len(sub_space_size) - 1)
    ])
    pareto_increase_gradient = diff_number_of_molecules_at_pareto_front / diff_sub_space_size_percent
    # print(pareto_increase_gradient)
    # print(number_of_molecules_at_pareto_front)
    # print(sub_space_size)
    # exit()

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    fig, ax1 = plt.subplots()

    color = 'c'
    ax1.set_xlabel('Subspace size (%)', fontsize=14)
    ax1.set_ylabel('Number of Pareto molecules', fontsize=14, color=color)
    ax1.plot(sub_space_size * 100, number_of_molecules_at_pareto_front, color=color)
    ax1.plot(sub_space_size * 100, number_of_molecules_at_pareto_front, f'{color}o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'm'
    ax2.set_ylabel('Gradient ($\%^{-1}$)', fontsize=14, color=color, rotation=270, va='bottom')  # va bottom not to overlap between axis and labels
    sub_space_middle_point_percent = np.array([
        (sub_space_size[idx + 1] + sub_space_size[idx]) / 2 * 100 for idx in range(len(sub_space_size) - 1)
    ])
    ax2.plot(sub_space_middle_point_percent, pareto_increase_gradient, color=color)
    ax2.plot(sub_space_middle_point_percent, pareto_increase_gradient, f'{color}o')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pareto_increase_gradient_active_learning_cycle_{uncertainty_sampling_cycle}.png'))


if __name__ == '__main__':
    """
    This function plots the number ratio of OMG polymers located to the Pareto front of the multi-dimensional (21) 
    uncertainty space to ensure the partial space is a good approximation to the original whole space.
    """
    save_dir = '/home/sk77/PycharmProjects/omg_database_publication/active_learning/figure/pareto_partial'
    uncertainty_sampling_cycle = 0  # uncertainty sampling with the trained model of the active learning train batch "n". 0 means initial training set.
    # job_num in uncertainty sampling outfile dir:
    # 48753 (256th), 48754 (128th), 48755 (64th), 48756 (32th), 48757 (16th), 48758 (14th), 48759 (12th), 48760 (10th), 48761 (8th), 48772 (6th)
    number_of_molecules_at_pareto_front = np.array([29917, 55361, 93468, 142983, 194292, 203574, 214229, 226882, 242770, 264721])
    sub_space_size = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/14, 1/12, 1/10, 1/8, 1/6])

    # plot
    plot_ratio_of_number_of_molecules_at_the_pareto_front(
        save_dir=save_dir,
        uncertainty_sampling_cycle=uncertainty_sampling_cycle,
        number_of_molecules_at_pareto_front=number_of_molecules_at_pareto_front,
        sub_space_size=sub_space_size,
    )
