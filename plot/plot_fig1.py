# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import os
import numpy as np
import matplotlib.pyplot as plt

from sampling_complexity import evaluate_sampling_complexity
from helper import setup_figure_latex_layout, COLORS


if __name__ == '__main__':

    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    deltas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    deltas_grid, epsilons_grid = np.meshgrid(deltas, epsilons)

    # determine figsize for paper
    figsize = setup_figure_latex_layout(2.6, single_column=True)
    # set up figure
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize,
                           gridspec_kw={'right': 0.88, 'left': -0.04, 'bottom': -0.03, 'top': 1.12},
                           subplot_kw={'projection': '3d'})

    # select models to compare
    models = ['C[16-2]', 'Q[14-4]', 'C[64-2]']
    models_label = ['small Classical', 'small Quantum', 'large Classical']
    models_ids = [range(100), range(100), range(100)]
    models_color = [COLORS[m] for m in models]

    # iterate over models to compare
    for model, model_ids, model_color, model_label in zip(models, models_ids, models_color, models_label):
        sampling_complexities = evaluate_sampling_complexity(
            path=os.path.join('results', 'dqn', 'a3-main', 'deg3'),
            filename=model,
            experiment_ids=model_ids,
            epsilons=epsilons,
            deltas=deltas,
            samples_per_epoch=2000,
            max_epochs=100
        )
        # show as surface plot
        ax.plot_surface(epsilons_grid, deltas_grid, sampling_complexities,
                        linewidth=.5, color=model_color, alpha=0.65, linestyle='-', edgecolor='black', label=model_label)

    # set labels and ticks
    ax.set_xlabel(r'$\textbf{relative error } \varepsilon$', fontsize=8, labelpad=-7.0)
    ax.set_xticks(epsilons, ['', 0.15, '', 0.25, '', 0.35, ''], va='center', ha='center')
    ax.set_xlim(left=0.1, right=0.4)
    ax.tick_params(axis='x', which='major', pad=-2.5)
    ax.set_ylabel(r'$\textbf{threshold probability } \delta$', fontsize=8, labelpad=-9.0)
    # ax.set_yticks(deltas, ['', 0.2, '', 0.4, '', 0.6, ''], va='center', ha='center')
    ax.set_yticks(deltas, ['', 0.4, '', 0.6, '', 0.8, ''], va='center', ha='center')
    ax.set_ylim(bottom=0.3, top=0.9)
    ax.tick_params(axis='y', which='major', pad=-3.0)
    ax.set_zlabel(r'$\textbf{sample complexity } \hat{S}$', fontsize=8, labelpad=-4.5, rotation=90)
    ax.set_zticks([0, 50000, 100000, 150000, 200000], ['', '$50$k', '$100$k', '$150$k', '$200$k'], va='center', ha='center')
    ax.tick_params(axis='z', which='major', pad=-0.5)
    # invert order of y-axis for nicer plotting
    plt.gca().invert_yaxis()

    # set viewing angle
    ax.view_init(elev=22.5, azim=135)

    # plot legend
    leg = fig.legend(ncol=1, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.4, 1.0), framealpha=1.0, title=r"$\textbf{Model}$")
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)

    # save figure
    path = os.path.join(os.path.dirname(__file__), 'plots', f'fig1.pdf')
    plt.savefig(path, format='pdf')
    print(f'Figure saved to {path}.')
