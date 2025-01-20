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
import matplotlib as mpl

from cluster_bootstrapping import evaluate_sampling_complexity_with_percentiles
from helper import setup_figure_latex_layout, COLORS


if __name__ == '__main__':

    # determine figsize for paper
    figsize = setup_figure_latex_layout(2.5, single_column=True)

    # set up figure, shift y-axis to right side
    plt.rcParams['ytick.right'], plt.rcParams['ytick.labelright'] = True, True
    plt.rcParams['ytick.left'], plt.rcParams['ytick.labelleft'] = False, False
    fig, ax = plt.subplots(2, 1, sharex=False, sharey=True, figsize=figsize,
                           gridspec_kw={'right': 0.86, 'left': 0.05, 'bottom': 0.125, 'top': 0.925, 'hspace': 0.75})

    # plot for complexity with trajectory degree
    epsilon, delta = 0.15, 0.75

    ##############################
    ### plot for increasing width (classical)

    # select models to compare
    models = ['C[16-2]', 'C[32-2]', 'C[64-2]', 'C[128-2]', 'C[256-2]']
    models_ids = [range(100), range(100), range(100), range(100), range(100)]

    # iterate over models to compare
    sampling_complexities_width = []
    for model, model_ids in zip(models, models_ids):
        # extract sampling complexities, post-process for plotting of percentiles
        sampling_complexities = evaluate_sampling_complexity_with_percentiles(
            path=os.path.join('results', 'dqn', 'a3-main', 'deg3'),
            filename=model,
            experiment_ids=model_ids,
            epsilons=[epsilon],
            deltas=[delta],
            samples_per_epoch=2000,
            max_epochs=100
        )
        sampling_complexities[:, :, 1] = sampling_complexities[:, :, 0] - sampling_complexities[:, :, 1]  # lower percentile
        sampling_complexities[:, :, 2] = sampling_complexities[:, :, 2] - sampling_complexities[:, :, 0]  # upper percentile
        sampling_complexities_width.append(sampling_complexities[0][0])
    sampling_complexities_width = np.array(sampling_complexities_width)

    # plot data point and error bar indicating percentiles
    ax[1].scatter([0], sampling_complexities_width[0, 0], color=COLORS['classical_model'], marker='^')
    ax[1].scatter([1], sampling_complexities_width[1, 0], color=COLORS['classical_model'], marker='<')
    ax[1].scatter([2], sampling_complexities_width[2, 0], color=COLORS['classical_model'], marker='>')
    ax[1].scatter([3], sampling_complexities_width[3, 0], color=COLORS['classical_model'], marker='.')
    ax[1].scatter([4], sampling_complexities_width[4, 0], color=COLORS['classical_model'], marker='.')
    ax[1].plot([0, 1, 2, 3, 4], sampling_complexities_width[:, 0], color=COLORS['classical_model'], linestyle='--', linewidth=.75)
    ax[1].errorbar([0, 1, 2, 3, 4], sampling_complexities_width[:, 0], yerr=np.transpose(sampling_complexities_width[:, 1:], (1, 0)),
                   fmt='none', capsize=3, capthick=.85, elinewidth=.85, ecolor='k')

    # enforce same limits for all plots and set ticks, margins for nicer plotting
    ax[1].set_ylim(bottom=0, top=215000)
    ax[1].set_yticks([50000, 100000, 150000, 200000], ['$50$k', '$100$k', '$150$k', '$200$k'])
    ax[1].set_xticks([0, 1, 2, 3, 4], ['$16$', '$32$', '$64$', '$128$', '$256$'])
    ax[1].set_xmargin(0.15)

    # plot epsilon and delta labels to top and left side
    ax[1].set_title(r'$\varepsilon=' + f'{epsilon:.2f}' + r'$', fontsize=8)
    ax[1].set_ylabel(r'$\delta=' + f'{delta:.2f}' + r'$', fontsize=8)
    ax[1].set_xlabel(r'$\textbf{width}$', fontsize=8, labelpad=1.0)

    # set y-grid and send to background (breaks if done multiple times)
    ax[1].grid(axis='y')
    ax[1].set_axisbelow(True)

    # place number of parameters next to labels
    ax[1].text(0, 162000, r'$387$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[1].text(1, 120000, r'$1283$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[1].text(2, 85000, r'$4611$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[1].text(3, 85000, r'$17411$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[1].text(4, 85000, r'$67587$', horizontalalignment='center', verticalalignment='center', fontsize=6)

    # set up and place legend
    legend_elements = [plt.plot([], marker="", ls="", label=r'Classical (depth 2)')[0]]
    leg_0 = ax[1].legend(handles=legend_elements+[],
                         loc='upper center', handletextpad=0.01, columnspacing=0.5, bbox_to_anchor=(0.792, 1.05),
                         fontsize=8)
    leg_0.get_frame().set_edgecolor('k')
    leg_0.get_frame().set_linewidth(.75)
    # remove spacing (https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows)
    for vpack in leg_0._legend_handle_box.get_children()[:1]:  # noqa
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
    # leg_0 = ax[0].legend(loc='upper center', handletextpad=0.01, columnspacing=0.5, bbox_to_anchor=(0.836, 1.05), fontsize=8)
    # leg_0.get_frame().set_edgecolor('k')
    # leg_0.get_frame().set_linewidth(.75)

    ##############################
    ### plot for increasing width (quantum)

    # select models to compare
    models = ['Q[6-4]', 'Q[8-4]', 'Q[10-4]', 'Q[12-4]', 'Q[14-4]']
    models_ids = [range(100), range(100), range(100), range(100), range(100)]

    # iterate over models to compare
    sampling_complexities_qubits = []
    for model, model_ids in zip(models, models_ids):
        # extract sampling complexities, post-process for plotting of percentiles
        sampling_complexities = evaluate_sampling_complexity_with_percentiles(
            path=os.path.join('results', 'dqn', 'a3-main', 'deg3'),
            filename=model,
            experiment_ids=model_ids,
            epsilons=[epsilon],
            deltas=[delta],
            samples_per_epoch=2000,
            max_epochs=100
        )
        sampling_complexities[:, :, 1] = sampling_complexities[:, :, 0] - sampling_complexities[:, :,
                                                                          1]  # lower percentile
        sampling_complexities[:, :, 2] = sampling_complexities[:, :, 2] - sampling_complexities[:, :,
                                                                          0]  # upper percentile
        sampling_complexities_qubits.append(sampling_complexities[0][0])
    sampling_complexities_qubits = np.array(sampling_complexities_qubits)

    # plot data point and error bar indicating percentiles
    ax[0].scatter([0], sampling_complexities_qubits[0, 0], color=COLORS['quantum_model'], marker='.',
                  label=r'$\text{Quantum}$')
    ax[0].scatter([1], sampling_complexities_qubits[1, 0], color=COLORS['quantum_model'], marker='.',
                  label=r'$\text{Quantum}$')
    ax[0].scatter([2], sampling_complexities_qubits[2, 0], color=COLORS['quantum_model'], marker='o',
                  label=r'$\text{Quantum}$')
    ax[0].scatter([3], sampling_complexities_qubits[3, 0], color=COLORS['quantum_model'], marker='.',
                  label=r'$\text{Quantum}$')
    ax[0].scatter([4], sampling_complexities_qubits[4, 0], color=COLORS['quantum_model'], marker='H',
                  label=r'$\text{Quantum}$')
    # ax[1].scatter([0, 1, 2, 3, 4], sampling_complexities_qubits[:, 0], color=COLORS['quantum'], marker='o',
    #               label=r'$\text{Quantum}$')
    ax[0].plot([0, 1, 2, 3, 4], sampling_complexities_qubits[:, 0], color=COLORS['quantum_model'], linestyle='--',
               linewidth=.75)
    ax[0].errorbar([0, 1, 2, 3, 4], sampling_complexities_qubits[:, 0],
                   yerr=np.transpose(sampling_complexities_qubits[:, 1:], (1, 0)),
                   fmt='none', capsize=3, capthick=.85, elinewidth=.85, ecolor='k')

    # enforce same limits for all plots and set ticks, margins for nicer plotting
    ax[0].set_ylim(bottom=0, top=215000)
    ax[0].set_yticks([50000, 100000, 150000, 200000], ['$50$k', '$100$k', '$150$k', '$200$k'])
    ax[0].set_xticks([0, 1, 2, 3, 4], ['$6$', '$8$', '$10$', '$12$', '$14$'])
    ax[0].set_xmargin(0.15)

    # plot epsilon and delta labels to top and left side
    ax[0].set_title(r'$\varepsilon=' + f'{epsilon:.2f}' + r'$', fontsize=8)
    ax[0].set_ylabel(r'$\delta=' + f'{delta:.2f}' + r'$', fontsize=8)
    ax[0].set_xlabel(r'$\textbf{qubits}$', fontsize=8, labelpad=1.0)

    # set y-grid and send to background (breaks if done multiple times)
    ax[0].grid(axis='y')
    ax[0].set_axisbelow(True)

    # place number of parameters next to labels
    ax[0].text(0, 124000, r'$189$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[0].text(1, 165000, r'$251$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[0].text(2, 117000, r'$313$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[0].text(3, 117000, r'$375$', horizontalalignment='center', verticalalignment='center', fontsize=6)
    ax[0].text(4, 32000, r'$437$', horizontalalignment='center', verticalalignment='center', fontsize=6)

    # set up and place legend
    legend_elements = [plt.plot([], marker="", ls="", label=r'Quantum (4 layers)')[0]]
    leg_1 = ax[0].legend(handles=legend_elements+[],
                         loc='upper center', handletextpad=0.01, columnspacing=0.5, bbox_to_anchor=(0.784, 1.05),
                         fontsize=8)
    leg_1.get_frame().set_edgecolor('k')
    leg_1.get_frame().set_linewidth(.75)
    # remove spacing (https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows)
    for vpack in leg_1._legend_handle_box.get_children()[:1]:  # noqa
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
    # leg_1 = ax[1].legend(loc='upper center', handletextpad=0.01, columnspacing=0.5, bbox_to_anchor=(0.836, 1.05),
    #                      fontsize=8)
    # leg_1.get_frame().set_edgecolor('k')
    # leg_1.get_frame().set_linewidth(.75)

    # set shared axis labels
    fig.supylabel(r'$\textbf{sample complexity}$', fontsize=8, x=0.97, rotation=-270)

    # save figure
    path = os.path.join(os.path.dirname(__file__), 'plots', 'fig7.pdf')
    plt.savefig(path, format='pdf')
    print(f'Figure saved to {path}.')
