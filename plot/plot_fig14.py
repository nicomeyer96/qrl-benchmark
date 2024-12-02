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
import matplotlib as mpl
import matplotlib.pyplot as plt

from cluster_bootstrapping import evaluate_sampling_complexity_with_percentiles
from helper import setup_figure_latex_layout, COLORS


if __name__ == '__main__':

    epsilons = [0.2, 0.15, 0.1]
    deltas = [0.5, 0.75]

    # determine figsize for paper
    figsize = setup_figure_latex_layout(3.0, single_column=False)

    # set up figure, shift y-axis to right side
    plt.rcParams['ytick.right'], plt.rcParams['ytick.labelright'] = True, True
    plt.rcParams['ytick.left'], plt.rcParams['ytick.labelleft'] = False, False
    fig, ax = plt.subplots(len(deltas), len(epsilons), sharex=True, sharey=True, figsize=figsize,
                           gridspec_kw={'right': 0.92, 'left': 0.028, 'bottom': 0.11, 'top': 0.85, 'wspace': 0.06, 'hspace': 0.1})

    # select models to compare, set up x-axis text, label, number of models, color, and marker
    models = ['Q[10-4]', 'C[16-2]', 'C[32-2]', 'C[64-2]']
    models_text = ['604', '740', '2500', '9092']
    models_label = ['Quantum (10-4)', 'Classical (16-2)', 'Classical (32-2)', 'Classical (64-2)']
    models_ids = [range(100), range(100), range(100), range(100)]
    models_color = [COLORS[m] for m in ['quantum_model', 'classical_model', 'classical_model', 'classical_model']]
    models_marker = ['o', '^', '<', '>']

    # iterate over models to compare
    for model, model_ids, model_color, model_marker, model_text in zip(models, models_ids, models_color, models_marker, models_text):
        # extract sampling complexities, post-process for plotting of percentiles
        sampling_complexities = evaluate_sampling_complexity_with_percentiles(
            path=os.path.join('results', 'ppo', 'a3-main', 'deg3'),
            filename=model,
            experiment_ids=model_ids,
            epsilons=epsilons,
            deltas=deltas,
            samples_per_epoch=2000,
            max_epochs=500
        )
        sampling_complexities[:, :, 1] = sampling_complexities[:, :, 0] - sampling_complexities[:, :, 1]  # lower percentile
        sampling_complexities[:, :, 2] = sampling_complexities[:, :, 2] - sampling_complexities[:, :, 0]  # upper percentile

        # iterate over all epsilon-delta combinations
        for _axis, _sampling_complexity, delta in zip(ax, np.transpose(sampling_complexities, (1, 0, 2)), deltas):
            for axis, sampling_complexity, epsilon in zip(_axis, _sampling_complexity, epsilons):

                # draw points in opaque color if upper percentile is maximum sample number
                alpha = 1 - 0.5 * (1000000 == sampling_complexity[0] + sampling_complexity[2])
                ecolor = 'gray' if 1000000 == sampling_complexity[0] + sampling_complexity[2] else 'black'

                # plot data point and error bar indicating percentiles
                axis.scatter(model_text, sampling_complexity[0], color=model_color, marker=model_marker, alpha=alpha)
                axis.errorbar(model_text, sampling_complexity[0], yerr=np.expand_dims(sampling_complexity[1:], axis=1),
                              fmt='none', capsize=3, capthick=.85, elinewidth=.85, ecolor=ecolor)

                # enforce same limits for all plots and set ticks, margins for nicer plotting
                axis.set_ylim(bottom=0, top=1075000)
                axis.set_yticks([250000, 500000, 750000, 1000000], ['$250$k', '$500$k', '$750$k', '$1000$k'])
                axis.set_xmargin(0.175)

                # plot epsilon and delta labels to top and left side
                if model == models[0] and delta == deltas[0]:
                    axis.set_title(r'$\varepsilon=' + f'{epsilon:.2f}' + r'$', fontsize=8)
                if model == models[0] and epsilon == epsilons[0]:
                    axis.set_ylabel(r'$\delta=' + f'{delta:.2f}' + r'$')

                # set y-grid and send to background (breaks if done multiple times)
                if model == models[0]:
                    axis.grid(axis='y')
                    axis.set_axisbelow(True)

    # set axis labels
    fig.supxlabel(r'$\textbf{model trainable parameters}$', fontsize=8, y=-0.0)
    fig.supylabel(r'$\textbf{sample complexity}$', fontsize=8, x=0.982, rotation=-270)

    # plot legend
    legend_title = [plt.plot([], marker="", ls="", label=r'\textbf{Model (width-depth)}:')[0]]
    legend_elements = [mpl.lines.Line2D([0], [0], color=model_color, marker=model_marker, lw=0, label=model_label)
                       for model_color, model_marker, model_label in zip(models_color, models_marker, models_label)]
    leg = fig.legend(handles=legend_title+legend_elements, loc='upper center', ncol=6,
                     handletextpad=0.01, columnspacing=0.5, bbox_to_anchor=(0.5, 1.014), fontsize=7.5)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.75)
    # remove spacing (https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows)
    for vpack in leg._legend_handle_box.get_children()[:1]:  # noqa
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    # save figure
    path = os.path.join(os.path.dirname(__file__), 'plots', f'fig14.pdf')
    plt.savefig(path, format='pdf')
    print(f'Figure saved to {path}.')
