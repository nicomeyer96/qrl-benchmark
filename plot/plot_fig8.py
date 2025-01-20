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
import pickle
import gymnasium as gym
import warnings

import gym_6G  # required for loading custom environment
from helper import setup_figure_latex_layout, test_model, COLORS


def find_best_models(model_path, model_name, model_ids, number=10):
    """ Find ids of best performing models (according to final validation results) """
    final_val_returns = []
    for model_id in model_ids:
        # load model and extract validation returns, average, and extract final one
        data = pickle.load(open(os.path.join(model_path, f'{model_name}-{model_id}.pkl'), 'rb'))
        final_val_returns.append(np.average(data['val_returns'], axis=1)[-1])
    return np.argsort(final_val_returns)[-number:]


def test_best_models(model_path, model_name, model_ids, number_best_performing=10, number_test_runs=100, mode='absolute',
                     env_configuration=None):
    """ Determine and test best performing models """
    print(f'\nDetermine and test {number_best_performing} best performing `{model}` models for {number_test_runs} runs each:')
    best_model_ids = find_best_models(model_path, model_name, model_ids, number=number_best_performing)
    test_results = [test_model(os.path.join(path, f'{model}-{best_model_id}.pkl'), runs=number_test_runs, mode=mode,
                               env_configuration=env_configuration)
                    for best_model_id in best_model_ids]
    return np.array(test_results)


def test_untrained_models(env_configuration, runs=1000, action_selection='optimal', mode='absolute'):
    """ Test optimal and random actions. """
    assert mode in ['absolute', 'relative']
    assert action_selection in ['optimal', 'random']
    # set up environment
    env = gym.make('6G-v0', size=env_configuration['size'], steps=env_configuration['steps'],
                   antennas=env_configuration['antennas'], trajectory_degree=env_configuration['degree'],
                   mode='test' if 'absolute' == mode else 'val')

    # perform testing
    test_results = []
    print(f'\nPerform testing with {action_selection} action for {runs} runs.')
    for run in range(runs):
        print(f'{run+1}/{runs} done', end='\r')
        _, _ = env.reset()
        while True:
            # select either optimal or random action
            action = env.optimal_action() if 'optimal' == action_selection else env.action_space.sample()  # noqa
            _, reward, terminated, _, _ = env.step(action)
            if terminated:  # if terminated extract result and break to reset environment
                test_results.append(reward)
                break
    print()
    return np.array(test_results)


if __name__ == '__main__':

    # standard environment configuration to test on (same that was trained on)
    configuration = {
        'size': (6.0, 6.0),
        'steps': 200,
        'degree': 3,
        'antennas': [((1.5, 1.0), (1.0, 0.0)), ((1.5, 5.0), (-1.0, 0.0)), ((4.5, 3.0), (0.0, 1.0))]
    }
    # path to train data
    path = os.path.join('results', 'dqn', 'a3-main', 'deg3')

    # determine figsize for paper
    figsize = setup_figure_latex_layout(4.0, single_column=True)

    # set up figure, shift y-axis to right side
    plt.rcParams['ytick.right'], plt.rcParams['ytick.labelright'] = True, True
    plt.rcParams['ytick.left'], plt.rcParams['ytick.labelleft'] = False, False
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=False, figsize=figsize,
                           gridspec_kw={'right': 0.855, 'left': 0.01, 'bottom': 0.1, 'top': 0.95, 'hspace': 0.65})

    # select models to compare, set up model, title, admissible ids, and color
    models = ['Q[14-4]', 'C[64-2]']
    models_title = ['Quantum Model (14-4)', 'Classical Model (64-2)']
    models_ids = [range(100), range(100)]
    models_color = [COLORS[m] for m in models]

    # iterate over and plot test results for best performing trained models
    # Note: The results in the paper were produced with 1000 test_runs, this however takes quite some time to compute
    best_models, test_runs = 10, 100
    if test_runs < 1000:
        warnings.warn('For reliable results the `test_runs` should be set to 1000.')
    for axis, model, model_id, model_color, model_title in zip(ax[:len(models)], models, models_ids, models_color, models_title):

        # determine and test best performing models
        test_result = test_best_models(path, model, model_id, number_best_performing=best_models,
                                       number_test_runs=test_runs, env_configuration=configuration)

        # determine average and std of test results for title
        test_result_per_model = np.average(test_result, axis=1)
        avg_test_result, std_test_result = np.average(test_result_per_model), np.std(test_result_per_model)
        axis.set_title(r'$\textbf{' + f'{model_title}' + r':}~$' + f'   {avg_test_result:.2f} ' + r'$\pm$' + f' {std_test_result:.2f}', fontsize=8)

        # flatten data
        test_result = test_result.flatten()

        # compute and plot (normalized) histogram
        counts, bins = np.histogram(test_result, bins=30, range=(0, 30))
        axis.stairs(counts, bins, fill=True, color=model_color, edgecolor='black')

        # set grid and limits for uniform plotting
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        axis.set_xlim(left=0.1, right=29.9)
        axis.set_ylim(bottom=0, top=best_models * test_runs / 10)

    # iterate over and plot test results for optimal and random actions
    modes = ['optimal', 'random']
    modes_color = [COLORS[m] for m in modes]
    modes_title = ['Optimal Actions', 'Random Actions']
    for axis, mode, mode_color, mode_title in zip(ax[len(models):], modes, modes_color, modes_title):

        # test with optimal / random action
        test_result = test_untrained_models(configuration, runs=best_models * test_runs, action_selection=mode, mode='absolute')

        # determine average of test results for title
        avg_test_result = np.average(test_result)
        axis.set_title(r'$\textbf{' + f'{mode_title}' + r':}~$' + f'   {avg_test_result:.2f}', fontsize=8)

        # compute and plot (normalized) histogram
        counts, bins = np.histogram(test_result, bins=30, range=(0, 30))
        axis.stairs(counts, bins, fill=True, color=mode_color, edgecolor='black')

        # set grid and limits for uniform plotting
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        axis.set_xlim(left=0.1, right=29.9)
        if 'random' == mode:
            axis.set_ylim(bottom=0, top=best_models * test_runs / 5)
        else:
            axis.set_ylim(bottom=0, top=best_models * test_runs / 10)

    # set axis labels
    fig.supxlabel(r'$\textbf{absolute energy} ~~(\pm 0.5)$', fontsize=8, y=-0.0)
    fig.supylabel(r'$\textbf{count}~~$' + f'(out of {best_models * test_runs})', fontsize=8, x=0.965, rotation=-270)

    # save figure
    path = os.path.join(os.path.dirname(__file__), 'plots', f'fig8.pdf')
    plt.savefig(path, format='pdf')
    print(f'Figure saved to {path}.')
