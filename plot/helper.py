# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Read and extract training data, configure plotting style, test trained model.
"""

import os
import pickle
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_6G  # required for loading custom environment
from gymnasium.wrappers import FrameStack
import tianshou as ts
from tianshou.env import DummyVectorEnv

import sys
sys.path.append(os.path.abspath(os.getcwd()))  # allow for imports above this directory
from trainers.logger import CustomLogger  # noqa
from networks import ClassicalNN, HybridNN  # noqa


def read(path, filename):
    """ Read validation returns from stored data. """
    filepath = os.path.join(path, f'{filename}.pkl')
    if not os.path.isfile(filepath):
        raise RuntimeError(f'The file {filepath} does not exist. '
                           f'Consider downloading the pre-computed data as indicated in the README file.')
    data = pickle.load(open(filepath, 'rb'))
    val_returns = data.get('val_returns', None)  # shape `epochs_train`+1 x `environments_val`
    if val_returns is None:
        raise RuntimeError(f'Validation data is missing in {filepath}.')
    return val_returns


def extract(path, filename, experiment_ids):
    """ Read and combine data for multiple runs of one setup. """
    val_returns = []
    for experiment_id in experiment_ids:
        val_returns.append(read(path, f'{filename}-{experiment_id}'))
    val_returns = np.array(val_returns)  # shape NUM_RUNS x `epochs_train`+1 x `environments_val`
    # ignore initial validation run
    val_returns = val_returns[:, 1:, :]  # shape NUM_RUNS x `epochs_train` x `environments_val`
    # average return for each experiment-epoch combination
    val_returns = np.average(val_returns, axis=2)  # shape NUM_RUNS x `epochs_train`+1
    return val_returns


def test_model(model_path, env_configuration=None, runs=1000, mode='absolute'):
    """ Test trained model and report energy distribution (either absolute or relative) """

    assert mode in ['absolute', 'relative']

    # load model data
    assert os.path.isfile(model_path), f'Model {model_path} not found.'
    model_data = pickle.load(open(model_path, 'rb'))
    print(f'Testing model `{model_path}` for {runs} runs:')

    # load environment configuration used for training and compare with optionally provided environment
    env_configuration_train = model_data.get('env_configuration', None)
    if env_configuration_train is None and env_configuration is None:
        raise RuntimeError('Model data contains no environment configuration and no environment was provided.')
    elif env_configuration_train is None:
        warnings.warn('Model data contains no environment configuration, cannot compare for consistency.')
    elif env_configuration_train is not None and env_configuration is None:
        env_configuration = env_configuration_train
    else:
        if env_configuration['size'] != env_configuration_train['size']:
            warnings.warn('Using different size than during training: {} vs. {}.'
                          .format(env_configuration['size'], env_configuration_train['size']))
        if env_configuration['steps'] != env_configuration_train['steps']:
            warnings.warn('Using different number of trajectory steps than during training: {} vs. {}.'
                          .format(env_configuration['steps'], env_configuration_train['steps']))
        if env_configuration['degree'] != env_configuration_train['degree']:
            warnings.warn('Using different trajectory degree than during training: {} vs. {}.'
                          .format(env_configuration['degree'], env_configuration_train['degree']))
        if env_configuration['antennas'] != env_configuration_train['antennas']:
            warnings.warn('Using different antenna placement than during training: {} vs. {}.'
                          .format(env_configuration['antennas'], env_configuration_train['antennas']))
        if len(env_configuration['antennas']) != len(env_configuration_train['antennas']):
            raise RuntimeError('Number of antennas (i.e. actions) is inconsistent between training and provided test'
                               'configuration: {} vs. {}.'.format(len(env_configuration['antennas']),
                                                                  len(env_configuration_train['antennas'])))

    # set up environment
    model_args = model_data['args']
    env = FrameStack(gym.make('6G-v0', size=env_configuration['size'], steps=env_configuration['steps'],
                              antennas=env_configuration['antennas'], trajectory_degree=env_configuration['degree']),
                     model_args.env_stack)
    train_env = DummyVectorEnv([
        lambda: FrameStack(gym.make('6G-v0', size=env_configuration['size'], steps=env_configuration['steps'],
                                    antennas=env_configuration['antennas'], trajectory_degree=env_configuration['degree']),
                           model_args.env_stack)
        for _ in range(1)])  # dummy
    test_env = DummyVectorEnv([
        lambda: FrameStack(gym.make('6G-v0', size=env_configuration['size'], steps=env_configuration['steps'],
                                    antennas=env_configuration['antennas'], trajectory_degree=env_configuration['degree'],
                                    mode='test' if 'absolute' == mode else 'val'),
                           model_args.env_stack)
        for _ in range(runs // 2)])  # divide by 2 as tianshou distributes in two test steps

    # set up model
    if vars(model_args).get('method', None) is not None and 'dqn' != model_args.method:  # some old trained models do not carry this flag
        raise NotImplementedError
    if 'classical' == model_args.model:
        # construct classical model
        net = ClassicalNN(env.observation_space.shape, env.action_space.n,
                          width=model_args.width_nn, depth=model_args.depth_nn, activation=model_args.activation_nn)
    else:
        # construct hybrid model
        net = HybridNN(env.observation_space.shape, env.action_space.n,
                       qubits=model_args.qubits_qnn, layers=model_args.layers_qnn, ansatz=model_args.ansatz_qnn,
                       gates=model_args.gates_qnn)

    # load trained weights
    net.load_state_dict(model_data['model'])

    # dummy optimizer
    optim = torch.optim.Adam(net.parameters(), lr=0.0)

    # set up policy
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,  # noqa  (suppress faulty WrongType warning)
        discount_factor=1.0,  # dummy
        target_update_freq=1  # dummy
    )

    # set up data collectors
    train_collector = ts.data.Collector(policy, train_env, ts.data.VectorReplayBuffer(1, 1),
                                        exploration_noise=True)  # dummy
    test_collector = ts.data.Collector(policy, test_env, exploration_noise=False)

    # set up logger
    logger = CustomLogger(train_interval=1, test_interval=1)

    # set up tianshou trainer
    trainer = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1,  # dummy
        step_per_epoch=1,  # dummy
        step_per_collect=1,  # dummy
        update_per_step=1.0,  # dummy
        batch_size=1,  # dummy
        episode_per_test=runs // 2,  # tianshou distributes testing over two instances
        test_fn=lambda epoch, env_step: policy.set_eps(0.0),
        test_in_train=False,
        logger=logger
    )

    # run training
    trainer.run()

    # extract test results
    test_results = np.array(logger.get_test_returns()).flatten()
    return test_results


# # colors for plots
# COLORS = {
#     'C[16-2]': 'red',
#     'C[32-2]': 'red',
#     'C[64-2]': 'red',
#     'Q[10-4]': 'royalblue',
#     'Q[14-4]': 'royalblue',
#     'optimal': 'orange',
#     'random': 'saddlebrown',
#     'depth2': 'royalblue',
#     'width32': 'red',
#     'width64': 'green',
#     'layers4': 'royalblue',
#     'qubits8': 'red',
#     'qubits10': 'green',
#     'classical': 'royalblue',
#     'quantum': 'green',
#     'default': 'black'
# }

# colors for plots
COLORS = {
    'C[16-2]': 'red',
    'C[32-2]': 'purple',
    'C[64-2]': 'royalblue',
    'Q[10-4]': 'darkcyan',
    'Q[14-4]': 'green',
    'optimal': 'orange',
    'random': 'saddlebrown',
    'depth2': 'royalblue',
    'width32': 'red',
    'width64': 'green',
    'layers4': 'royalblue',
    'qubits8': 'red',
    'qubits10': 'green',
    'classical': 'royalblue',
    'quantum': 'green',
    'default': 'black',
    'quantum_model': 'royalblue',
    'classical_model': 'red'
}


# compute figsize for paper
def setup_figure_latex_layout(height_inch, single_column=True):
    tex_fonts = {
        # Use LaTeX to write all text, load fonts
        "text.usetex": True,
        'text.latex.preamble': r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": "STIX",
        "mathtext.fontset": 'stix',
        # Set font sizes
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        'xtick.major.size': 3,
        'xtick.major.width': .5,
        'ytick.major.size': 3,
        'ytick.major.width': .5,
    }
    plt.style.use('default')
    plt.rcParams.update(tex_fonts)

    width_inch = 3.25 if single_column else 6.75
    assert height_inch <= 9.0

    return width_inch, height_inch
