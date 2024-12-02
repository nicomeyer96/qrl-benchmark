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
import pickle
import argparse

from trainers import DQNTrainer, PPOTrainer


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env_load', '-env', type=str, default=None,
                        help='Environment configuration to load (otherwise randomly initialized).')
    parser.add_argument('--env_antennas', '-antennas', type=int, default=3,
                        help='Number of antennas to randomly sample (only if `env_load` is None).')
    parser.add_argument('--env_degree', '-degree', type=int, default=3,
                        help='Number of support points for sampling trajectory.')
    parser.add_argument('--env_steps', '-steps', type=int, default=200,
                        help='Number of steps in one trajectory.')
    parser.add_argument('--env_stack', '-stack', type=int, default=1,
                        help='Stack multiple (past) observations.')

    # Algorithm
    parser.add_argument('--method', '-method', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='Algorithm to use (`dqn`: double deep Q-networks, `ppo`: proximal policy optimization).')
    parser.add_argument('--batch_update', '-batch', type=int, default=64,
                        help='Number of data samples for each update.')
    parser.add_argument('--discount_factor', '-gamma', type=float, default=0.95,
                        help='Reward discount factor.')
    parser.add_argument('--epochs_train', '-epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--environments_train', '-envs', type=int, default=10,
                        help='Number of parallel environments to use in each epoch.')
    parser.add_argument('--environments_val', '-val', type=int, default=100,
                        help='Number of environments to perform validation on (after each epoch).')
    parser.add_argument('--epsilon_greedy', '-epsilon', type=float, default=0.1,
                        help='[dqn] Hyperparameter for epsilon-greedy action selection.')
    parser.add_argument('--epsilon_clip', '-clip', type=float, default=0.1,
                        help='[ppo] Hyperparameter for clipping gradients.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005,
                        help='Learning rate for classical parameters.')
    parser.add_argument('--learning_rate_quantum', '-lrq', type=float, default=0.001,
                        help='[quantum] Learning rate for quantum parameters.')

    # Model
    parser.add_argument('--model', type=str, default='classical', choices=['classical', 'quantum'],
                        help='Use classical or quantum (hybrid) model.')
    parser.add_argument('--width_nn', '-width', type=int, default=64,
                        help='[classical] Width of hidden layer(s).')
    parser.add_argument('--depth_nn', '-depth', type=int, default=2,
                        help='[classical] Number of hidden layers.')
    parser.add_argument('--activation_nn', '-act', type=str, default='relu', choices=['relu', 'tanh'],
                        help='[classical] Activation function to use between layers.')
    parser.add_argument('--qubits_qnn', '-qubits', type=int, default=10,
                        help='[quantum] Number of qubits.')
    parser.add_argument('--layers_qnn', '-layers', type=int, default=4,
                        help='[quantum] Number of variational layers.')
    parser.add_argument('--ansatz_qnn', '-ansatz', type=str, default='iqp', choices=['iqp', 'cx', 'cz'],
                        help='[quantum] Structure of variational ansatz.')
    parser.add_argument('--gates_qnn', '-gates', type=str, default='rot', choices=['rot', 'u3', 'xyz'],
                        help='[quantum] Type of parametrized gates to use for variational ansatz.')

    # General
    parser.add_argument('--experiment', '-exp', type=str, default='0',
                        help='Suffix for storing results.')
    parser.add_argument('--overwrite', '-over', action='store_true',
                        help='Overwrite result if file already exists.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse()
    # construct path and filename for storing results
    path = os.path.join(
        'results',
        _args.method,
        f'a{_args.env_antennas}' if _args.env_load is None else _args.env_load,
        f'deg{_args.env_degree}'
    )
    file_name = f'C[{_args.width_nn}-{_args.depth_nn}]-{_args.experiment}.pkl' \
        if 'classical' == _args.model \
        else f'Q[{_args.qubits_qnn}-{_args.layers_qnn}]-{_args.experiment}.pkl'
    file_path = os.path.join(path, file_name)

    # generate result path (if not already existing)
    if not os.path.exists(path):
        os.makedirs(path)

    # see if file already exists (or overwrite flag is set)
    if os.path.isfile(file_path) and not _args.overwrite:
        raise ValueError(f'File {file_path} already exists. Either change `--experiment`, or use `--overwrite`.')

    print(f'Result will be saved to {file_path}.')

    # set up selected algorithm
    if 'dqn' == _args.method:
        trainer = DQNTrainer(_args.env_load, _args.env_antennas, _args.env_degree, _args.env_steps, _args.env_stack, False,
                             _args.environments_train, _args.environments_val,
                             _args.model, _args.learning_rate, _args.learning_rate_quantum,
                             _args.width_nn, _args.depth_nn, _args.activation_nn,
                             _args.qubits_qnn, _args.layers_qnn, _args.ansatz_qnn, _args.gates_qnn,
                             _args.discount_factor, _args.epochs_train, _args.batch_update, _args.epsilon_greedy)
    else:
        trainer = PPOTrainer(_args.env_load, _args.env_antennas, _args.env_degree, _args.env_steps, _args.env_stack, False,
                             _args.environments_train, _args.environments_val,
                             _args.model, _args.learning_rate, _args.learning_rate_quantum,
                             _args.width_nn, _args.depth_nn, _args.activation_nn,
                             _args.qubits_qnn, _args.layers_qnn, _args.ansatz_qnn, _args.gates_qnn,
                             _args.discount_factor, _args.epochs_train, _args.batch_update, _args.epsilon_clip)

    # run training
    result = trainer.train()

    # add argument settings to result
    result['args'] = _args

    # store result
    with open(file_path, 'wb') as ff:
        pickle.dump(result, ff)

    print(f'Result has been saved to {file_path}.')
