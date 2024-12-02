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
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from tianshou.env import DummyVectorEnv

import gym_6G  # required for loading custom environment
from .logger import CustomLogger


class Trainer(ABC):

    def __init__(self,
                 env_load: None | str, env_antennas: int, env_degree: int, env_steps: int, env_stack: int, env_show: bool,
                 environments_train: int, environments_val: int):

        if env_load is None:  # used for later sampling random antennas
            antennas = env_antennas
        else:  # load explicit antenna configuration
            # with open(os.path.join(os.path.dirname(__file__), 'env_configuration', f'{env_load}.pkl'), 'rb') as f:
            with open(os.path.join('env_configuration', f'{env_load}.pkl'), 'rb') as f:
                antennas = pickle.load(f)
            print(f'Loaded environment {env_load}: {antennas}')

        # dummy environment for fixing antenna configuration
        self.env = FrameStack(gym.make('6G-v0', steps=env_steps, antennas=antennas, enforce_distance=2.0,
                                       trajectory_degree=env_degree), env_stack)
        # vectorized environment for training
        self.train_env = DummyVectorEnv([
            lambda: FrameStack(gym.make('6G-v0', steps=env_steps, antennas=self.env.get_antenna_configuration(),
                                        trajectory_degree=env_degree), env_stack)
            for _ in range(environments_train)])
        # vectorized environment for validation
        self.val_env = DummyVectorEnv([
            lambda: FrameStack(gym.make('6G-v0', steps=env_steps, antennas=self.env.get_antenna_configuration(),
                                        trajectory_degree=env_degree, mode='val'), env_stack)
            for _ in range(environments_val)])

        if env_show:
            self.env.antennas.plot_ground_truth()

        # set up logger, reports all returns in order to be able to compute the sampling efficiency afterward
        self.logger = CustomLogger(train_interval=environments_train * env_steps, test_interval=1)

    @abstractmethod
    def train(self) -> dict:
        pass
