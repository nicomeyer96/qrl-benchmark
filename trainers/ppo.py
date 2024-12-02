# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import tianshou as ts
from tianshou.utils.net.common import ActorCritic
import numpy as np
from typing import Optional
import torch

from .trainer import Trainer
from networks import ClassicalNN, HybridNN


class PPOTrainer(Trainer):

    def __init__(self,
                 env_load: None | str, env_antennas: int, env_degree: int, env_steps: int, env_stack: int,
                 env_show: bool,
                 environments_train: int, environments_val: int,
                 model: str, lr_classical: float, lr_quantum: float,
                 width_nn: Optional[int], depth_nn: Optional[int], activation_nn: Optional[str],
                 qubits_qnn: Optional[int], layers_qnn: Optional[int], ansatz_qnn: Optional[str],
                 gates_qnn: Optional[str],
                 discount_factor: float, epochs_train: int, batch_update: int, epsilon_clip: float):
        super().__init__(env_load, env_antennas, env_degree, env_steps, env_stack, env_show,
                         environments_train, environments_val)

        if 'classical' == model:
            # construct classical model
            actor = ClassicalNN(self.env.observation_space.shape, self.env.action_space.n,
                                width=width_nn, depth=depth_nn, activation=activation_nn,
                                softmax_output=True)
            critic = ClassicalNN(self.env.observation_space.shape, 1,
                                 width=width_nn, depth=depth_nn, activation=activation_nn,
                                 forward_state=False)
            self.net = ActorCritic(actor, critic)
            # set learning rates
            self.optim = torch.optim.Adam(self.net.parameters(), lr=lr_classical)
        else:
            # construct hybrid model
            actor = HybridNN(self.env.observation_space.shape, self.env.action_space.n,
                             qubits=qubits_qnn, layers=layers_qnn, ansatz=ansatz_qnn,
                             gates=gates_qnn,
                             softmax_output=True)
            critic = HybridNN(self.env.observation_space.shape, 1,
                              qubits=qubits_qnn, layers=layers_qnn, ansatz=ansatz_qnn,
                              gates=gates_qnn,
                              forward_state=False)
            self.net = ActorCritic(actor, critic)
            # set learning rates individually for classical and quantum layers
            self.optim = torch.optim.Adam([
                {'params': self.net.actor.model[0].weight}, {'params': self.net.actor.model[0].bias},
                {'params': self.net.actor.model[1].weights, 'lr': lr_quantum},
                {'params': self.net.actor.model[1].scaling, 'lr': lr_quantum},
                {'params': self.net.actor.model[2].weight}, {'params': self.net.actor.model[2].bias},
                {'params': self.net.critic.model[0].weight}, {'params': self.net.critic.model[0].bias},
                {'params': self.net.critic.model[1].weights, 'lr': lr_quantum},
                {'params': self.net.critic.model[1].scaling, 'lr': lr_quantum},
                {'params': self.net.critic.model[2].weight}, {'params': self.net.critic.model[2].bias},
            ], lr=lr_classical)

        # set up policy
        policy = ts.policy.PPOPolicy(
            actor=actor,
            critic=critic,
            optim=self.optim,
            dist_fn=torch.distributions.Categorical,
            action_space=env.action_space,  # noqa  (suppress faulty WrongType warning)
            eps_clip=epsilon_clip,
            action_scaling=False,
            discount_factor=discount_factor
        )

        # set up data collectors
        train_collector = ts.data.Collector(policy, self.train_env,
                                            ts.data.VectorReplayBuffer(environments_train * env_steps, environments_train),
                                            exploration_noise=True)
        test_collector = ts.data.Collector(policy, self.val_env, exploration_noise=False)

        # set up tianshou trainer
        self.trainer = ts.trainer.OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epochs_train,
            step_per_epoch=environments_train * env_steps,
            episode_per_collect=environments_train,
            repeat_per_collect=10,
            batch_size=batch_update,
            episode_per_test=environments_val,
            test_in_train=False,
            logger=self.logger
        )

    def train(self):
        # perform training
        timings = self.trainer.run()

        # extract result
        result = {
            'env_configuration': self.env.get_configuration(),
            'train_returns': np.array(self.logger.get_train_returns()),
            'val_returns': np.array(self.logger.get_test_returns()),
            'train_time': timings.timing.train_time,
            'train_time_collect': timings.timing.train_time_collect,
            'train_time_update': timings.timing.train_time_update,
            'test_time': timings.timing.test_time,
            'optim': self.optim.state_dict(),
            'model': self.net.state_dict()
        }

        return result
