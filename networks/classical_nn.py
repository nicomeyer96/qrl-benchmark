# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import torch
from torch import nn


class ClassicalNN(nn.Module):
    def __init__(self, input_shape, output_shape, width, depth, softmax_output=False, forward_state=True, activation='relu'):
        super().__init__()
        self.input_dim = np.prod(input_shape)
        self.output_dim = np.prod(output_shape)
        self.forward_state = forward_state
        layers = [nn.Linear(self.input_dim, width, bias=True)]
        for _ in range(depth - 1):
            layers.append(nn.ReLU(inplace=True) if 'relu' == activation else nn.Tanh())
            layers.append(nn.Linear(width, width, bias=True))
        layers.append(nn.ReLU(inplace=True) if 'relu' == activation else nn.Tanh())
        layers.append(nn.Linear(width, self.output_dim, bias=True))
        if softmax_output:
            layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs, state=None, info=None):
        info = {} if info is None else info
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        if not self.forward_state:
            return logits
        return logits, state
