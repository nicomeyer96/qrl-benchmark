# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="6G-v0",
    entry_point="gym_6G.envs:BeamManagement6G",
    kwargs={
        'size': (6.0, 6.0),
        'steps': 100,
        'antennas':  None,
        'enforce_distance': 0.0,
        'trajectory_degree': 3,
        'mode': 'train',
        'seed': None,
        'render_mode': None
    }
)
