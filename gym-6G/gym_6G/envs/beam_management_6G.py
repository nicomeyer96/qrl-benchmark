# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import warnings
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from ..helpers import AntennaCollection, Trajectory


class BeamManagement6G(gym.Env):
    """ Object to define a BeamManagement6G environment.

           INPUT:
                size: layout of environment
                steps: number of steps each trajectory is split into
                antennas: number of randomly placed antennas, or list of explicit antenna positions and orientations
                enforce_distance: if randomly place antennas, enforce this distance between all pairs
                trajectory_degree: number of support points to sample trajectory on
                mode: in which mode to report the rewards, choose from
                    - 'train': returns raw reward in each timestep
                    - 'val': returns 0 if non-terminal, otherwise sum of all received rewards in this episode
                    - 'test': returns 0 in all non-terminal steps, otherwise fraction of achievable reward in this episode
                seed: random seed for antenna placement
                render_mode: how to render environment, choose from None, 'text', 'plot'

        Once set up, the object can be used as any gymnasium environment.
    """

    metadata = {"render_modes": ["plot", "text"], "render_fps": 1}

    def __init__(self,
                 size: (float, float) = (6.0, 6.0),
                 steps: int = 100,
                 antennas: int | list[((float, float), (float, float))] = 3,
                 enforce_distance: float = 0.0,
                 trajectory_degree: int = 3,
                 mode: str = 'train',
                 seed: int = None,
                 render_mode: str = None
                 ):
        super(BeamManagement6G, self).__init__()

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Mode {mode} unknown.')

        self.size = size
        self.steps = steps
        self.timestep_index = None  # will be initialized in `reset`
        self.timesteps = np.linspace(0.0, 1.0, num=steps+1, endpoint=True)

        # create random antenna placements
        if isinstance(antennas, int):
            antennas = self.create_random(size, num_antennas=antennas, enforce_distance=enforce_distance, seed=seed)
        self.configuration_antennas = antennas  # for re-storing configuration
        self.num_antennas = len(antennas)
        self.size_codebook = 9  # number of codebook elements, should be odd number
        # set up environment
        self.antennas = AntennaCollection(size, antennas, size_codebook=self.size_codebook)

        self.observation_space = spaces.Box(np.array(3 * [0.0], dtype=np.float32), np.array(3 * [1.0], dtype=np.float32))
        self.action_space = spaces.Discrete(len(antennas))

        self.trajectory_degree = trajectory_degree
        self.trajectory = None

        # logging and plotting
        self.energy_sum_selected, self.energy_sum_optimal = None, None
        self.path = None
        if render_mode is not None:
            assert render_mode in self.metadata['render_modes'], 'render mode unknown'
        self.render_mode = render_mode
        self.render_data = None
        self.mode = mode

    def step(self, action: int):

        # update timestep counter and check for termination
        terminated, truncated = False, False
        if self.timestep_index >= self.steps:
            terminated, truncated = True, True

        # determine current position
        position = self.trajectory.trajectory(self.timesteps[self.timestep_index])
        self.path.append(position)

        # determine optimal codebook element for selected antenna (in principle this could be also done by the agent,
        # but for simplification this environment does this automatically)
        selected_codebook, selected_energy = self.antennas.get_optimal_codebook_element_for_antenna(action, *position)

        # determine optimal antenna, energy, and codebook element, ONLY USED FOR LOGGING
        optimal_antenna, optimal_energy, optimal_codebook = self.antennas.get_optimal_antenna(*position)

        # log some info
        self.energy_sum_selected += selected_energy
        self.energy_sum_optimal += optimal_energy

        self.render_data = {'selected_antenna': action, 'selected_codebook': selected_codebook,
                            'selected_energy': selected_energy, 'optimal_energy': optimal_energy,
                            'running_selected_energy': self.energy_sum_selected, 'running_optimal_energy': self.energy_sum_optimal,
                            'steps': self.steps, 'step': self.timestep_index}

        self.timestep_index += 1

        # normalize antenna and codebook indices to [0, 1], energy is inherently bounded by [0,1]
        state = [action / (self.num_antennas - 1), selected_energy, selected_codebook / (self.size_codebook - 1)]

        # determine reward
        reward = selected_energy
        if 'val' == self.mode:  # This ensures that only is the final step the fraction of actually received energy is returned
            # (ensures correct interpretation within Tianshou's TestCollector)
            reward = self.energy_sum_selected / self.energy_sum_optimal if terminated else 0.0
        if 'test' == self.mode:  # This ensures that only is the final step the actually received energy is returned
            # (ensures correct interpretation within Tianshou's TestCollector)
            reward = self.energy_sum_selected if terminated else 0.0

        # return state, reward, termination and truncation flag, and some additional info
        return np.array(state, dtype=np.float32), reward, terminated, truncated, {'optimal': optimal_energy}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None
              ):
        super().reset(seed=seed)

        # sample random trajectory
        self.trajectory = Trajectory(self.size, self.trajectory_degree, seed=seed)

        # determine initial position
        position = self.trajectory.trajectory(0)
        self.path = [position]

        # determine optimal antenna, energy, and codebook element for the initial position, reset(..) starts optimal
        optimal_antenna, optimal_energy, optimal_codebook = self.antennas.get_optimal_antenna(*position)

        # initially we always pre-select the optimal action
        self.energy_sum_selected, self.energy_sum_optimal = optimal_energy, optimal_energy
        self.timestep_index = 1

        self.render_data = {'selected_antenna': optimal_antenna, 'selected_codebook': optimal_codebook,
                            'selected_energy': optimal_energy, 'optimal_energy': optimal_energy,
                            'running_selected_energy': self.energy_sum_selected, 'running_optimal_energy': self.energy_sum_optimal,
                            'steps': self.steps, 'step': 0}

        # normalize antenna and codebook indices to [0, 1], energy is inherently bounded by [0,1]
        state = [optimal_antenna / (self.num_antennas - 1), optimal_energy, optimal_codebook / (self.size_codebook - 1)]
        return np.array(state, dtype=np.float32), {'optimal': optimal_energy}

    def optimal_action(self):
        """
        Helper for determining optimal action for current environment state (e.g. for testing).
        """
        position = self.trajectory.trajectory(self.timesteps[self.timestep_index])
        optimal_antenna, optimal_energy, optimal_codebook = self.antennas.get_optimal_antenna(*position)
        return optimal_antenna

    def optimal_antenna(self):
        """
        Helper for determining optimal antenna, energy, and codebook element for current environment state.
        """
        position = self.trajectory.trajectory(self.timesteps[self.timestep_index])
        optimal_antenna, optimal_energy, optimal_codebook = self.antennas.get_optimal_antenna(*position)
        return optimal_antenna, optimal_energy, optimal_codebook

    @staticmethod
    def create_random(env_size: (float, float), num_antennas: int, enforce_distance: float, seed: int):
        """
        Create random antenna configuration
        """

        def run_create_random():
            for _ in range(1000):
                positions = try_create_random()
                if positions is not None:
                    return positions
            raise RuntimeError('Did not find antenna placement satisfying the constraints.'
                               'Increase either `size` or reduce `num_antennas` or `enforce_distance`')

        def try_create_random():
            positions = [(a, b) for a, b in zip(np.random.uniform(low=0.0, high=env_size[0], size=num_antennas),
                                                np.random.uniform(low=0.0, high=env_size[1], size=num_antennas))]
            for index_0 in range(num_antennas - 1):
                for index_1 in range(index_0 + 1, num_antennas):
                    dist = np.linalg.norm(np.array(positions[index_0]) - np.array(positions[index_1]))
                    if dist < enforce_distance:
                        return None
            return positions

        np.random.seed(seed)

        _positions = run_create_random()
        _rotations = [(a, b) for a, b in zip(np.random.uniform(low=-1.0, high=1.0, size=num_antennas),
                                             np.random.uniform(low=-1.0, high=1.0, size=num_antennas))]
        _rotations = [(a / np.sqrt(a ** 2 + b ** 2), b / np.sqrt(a ** 2 + b ** 2)) for (a, b) in _rotations]
        _antennas = [(_p, _r) for _p, _r in zip(_positions, _rotations)]
        return _antennas

    def get_antenna_configuration(self):
        return self.configuration_antennas

    def get_configuration(self):
        return {'size': self.size, 'steps': self.steps, 'antennas': self.configuration_antennas, 'degree': self.trajectory_degree}

    def render(self):
        if self.render_mode is None:
            warnings.warn('No `render_mode` has been specified.')
        elif 'text' == self.render_mode:
            print('[Step {}/{}]'.format(self.render_data['step'], self.render_data['steps'])
                  + ' Selected Antenna {}, Beam {}'.format(self.render_data['selected_antenna'], self.render_data['selected_codebook'])
                  + ' -> Energy {:.5f}/{:.5f}'.format(self.render_data['selected_energy'], self.render_data['optimal_energy'])
                  + f' (Overall Energy {self.energy_sum_selected:.5f}/{self.energy_sum_optimal:.5f})')
        elif 'plot' == self.render_mode:
            return self.antennas.render(self.render_data['selected_antenna'], self.render_data['selected_codebook'],
                                        np.array(self.path), self.render_data)

    def close(self):
        pass

    @staticmethod
    def seed(seed):
        np.random.seed(seed)
