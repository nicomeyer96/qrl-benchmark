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
import matplotlib.pyplot as plt
import warnings


class Antenna:
    """ Object to define a single antenna.

           INPUT:
               position: position of antenna
               orientation: orientation of antenna, has to be normalized
               num_senders: number of sender elements of antenna
               size_codebook: number of elements in codebook of antenna

           The antenna is placed within the environment and its codebook is pre-computed based on the provided
           arguments. The intensity is only computed once called.
    """

    def __init__(self,
                 position: (float, float),
                 orientation: (float, float),
                 num_senders: int = 17,
                 size_codebook: int = 9
                 ):
        assert np.linalg.norm(np.array(orientation)) - 1 < 1e-10, '`orientation` must be normalized'
        assert 0 != num_senders % 2, '`num_senders` must be an odd number'
        assert 0 != size_codebook % 2, '`size_codebook` must be an odd number'
        self.num_senders = num_senders
        self.k = 580
        self.position = np.array(position)

        # determine setup as explained in derivations of the associated paper
        self.orientation = orientation
        self.direction = np.array(orientation) * np.pi / self.k
        self.epsilon = 1.0
        self.I0 = 1.0
        self.codebook = self._generate_codebook(size_codebook)

    @staticmethod
    def _generate_codebook(
            size_codebook: int
    ) -> np.array(float):
        """ Codebook generator, generates size_codebook cosine-equidistant spaced angles in (pi, -pi)

        :param size_codebook: number of codebook elements
        :return: codebook instance (containing angles phi)
        """
        return np.round(np.pi * np.cos(np.linspace(0.1, np.pi-0.1, size_codebook)), 2)

    def intensity(self,
                  x: float | np.ndarray,
                  y: float | np.ndarray,
                  phi: float
                  ) -> float | np.ndarray:
        """ Calculates the intensity for a given angle phi

        :param x: x coordinate / array of x coordinates
        :param y: y coordinate / array of y coordinates
        :param phi: Phase difference between adjacent sources. It is related to the direction of the
                    main lobe by cos(theta)=phi/pi, where theta is the angle between d and the
                    main lobe.
        :return: intensity at specified point / array of points for given angle
        """
        # coordinate-wise distance to antenna
        offset_x, offset_y = x - self.position[0], y - self.position[1]
        # Euclidean distance to antenna (with small epsilon to avoid zero-distance)
        R = np.sqrt(offset_x**2 + offset_y**2 + self.epsilon)
        # determine xi = k * (R * direction) / |R| - phi
        xi = self.k * (offset_x*self.direction[0] + offset_y*self.direction[1]) / R - phi
        # determine denominator sin^2(xi/2)
        denom = np.sin(xi / 2) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = np.where(
                denom <= 1e-10,
                self.I0 / R ** 2,  # default intensity
                (2 / self.num_senders) ** 2 * (self.I0 / (2 * R ** 2) * np.sin(self.num_senders * xi / 2)) ** 2 / denom  # compute intensity
            )
        return intensity

    def codebook_element(self, index: int) -> float:
        """
        Return codebook element (i.e. associated angle) for specified index.
        """
        return self.codebook[index]

    def optimal_codebook_element(self,
                                 x: float | np.ndarray,
                                 y: float | np.ndarray,
                                 ):
        """ For given position(s) determine the highest-energy codebook element and respective energy

        :param x: x coordinate / array of x coordinates
        :param y: y coordinate / array of y coordinates
        :return: index of optimal codebook element(s), intensity achieved by optimal codebook element(s)
        """
        intensities = np.array([self.intensity(x, y, phi) for phi in self.codebook])
        return np.argmax(intensities, axis=0), np.max(intensities, axis=0)


class AntennaCollection:
    """
    A collection of antennas that makes up the RL environment.

           INPUT:
               size: environment size
               antennas: list of antenna positions and orientations
               size_codebook: number of codebook elements, same for all antennas

           Individual Antenna elements are constructed for the provided setup. Intensities, optimal codebook elements,
           and optimal antennas are only computed once called.
    """

    def __init__(self, size: (float, float), antennas: list[((float, float), (float, float))], size_codebook: int = 9):
        self.size = size
        self.antennas = []
        for antenna in antennas:
            if antenna[0][0] < 0 or antenna[0][0] > size[0] or antenna[0][1] < 0 or antenna[0][1] > size[1]:
                warnings.warn('Antenna is located outside of environment boundaries (might be plotted incorrectly).')
            self.antennas.append(Antenna(antenna[0], antenna[1], size_codebook=size_codebook))

    @property
    def num_antennas(self):
        return len(self.antennas)

    def get_optimal_codebook_element_for_antenna(self, index: int, x: float | np.ndarray, y: float | np.ndarray):
        """
        Get optimal codebook element and respective energy for antenna with provided index
        """
        return self.antennas[index].optimal_codebook_element(x, y)

    def get_optimal_codebook_elements(self, x: float | np.ndarray, y: float | np.ndarray):
        """
        Get optimal codebook element and respective energy for all antennas
        """
        return [antenna.optimal_codebook_element(x, y) for antenna in self.antennas]

    def get_optimal_antenna(self, x: float | np.ndarray, y: float | np.ndarray):
        """
        Get optimal antenna, respective energy, and associated codebook element.
        """

        # optimal energies achievable from each antenna
        optimal_intensities = self.get_optimal_codebook_elements(x, y)
        optimal_intensities_values = np.array([oi[1] for oi in optimal_intensities])

        # extract overall optimal energy and determine emitting antenna
        optimal_antenna = np.argmax(optimal_intensities_values, axis=0)
        optimal_intensity = np.max(optimal_intensities_values, axis=0)

        # determine optimal beam that emits optimal energy for optimal antenna
        optimal_codebook_elements = np.array([oi[0] for oi in optimal_intensities])
        optimal_beam = np.take_along_axis(optimal_codebook_elements, np.expand_dims(optimal_antenna, axis=0), axis=0)[0]

        # output in respective format, depending on if only single position or multiple positions were asked for
        if isinstance(optimal_intensity, float):
            return optimal_antenna, optimal_intensity, optimal_beam
        else:
            return list(zip(optimal_antenna, optimal_intensity, optimal_beam))

    def render(self,
               antenna_index: int,
               codebook_index: int,
               path: np.ndarray,
               info: dict,
               resolution: (int, int) = (250, 250),
               power: float = 0.2,
               eps: float = 0.5,
               show: bool = True
               ):
        """
        Helper for rendering of current environment setup.
        """

        # initialize figure and discrete grid
        x, y = np.linspace(0.0 - eps, self.size[0] + eps, resolution[0]), np.linspace(0.0 - eps, self.size[1] + eps, resolution[1])
        X, Y = np.meshgrid(x, y)

        # extract and plot intensities for selected antenna and codebook element
        I = self.antennas[antenna_index].intensity(X, Y, phi=self.antennas[antenna_index].codebook_element(codebook_index)) ** power

        if show:
            plt.figure(figsize=self.size)
            plt.pcolor(X, Y, I, cmap='Blues', vmin=0.0, vmax=1.0)
            # plt.colorbar()

            # plot environment boundaries
            plt.vlines(x=0, ymin=0, ymax=self.size[1], ls='--', color='gray')
            plt.vlines(x=self.size[0], ymin=0, ymax=self.size[1], ls='--', color='gray')
            plt.plot([0.0, self.size[0]], [0.0, 0.0], color='gray', linestyle='--')
            plt.plot([0.0, self.size[0]], [self.size[1], self.size[1]], color='gray', linestyle='--')

            # plot trajectory
            plt.plot(path[:, 0], path[:, 1], marker='X', color='g', linewidth=3)

            # plot antenna positions
            for i, antenna in enumerate(self.antennas):
                color = 'k' if i == antenna_index else 'gray'
                plt.plot([antenna.position[0] - 0.25 * antenna.orientation[0], antenna.position[0] + 0.25 * antenna.orientation[0]],
                         [antenna.position[1] - 0.25 * antenna.orientation[1], antenna.position[1] + 0.25 * antenna.orientation[1]],
                         '-', color=color, linewidth=10.0)

            plt.title('[Step {}/{}]'.format(info['step'], info['steps'])
                      + ' Antenna {}, Beam {}\n'.format(info['selected_antenna'], info['selected_codebook'])
                      + 'Energy {:.2f}/{:.2f}'.format(1000 * info['selected_energy'], 1000 * info['optimal_energy'])
                      + ' | Total {:.2f}/{:.2f} [1e-3]'.format(1000 * info['running_selected_energy'], 1000 * info['running_optimal_energy']))
            plt.show()
        return I

    def plot_ground_truth(self,
                          resolution: (int, int) = (250, 250),
                          power: float = 0.3,
                          show: bool = True
                          ):
        """
        Plot energy landscape of selected environment (more concretely energy^power for improved visibility)
        """

        # initialize figure and discrete grid
        x, y = np.linspace(0.0, self.size[0], resolution[0]), np.linspace(0.0, self.size[1], resolution[1])
        X, Y = np.meshgrid(x, y)

        # extract optimal energies (across all antennas) and store emitting antenna
        gt = self.get_optimal_antenna(np.ravel(X), np.ravel(Y))
        I_max = np.array([tmp[1] for tmp in gt]).reshape(X.shape) ** power
        Antenna_max = np.array([tmp[0] for tmp in gt]).reshape(X.shape)

        # plot optimal energies, color-coded for different emitting antennas
        picture = np.ones((*I_max.shape, 3))
        picture[:, :, 0] = Antenna_max / self.num_antennas
        picture[:, :, 2] = I_max
        from matplotlib.colors import hsv_to_rgb
        picture = hsv_to_rgb(picture)  # ensures correct color-coding of antennas

        if show:
            plt.figure(figsize=self.size)
            plt.axis('off')
            plt.imshow(picture, origin='lower')  # flip to be consistent with environment's render functionality
            plt.show()

        return picture
