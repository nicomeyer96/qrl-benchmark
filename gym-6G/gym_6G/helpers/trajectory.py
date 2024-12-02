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
import warnings
from scipy.interpolate import UnivariateSpline


class Trajectory:
    """ Object to create trajectories.
           INPUT:
               size: size of environment
               support_points: number of randomly sampled points in the rectangular (>=2), the trajectory passes
                         through. The path between two support points is a spline of degree min(4, support_points-1)
               suppress_loops: bool, if true, trajectories with loops are suppressed (although not completely avoided).

           First, it is checked that this trajectory does not leave the rectangle. If it does, the points are
           resampled until the trajectory stays in the rectangle.

           Next, the spline is re-parametrized such that it is described by
           r(t)=(x(t),y(t)) with x(0)=0, x(1)=Lx and |d/dt(r(t))|=const, i.e. constant velocity.
    """

    def __init__(self,
                 size: (float, float),
                 support_points: int = 3,
                 suppress_loops: bool = True,
                 seed: bool = None):

        self.width, self.height = size[0], size[1]
        assert support_points >= 2, '`support_points` must be >=2'

        np.random.seed(seed)

        # sample random trajectories until one is found that does not leave environment
        counter = 0
        while True:
            sxt, syt = self._get_random_trajectory(support_points, suppress_loops)
            if self._check_trajectory_boundaries(sxt, syt):
                break
            counter += 1
            if 100 == counter:
                warnings.warn('Searching for valid trajectory seems to take a long time, '
                              'consider to change environment layout.')

        # determine re-parameterized splines (ensures constant velocity over all timesteps)
        self.sxt, self.syt = self._reparametrize(sxt, syt)

    def trajectory(self, timestep: float) -> (float, float):
        """Function returns the vector (x(t), y(t))"""

        assert 0 <= timestep <= 1, '`timestep` must be between zero and 1'
        return self.sxt(timestep), self.syt(timestep)

    def plot(self, steps: int = 100):
        """Visualize trajectory within environment."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(self.width, self.height))
        plt.vlines(x=0, ymin=0, ymax=self.height, ls='--', color='k')
        plt.vlines(x=self.width, ymin=0, ymax=self.height, ls='--', color='k')
        plt.plot([0.0, self.width], [0.0, 0.0], 'k--')
        plt.plot([0.0, self.width], [self.height, self.height], 'k--')
        timesteps = np.linspace(0, 1, steps + 1)
        traj = np.array([self.trajectory(timestep) for timestep in timesteps])
        plt.plot(traj[:, 0], traj[:, 1], marker='.')
        plt.show()

    @staticmethod
    def _reparametrize(sxt: UnivariateSpline, syt: UnivariateSpline, resolution=1000):
        """
        Given an x trajectory sxt and y trajectory syt parametrized by t and in spline form,
        the function re-parametrizes the spline such that the absolute value of the vector velocity
        stays constant. `resolution` is the number of points, the spline is evaluated at during re-parametrization.
        """

        # calculate velocities
        svxt, svyt = sxt.derivative(), syt.derivative()

        # calculate the absolute value of the velocity as spline function and integrate it
        t = np.linspace(0, 1, resolution)
        K = UnivariateSpline(t, np.sqrt(svxt(t) ** 2 + svyt(t) ** 2), s=0).antiderivative()

        # solve the differential equation with correct free parameters
        taut = (K(t)-K(0))/(K(1)-K(0))

        # return re-parametrized splines
        return UnivariateSpline(taut, sxt(t), s=0), UnivariateSpline(taut, syt(t), s=0)

    def _check_trajectory_boundaries(self, sxt: UnivariateSpline, syt: UnivariateSpline, resolution=1000) -> bool:
        """Checks if the spline stays in the rectangle [0,width]x[0,height]"""

        # evaluate at multiple uniformly distributed points
        x, y = sxt(np.linspace(0, 1, resolution)), syt(np.linspace(0, 1, resolution))
        return np.all(0 <= x) and np.all(x <= self.width) and np.all(0 <= y) and np.all(y <= self.height)

    def _get_random_trajectory(self, support_points: int, suppress_loops: bool) -> (UnivariateSpline, UnivariateSpline):
        """
        Samples a random trajectory, a spline function through n_points points.
        Note that the spline function can be > Ly and < 0 for some values of x.
        """

        # randomly sample x-coordinates of support points
        points_x = np.random.uniform(0, self.width, support_points)
        # force trajectory to start at left side and finish at right side
        points_x[0], points_x[-1] = 0, self.width

        # sort x-coordinates in ascending order to suppress loops (no hard constraint, but reduces likelihood)
        if suppress_loops:
            points_x = np.sort(points_x)

        # randomly sample y-coordinates of support points
        points_y = np.random.uniform(0, self.height, support_points)

        # choose order depending on number of support points, upper bounded by 4 (in principle also higher degrees
        # would be possible, but want to ensure somewhat smooth trajectories to mimic human movement)
        order = min(4, support_points - 1)

        # set up splines (will be re-parameterized later)
        sxt = UnivariateSpline(np.linspace(0, 1, support_points), points_x, s=0, k=order)
        syt = UnivariateSpline(np.linspace(0, 1, support_points), points_y, s=0, k=order)
        return sxt, syt
