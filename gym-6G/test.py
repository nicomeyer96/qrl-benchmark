# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from gym_6G.envs import BeamManagement6G

# set up the environment
env = BeamManagement6G(
    size=(6.0, 6.0),  # size of environment
    steps=250,  # number of steps in each trajectory
    antennas=4,  # randomly sample 4 antennas ...
    enforce_distance=1.5,  # ... which are at least 1.5 apart from each other
    trajectory_degree=4,  # use a trajectory with 4 support points
    render_mode='plot'  # visual rendering mode
)

# reset environment
env.reset()

# visualize underlying ground truth
env.antennas.plot_ground_truth()

# plot random trajectory
env.trajectory.plot()

# perform some steps with optimal actions and display status after these steps
for _ in range(100):
    env.step(env.optimal_action())
env.render()

# perform more steps with random action
for _ in range(100):
    env.step(env.action_space.sample())
env.render()
